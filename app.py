import os
import textract  # To extract text from documents
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import pytesseract  # For OCR when dealing with images
from PIL import Image  # To open images if necessary
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Define NER pipelines
ner_model_pii = pipeline("token-classification", model="iiiorg/piiranha-v1-detect-personal-information")
ner_model_pci = pipeline("token-classification", model="lakshyakh93/deberta_finetuned_pii")
ner_model_phi = pipeline("token-classification", model="obi/deid_roberta_i2b2")
ner_model_medical = pipeline("token-classification", model="blaze999/Medical-NER")

# Function to extract text from different document types
def extract_text_from_documents(doc_path):
    if doc_path.endswith(('.png', '.jpg', '.jpeg')):
        # Perform OCR on image files
        text = pytesseract.image_to_string(Image.open(doc_path))
    else:
        try:
            text = textract.process(doc_path).decode('utf-8')
        except Exception as e:
            print(f"Error extracting text from {doc_path}: {e}")
            return None
    return text

# Simulated LLM prompt-based classification using fixed classes
def classify_document_with_llm(text, possible_classes):
    # A placeholder function to simulate LLM interaction
    for doc_class in possible_classes:
        # Simulating yes/no prompts for classification
        if doc_class.lower() in text.lower():
            return doc_class
    return "Unknown"

# Function to generate embeddings using HuggingFace models
def get_text_embeddings(text_list, model_name="sentence-transformers/all-mpnet-base-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    return np.array(embeddings)

# Function to perform document classification using SVM
def classify_documents_svm(embeddings, labels):
    # Using SVM for classification
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    clf.fit(embeddings, labels)

    # Predict on the same data to simulate classification (usually you'd have test data)
    predictions = clf.predict(embeddings)
    prediction_probabilities = clf.predict_proba(embeddings)

    # Generate classification report
    print("Classification report:")
    print(classification_report(labels, predictions))

    return predictions, prediction_probabilities

# Function to apply NER based on document classification
def apply_ner(text, classification):
    if classification == 'Resume':
        ner_results = ner_model_pii(text)
    elif classification == 'Invoice':
        ner_results = ner_model_pci(text)
    elif classification == 'Report':
        ner_results = ner_model_phi(text)
    elif classification == 'Offer Letter':
        ner_results = ner_model_medical(text)
    else:
        ner_results = []  # Default empty for unknown classifications

    return ner_results

# Main function to run the document processing and classification
def main(doc_folder_path, possible_classes):
    documents = []
    labels = []

    # Step 1: Extract text from all documents
    for file_name in os.listdir(doc_folder_path):
        file_path = os.path.join(doc_folder_path, file_name)
        text = extract_text_from_documents(file_path)
        if text:
            documents.append(text)

            # Step 2: Classify the document using LLM (simulated prompt approach)
            doc_class = classify_document_with_llm(text, possible_classes)
            labels.append(doc_class)

    if not documents:
        print("No valid documents found.")
        return

    # Step 3: Generate embeddings for text documents
    print("Generating embeddings...")
    embeddings = get_text_embeddings(documents)

    # Step 4: Use SVM for classification
    print("Classifying documents with SVM...")
    predictions, _ = classify_documents_svm(embeddings, labels)

    # Step 5: Apply NER based on classification results
    for i, doc_text in enumerate(documents):
        classification = predictions[i]
        print(f"\nNER results for document classified as '{classification}':")
        ner_results = apply_ner(doc_text, classification)
        for entity in ner_results:
            print(f"Entity: {entity['word']}, Label: {entity['entity']}, Confidence: {entity['score']:.2f}")

if __name__ == "__main__":
    # Folder containing documents
    document_folder = "path_to_your_documents_folder"

    # List of possible classes for document classification
    possible_classes = ['Resume', 'Invoice', 'CV', 'Report', 'Offer Letter']

    main(document_folder, possible_classes)
