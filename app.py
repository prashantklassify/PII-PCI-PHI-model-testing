import streamlit as st
import os
import zipfile
import tempfile
import textract  # To extract text from documents
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# Define NER pipelines
ner_model_pii = pipeline("token-classification", model="iiiorg/piiranha-v1-detect-personal-information")
ner_model_pci = pipeline("token-classification", model="lakshyakh93/deberta_finetuned_pii")
ner_model_phi = pipeline("token-classification", model="obi/deid_roberta_i2b2")
ner_model_medical = pipeline("token-classification", model="blaze999/Medical-NER")

# Function to extract text from different document types
def extract_text_from_documents(doc_path):
    try:
        text = textract.process(doc_path).decode('utf-8')
    except Exception as e:
        print(f"Error extracting text from {doc_path}: {e}")
        return None
    return text

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
def classify_documents_svm(embeddings, possible_classes):
    # Simulate binary classification using text matches for clustering
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    clf.fit(embeddings, possible_classes)

    # Predict on the same data to simulate classification (usually you'd have test data)
    predictions = clf.predict(embeddings)
    return predictions

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

# Streamlit interface for uploading the folder and displaying results
def main():
    st.title("Document Clustering and Sensitive Information Extraction")

    # Step 1: Upload a ZIP file containing documents
    uploaded_zip = st.file_uploader("Upload a zip file containing documents", type="zip")

    if uploaded_zip is not None:
        # Create a temporary directory to extract the uploaded zip file
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, uploaded_zip.name)

            # Save the uploaded zip file
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())

            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Get list of all documents
            doc_files = [os.path.join(temp_dir, file) for file in os.listdir(temp_dir)]

            # Step 2: Extract text from each document
            st.write("Extracting text from documents...")
            documents = []
            for file_path in doc_files:
                text = extract_text_from_documents(file_path)
                if text:
                    documents.append((file_path, text))

            if not documents:
                st.write("No valid documents found.")
                return

            # Step 3: Generate embeddings for text documents
            st.write("Generating embeddings for documents...")
            texts = [doc[1] for doc in documents]  # Extract text for embeddings
            embeddings = get_text_embeddings(texts)

            # Step 4: Use SVM for classification
            st.write("Classifying documents using SVM...")
            possible_classes = ['Resume', 'Invoice', 'Report', 'Offer Letter']
            classifications = classify_documents_svm(embeddings, possible_classes)

            # Step 5: Apply NER to classified documents
            st.write("Applying Named Entity Recognition (NER) to extract sensitive information...")

            # Show the NER results document-wise
            for i, (file_path, doc_text) in enumerate(documents):
                classification = classifications[i]
                st.subheader(f"Document: {os.path.basename(file_path)} (Classified as: {classification})")

                # Apply NER based on the classification
                ner_results = apply_ner(doc_text, classification)
                
                # Display NER results
                if ner_results:
                    for entity in ner_results:
                        st.write(f"Entity: {entity['word']}, Label: {entity['entity']}, Confidence: {entity['score']:.2f}")
                else:
                    st.write("No sensitive information found.")

if __name__ == "__main__":
    main()
