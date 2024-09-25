import streamlit as st
import pdfplumber  # Use PDFPlumber for PDF extraction
import docx  # python-docx for Word file extraction
import os
from transformers import pipeline

# Load Hugging Face models
pii_model = pipeline("ner", model="iiiorg/piiranha-v1-detect-personal-information")
pci_model = pipeline("ner", model="lakshyakh93/deberta_finetuned_pii")
phi_model = pipeline("ner", model="obi/deid_roberta_i2b2")
medical_ner_model = pipeline("ner", model="blaze999/Medical-NER")

# Function to extract text from different file formats
def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1]
    text = ""
    
    if ext == '.pdf':
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + '\n'
    elif ext == '.docx':
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + '\n'
    elif ext in ['.txt', '.csv']:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    return text

# Function to process the document through the NER pipeline
def process_ner_pipeline(text):
    # Step 1: PII detection
    pii_results = pii_model(text)
    pii_tokens = [r['word'] for r in pii_results]
    
    # If certain tokens are found, it goes to PCI
    pci_tokens = {'accountnum', 'creditcardnumber', 'idcardnumber'}
    if any(token in pii_tokens for token in pci_tokens):
        pci_results = pci_model(text)
        return 'PCI', pci_results

    # Step 2: PHI detection if no PCI-related tokens found
    phi_results = phi_model(text)
    if phi_results:
        # If PHI is detected, go to medical NER model
        medical_results = medical_ner_model(text)
        return 'Medical NER', medical_results
    
    # Default if no PHI is detected
    return 'PHI', phi_results

# Streamlit UI
st.title("NER Pipeline Testing")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt', 'csv'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract and display text
    extracted_text = extract_text_from_file(uploaded_file.name)
    st.write("Extracted Text:")
    st.write(extracted_text)

    # Run NER pipeline
    category, ner_results = process_ner_pipeline(extracted_text)
    st.write(f"Document Category: {category}")
    
    st.write("Named Entities:")
    for result in ner_results:
        st.write(f"{result['word']} - {result['entity']}")
