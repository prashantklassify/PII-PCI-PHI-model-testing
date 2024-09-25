import streamlit as st
from transformers import pipeline
import re

# Load Hugging Face models
pii_model = pipeline("ner", model="iiiorg/piiranha-v1-detect-personal-information")
pci_model = pipeline("ner", model="lakshyakh93/deberta_finetuned_pii")
phi_model = pipeline("ner", model="obi/deid_roberta_i2b2")

# Function to process the document through the NER pipeline
def process_ner_pipeline(text):
    results = {
        "PII": [],
        "PCI": [],
        "PHI": []
    }
    
    # Step 1: PII detection
    pii_results = pii_model(text)
    results["PII"] = [r['word'] for r in pii_results]
    
    # Define PCI-related tokens
    pci_related_tokens = {
        'accountnum': 'Account-related information',
        'creditcardnumber': 'Financial data',
        'idcardnumber': 'Account-related information',
        'email': 'Contact information',
        'telephonenumber': 'Contact information',
        # Add more as needed
    }
    
    # Check if any PII tokens should be classified as PCI
    for token in results["PII"]:
        if token.lower() in pci_related_tokens:
            results["PCI"].append(token)
    
    # Step 2: PHI detection if no PCI-related tokens found
    if not results["PCI"]:
        phi_results = phi_model(text)
        results["PHI"] = [r['word'] for r in phi_results]
    
    return results

# Function to highlight text with color coding
def highlight_text(text, tokens):
    highlighted_text = text
    # Highlight PII tokens
    for word in tokens["PII"]:
        highlighted_text = re.sub(r'\b' + re.escape(word) + r'\b', 
                                   f'<mark style="background-color: yellow;">{word}</mark>', highlighted_text)
    # Highlight PCI tokens
    for word in tokens["PCI"]:
        highlighted_text = re.sub(r'\b' + re.escape(word) + r'\b', 
                                   f'<mark style="background-color: red;">{word}</mark>', highlighted_text)
    # Highlight PHI tokens
    for word in tokens["PHI"]:
        highlighted_text = re.sub(r'\b' + re.escape(word) + r'\b', 
                                   f'<mark style="background-color: green;">{word}</mark>', highlighted_text)
    return highlighted_text

# Streamlit UI
st.title("Dynamic NER Detection in Text")

# Text area for user input
user_input = st.text_area("Enter text here:", height=300)

if st.button("Analyze"):
    # Run NER pipeline
    tokens = process_ner_pipeline(user_input)
    
    # Highlight detected tokens in the text
    highlighted_text = highlight_text(user_input, tokens)
    
    # Display the highlighted text
    st.markdown("### Analyzed Text:")
    st.markdown(highlighted_text, unsafe_allow_html=True)

    # Display categories found
    st.write("### Detected Categories:")
    for category, words in tokens.items():
        if words:
            st.write(f"{category}: {', '.join(words)}")

# Information on token categories for user reference
st.sidebar.title("Token Categories")
st.sidebar.write("**PII (Personally Identifiable Information):**")
st.sidebar.write("Names, emails, phone numbers, etc.")
st.sidebar.write("**PCI (Payment Card Information):**")
st.sidebar.write("Credit card numbers, account numbers, etc.")
st.sidebar.write("**PHI (Protected Health Information):**")
st.sidebar.write("Patient data, health records, etc.")
