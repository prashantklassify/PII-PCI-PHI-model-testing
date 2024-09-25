import streamlit as st
from transformers import pipeline
import re

# Load Hugging Face models
pii_model = pipeline("ner", model="iiiorg/piiranha-v1-detect-personal-information")
pci_model = pipeline("ner", model="lakshyakh93/deberta_finetuned_pii")
phi_model = pipeline("ner", model="obi/deid_roberta_i2b2")
medical_ner_model = pipeline("ner", model="blaze999/Medical-NER")

# Common tokens to filter out from results
common_tokens = set([
    'accountnum', 'creditcardnumber', 'idcardnumber',
    'email', 'telephonenumber', 'dateofbirth', 'surname',
    'givenname', 'buildingnum', 'street', 'city', 'zipcode'
])

# Function to process the document through the NER pipeline
def process_ner_pipeline(text):
    results = {
        "PII": [],
        "PCI": [],
        "PHI": [],
        "Medical NER": []
    }

    # Step 1: PII detection
    pii_results = pii_model(text)
    results["PII"] = [r['word'] for r in pii_results]

    # Step 2: PCI detection
    pci_related_tokens = {
        'accountnum', 'creditcardnumber', 'idcardnumber', 
        'email', 'telephonenumber'
    }

    # Check if any PII tokens should be classified as PCI
    for token in results["PII"]:
        if token.lower() in pci_related_tokens:
            results["PCI"].append(token)

    # Step 3: PHI detection
    phi_results = phi_model(text)
    results["PHI"] = [r['word'] for r in phi_results]

    # Filter out common tokens from results
    results["PII"] = [token for token in results["PII"] if token.lower() not in common_tokens]
    results["PCI"] = [token for token in results["PCI"] if token.lower() not in common_tokens]
    results["PHI"] = [token for token in results["PHI"] if token.lower() not in common_tokens]

    # Prioritize PCI if all models return common tokens
    if results["PCI"]:
        results["PII"] = []  # Clear PII if PCI tokens are found
        results["PHI"] = []  # Clear PHI if PCI tokens are found
    else:
        # Analyze Medical NER if conditions are met
        if len(results["PHI"]) > len(results["PII"]) and len(results["PHI"]) > len(results["PCI"]):
            medical_results = medical_ner_model(text)
            results["Medical NER"] = [r['word'] for r in medical_results]

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
    # Highlight Medical NER tokens
    for word in tokens["Medical NER"]:
        highlighted_text = re.sub(r'\b' + re.escape(word) + r'\b', 
                                   f'<mark style="background-color: blue;">{word}</mark>', highlighted_text)
    return highlighted_text

# Streamlit UI
st.title("Dynamic NER Detection in Text")

# Text area for user input
user_input = st.text_area("Enter text here:", height=300)

if st.button("Analyze"):
    # Clean input text to prevent unwanted characters
    cleaned_input = re.sub(r'[^\w\s.,-]', '', user_input)  # Keep only words, spaces, commas, periods, and hyphens

    # Run NER pipeline
    tokens = process_ner_pipeline(cleaned_input)
    
    # Highlight detected tokens in the text
    highlighted_text = highlight_text(cleaned_input, tokens)
    
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
st.sidebar.write("**Medical NER:**")
st.sidebar.write("Medical-related information, conditions, etc.")
