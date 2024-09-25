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
    
    # If certain tokens are found, it goes to PCI
    pci_tokens = {'accountnum', 'creditcardnumber', 'idcardnumber'}
    if any(token in results["PII"] for token in pci_tokens):
        pci_results = pci_model(text)
        results["PCI"] = [r['word'] for r in pci_results]

    # Step 2: PHI detection if no PCI-related tokens found
    if not results["PCI"]:
        phi_results = phi_model(text)
        results["PHI"] = [r['word'] for r in phi_results]
    
    return results

# Function to highlight text with color coding
def highlight_text(text, tokens):
    for category, words in tokens.items():
        for word in words:
            # Create regex pattern to find the word and highlight it
            pattern = re.escape(word)
            if category == "PII":
                text = re.sub(r'\b' + pattern + r'\b', f'<mark style="background-color: yellow;">{word}</mark>', text)
            elif category == "PCI":
                text = re.sub(r'\b' + pattern + r'\b', f'<mark style="background-color: red;">{word}</mark>', text)
            elif category == "PHI":
                text = re.sub(r'\b' + pattern + r'\b', f'<mark style="background-color: green;">{word}</mark>', text)
    return text

# Streamlit UI
st.title("Real-Time Text Editor with NER Detection")

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
