import streamlit as st
from transformers import pipeline

# Load NER models from Hugging Face
pii_model = pipeline("ner", model="path/to/pii-model")
phi_model = pipeline("ner", model="path/to/phi-model")
pci_model = pipeline("ner", model="path/to/pci-model")
medical_ner_model = pipeline("ner", model="path/to/medical-ner-model")

# Function to safely extract entity groups, handling missing keys
def get_entity_group(token):
    return token.get('entity_group', 'UNKNOWN')

# Function to run all NER models on the input text
def run_ner_models(text):
    # Run PII, PHI, and PCI models
    pii_results = pii_model(text)
    phi_results = phi_model(text)
    pci_results = pci_model(text)
    
    # Check if there are significant PHI tokens to run medical NER
    significant_phi_tokens = [token for token in phi_results if get_entity_group(token) in ['PATIENT', 'LOC', 'ID']]
    if len(significant_phi_tokens) > 5:
        medical_results = medical_ner_model(text)
    else:
        medical_results = []

    return pii_results, phi_results, pci_results, medical_results

# Function to resolve token overlap by prioritizing PCI > PII > PHI
def resolve_conflicts(pii, phi, pci, medical):
    resolved_tokens = {}
    
    # Add PCI tokens first (highest priority)
    for token in pci:
        resolved_tokens[(token['start'], token['end'])] = token
    
    # Add PII tokens if they don't overlap with PCI tokens
    for token in pii:
        if (token['start'], token['end']) not in resolved_tokens:
            resolved_tokens[(token['start'], token['end'])] = token
    
    # Add PHI tokens if they don't overlap with PCI or PII tokens
    for token in phi:
        if (token['start'], token['end']) not in resolved_tokens:
            resolved_tokens[(token['start'], token['end'])] = token
    
    # Add Medical tokens if they don't overlap with other tokens
    for token in medical:
        if (token['start'], token['end']) not in resolved_tokens:
            resolved_tokens[(token['start'], token['end'])] = token
    
    # Return the final resolved list of tokens
    return list(resolved_tokens.values())

# Function to mark tokens in the text dynamically with different colors for each category
def mark_text(text, tokens, label_colors):
    for token in tokens:
        label = get_entity_group(token)
        word = token['word']
        color = label_colors.get(label, "#ffffff")  # Default color if no label found
        text = text.replace(word, f'<mark style="background-color:{color};">{word} ({label})</mark>')
    return text

# Streamlit app UI
st.title("Multi-NER Token Classifier")
st.write("Enter text to classify tokens by categories like PII, PHI, PCI, and Medical.")

# Text input area
input_text = st.text_area("Enter text here:", height=200)

# Label colors for each category
label_colors = {
    "ACCOUNTNUM": "#ffcccb",  # Red for PCI
    "CREDITCARDNUMBER": "#ffcccb",  # Red for PCI
    "CITY": "#ffff99",  # Yellow for PII
    "PATIENT": "#b3e5fc",  # Light Blue for PHI
    "EMAIL": "#ffff99",  # Yellow for PII
    "USERNAME": "#d1c4e9",  # Light Purple for Medical NER
}

# Run NER and display results
if input_text:
    st.subheader("NER Results")

    # Run all NER models on the input text
    try:
        pii_results, phi_results, pci_results, medical_results = run_ner_models(input_text)

        # Resolve token conflicts
        resolved_tokens = resolve_conflicts(pii_results, phi_results, pci_results, medical_results)

        # Mark the text with detected entities
        highlighted_text = mark_text(input_text, resolved_tokens, label_colors)

        # Display the marked text in the Streamlit app
        st.markdown(highlighted_text, unsafe_allow_html=True)
    except KeyError as e:
        st.error(f"An error occurred while processing: {e}")
