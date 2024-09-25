import streamlit as st
from transformers import pipeline
from itertools import chain
import spacy

# Load NER models from Hugging Face
pii_ner = pipeline("ner", model="your-pii-model")
phi_ner = pipeline("ner", model="your-phi-model")
hipaa_ner = pipeline("ner", model="your-hipaa-model")
medical_ner = pipeline("ner", model="your-medical-model")

# Categories for classification
PCI_CATEGORIES = [
    "Account name", "Account number", "Transaction amounts", "BIC", "IBAN",
    "Credit card number", "CVV", "Email", "SSN", "MAC addresses", "IP addresses"
]
PHI_CATEGORIES = ["DATE", "STAFF", "AGE", "LOC", "PATIENT", "PHONE", "ID", "EMAIL"]

# Function to highlight text
def highlight_text(text, spans):
    for span in sorted(spans, key=lambda x: x['start'], reverse=True):
        label = span['entity']
        category = span['category']
        text = text[:span['start']] + f"**[{label}]**({category})" + text[span['end']:]
    return text

# Define a function to resolve token overlap by prioritizing
def resolve_overlap(spans):
    spans = sorted(spans, key=lambda x: (x['start'], -x['end']))  # Sort by start and end positions
    resolved_spans = []
    current_end = -1
    
    for span in spans:
        if span['start'] >= current_end:  # No overlap
            resolved_spans.append(span)
            current_end = span['end']
        else:
            # If there's an overlap, prioritize based on model or category (can be customized)
            pass
    
    return resolved_spans

# Define function to map tokens to the correct category (e.g., PCI/PHI/PII)
def classify_tokens(text, pii_entities, phi_entities, hipaa_entities, medical_entities):
    all_entities = []

    # Combine results from all NER models
    for entity in chain(pii_entities, phi_entities, hipaa_entities, medical_entities):
        category = "Unknown"
        if entity['entity'] in PCI_CATEGORIES:
            category = "PCI"
        elif entity['entity'] in PHI_CATEGORIES:
            category = "PHI"
        # Add more rules as needed for HIPAA and medical
        all_entities.append({
            'start': entity['start'],
            'end': entity['end'],
            'entity': entity['entity'],
            'category': category
        })

    # Resolve overlapping tokens
    resolved_entities = resolve_overlap(all_entities)
    return resolved_entities

# Streamlit interface
st.title("NER Text Classifier")

# Text input
text = st.text_area("Enter your text here:")

if text:
    # Run NER models
    pii_entities = pii_ner(text)
    phi_entities = phi_ner(text)
    hipaa_entities = hipaa_ner(text)
    medical_entities = medical_ner(text) if len(phi_entities) > 5 else []  # Run medical NER only if needed

    # Classify and annotate the text
    entities = classify_tokens(text, pii_entities, phi_entities, hipaa_entities, medical_entities)
    highlighted_text = highlight_text(text, entities)

    # Display annotated text
    st.markdown(highlighted_text)
