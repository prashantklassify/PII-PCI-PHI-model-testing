import streamlit as st
from transformers import pipeline
import pandas as pd

# Define the models for NER
models = {
    "PII": "iiiorg/piiranha-v1-detect-personal-information",
    "PCI": "lakshyakh93/deberta_finetuned_pii",
    "PHI": "obi/deid_roberta_i2b2",
    "Medical NER": "blaze999/Medical-NER"
}

# Load the models
model_pii = pipeline("token-classification", model=models["PII"])
model_pci = pipeline("token-classification", model=models["PCI"])
model_phi = pipeline("token-classification", model=models["PHI"])
model_medical = pipeline("token-classification", model=models["Medical NER"])

# Accepted labels for each model
accepted_pii_labels = set()  # Accept all categories under PII
accepted_pci_labels = {
    "JOBDESCRIPTOR", "JOBTITLE", "JOBAREA", "BITCOINADDRESS", "ETHEREUMADDRESS",
    "ACCOUNTNAME", "ACCOUNTNUMBER", "IBAN", "BIC", "IPV4", "IPV6",
    "CREDITCARDNUMBER", "VEHICLEVIN", "AMOUNT", "CURRENCY", "PASSWORD",
    "PHONEIMEI", "CURRENCYSYMBOL", "CURRENCYNAME", "CURRENCYCODE",
    "LITECOINADDRESS", "MAC", "CREDITCARDISSUER", "CREDITCARDCVV",
    "NEARBYGPSCOORDINATE", "SEXTYPE"
}
accepted_phi_labels = {"staff", "HOSP", "AGE"}
accepted_medical_labels = {
    "BIOLOGICAL_ATTRIBUTE", "BIOLOGICAL_STRUCTURE", "CLINICAL_EVENT",
    "DISEASE_DISORDER", "DOSAGE", "FAMILY_HISTORY", "LAB_VALUE", "MASS",
    "MEDICATION", "OUTCOME", "SIGN_SYMPTOM", "THERAPUTIC_PROCEDURE"
}

# Helper function to clean tokens and merge adjacent entities
def clean_and_merge_tokens(entities):
    cleaned_entities = []
    current_entity = None

    for entity in entities:
        entity_type = entity['entity'].split('-')[-1]
        token = entity['word'].replace('▁', '').replace('Ġ', '')

        if current_entity and token == current_entity['word'] and entity_type == current_entity['entity_type']:
            current_entity['score'] = max(current_entity['score'], entity['score'])
        else:
            if current_entity:
                cleaned_entities.append(current_entity)
            current_entity = {'word': token, 'entity_type': entity_type, 'score': entity['score']}

    if current_entity:
        cleaned_entities.append(current_entity)

    return cleaned_entities

# Function to filter and prioritize NER results
def filter_and_prioritize_entities(*args):
    merged_entities = {}
    for entities in args:
        for entity in entities:
            if entity['word'] not in merged_entities:
                merged_entities[entity['word']] = entity
    return list(merged_entities.values())

# Custom NER pipeline
def custom_pipeline(text):
    pii_results = clean_and_merge_tokens(model_pii(text))
    pci_results = clean_and_merge_tokens(model_pci(text))
    phi_results = clean_and_merge_tokens(model_phi(text))
    medical_results = clean_and_merge_tokens(model_medical(text))

    # Prioritize entities
    final_results = filter_and_prioritize_entities(pii_results, phi_results, pci_results, medical_results)
    return final_results

# Streamlit App Layout
st.title("Named Entity Recognition (NER) with Prioritization")

# User input for text
text = st.text_area("Enter text for NER processing", "Sample text goes here...")

# Process text button
if st.button("Run NER Models"):
    st.subheader("NER Results")
    results = custom_pipeline(text)

    # Display results
    if results:
        results_table = pd.DataFrame([{
            "Entity": res['word'],
            "Entity Type": res['entity_type'],
            "Confidence": f"{res['score'] * 100:.2f}%"
        } for res in results])
        st.table(results_table)
    else:
        st.write("No entities detected.")
