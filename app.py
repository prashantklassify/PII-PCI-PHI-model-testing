import streamlit as st
from transformers import pipeline
import pandas as pd
import re

# Load the models for NER
models = {
    "PII": "iiiorg/piiranha-v1-detect-personal-information",
    "PCI": "lakshyakh93/deberta_finetuned_pii",
    "PHI": "obi/deid_roberta_i2b2",
    "Medical NER": "blaze999/Medical-NER"
}

model_pii = pipeline("token-classification", model=models["PII"])
model_pci = pipeline("token-classification", model=models["PCI"])
model_phi = pipeline("token-classification", model=models["PHI"])
model_medical = pipeline("token-classification", model=models["Medical NER"])

# Define the accepted labels for each category
accepted_pii_labels = set()  # Accept all categories under PII
accepted_pci_labels = {"JOBDESCRIPTOR", "JOBTITLE", "JOBAREA", "BITCOINADDRESS", "ETHEREUMADDRESS",
                       "ACCOUNTNAME", "ACCOUNTNUMBER", "IBAN", "BIC", "IPV4", "IPV6",
                       "CREDITCARDNUMBER", "VEHICLEVIN", "AMOUNT", "CURRENCY", "PASSWORD",
                       "PHONEIMEI", "CURRENCYSYMBOL", "CURRENCYNAME", "CURRENCYCODE",
                       "LITECOINADDRESS", "MAC", "CREDITCARDISSUER", "CREDITCARDCVV",
                       "NEARBYGPSCOORDINATE", "SEXTYPE"}

accepted_phi_labels = {"staff", "HOSP", "AGE"}

accepted_medical_labels = {"BIOLOGICAL_ATTRIBUTE", "BIOLOGICAL_STRUCTURE", "CLINICAL_EVENT",
                           "DISEASE_DISORDER", "DOSAGE", "FAMILY_HISTORY", "LAB_VALUE", "MASS",
                           "MEDICATION", "OUTCOME", "SIGN_SYMPTOM", "THERAPUTIC_PROCEDURE"}

# Define a function to clean and merge tokens
def clean_and_merge_tokens(entities):
    cleaned_entities = []
    for entity in entities:
        # Clean token
        token = entity['word'].replace("▁", "").replace("Ġ", "")
        entity['word'] = token
        
        # Merge contiguous entities of the same type
        if cleaned_entities and cleaned_entities[-1]['entity'] == entity['entity'] \
                and cleaned_entities[-1]['end'] == entity['start']:
            cleaned_entities[-1]['word'] += token
            cleaned_entities[-1]['end'] = entity['end']
            cleaned_entities[-1]['score'] = max(cleaned_entities[-1]['score'], entity['score'])  # Take the max confidence
        else:
            cleaned_entities.append(entity)
    return cleaned_entities

# Custom NER pipeline function
def custom_pipeline(text):
    # Run the text through the PII model
    pii_results = model_pii(text)
    # Accepting all categories for PII as no specific labels were mentioned
    pii_results = clean_and_merge_tokens(pii_results)

    # Run the text through the PHI model
    phi_results = model_phi(text)
    phi_results = [entity for entity in phi_results if entity['entity'].split("-")[-1] in accepted_phi_labels]
    phi_results = clean_and_merge_tokens(phi_results)

    # Run the text through the PCI model
    pci_results = model_pci(text)
    pci_results = [entity for entity in pci_results if entity['entity'].split("-")[-1] in accepted_pci_labels]
    pci_results = clean_and_merge_tokens(pci_results)

    # Run Medical NER model independently on the original text
    medical_results = model_medical(text)
    medical_results = [entity for entity in medical_results if entity['entity'].split("-")[-1] in accepted_medical_labels]
    medical_results = clean_and_merge_tokens(medical_results)

    # Combine all results
    combined_results = pii_results + phi_results + pci_results + medical_results

    return combined_results

# Function to display results in a table format
def display_results(results):
    table_data = [{
        "Entity": result['word'],
        "Entity Type": result['entity'].split("-")[-1],
        "Confidence (%)": f"{result['score'] * 100:.2f}"
    } for result in results]
    return pd.DataFrame(table_data)

# Streamlit app layout
st.title("Enhanced Named Entity Recognition (NER) Streamlit App")

# User input for text
text = st.text_area("Enter text for NER processing", "Patient John Doe admitted to the hospital has a heart rate of 80bpm and no known allergies. He is taking medication Metformin.")

# Button to process the text
if st.button("Run NER Models"):
    st.subheader("Custom NER Pipeline Results")
    
    # Run the custom pipeline
    predictions = custom_pipeline(text)

    # Display results in a table
    if predictions:
        results_table = display_results(predictions)
        st.table(results_table)
    else:
        st.write("No entities detected.")
