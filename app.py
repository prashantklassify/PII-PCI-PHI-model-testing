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

# Define accepted tokens
accepted_pii_labels = {"ALL CATEGORIES UNDER PII"}  # Specify all accepted PII labels if needed
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
    "DISEASE_DISORDER", "DOSAGE", "FAMILY_HISTORY", "LAB_VALUE",
    "MASS", "MEDICATION", "OUTCOME", "SIGN_SYMPTOM", "THERAPUTIC_PROCEDURE"
}

# Function to clean tokens
def clean_token(token):
    """Clean token by removing unwanted characters like '▁' and 'Ġ'."""
    return token.replace("▁", "").replace("Ġ", "")

# Function to format NER results into a DataFrame
def format_ner_results_as_table(ner_results):
    entity_list = []
    
    for entity in ner_results:
        token = clean_token(entity['word'])
        confidence = entity['score'] * 100  # Convert to percentage
        entity_type = entity['entity'].split("-")[-1]  # Get entity type
        
        # Append to the list as a dictionary
        entity_list.append({
            "Entity": token,
            "Entity Type": entity_type,
            "Confidence (%)": f"{confidence:.2f}"
        })
    
    # Convert list to DataFrame for tabular display
    return pd.DataFrame(entity_list)

# Function to filter predictions by confidence threshold
def filter_by_confidence(predictions, threshold=0.5):
    """Filter predictions to only include those with a confidence above the threshold."""
    return [prediction for prediction in predictions if prediction['score'] > threshold]

# Custom NER pipeline function
def custom_pipeline(text):
    # Run the text through the PII model
    pii_results = model_pii(text)
    filtered_pii_results = [res for res in pii_results if res['label'] in accepted_pii_labels]

    # If PII labels are detected, process with the PCI model
    if filtered_pii_results:
        pci_results = model_pci(text)
        filtered_pci_results = [res for res in pci_results if res['label'] in accepted_pci_labels]
        return filtered_pci_results

    # If no PII labels found, process with the PHI model
    phi_results = model_phi(text)
    filtered_phi_results = [res for res in phi_results if res['label'] in accepted_phi_labels]

    # If PHI results have unique tokens, proceed to the Medical model
    if filtered_phi_results:
        medical_results = model_medical(text)
        filtered_medical_results = [res for res in medical_results if res['label'] in accepted_medical_labels]
        return filtered_medical_results
    
    return filtered_phi_results  # Return PHI results if no other conditions are met

# Streamlit App Layout
st.title("Named Entity Recognition (NER) Streamlit App")

# User input for text
text = st.text_area("Enter text for NER processing", "Patient Brijesh Kumar admitted in the room no 101 in glacier hospital has blood pressure over 140 and heart rate of 83bpm. The patient wants to avail no txn cost from insurance provider. Insurance number of FHZPB1650J and rest of the payment will be done by card number 4111 1111 1111 1111.")

# Confidence threshold input
confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

# Button to process the text
if st.button("Run NER Models"):
    st.subheader("Custom NER Pipeline Results")
    
    # Run the custom pipeline
    predictions = custom_pipeline(text)

    # Filter predictions by confidence
    filtered_predictions = filter_by_confidence(predictions, confidence_threshold)

    # Convert NER results to a table format
    if filtered_predictions:
        ner_table = format_ner_results_as_table(filtered_predictions)
        st.table(ner_table)
    else:
        st.write("No entities detected above the confidence threshold.")
