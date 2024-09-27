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
accepted_pii_labels = {"ACCOUNTNUM", "BUILDINGNUM", "CITY", "CREDITCARDNUMBER", "DATEOFBIRTH", "DRIVERLICENSENUM", "EMAIL", "GIVENNAME", "IDCARDNUM", "PASSWORD", "SOCIALNUM", "STREET", "SURNAME", "TAXNUM", "TELEPHONENUM", "USERNAME", "ZIPCODE"} 
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

# Function to check overlapping tokens
def is_overlapping(token1, token2):
    """Check if two tokens (text spans) are overlapping."""
    start1, end1 = token1['start'], token1['end']
    start2, end2 = token2['start'], token2['end']
    return not (end1 < start2 or end2 < start1)

# Function to resolve overlapping tokens by keeping the highest confidence one
def resolve_overlapping_tokens(entities):
    """Resolve overlapping tokens by keeping the one with the highest confidence."""
    resolved_entities = []

    for entity in entities:
        # Check if there's an overlapping token already in the resolved list
        overlapping_entity = None
        for res in resolved_entities:
            if is_overlapping(entity, res):
                overlapping_entity = res
                break

        if overlapping_entity:
            # Compare confidence scores and keep the one with the higher score
            if entity['score'] > overlapping_entity['score']:
                resolved_entities.remove(overlapping_entity)
                resolved_entities.append(entity)
        else:
            # No overlap, just add the entity
            resolved_entities.append(entity)

    return resolved_entities

# Custom NER pipeline function
def custom_pipeline(text):
    # Run the text through the PII model
    pii_results = model_pii(text)
    filtered_pii_results = [res for res in pii_results if res['entity'].split("-")[-1] in accepted_pii_labels]
    
    # Run PCI model to capture additional financial or sensitive information
    pci_results = model_pci(text)
    filtered_pci_results = [res for res in pci_results if res['entity'].split("-")[-1] in accepted_pci_labels]
    
    # Run PHI model if relevant, even if PCI and PII were detected
    phi_results = model_phi(text)
    filtered_phi_results = [res for res in phi_results if res['entity'].split("-")[-1] in accepted_phi_labels]
    
    # Run Medical model to capture medical-related entities
    medical_results = model_medical(text)
    filtered_medical_results = [res for res in medical_results if res['entity'].split("-")[-1] in accepted_medical_labels]
    
    # Combine results from all models
    combined_results = filtered_pii_results + filtered_pci_results + filtered_phi_results + filtered_medical_results
    
    # Resolve overlapping tokens
    resolved_results = resolve_overlapping_tokens(combined_results)
    
    return resolved_results

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
