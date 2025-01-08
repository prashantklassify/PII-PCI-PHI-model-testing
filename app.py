import streamlit as st
from transformers import pipeline
import pandas as pd

# Load the models for NER and classification
models = {
    "PII": "iiiorg/piiranha-v1-detect-personal-information",
    "PCI": "lakshyakh93/deberta_finetuned_pii",
    "PHI": "obi/deid_roberta_i2b2",
    "Medical NER": "blaze999/Medical-NER"
}
accepted_pii_labels = {'ACCOUNTNUM','BUILDINGNUM','CITY','CREDITCARDNUMBER','DATEOFBIRTH','DRIVERLICENSENUM','EMAIL','GIVENNAME','IDCARDNUM','PASSWORD','SOCIALNUM','STREET','SURNAME','TAXNUM','TELEPHONENUM','USERNAME'}
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
model_pii = pipeline("token-classification", model=models["PII"])
model_pci = pipeline("token-classification", model=models["PCI"])
model_phi = pipeline("token-classification", model=models["PHI"])
model_medical = pipeline("token-classification", model=models["Medical NER"])
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define sliders for confidence thresholds
threshold_pii = st.slider("Confidence Threshold for PII Model", 0.0, 1.0, 0.75, 0.05)
threshold_pci = st.slider("Confidence Threshold for PCI Model", 0.0, 1.0, 0.75, 0.05)
threshold_phi = st.slider("Confidence Threshold for PHI Model", 0.0, 1.0, 0.75, 0.05)
threshold_medical = st.slider("Confidence Threshold for Medical NER Model", 0.0, 1.0, 0.75, 0.05)

# Function to clean and merge tokens
def clean_and_merge_tokens(entities):
    cleaned_entities = []
    for entity in entities:
        token = entity['word'].replace("▁", "").replace("Ġ", "")
        entity['word'] = token
        if cleaned_entities and cleaned_entities[-1]['entity'] == entity['entity'] and cleaned_entities[-1]['end'] == entity['start']:
            cleaned_entities[-1]['word'] += token
            cleaned_entities[-1]['end'] = entity['end']
            cleaned_entities[-1]['score'] = max(cleaned_entities[-1]['score'], entity['score'])
        else:
            cleaned_entities.append(entity)
    return cleaned_entities

# Custom pipeline function to calculate category-wise percentages
def custom_pipeline(text):
    # Run text through the models and clean results
    pii_results = model_pii(text)
    pii_results = [entity for entity in pii_results if entity['entity'].split("-")[-1] in accepted_pii_labels]
    pii_results = clean_and_merge_tokens(pii_results)

    phi_results = model_phi(text)
    phi_results = [entity for entity in phi_results if entity['entity'].split("-")[-1] in accepted_phi_labels]
    phi_results = clean_and_merge_tokens(phi_results)

    pci_results = model_pci(text)
    pci_results = [entity for entity in pci_results if entity['entity'].split("-")[-1] in accepted_pci_labels]
    pci_results = clean_and_merge_tokens(pci_results)

    medical_results = model_medical(text)
    medical_results = [entity for entity in medical_results if entity['entity'].split("-")[-1] in accepted_medical_labels]
    medical_results = clean_and_merge_tokens(medical_results)

    # Combine all results with a label for category
    combined_results = (
        [("PII", result) for result in pii_results] +
        [("PHI", result) for result in phi_results] +
        [("PCI", result) for result in pci_results] +
        [("Medical", result) for result in medical_results]
    )

    return combined_results

# Streamlit App layout
st.title("Document Classification and NER")

# Input text
input_text = st.text_area("Enter text for classification and NER:")

# Add a button to trigger classification and NER
if st.button("Classify and Extract Entities"):
    if input_text:
        # Perform NER
        ner_results = custom_pipeline(input_text)

        # Calculate percentages for each category
        category_counts = {"PII": 0, "PHI": 0, "PCI": 0, "Medical": 0, "Others": 0}
        total_entities = len(ner_results)

        for category, entity in ner_results:
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts["Others"] += 1

        percentages = {key: (count / total_entities) * 100 if total_entities > 0 else 0 for key, count in category_counts.items()}

        # Display percentages
        st.subheader("Entity Category Percentages:")
        for category, percentage in percentages.items():
            st.write(f"{category}: {percentage:.2f}%")

        # Display detailed entity results
        st.subheader("Named Entities:")
        if ner_results:
            table_data = [{
                "Entity": entity['word'],
                "Category": category,
                "Entity Type": entity['entity'].split("-")[-1],
                "Confidence (%)": f"{entity['score'] * 100:.2f}"
            } for category, entity in ner_results]
            results_table = pd.DataFrame(table_data)
            st.table(results_table)
        else:
            st.write("No entities detected.")
    else:
        st.write("Please enter some text for classification and NER.")
