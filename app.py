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

# Accepted labels for each model
accepted_labels = {
    "PII": {'ACCOUNTNUM', 'BUILDINGNUM', 'CITY', 'CREDITCARDNUMBER', 'DATEOFBIRTH',
            'DRIVERLICENSENUM', 'EMAIL', 'GIVENNAME', 'IDCARDNUM', 'PASSWORD',
            'SOCIALNUM', 'STREET', 'SURNAME', 'TAXNUM', 'TELEPHONENUM', 'USERNAME'},
    "PCI": {"JOBDESCRIPTOR", "JOBTITLE", "JOBAREA", "BITCOINADDRESS", "ETHEREUMADDRESS",
            "ACCOUNTNAME", "ACCOUNTNUMBER", "IBAN", "BIC", "IPV4", "IPV6",
            "CREDITCARDNUMBER", "VEHICLEVIN", "AMOUNT", "CURRENCY", "PASSWORD",
            "PHONEIMEI", "CURRENCYSYMBOL", "CURRENCYNAME", "CURRENCYCODE",
            "LITECOINADDRESS", "MAC", "CREDITCARDISSUER", "CREDITCARDCVV",
            "NEARBYGPSCOORDINATE", "SEXTYPE"},
    "PHI": {"staff", "HOSP", "AGE"},
    "Medical": {"BIOLOGICAL_ATTRIBUTE", "BIOLOGICAL_STRUCTURE", "CLINICAL_EVENT",
                "DISEASE_DISORDER", "DOSAGE", "FAMILY_HISTORY", "LAB_VALUE", "MASS",
                "MEDICATION", "OUTCOME", "SIGN_SYMPTOM", "THERAPUTIC_PROCEDURE"}
}

# Load models
model_pii = pipeline("token-classification", model=models["PII"])
model_pci = pipeline("token-classification", model=models["PCI"])
model_phi = pipeline("token-classification", model=models["PHI"])
model_medical = pipeline("token-classification", model=models["Medical NER"])

# Threshold sliders
thresholds = {
    "PII": st.slider("Confidence Threshold for PII Model", 0.0, 1.0, 0.75, 0.05),
    "PCI": st.slider("Confidence Threshold for PCI Model", 0.0, 1.0, 0.75, 0.05),
    "PHI": st.slider("Confidence Threshold for PHI Model", 0.0, 1.0, 0.75, 0.05),
    "Medical": st.slider("Confidence Threshold for Medical NER Model", 0.0, 1.0, 0.75, 0.05),
}

# Function to clean and merge tokens
def clean_and_merge_tokens(entities, threshold, accepted_labels):
    cleaned_entities = []
    for entity in entities:
        if entity['score'] < threshold or entity['entity'].split("-")[-1] not in accepted_labels:
            continue
        token = entity['word'].replace("▁", "").replace("Ġ", "")
        entity['word'] = token
        if cleaned_entities and cleaned_entities[-1]['entity'] == entity['entity'] and cleaned_entities[-1]['end'] == entity['start']:
            cleaned_entities[-1]['word'] += token
            cleaned_entities[-1]['end'] = entity['end']
            cleaned_entities[-1]['score'] = max(cleaned_entities[-1]['score'], entity['score'])
        else:
            cleaned_entities.append(entity)
    return cleaned_entities

# Resolve token conflicts by confidence
def resolve_conflicts(entities):
    resolved = {}
    for entity in entities:
        span = (entity['start'], entity['end'])
        if span not in resolved or resolved[span]['score'] < entity['score']:
            resolved[span] = entity
    return list(resolved.values())

# Normalize scores to avoid discrepancies
def normalize_scores(entities):
    max_score = max(entity['score'] for entity in entities) if entities else 1.0
    for entity in entities:
        entity['score'] /= max_score
    return entities

# Filter overlapping entities
def filter_overlaps(entities):
    sorted_entities = sorted(entities, key=lambda x: (x['start'], -x['end']))
    filtered = []
    for entity in sorted_entities:
        if not filtered or filtered[-1]['end'] <= entity['start']:
            filtered.append(entity)
    return filtered

# Custom pipeline function
def custom_pipeline(text):
    results = []
    for model_name, model in [
        ("PII", model_pii), ("PCI", model_pci), ("PHI", model_phi), ("Medical", model_medical)
    ]:
        model_results = model(text)
        model_results = clean_and_merge_tokens(model_results, thresholds[model_name], accepted_labels[model_name])
        model_results = normalize_scores(model_results)
        model_results = filter_overlaps(model_results)
        for res in model_results:
            res["entity"] = res["entity"].split('-')[-1]
        results.extend(model_results)
    return resolve_conflicts(results)

# Streamlit App layout
st.title("Document Classification and NER")

input_text = st.text_area("Enter text for classification and NER:")

if st.button("Classify and Extract Entities"):
    if input_text:
        ner_results = custom_pipeline(input_text)

        st.subheader("Extracted Entities:")
        if ner_results:
            table_data = [{
                "Entity": entity['word'],
                "Entity Type": entity['entity'],
                "Start": entity['start'],
                "End": entity['end'],
                "Confidence (%)": f"{entity['score'] * 100:.2f}"
            } for entity in ner_results]
            st.table(pd.DataFrame(table_data))
        else:
            st.write("No entities detected.")
    else:
        st.write("Please enter some text for classification and NER.")
