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

# Define the hierarchical categories for document classification
hierarchy = {
    "Hiring Department": {
        "Resumes": ["Intern", "Experienced", "Manager"],
        "Job Descriptions": ["Full-time", "Part-time", "Temporary"],
        "Onboarding Documents": ["HR forms", "Welcome letter"],
        "Offboarding Documents": ["Exit survey", "Resignation letter"]
    },
    "Finance": {
        "Cash Flow Statement": ["Annual", "Quarterly"],
        "Income Statement": ["Annual", "Quarterly"],
        "Notes": ["Accounting Notes", "Tax Notes"],
        "Others": ["Miscellaneous"]
    },
    "Sales": {
        "Sales Proposal": ["Formal", "Informal"],
        "Sales Pitch": ["Presentation", "Email"],
        "Follow-up Email": ["Reminder", "Thank You"],
        "Plausible Lists": ["Leads", "Prospects"]
    },
    "Others": {
        "others": ["Unclassified"]
    }
}

# Define sliders for confidence thresholds
threshold_pii = st.slider("Confidence Threshold for PII Model", 0.0, 1.0, 0.75, 0.05)
threshold_pci = st.slider("Confidence Threshold for PCI Model", 0.0, 1.0, 0.75, 0.05)
threshold_phi = st.slider("Confidence Threshold for PHI Model", 0.0, 1.0, 0.75, 0.05)
threshold_medical = st.slider("Confidence Threshold for Medical NER Model", 0.0, 1.0, 0.75, 0.05)

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
    pii_results = [entity for entity in pii_results if entity['entity'].split("-")[-1] in accepted_pii_labels]
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


# Recursive Document classification based on hierarchy
def classify_document_recursive(text, current_level):
    if isinstance(current_level, dict):
        classification = classifier(text, candidate_labels=list(current_level.keys()), multi_label=False)
        top_category = classification['labels'][0]
        return top_category, classify_document_recursive(text, current_level[top_category])
    else:
        # When we reach the final level with specific document types
        final_classification = classifier(text, candidate_labels=current_level, multi_label=False)
        return final_classification['labels'][0]

# Streamlit App layout
st.title("Document Classification and NER")

# Input text
input_text = st.text_area("Enter text for classification and NER:")

# Add a button to trigger classification and NER
if st.button("Classify and Extract Entities"):
    if input_text:
        # Perform document classification
        main_category, sub_category = classify_document_recursive(input_text, hierarchy)
        
        st.write(f"Main Category: {main_category}")
        if isinstance(sub_category, str):
            st.write(f"Sub Category: {sub_category}")
        
        # Perform NER
        ner_results = custom_pipeline(input_text)
        st.subheader("Named Entities:")
        
        # Display NER results in a table format
        if ner_results:
            table_data = [{
                "Entity": result['word'],
                "Entity Type": result['entity'].split("-")[-1],
                "Confidence (%)": f"{result['score'] * 100:.2f}"
            } for result in ner_results]
            results_table = pd.DataFrame(table_data)
            st.table(results_table)
        else:
            st.write("No entities detected.")
    else:
        st.write("Please enter some text for classification and NER.")
