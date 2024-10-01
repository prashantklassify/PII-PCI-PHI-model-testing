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
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define possible categories and subcategories for classification
classification_hierarchy = {
    "Legal": {
        "Contracts": ["Sales Contracts", "Employment Contracts"],
        "Legal Briefs": ["Case Briefs", "Motion Briefs"],
        "Memorandums": ["Legal Memos", "Opinion Memos"],
        "Compliance Documents": ["Compliance Checklists", "Audit Reports"],
        "Court Filings": ["Pleadings", "Court Orders"],
        "Legal Policies": ["Workplace Policies", "Privacy Policies"]
    },
    "HR (Human Resources)": {
        "Resumes": ["CVs", "Cover Letters"],
        "Employee Handbooks": ["Company Policies", "Code of Conduct"],
        "Performance Reviews": ["Annual Review Forms", "Feedback Forms"],
        "Job Descriptions": ["Position Specifications", "Role Summaries"],
        "Onboarding Documents": ["Orientation Guides", "Training Materials"],
        "Termination Letters": ["Resignation Letters", "Dismissal Notices"]
    },
    "Finance": {
        "Financial Statements": ["Balance Sheets", "Income Statements"],
        "Budgets": ["Annual Budgets", "Project Budgets"],
        "Invoices": ["Service Invoices", "Purchase Invoices"],
        "Tax Returns": ["Personal Returns", "Corporate Returns"],
        "Expense Reports": ["Travel Expenses", "Reimbursement Forms"],
        "Investment Proposals": ["Business Investment Plans", "Risk Assessments"]
    }
}

accepted_pii_labels = set()  # Accept all categories under PII
accepted_pci_labels = {"JOBDESCRIPTOR", "JOBTITLE", "CREDITCARDNUMBER", "ACCOUNTNUMBER", "AMOUNT"}
accepted_phi_labels = {"AGE", "HOSP"}
accepted_medical_labels = {"DISEASE_DISORDER", "MEDICATION"}

# Define confidence sliders for models
threshold_pii = st.slider("Confidence Threshold for PII Model", 0.0, 1.0, 0.75, 0.05)
threshold_pci = st.slider("Confidence Threshold for PCI Model", 0.0, 1.0, 0.75, 0.05)
threshold_phi = st.slider("Confidence Threshold for PHI Model", 0.0, 1.0, 0.75, 0.05)
threshold_medical = st.slider("Confidence Threshold for Medical NER Model", 0.0, 1.0, 0.75, 0.05)

# Function to clean and merge tokens
def clean_and_merge_tokens(entities, threshold):
    cleaned_entities = []
    for entity in entities:
        if entity['score'] >= threshold:
            token = entity['word'].replace("▁", "").replace("Ġ", "")
            entity['word'] = token
            if cleaned_entities and cleaned_entities[-1]['entity'] == entity['entity'] and cleaned_entities[-1]['end'] == entity['start']:
                cleaned_entities[-1]['word'] += token
                cleaned_entities[-1]['end'] = entity['end']
                cleaned_entities[-1]['score'] = max(cleaned_entities[-1]['score'], entity['score'])
            else:
                cleaned_entities.append(entity)
    return cleaned_entities

# Custom NER pipeline
def custom_pipeline(text):
    # Run text through the PII model
    pii_results = clean_and_merge_tokens(model_pii(text), threshold_pii)

    # Run text through PHI model
    phi_results = clean_and_merge_tokens([entity for entity in model_phi(text) if entity['entity'].split("-")[-1] in accepted_phi_labels], threshold_phi)

    # Run text through PCI model
    pci_results = clean_and_merge_tokens([entity for entity in model_pci(text) if entity['entity'].split("-")[-1] in accepted_pci_labels], threshold_pci)

    # Run Medical NER model
    medical_results = clean_and_merge_tokens([entity for entity in model_medical(text) if entity['entity'].split("-")[-1] in accepted_medical_labels], threshold_medical)

    return pii_results + phi_results + pci_results + medical_results

# Three-tier document classification
def classify_document_with_three_tier_hierarchy(text):
    # Step 1: Classify the main category
    main_classifications = classifier(text, candidate_labels=list(classification_hierarchy.keys()), multi_label=True)
    main_classes = [label for label, score in zip(main_classifications['labels'], main_classifications['scores']) if score >= 0.5]

    sub_classifications = {}
    sub_sub_classifications = {}

    for main_class in main_classes:
        # Step 2: Classify sub-categories
        sub_classes = list(classification_hierarchy[main_class].keys())
        sub_class = classifier(text, candidate_labels=sub_classes, multi_label=True)
        selected_subclasses = [label for label, score in zip(sub_class['labels'], sub_class['scores']) if score >= 0.5]
        sub_classifications[main_class] = selected_subclasses

        # Step 3: Classify sub-sub-categories
        sub_sub_classifications[main_class] = {}
        for sub_category in selected_subclasses:
            sub_sub_classes = classification_hierarchy[main_class][sub_category]
            sub_sub_class = classifier(text, candidate_labels=sub_sub_classes, multi_label=True)
            sub_sub_classifications[main_class][sub_category] = [label for label, score in zip(sub_sub_class['labels'], sub_sub_class['scores']) if score >= 0.5]

    return main_classes, sub_classifications, sub_sub_classifications

# Function to display results in a table
def display_results(results):
    table_data = [{
        "Entity": result['word'],
        "Entity Type": result['entity'].split("-")[-1],
        "Confidence (%)": f"{result['score'] * 100:.2f}"
    } for result in results]
    return pd.DataFrame(table_data)

# Streamlit app layout
st.title("NER and Three-Tier Document Classification Streamlit App")

# User input for text
text = st.text_area("Enter text for NER and classification", "")

# Button to process the text
if st.button("Run NER Models and Classify Document"):
    st.subheader("Custom NER Pipeline Results")

    # Run the custom pipeline
    predictions = custom_pipeline(text)
    if predictions:
        results_table = display_results(predictions)
        st.table(results_table)
    else:
        st.write("No entities detected.")

    st.subheader("Document Classification Results")

    # Classify document with three-tier hierarchy
    main_classes, sub_classifications, sub_sub_classifications = classify_document_with_three_tier_hierarchy(text)

    # Display main categories
    st.write("Main Categories:")
    for main_class in main_classes:
        st.write(f"- {main_class}")

    # Display sub-categories and sub-sub-categories for each main category
    st.write("Sub-categories and Sub-sub-categories:")
    for main_class, sub_categories in sub_classifications.items():
        st.write(f"{main_class}:")
        for sub_category in sub_categories:
            st.write(f"  - {sub_category}")
            sub_sub_categories = sub_sub_classifications[main_class][sub_category]
            for sub_sub_category in sub_sub_categories:
                st.write(f"    - {sub_sub_category}")
