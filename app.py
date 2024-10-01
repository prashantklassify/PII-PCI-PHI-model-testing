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

# Define the main categories and their sub-categories
category_hierarchy = {
    'Legal': [
        'Contracts', 'Legal Briefs', 'Memorandums', 'Compliance Documents', 
        'Court Filings', 'Legal Policies'
    ],
    'HR': [
        'Resumes', 'Employee Handbooks', 'Performance Reviews', 
        'Job Descriptions', 'Onboarding Documents', 'Termination Letters'
    ],
    'Finance': [
        'Financial Statements', 'Budgets', 'Invoices', 'Tax Returns', 
        'Expense Reports', 'Investment Proposals'
    ],
    'Insurance': [
        'Policy Documents', 'Claim Forms', 'Underwriting Guidelines', 
        'Coverage Summaries', 'Renewal Notices', 'Incident Reports'
    ],
    'Operations': [
        'Standard Operating Procedures (SOPs)', 'Inventory Reports', 
        'Project Plans', 'Quality Assurance Documents', 
        'Logistics Plans', 'Operational Audits'
    ],
    'Personal': [
        'Personal Statements', 'Life Plans', 'Health Records', 
        'Financial Plans', 'Journals', 'Travel Itineraries'
    ],
    'Sales': [
        'Sales Proposals', 'Sales Reports', 'Customer Contracts', 
        'Lead Lists', 'Sales Scripts', 'Market Analysis Reports'
    ],
    'Product': [
        'Product Specifications', 'User Manuals', 'Product Roadmaps', 
        'Market Research Reports', 'Design Documents', 'Release Notes'
    ]
}

# Flatten the category hierarchy for classification
possible_classes = []
for main_category, sub_categories in category_hierarchy.items():
    possible_classes.append(main_category)  # Add main category
    for sub_category in sub_categories:
        possible_classes.append(f"{main_category}: {sub_category}")  # Add sub-category

# Define accepted labels for NER
accepted_pii_labels = set()  # Accept all categories under PII
accepted_pci_labels = {"JOBDESCRIPTOR", "JOBTITLE", "JOBAREA", "BITCOINADDRESS", 
                       "ETHEREUMADDRESS", "ACCOUNTNAME", "ACCOUNTNUMBER", 
                       "IBAN", "BIC", "IPV4", "IPV6", "CREDITCARDNUMBER", 
                       "VEHICLEVIN", "AMOUNT", "CURRENCY", "PASSWORD", 
                       "PHONEIMEI", "CURRENCYSYMBOL", "CURRENCYNAME", 
                       "CURRENCYCODE", "LITECOINADDRESS", "MAC", 
                       "CREDITCARDISSUER", "CREDITCARDCVV", 
                       "NEARBYGPSCOORDINATE", "SEXTYPE"}

accepted_phi_labels = {"staff", "HOSP", "AGE"}
accepted_medical_labels = {"BIOLOGICAL_ATTRIBUTE", "BIOLOGICAL_STRUCTURE", 
                           "CLINICAL_EVENT", "DISEASE_DISORDER", "DOSAGE", 
                           "FAMILY_HISTORY", "LAB_VALUE", "MASS", 
                           "MEDICATION", "OUTCOME", "SIGN_SYMPTOM", 
                           "THERAPUTIC_PROCEDURE"}

# Define sliders for confidence thresholds
threshold_pii = st.slider("Confidence Threshold for PII Model", 0.0, 1.0, 0.75, 0.05)
threshold_pci = st.slider("Confidence Threshold for PCI Model", 0.0, 1.0, 0.75, 0.05)
threshold_phi = st.slider("Confidence Threshold for PHI Model", 0.0, 1.0, 0.75, 0.05)
threshold_medical = st.slider("Confidence Threshold for Medical NER Model", 0.0, 1.0, 0.75, 0.05)

# Define a function to clean and merge tokens
def clean_and_merge_tokens(entities, threshold):
    cleaned_entities = []
    for entity in entities:
        if entity['score'] >= threshold:
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
    pii_results = clean_and_merge_tokens(pii_results, threshold_pii)

    # Run the text through the PHI model
    phi_results = model_phi(text)
    phi_results = [entity for entity in phi_results if entity['entity'].split("-")[-1] in accepted_phi_labels]
    phi_results = clean_and_merge_tokens(phi_results, threshold_phi)

    # Run the text through the PCI model
    pci_results = model_pci(text)
    pci_results = [entity for entity in pci_results if entity['entity'].split("-")[-1] in accepted_pci_labels]
    pci_results = clean_and_merge_tokens(pci_results, threshold_pci)

    # Run Medical NER model independently on the original text
    medical_results = model_medical(text)
    medical_results = [entity for entity in medical_results if entity['entity'].split("-")[-1] in accepted_medical_labels]
    medical_results = clean_and_merge_tokens(medical_results, threshold_medical)

    # Combine all results
    combined_results = pii_results + phi_results + pci_results + medical_results

    return combined_results

def classify_document(text):
    # Classify the main categories
    main_classifications = classifier(text, candidate_labels=list(category_hierarchy.keys()), multi_label=True)

    # Get the top main classes based on scores
    main_classes = [label for label, score in zip(main_classifications['labels'], main_classifications['scores']) if score >= 0.5]

    sub_classifications = []
    
    # Classify sub-categories for each identified main category
    for main_class in main_classes:
        sub_classifications.extend(classifier(text, candidate_labels=category_hierarchy[main_class], multi_label=True))

    # Combine results
    return {
        'main_classes': main_classes,
        'sub_classes': [label for label, score in zip(sub_classifications['labels'], sub_classifications['scores']) if score >= 0.5]
    }
