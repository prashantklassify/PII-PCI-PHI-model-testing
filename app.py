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
model_pii = pipeline("token-classification", model=models["PII"])
model_pci = pipeline("token-classification", model=models["PCI"])
model_phi = pipeline("token-classification", model=models["PHI"])
model_medical = pipeline("token-classification", model=models["Medical NER"])
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the hierarchical categories for document classification
hierarchy = {
    "Legal": {
        "Contracts": ["Sales Contracts", "Employment Contracts"],
        "Legal Briefs": ["Case Briefs", "Motion Briefs"],
        "Memorandums": ["Legal Memos", "Opinion Memos"],
        "Compliance Documents": ["Compliance Checklists", "Audit Reports"],
        "Court Filings": ["Pleadings", "Court Orders"],
        "Legal Policies": ["Workplace Policies", "Privacy Policies"]
    },
    "Human Resources Department": {
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
    },
    "Insurance": {
        "Policy Documents": ["Insurance Policies", "Endorsements"],
        "Claim Forms": ["Health Claim Forms", "Auto Insurance Claims"],
        "Underwriting Guidelines": ["Risk Assessment Guidelines", "Criteria Lists"],
        "Coverage Summaries": ["Benefits Overview", "Coverage Explanations"],
        "Renewal Notices": ["Policy Renewal Forms", "Reminder Letters"],
        "Incident Reports": ["Claims Incident Reports", "Loss Assessments"]
    },
    "Operations": {
        "Standard Operating Procedures": ["Process Manuals", "Workflow Guides"],
        "Inventory Reports": ["Stock Levels", "Inventory Audits"],
        "Project Plans": ["Project Charters", "Gantt Charts"],
        "Quality Assurance Documents": ["Inspection Reports", "Testing Protocols"],
        "Logistics Plans": ["Delivery Schedules", "Shipping Documents"],
        "Operational Audits": ["Performance Assessments", "Compliance Reports"]
    },
    "Sales": {
        "Sales Proposals": ["Business Proposals", "Bids"],
        "Sales Reports": ["Weekly Sales Reports", "Performance Dashboards"],
        "Customer Contracts": ["Service Agreements", "Client Contracts"],
        "Lead Lists": ["Prospect Lists", "Contact Databases"],
        "Sales Scripts": ["Cold Call Scripts", "Follow-up Templates"],
        "Market Analysis Reports": ["Competitive Analysis", "Market Research"]
    }
}

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
# Custom NER pipeline function
def custom_pipeline(text):
    # Run the text through the PII model
    pii_results = model_pii(text)
    pii_results = clean_and_merge_tokens(pii_results, threshold_pii)
    # Filter PII results
    pii_results = [entity for entity in pii_results if entity['entity'] in accepted_pii_labels]

    # Run the text through the PHI model
    phi_results = model_phi(text)
    phi_results = clean_and_merge_tokens(phi_results, threshold_phi)
    # Filter PHI results
    phi_results = [entity for entity in phi_results if entity['entity'].split("-")[-1] in accepted_phi_labels]

    # Run the text through the PCI model
    pci_results = model_pci(text)
    pci_results = clean_and_merge_tokens(pci_results, threshold_pci)
    # Filter PCI results
    pci_results = [entity for entity in pci_results if entity['entity'].split("-")[-1] in accepted_pci_labels]

    # Run the text through the Medical NER model
    medical_results = model_medical(text)
    medical_results = clean_and_merge_tokens(medical_results, threshold_medical)
    # Filter Medical NER results
    medical_results = [entity for entity in medical_results if entity['entity'].split("-")[-1] in accepted_medical_labels]

    # Combine all results
    combined_results = pii_results + phi_results + pci_results + medical_results

    return combined_results


# Document classification based on hierarchy
def classify_document(text):
    # First classify the main categories
    main_classifications = classifier(text, candidate_labels=list(hierarchy.keys()), multi_label=False)
    main_category = main_classifications['labels'][0]
    
    # Now classify the sub-categories within the identified main category
    sub_category_classes = list(hierarchy[main_category].keys())
    sub_classifications = classifier(text, candidate_labels=sub_category_classes, multi_label=False)
    sub_category = sub_classifications['labels'][0]
    
    # Classify at the third level (specific documents)
    third_level_classes = hierarchy[main_category][sub_category]
    third_classifications = classifier(text, candidate_labels=third_level_classes, multi_label=False)
    specific_doc_type = third_classifications['labels'][0]
    
    return main_category, sub_category, specific_doc_type

# Streamlit App layout
st.title("Document Classification and NER")

# Input text
input_text = st.text_area("Enter text for classification and NER:")

# Add a button to trigger classification and NER
if st.button("Classify and Extract Entities"):
    if input_text:
        # Perform document classification
        main_category, sub_category, specific_doc_type = classify_document(input_text)
        
        st.write(f"Main Category: {main_category}")
        st.write(f"Sub Category: {sub_category}")
        st.write(f"Specific Document Type: {specific_doc_type}")
        
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
