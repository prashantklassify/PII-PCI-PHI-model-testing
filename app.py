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
possible_classes = [
    # Business
    'Business Proposal', 'Invoice', 'Receipt', 'Contract', 'Purchase Order',
    'Quotation', 'Memorandum', 'Report', 'Sales Agreement', 'Expense Report',
    'Marketing Plan', 'Business Plan', 'Meeting Minutes', 'Annual Report',
    'Business Letter', 'Non-Disclosure Agreement', 'Partnership Agreement',
    'Stock Purchase Agreement', 'Service Agreement', 'Employment Agreement',
    'Loan Agreement', 'Termination Notice', 'Resignation Letter', 'Credit Application',
    'Market Research Report', 'Feasibility Study', 'Project Charter',
    'Vendor Agreement', 'Statement of Work', 'Confidentiality Agreement',

    # Education
    'Transcript', 'Diploma', 'Course Syllabus', 'Thesis', 'Research Paper',
    'Dissertation', 'Curriculum Vitae', 'Application Letter', 'Scholarship Application',
    'Grant Proposal', 'Course Registration Form', 'Student Handbook',
    'Faculty Handbook', 'Academic Calendar', 'Grade Report', 'Student Evaluation',
    'Internship Report', 'Field Study Report', 'Project Report', 'Lesson Plan',
    'Course Completion Certificate', 'Certificate of Achievement',

    # Health
    'Patient Record', 'Medical Report', 'Prescription', 'Surgical Report',
    'Health Assessment', 'Insurance Claim', 'Appointment Reminder',
    'Lab Results', 'Medical History', 'Referral Letter', 'Treatment Plan',
    'Discharge Summary', 'Billing Statement', 'Consent Form', 'Clinical Trial Report',
    'Health Education Material', 'Wellness Program Description',

    # Legal
    'Legal Brief', 'Court Document', 'Filing Notice', 'Motion', 'Affidavit',
    'Deposition', 'Subpoena', 'Summons', 'Judgment', 'Legal Opinion',
    'Statute', 'Regulation', 'License Agreement', 'Power of Attorney',
    'Will', 'Trust Document', 'Settlement Agreement', 'Trademark Registration',
    'Copyright Registration', 'Patent Application',

    # Technical
    'Technical Specification', 'User Manual', 'Installation Guide', 'Troubleshooting Guide',
    'API Documentation', 'Software Requirement Specification', 'System Architecture Document',
    'Test Plan', 'Test Report', 'Release Notes', 'Change Log', 'Technical Proposal',
    'Engineering Report', 'Design Document', 'Prototype Report',

    # Personal
    'Resume', 'Cover Letter', 'Personal Statement', 'Portfolio', 'Job Application',
    'Thank You Letter', 'Letter of Recommendation', 'Reference Letter',
    'Identity Proof', 'Bank Statement', 'Credit Report', 'Insurance Policy',
    'Travel Itinerary', 'Rental Agreement', 'Lease Agreement', 'Property Deed',
    'Will', 'Personal Diary',

    # Government
    'Tax Return', 'Policy Document', 'Government Report', 'Regulatory Compliance Document',
    'Public Notice', 'Statistical Report', 'Census Data', 'Public Health Report',
    'Environmental Impact Assessment', 'Grant Application',

    # Marketing
    'Marketing Strategy', 'Product Brochure', 'Press Release', 'Social Media Plan',
    'Content Marketing Plan', 'Email Marketing Campaign', 'Ad Copy', 'Market Analysis',
    'Brand Strategy Document', 'Customer Feedback Report',

    # Miscellaneous
    'Recipe', 'Travel Guide', 'Event Program', 'User Feedback Form', 'Survey',
    'Quality Assurance Document', 'Event Report', 'Workshop Material', 'Presentation Slides',
    'FAQ Document', 'Newsletter', 'Instructional Material', 'Guidebook',
    'Field Study Report', 'Data Analysis Report', 'Conference Proceedings',
    'Workshop Report', 'Presentation', 'Abstract',

    # Adding more unique categories (to reach around 1000)
    'Application Form', 'Job Description', 'Interview Guide', 'Employee Handbook',
    'Safety Manual', 'Crisis Management Plan', 'Business Continuity Plan',
    'Procurement Plan', 'Risk Assessment Report', 'Asset Valuation Report',
    'Feasibility Study', 'Performance Review', 'Customer Profile', 'Sales Strategy',
    'Market Entry Strategy', 'User Acceptance Testing Report', 'Product Specification',
    'User Research Report', 'A/B Testing Report', 'Customer Journey Map',
    'Corporate Social Responsibility Report', 'Training Manual',

    # (More unique categories can be added to meet the desired count)
] * 45  # This will replicate the list above until we reach around 1000

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
    # Accepting all categories for PII as no specific labels were mentioned
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
    # Classify multiple main categories
    main_classifications = classifier(text, candidate_labels=main_categories, multi_label=True)

    # Get the top main classes based on scores
    main_classes = [label for label, score in zip(main_classifications['labels'], main_classifications['scores']) if score >= 0.5]

    sub_classifications = []
    
    # Classify sub-categories for each identified main category
    for main_class in main_classes:
        sub_classes = [sub for sub in possible_classes if sub.startswith(main_class.split()[0])]
        if sub_classes:
            sub_class = classifier(text, candidate_labels=sub_classes, multi_label=True)
            sub_classifications.append({
                "main_class": main_class,
                "sub_classes": [label for label, score in zip(sub_class['labels'], sub_class['scores']) if score >= 0.5]
            })

    return main_classes, sub_classifications
# Function to display results in a table format
def display_results(results):
    table_data = [{
        "Entity": result['word'],
        "Entity Type": result['entity'].split("-")[-1],
        "Confidence (%)": f"{result['score'] * 100:.2f}"
    } for result in results]
    return pd.DataFrame(table_data)

# Streamlit app layout
st.title("Multi-Label Named Entity Recognition (NER) Streamlit App")

# User input for text
text = st.text_area("Enter text for NER processing", "")
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

    # Classify main and sub-categories
    main_classes, sub_classifications = classify_document(text)
    
    # Display the main categories
    st.write(f"Main Categories: {', '.join(main_classes)}")

    # Display sub-categories for each main category
    for sub_class in sub_classifications:
        st.write(f"Sub-categories for {sub_class['main_class']}: {', '.join(sub_class['sub_classes'])}")
