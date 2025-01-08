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
            'SOCIALNUM', 'STREET', 'SURNAME', 'TAXNUM', 'TELEPHONENUM','CREDITCARDNUMBER', 'USERNAME'},
    "PCI": {"JOBDESCRIPTOR", "JOBTITLE", "JOBAREA", "BITCOINADDRESS", "ETHEREUMADDRESS",
            "ACCOUNTNAME", "ACCOUNTNUMBER", "IBAN", "BIC", "IPV4", "IPV6",
            "CREDITCARDNUMBER", "VEHICLEVIN", "AMOUNT", "CURRENCY", "PASSWORD",
            "PHONEIMEI", "CURRENCYSYMBOL", "CURRENCYNAME", "CURRENCYCODE",
            "LITECOINADDRESS", "MAC", "CREDITCARDISSUER","PREFIX", "CREDITCARDCVV","CREDITCARDISSUER","IP","STREETADDRESS","SSN","SECONDARYADDRESS","USERAGENT","PIN",
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

# Custom pipeline function
def custom_pipeline(text):
    results = []
    for model_name, model in [("PII", model_pii), ("PCI", model_pci), ("PHI", model_phi), ("Medical", model_medical)]:
        model_results = model(text)
        model_results = clean_and_merge_tokens(model_results, thresholds[model_name], accepted_labels[model_name])
        for res in model_results:
            res["entity"] = model_name
        results.extend(model_results)
    return resolve_conflicts(results)

# Highlight text with colors
def highlight_text(text, entities):
    colors = {
        "PII": "#FFA07A",  # Light Salmon
        "PCI": "#ADD8E6",  # Light Blue
        "PHI": "#FFD700",  # Gold (shared with Medical)
        "Medical": "#FFD700"  # Gold
    }
    highlighted_text = ""
    current_pos = 0

    # Sort entities by start position
    entities = sorted(entities, key=lambda x: x['start'])
    for entity in entities:
        category = entity['entity']
        color = colors.get(category, "#FFFFFF")  # Default to white
        highlighted_text += text[current_pos:entity['start']]
        highlighted_text += f"<span style='background-color:{color}'>{text[entity['start']:entity['end']]}</span>"
        current_pos = entity['end']
    highlighted_text += text[current_pos:]
    return highlighted_text

# Categorize tokens
def categorize_tokens(text, entities):
    total_tokens = len(text.split())
    categories = {"PII": 0, "PCI": 0, "PHI": 0, "Others": 0}

    covered_positions = set()
    for entity in entities:
        category = "PHI" if entity['entity'] in ["PHI", "Medical"] else entity['entity']
        categories[category] += len(text[entity['start']:entity['end']].split())
        covered_positions.update(range(entity['start'], entity['end']))

    uncovered_tokens = [word for i, word in enumerate(text.split()) if i not in covered_positions]
    categories["Others"] += len(uncovered_tokens)

    percentages = {key: (count / total_tokens) * 100 for key, count in categories.items()}
    return percentages

# Streamlit App layout
st.title("Document Classification and NER")

input_text = st.text_area("Enter text for classification and NER:")

if st.button("Classify and Extract Entities"):
    if input_text:
        ner_results = custom_pipeline(input_text)
        
        st.subheader("Highlighted Text with Entities:")
        highlighted_html = highlight_text(input_text, ner_results)
        st.markdown(highlighted_html, unsafe_allow_html=True)

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

        st.subheader("Category Percentages:")
        percentages = categorize_tokens(input_text, ner_results)
        for category, percentage in percentages.items():
            st.write(f"{category}: {percentage:.2f}%")
    else:
        st.write("Please enter some text for classification and NER.")
