import streamlit as st
from transformers import pipeline
import pandas as pd
from functools import lru_cache
import torch

# Configure the app
st.set_page_config(page_title="Document Classification and NER", layout="wide")

# Display app title
st.title("Document Classification and NER")

# Add a warning about initial load time
with st.expander("ℹ️ Important Note"):
    st.write("""
    - The first run will take longer as models download (up to 5 minutes)
    - Subsequent runs will be faster due to caching
    - Large documents may take time to process
    """)

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

# Load models with caching and proper device handling
@st.cache_resource(show_spinner="Loading PII model...")
def load_pii_model():
    try:
        # First try loading with default settings
        return pipeline("token-classification", 
                       model="iiiorg/piiranha-v1-detect-personal-information",
                       device="cpu")
    except Exception as e:
        st.error(f"Failed to load PII model with default settings: {str(e)}")
        try:
            # Try loading with low_cpu_mem_usage=True
            return pipeline("token-classification", 
                          model="iiiorg/piiranha-v1-detect-personal-information",
                          device="cpu",
                          low_cpu_mem_usage=True)
        except Exception as e:
            st.error(f"Failed to load PII model with low_cpu_mem_usage: {str(e)}")
            return None

@st.cache_resource(show_spinner="Loading PCI model...")
def load_pci_model():
    try:
        return pipeline("token-classification", 
                      model="lakshyakh93/deberta_finetuned_pii",
                      device="cpu")
    except Exception as e:
        st.error(f"Failed to load PCI model with default settings: {str(e)}")
        try:
            return pipeline("token-classification", 
                          model="lakshyakh93/deberta_finetuned_pii",
                          device="cpu",
                          low_cpu_mem_usage=True)
        except Exception as e:
            st.error(f"Failed to load PCI model with low_cpu_mem_usage: {str(e)}")
            return None

@st.cache_resource(show_spinner="Loading PHI model...")
def load_phi_model():
    try:
        return pipeline("token-classification", 
                      model="obi/deid_roberta_i2b2",
                      device="cpu")
    except Exception as e:
        st.error(f"Failed to load PHI model with default settings: {str(e)}")
        try:
            return pipeline("token-classification", 
                          model="obi/deid_roberta_i2b2",
                          device="cpu",
                          low_cpu_mem_usage=True)
        except Exception as e:
            st.error(f"Failed to load PHI model with low_cpu_mem_usage: {str(e)}")
            return None

@st.cache_resource(show_spinner="Loading Medical NER model...")
def load_medical_model():
    try:
        return pipeline("token-classification", 
                      model="blaze999/Medical-NER",
                      device="cpu")
    except Exception as e:
        st.error(f"Failed to load Medical NER model with default settings: {str(e)}")
        try:
            return pipeline("token-classification", 
                          model="blaze999/Medical-NER",
                          device="cpu",
                          low_cpu_mem_usage=True)
        except Exception as e:
            st.error(f"Failed to load Medical NER model with low_cpu_mem_usage: {str(e)}")
            return None

# Load models with progress indicators
with st.spinner("Loading models (this may take a few minutes)..."):
    model_pii = load_pii_model()
    model_pci = load_pci_model()
    model_phi = load_phi_model()
    model_medical = load_medical_model()

# Check if all models loaded successfully
if None in [model_pii, model_pci, model_phi, model_medical]:
    st.error("One or more models failed to load. Please check the error messages above.")
    st.stop()

# Threshold sliders
st.sidebar.header("Model Thresholds")
thresholds = {
    "PII": st.sidebar.slider("Confidence Threshold for PII Model", 0.0, 1.0, 0.75, 0.05),
    "PCI": st.sidebar.slider("Confidence Threshold for PCI Model", 0.0, 1.0, 0.75, 0.05),
    "PHI": st.sidebar.slider("Confidence Threshold for PHI Model", 0.0, 1.0, 0.75, 0.05),
    "Medical": st.sidebar.slider("Confidence Threshold for Medical NER Model", 0.0, 1.0, 0.75, 0.05),
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
        try:
            model_results = model(text)
            model_results = clean_and_merge_tokens(model_results, thresholds[model_name], accepted_labels[model_name])
            for res in model_results:
                res["entity"] = model_name
            results.extend(model_results)
        except Exception as e:
            st.warning(f"Error processing text with {model_name} model: {str(e)}")
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
        highlighted_text += f"<span style='background-color:{color}' title='{category} ({entity['score']:.2f})'>{text[entity['start']:entity['end']]}</span>"
        current_pos = entity['end']
    highlighted_text += text[current_pos:]
    return highlighted_text

# Categorize tokens
def categorize_tokens(text, entities):
    total_tokens = len(text.split())
    if total_tokens == 0:
        return {}
    
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

# Main app interface
input_text = st.text_area("Enter text for classification and NER:", height=200)

if st.button("Classify and Extract Entities"):
    if not input_text.strip():
        st.warning("Please enter some text for classification and NER.")
        st.stop()
        
    with st.spinner("Processing text..."):
        try:
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
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            else:
                st.info("No entities detected above the confidence thresholds.")

            st.subheader("Category Percentages:")
            percentages = categorize_tokens(input_text, ner_results)
            if percentages:
                df = pd.DataFrame.from_dict(percentages, orient='index', columns=['Percentage'])
                df['Percentage'] = df['Percentage'].round(2)
                st.bar_chart(df)
                st.table(df)
            else:
                st.info("No categorization data available.")
                
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
