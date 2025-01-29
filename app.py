import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# Load Sentence Transformer model
st_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define model descriptions
MODEL_CATALOG = {
    "iiiorg/piiranha-v1-detect-personal-information": ['ACCOUNTNUM', 'BUILDINGNUM', 'CITY', 'DATEOFBIRTH','DRIVERLICENSENUM', 'EMAIL', 'GIVENNAME', 'IDCARDNUM', 'PASSWORD','SOCIALNUM', 'STREET', 'SURNAME', 'TAXNUM', 'TELEPHONENUM', 'USERNAME'],
    "blaze999/Medical-NER": ["BIOLOGICAL_ATTRIBUTE", "BIOLOGICAL_STRUCTURE", "CLINICAL_EVENT","DISEASE_DISORDER", "DOSAGE", "FAMILY_HISTORY", "LAB_VALUE", "MASS","MEDICATION", "OUTCOME", "SIGN_SYMPTOM", "THERAPUTIC_PROCEDURE"],
    "obi/deid_roberta_i2b2": ["staff", "HOSP", "AGE"],
    "lakshyakh93/deberta_finetuned_pii":["JOBDESCRIPTOR", "JOBTITLE", "JOBAREA", "BITCOINADDRESS", "ETHEREUMADDRESS","ACCOUNTNAME", "ACCOUNTNUMBER", "IBAN", "BIC", "IPV4", "IPV6","CREDITCARDNUMBER", "VEHICLEVIN", "AMOUNT", "CURRENCY", "PASSWORD","PHONEIMEI", "CURRENCYSYMBOL", "CURRENCYNAME", "CURRENCYCODE","LITECOINADDRESS", "MAC", "CREDITCARDISSUER", "CREDITCARDCVV","NEARBYGPSCOORDINATE", "SEXTYPE"]
}

models = {
    "PII": "iiiorg/piiranha-v1-detect-personal-information",
    "PCI": "lakshyakh93/deberta_finetuned_pii",
    "PHI": "obi/deid_roberta_i2b2",
    "Medical NER": "blaze999/Medical-NER"
}

# Accepted labels for each model
accepted_labels = {
    "PII": {'ACCOUNTNUM', 'BUILDINGNUM', 'CITY', 'DATEOFBIRTH',
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


def select_model(user_query):
    """
    Select the best model based on semantic similarity using Sentence Transformers.
    """
    config = {"model": None}
    query_embedding = st_model.encode(user_query, convert_to_tensor=True)
    best_model, best_score = None, -1

    for model, descriptions in MODEL_CATALOG.items():
        desc_embeddings = st_model.encode(descriptions, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, desc_embeddings).max().item()
        if score > best_score:
            best_model, best_score = model, score

    config["model"] = best_model
    return config

def run_ner_analysis(user_query, user_text):
    """Run NER model based on selected configuration."""
    config = select_model(user_query)
    model_name = config.get("model", "blaze999/General-NER")
    
    ner_pipeline = pipeline("ner", model=model_name)
    results = ner_pipeline(user_text)
    return results, model_name

# Streamlit UI
st.title("🤖 Smart AI NER Chatbot")
st.subheader("Detect Personal, Financial, and Medical Data Intelligently")

# User input fields
user_query = st.text_input("Describe what you need (e.g., 'Find all personal data'):")
user_text = st.text_area("Enter text for analysis:")

if st.button("Analyze"):
    if user_query and user_text:
        with st.spinner("Analyzing text..."):
            extracted_entities, selected_model = run_ner_analysis(user_query, user_text)
        
        st.success(f"Model Used: `{selected_model}`")
        st.json(extracted_entities)
    else:
        st.warning("Please enter both a query and text for analysis.")
