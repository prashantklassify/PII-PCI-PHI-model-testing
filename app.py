import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import re

# Load Sentence Transformer model
st_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define model descriptions
MODEL_CATALOG = {
    "iiiorg/piiranha-v1-detect-personal-information": ['ACCOUNTNUM', 'BUILDINGNUM', 'CITY', 'DATEOFBIRTH', 'DRIVERLICENSENUM', 'EMAIL', 'GIVENNAME', 'IDCARDNUM', 'PASSWORD', 'SOCIALNUM', 'STREET', 'SURNAME', 'TAXNUM', 'TELEPHONENUM', 'USERNAME'],
    "blaze999/Medical-NER": ["BIOLOGICAL_ATTRIBUTE", "BIOLOGICAL_STRUCTURE", "CLINICAL_EVENT", "DISEASE_DISORDER", "DOSAGE", "FAMILY_HISTORY", "LAB_VALUE", "MASS", "MEDICATION", "OUTCOME", "SIGN_SYMPTOM", "THERAPUTIC_PROCEDURE"],
    "obi/deid_roberta_i2b2": ["staff", "HOSP", "AGE"],
    "lakshyakh93/deberta_finetuned_pii": ["JOBDESCRIPTOR", "JOBTITLE", "JOBAREA", "BITCOINADDRESS", "ETHEREUMADDRESS", "ACCOUNTNAME", "ACCOUNTNUMBER", "IBAN", "BIC", "IPV4", "IPV6", "CREDITCARDNUMBER", "VEHICLEVIN", "AMOUNT", "CURRENCY", "PASSWORD", "PHONEIMEI", "CURRENCYSYMBOL", "CURRENCYNAME", "CURRENCYCODE", "LITECOINADDRESS", "MAC", "CREDITCARDISSUER", "CREDITCARDCVV", "NEARBYGPSCOORDINATE", "SEXTYPE"]
}

# Function to select relevant models for a query
def select_models(user_query):
    query_embedding = st_model.encode(user_query, convert_to_tensor=True)
    model_scores = {}
    
    for model, descriptions in MODEL_CATALOG.items():
        desc_embeddings = st_model.encode(descriptions, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, desc_embeddings).max().item()
        if score > 0.4:  # Threshold to filter relevant models
            model_scores[model] = score
    
    return sorted(model_scores, key=model_scores.get, reverse=True)  # Return models sorted by relevance

# Post-processing function for NER output
def process_entities(entities):
    cleaned_entities = []
    for ent in entities:
        entity_text = re.sub(r'▁', '', ent['word'])  # Remove subword markers
        if entity_text:
            cleaned_entities.append({
                'entity': ent['entity'],
                'text': entity_text,
                'confidence': round(ent['score'], 4)
            })
    return cleaned_entities

# Function to handle complex queries
def handle_complex_queries(user_query, extracted_entities):
    query = user_query.lower()
    
    if "names" in query:
        return [ent for ent in extracted_entities if ent['entity'] in ['GIVENNAME', 'SURNAME']]
    elif "exclude names" in query:
        return [ent for ent in extracted_entities if ent['entity'] not in ['GIVENNAME', 'SURNAME']]
    elif "PII and PCI" in query:
        pii_pci_labels = MODEL_CATALOG["iiiorg/piiranha-v1-detect-personal-information"] + MODEL_CATALOG["lakshyakh93/deberta_finetuned_pii"]
        return [ent for ent in extracted_entities if ent['entity'] in pii_pci_labels]
    return extracted_entities

# Streamlit UI
st.title("💬 AI NER Chatbot")
st.markdown("### Detect Personal, Financial, and Medical Data Intelligently")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    role, text = message
    if role == "bot":
        st.markdown(f"**🤖 Bot:** {text}")
    else:
        st.markdown(f"**🧑 You:** {text}")

# User input
user_query = st.text_input("Ask me to analyze something:")
user_text = st.text_area("Paste your text here:")

if st.button("Send"):
    if user_query and user_text:
        with st.spinner("Processing..."):
            selected_models = select_models(user_query)
            extracted_entities = []
            
            for model_name in selected_models:
                ner_pipeline = pipeline("ner", model=model_name)
                raw_entities = ner_pipeline(user_text)
                processed_entities = process_entities(raw_entities)
                extracted_entities.extend(processed_entities)
            
            filtered_entities = handle_complex_queries(user_query, extracted_entities)
        
        # Add messages to chat history
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("bot", f"Models Used: {selected_models}"))
        st.session_state.chat_history.append(("bot", f"Extracted Entities: {filtered_entities}"))
        
        # Refresh UI
        st.rerun()
    else:
        st.warning("Please enter both a query and text for analysis.")
