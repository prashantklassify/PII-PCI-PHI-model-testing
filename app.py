import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import random

# Load Sentence Transformer model
st_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define model descriptions
MODEL_CATALOG = {
    "iiiorg/piiranha-v1-detect-personal-information": ['ACCOUNTNUM', 'BUILDINGNUM', 'CITY', 'DATEOFBIRTH','DRIVERLICENSENUM', 'EMAIL', 'GIVENNAME', 'IDCARDNUM', 'PASSWORD','SOCIALNUM', 'STREET', 'SURNAME', 'TAXNUM', 'TELEPHONENUM', 'USERNAME'],
    "blaze999/Medical-NER": ["BIOLOGICAL_ATTRIBUTE", "BIOLOGICAL_STRUCTURE", "CLINICAL_EVENT","DISEASE_DISORDER", "DOSAGE", "FAMILY_HISTORY", "LAB_VALUE", "MASS","MEDICATION", "OUTCOME", "SIGN_SYMPTOM", "THERAPUTIC_PROCEDURE"],
    "obi/deid_roberta_i2b2": ["staff", "HOSP", "AGE"],
    "lakshyakh93/deberta_finetuned_pii":["JOBDESCRIPTOR", "JOBTITLE", "JOBAREA", "BITCOINADDRESS", "ETHEREUMADDRESS","ACCOUNTNAME", "ACCOUNTNUMBER", "IBAN", "BIC", "IPV4", "IPV6","CREDITCARDNUMBER", "VEHICLEVIN", "AMOUNT", "CURRENCY", "PASSWORD","PHONEIMEI", "CURRENCYSYMBOL", "CURRENCYNAME", "CURRENCYCODE","LITECOINADDRESS", "MAC", "CREDITCARDISSUER", "CREDITCARDCVV","NEARBYGPSCOORDINATE", "SEXTYPE"]
}

# Function to select the most relevant model based on the user query
def select_model(user_query):
    query_embedding = st_model.encode(user_query, convert_to_tensor=True)
    best_model, best_score = None, -1
    
    for model, descriptions in MODEL_CATALOG.items():
        desc_embeddings = st_model.encode(descriptions, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, desc_embeddings).max().item()
        if score > best_score:
            best_model, best_score = model, score
    
    return best_model

# Function to process complex queries
def handle_complex_queries(user_query, extracted_entities):
    if "last names" in user_query.lower():
        return [ent for ent in extracted_entities if ent['entity'] in ['SURNAME']]
    elif "PII entries excluding names" in user_query.lower():
        return [ent for ent in extracted_entities if ent['entity'] not in ['GIVENNAME', 'SURNAME']]
    elif "PII and PCI entries" in user_query.lower():
        return [ent for ent in extracted_entities if ent['entity'] in MODEL_CATALOG["iiiorg/piiranha-v1-detect-personal-information"] or ent['entity'] in MODEL_CATALOG["lakshyakh93/deberta_finetuned_pii"]]
    else:
        return extracted_entities

# Interactive chatbot-style UI
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
            model_name = select_model(user_query)
            ner_pipeline = pipeline("ner", model=model_name)
            extracted_entities = ner_pipeline(user_text)
            filtered_entities = handle_complex_queries(user_query, extracted_entities)
        
        # Add messages to chat history
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("bot", f"Model Used: `{model_name}`"))
        st.session_state.chat_history.append(("bot", f"Extracted Entities: {filtered_entities}"))
        
        # Refresh UI
        st.experimental_rerun()
    else:
        st.warning("Please enter both a query and text for analysis.")
