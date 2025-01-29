import streamlit as st
from transformers import pipeline
import pandas as pd

# Define available models
models = {
    "PII": "iiiorg/piiranha-v1-detect-personal-information",
    "PCI": "lakshyakh93/deberta_finetuned_pii",
    "PHI": "obi/deid_roberta_i2b2",
    "Medical NER": "blaze999/Medical-NER"
}

# Store user conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Interactive NER Chatbot")
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose a Model", list(models.keys()))

# Load the selected model lazily
if "active_model" not in st.session_state or st.session_state.active_model != selected_model:
    st.session_state.pipeline = pipeline("token-classification", model=models[selected_model])
    st.session_state.active_model = selected_model

# Confidence threshold
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.75, 0.05)

# User input
user_input = st.chat_input("Type a sentence...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    
    # Perform NER
    ner_results = st.session_state.pipeline(user_input)
    
    # Filter entities by threshold
    filtered_results = [ent for ent in ner_results if ent['score'] >= threshold]
    
    response = ""
    if filtered_results:
        response = "Detected Entities:\n" + "\n".join(
            [f"- {ent['word']} ({ent['entity']}) - {ent['score']:.2f}" for ent in filtered_results]
        )
    else:
        response = "No significant entities detected."
    
    st.session_state.chat_history.append(("bot", response))

# Display chat history
for role, message in st.session_state.chat_history:
    with st.chat_message("assistant" if role == "bot" else "user"):
        st.markdown(message)
