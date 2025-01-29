import streamlit as st
from transformers import pipeline
import json

# Load models dynamically
def load_model(model_name):
    return pipeline("token-classification", model=model_name)

# Model mapping
MODEL_MAP = {
    "PII": "iiiorg/piiranha-v1-detect-personal-information",
    "PCI": "lakshyakh93/deberta_finetuned_pii",
    "PHI": "obi/deid_roberta_i2b2",
    "Medical NER": "blaze999/Medical-NER"
}

# Simulated GPT function to interpret user queries
def generate_model_config(user_query):
    """Uses an LLM to generate model selection and parameters based on the query."""
    # Simulated JSON response from an LLM
    gpt_response = {
        "models": ["PII", "Medical NER"],
        "thresholds": {
            "PII": 0.8,
            "Medical NER": 0.7
        }
    }
    return gpt_response

# Function to process text based on LLM-configured models
def process_text(text, user_query):
    config = generate_model_config(user_query)
    selected_models = config["models"]
    thresholds = config["thresholds"]
    
    results = []
    for model_name in selected_models:
        model = load_model(MODEL_MAP[model_name])
        model_results = model(text)
        # Filter by threshold
        filtered_results = [ent for ent in model_results if ent['score'] >= thresholds.get(model_name, 0.5)]
        for ent in filtered_results:
            ent["model"] = model_name
        results.extend(filtered_results)
    return results

# Streamlit UI
st.title("Smart AI NER Chatbot")
user_query = st.text_input("Describe what you need (e.g., 'Find all personal data'):")
input_text = st.text_area("Enter text for analysis:")

if st.button("Run Analysis"):
    if user_query and input_text:
        entities = process_text(input_text, user_query)
        st.subheader("Extracted Entities:")
        st.json(entities)  # Display results as JSON
    else:
        st.warning("Please enter a query and text for analysis.")
