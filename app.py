import streamlit as st
from transformers import pipeline
import pandas as pd

# Define the models for NER
models = {
    "PII": "iiiorg/piiranha-v1-detect-personal-information",
    "PHI": "obi/deid_roberta_i2b2",
    "Medical NER": "blaze999/Medical-NER",
    "PCI": "lakshyakh93/deberta_finetuned_pii"
}

# Function to clean tokens
def clean_token(token):
    """Clean token by removing unwanted characters like '▁' and 'Ġ'."""
    return token.replace("▁", "").replace("Ġ", "")

# Function to format NER results into a DataFrame
def format_ner_results_as_table(ner_results):
    entity_list = []
    
    for entity in ner_results:
        token = clean_token(entity['word'])
        confidence = entity['score'] * 100  # Convert to percentage
        entity_type = entity['entity'].split("-")[-1]  # Get entity type
        
        # Append to the list as a dictionary
        entity_list.append({
            "Entity": token,
            "Entity Type": entity_type,
            "Confidence (%)": f"{confidence:.2f}"
        })
    
    # Convert list to DataFrame for tabular display
    return pd.DataFrame(entity_list)

# Function to filter predictions by confidence threshold
def filter_by_confidence(predictions, threshold=0.5):
    """Filter predictions to only include those with a confidence above the threshold."""
    return [prediction for prediction in predictions if prediction['score'] > threshold]

# Streamlit App Layout
st.title("Named Entity Recognition (NER) Streamlit App")

# User input for text
text = st.text_area("Enter text for NER processing", "Patient Brijesh Kumar admitted in the room no 101 in glacier hospital has blood pressure over 140 and heart rate of 83bpm. The patient wants to avail no txn cost from insurance provider. Insurance number of FHZPB1650J and rest of the payment will be done by card number 4111 1111 1111 1111.")

# Confidence threshold input
confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

# Button to process the text
if st.button("Run NER Models"):
    for model_name, model_path in models.items():
        st.subheader(f"{model_name} Model Results")

        # Load the model
        pipe = pipeline("token-classification", model=model_path)
        
        # Get predictions
        predictions = pipe(text)

        # Filter predictions by confidence
        filtered_predictions = filter_by_confidence(predictions, confidence_threshold)

        # Convert NER results to a table format
        if filtered_predictions:
            ner_table = format_ner_results_as_table(filtered_predictions)
            st.table(ner_table)
        else:
            st.write(f"No entities detected above the confidence threshold for {model_name}.")
