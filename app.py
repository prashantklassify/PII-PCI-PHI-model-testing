import streamlit as st
from transformers import pipeline

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

# Function to format NER results
def format_ner_results(ner_results, model_name):
    formatted_output = f"### {model_name} Results ###\n"
    entity_buffer = ""
    confidences = []
    last_entity_type = None

    for entity in ner_results:
        token = clean_token(entity['word'])  # Clean the token
        confidence = entity['score'] * 100  # Convert to percentage
        entity_type = entity['entity']

        # If continuing the same entity type, concatenate tokens
        if last_entity_type and last_entity_type.split("-")[-1] == entity_type.split("-")[-1]:
            entity_buffer += token
            confidences.append(confidence)
        else:
            # If a new entity starts, print the previous one
            if entity_buffer:
                avg_confidence = sum(confidences) / len(confidences)
                formatted_output += f"- **{entity_buffer}**\n  - Confidence: {', '.join([f'{c:.2f}%' for c in confidences])}\n  - Detected as: {last_entity_type.split('-')[-1]}\n"
            
            # Start a new entity
            entity_buffer = token
            confidences = [confidence]
        last_entity_type = entity_type

    # Output the last entity if available
    if entity_buffer:
        avg_confidence = sum(confidences) / len(confidences)
        formatted_output += f"- **{entity_buffer}**\n  - Confidence: {', '.join([f'{c:.2f}%' for c in confidences])}\n  - Detected as: {last_entity_type.split('-')[-1]}\n"

    return formatted_output

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

        # Display formatted NER results
        formatted_results = format_ner_results(filtered_predictions, model_name)
        st.markdown(formatted_results)

