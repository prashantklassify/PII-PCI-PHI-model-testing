import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import json

# Load Sentence Transformer model for semantic search
st_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load GPT-Neo model from Hugging Face
model_name = "EleutherAI/gpt-neo-2.7B"  # You can use other available models as well
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to process the query using GPT-Neo and return a JSON-like structure
def query_gpt_neo_for_model_selection(user_query):
    input_text = f"Given the following query, suggest the models to use for NER extraction and any specific entities to focus on. Return a JSON with 'selected_model' and 'enabled_entities'. Query: {user_query}"
    
    # Tokenize input and generate a response
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse the response to extract JSON-like structure
    try:
        start_index = decoded_output.find('{')
        end_index = decoded_output.rfind('}') + 1
        json_output = decoded_output[start_index:end_index]
        return json.loads(json_output)
    except Exception as e:
        print("Error parsing GPT-Neo response:", e)
        return {}

# Function to process extracted entities based on the LLM response
def handle_entities_based_on_response(user_query, extracted_entities, model_response):
    enabled_entities = model_response.get("enabled_entities", [])
    return [ent for ent in extracted_entities if ent['entity'] in enabled_entities]

# Function to process complex queries
def handle_complex_queries(user_query, extracted_entities, model_response):
    if "last names" in user_query.lower():
        return [ent for ent in extracted_entities if ent['entity'] in ['SURNAME']]
    elif "PII entries excluding names" in user_query.lower():
        return [ent for ent in extracted_entities if ent['entity'] not in ['GIVENNAME', 'SURNAME']]
    elif "PII and PCI entries" in user_query.lower():
        return [ent for ent in extracted_entities if ent['entity'] in model_response.get("enabled_entities", [])]
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
            # Step 1: Query GPT-Neo for model and configuration suggestions
            model_response = query_gpt_neo_for_model_selection(user_query)
            
            # Step 2: Select the right model based on the GPT-Neo response
            if model_response:
                model_name = model_response.get("selected_model", "iiiorg/piiranha-v1-detect-personal-information")  # Default to a PII model
                enabled_entities = model_response.get("enabled_entities", [])
                
                # Run the NER pipeline for the selected model
                ner_pipeline = pipeline("ner", model=model_name)
                extracted_entities = ner_pipeline(user_text)
                
                # Filter entities based on the LLM's response
                filtered_entities = handle_entities_based_on_response(user_query, extracted_entities, model_response)
                
                # Process complex queries for additional filtering
                final_entities = handle_complex_queries(user_query, filtered_entities, model_response)
                
                # Add messages to chat history
                st.session_state.chat_history.append(("user", user_query))
                st.session_state.chat_history.append(("bot", f"Model Used: `{model_name}`"))
                st.session_state.chat_history.append(("bot", f"Extracted Entities: {final_entities}"))
            
            else:
                st.warning("Error in processing the query with the model response.")
        
        # Refresh UI
        st.rerun()
    else:
        st.warning("Please enter both a query and text for analysis.")
