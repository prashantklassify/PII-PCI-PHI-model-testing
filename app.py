import streamlit as st
from gliner import GLiNER
import re

@st.cache_resource
def load_model():
    return GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

model = load_model()

ALL_LABELS = [
    "person", "organization", "phone number", "address", "passport number", "email", "credit card number",
    "social security number", "health insurance id number", "date of birth", "mobile phone number",
    "bank account number", "medication", "cpf", "driver's license number", "tax identification number",
    "medical condition", "identity card number", "national id number", "ip address", "email address", "iban",
    "credit card expiration date", "username", "health insurance number", "registration number", "student id number",
    "insurance number", "flight number", "landline phone number", "blood type", "cvv", "reservation number",
    "digital signature", "social media handle", "license plate number", "cnpj", "postal code", "passport_number",
    "serial number", "vehicle registration number", "credit card brand", "fax number", "visa number",
    "insurance company", "identity document number", "transaction number", "national health insurance number",
    "cvc", "birth certificate number", "train ticket number", "passport expiration date", "social_security_number"
]

# Regex validation rules for numeric PII types
VALIDATION_REGEX = {
    "phone number": r"^\+?\d[\d\s\-()]{7,}$",
    "mobile phone number": r"^\+?\d[\d\s\-()]{7,}$",
    "landline phone number": r"^\+?\d[\d\s\-()]{7,}$",
    "email": r"^[\w\.-]+@[\w\.-]+\.\w+$",
    "email address": r"^[\w\.-]+@[\w\.-]+\.\w+$",
    "credit card number": r"^(?:\d[ -]*?){13,19}$",
    "cvv": r"^\d{3,4}$",
    "cvc": r"^\d{3,4}$",
    "iban": r"^[A-Z]{2}\d{2}[A-Z0-9]{1,30}$",
    "passport number": r"^[A-Z0-9]{6,9}$",
    "passport_number": r"^[A-Z0-9]{6,9}$",
    "social security number": r"^\d{3}-\d{2}-\d{4}$",
    "social_security_number": r"^\d{3}-\d{2}-\d{4}$",
    "ip address": r"^(\d{1,3}\.){3}\d{1,3}$",
    "credit card expiration date": r"^(0[1-9]|1[0-2])\/\d{2,4}$"
}

def validate_entity(entity):
    label = entity["label"].lower()
    value = entity["text"].strip()
    if label in VALIDATION_REGEX:
        return re.match(VALIDATION_REGEX[label], value) is not None
    return True  # No regex: assume valid

# Streamlit UI
st.set_page_config(page_title="GLiNER + Regex Validator", layout="wide")
st.title("🔐 GLiNER Entity Extraction with Regex Verification")

text_input = st.text_area("📝 Enter text to analyze:", height=250, value="""
Harilala Rasoanaivo, un homme d'affaires local d'Antananarivo, a enregistré une nouvelle société nommée "Rasoanaivo Enterprises" au Lot II M 92 Antohomadinika. Son numéro est le +261 32 22 345 67, et son adresse électronique est harilala.rasoanaivo@telma.mg. Il a fourni son numéro de sécu 501-02-1234 pour l'enregistrement.
""")

selected_labels = st.multiselect("🏷️ Select entity labels to detect:", options=ALL_LABELS, default=[
    "person", "organization", "phone number", "email", "social security number"
])

if st.button("🚀 Extract Entities"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    elif not selected_labels:
        st.warning("Please select at least one label.")
    else:
        with st.spinner("Running GLiNER model..."):
            raw_entities = model.predict_entities(text_input, selected_labels)

        validated_entities = []
        for ent in raw_entities:
            is_valid = validate_entity(ent)
            ent["valid"] = is_valid
            validated_entities.append(ent)

        if validated_entities:
            st.success(f"✅ Found {len(validated_entities)} entity{'ies' if len(validated_entities) > 1 else ''}.")
            st.dataframe([
                {
                    "Entity Text": ent["text"],
                    "Label": ent["label"],
                    "Valid Format": "✅" if ent["valid"] else "❌"
                }
                for ent in validated_entities
            ])
        else:
            st.info("No entities found.")
