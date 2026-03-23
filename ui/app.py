import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def get_models() -> list[str]:
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return models or ["distilbert_model", "lstm_model"]
    except requests.RequestException:
        return ["distilbert_model", "lstm_model"]


st.set_page_config(page_title="ABTE Demo", page_icon="🍽️", layout="centered")
st.title("Aspect-Based Term Extraction")
st.caption("Infer aspect terms from restaurant reviews using FastAPI + Streamlit.")

available_models = get_models()
selected_model = st.selectbox("Choose model", available_models, index=0)

default_text = "The bread is top notch as well but the room smells very bad"
text = st.text_area("Input sentence", value=default_text, height=120)

if st.button("Run inference", type="primary"):
    if not text.strip():
        st.warning("Please enter a non-empty sentence.")
    else:
        payload = {"text": text.strip(), "model_name": selected_model}
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
            if response.status_code >= 400:
                detail = None
                try:
                    detail = response.json().get("detail")
                except ValueError:
                    detail = response.text
                st.error(f"API error ({response.status_code}): {detail or 'Unknown error'}")
                st.stop()

            result = response.json()

            st.subheader("Extracted terms")
            terms = result.get("terms", [])
            st.write(terms if terms else "No aspect term detected.")

            st.subheader("Token labels")
            tokens = result.get("tokens", [])
            labels = result.get("labels", [])
            st.table({"token": tokens, "label": labels})
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
