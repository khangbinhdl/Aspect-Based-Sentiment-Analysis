import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def get_models() -> dict:
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        resp.raise_for_status()
        models = resp.json()
        if not isinstance(models, dict):
            return {"abte": ["abte-minilm"], "absc": ["absc-minilm"]}
        return {
            "abte": models.get("abte", []) or ["abte-minilm"],
            "absc": models.get("absc", []) or ["absc-minilm"],
        }
    except requests.RequestException:
        return {"abte": ["abte-minilm"], "absc": ["absc-minilm"]}


st.set_page_config(page_title="ABSA Demo", page_icon="🍽️", layout="centered")
st.title("Aspect-Based Sentiment Analysis")
st.caption("Extract aspect terms (ABTE) and classify sentiment (ABSC).")

available_models = get_models()
abte_model = st.selectbox("ABTE model", available_models["abte"], index=0)
absc_model = st.selectbox("ABSC model", available_models["absc"], index=0)
device = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)

default_text = "The bread is top notch as well but the room smells very bad"
text = st.text_area("Input sentence", value=default_text, height=120)
term_override = st.text_input("Optional aspect term", value="")

if st.button("Run inference", type="primary"):
    if not text.strip():
        st.warning("Please enter a non-empty sentence.")
    else:
        payload = {
            "text": text.strip(),
            "abte_model_name": abte_model,
            "absc_model_name": absc_model,
            "term": term_override.strip() or None,
            "device": device,
        }
        try:
            progress = st.progress(0, text="Running inference...")
            with st.spinner("Calling API..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
                progress.progress(100, text="Done")
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

            st.subheader("Sentiment results")
            results = result.get("results", [])
            if results:
                st.table(results)
            else:
                st.write("No sentiment results.")

            tokens = result.get("tokens", [])
            labels = result.get("labels", [])
            if tokens and labels:
                st.subheader("Token labels")
                st.table({"token": tokens, "label": labels})
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
        finally:
            try:
                progress.empty()
            except Exception:
                pass
