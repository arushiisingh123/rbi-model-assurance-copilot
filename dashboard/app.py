"""Phase 0 Streamlit skeleton (owner: Khushi).

Displays the FastAPI mock endpoint's data. No real analytics yet --
that is Phase 4 work (see docs/development-phases.md).

Run locally (with the API already running separately):
    streamlit run dashboard/app.py
"""
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/mock-assurance-result"

FALLBACK_MOCK_RESULT = {
    "model": {"status": "PASS", "is_mock": True},
    "explainability": {"status": "PASS", "is_mock": True},
    "fairness_drift": {"status": "WARNING", "is_mock": True},
    "compliance": {"status": "PENDING", "is_mock": True},
    "note": "SYNTHETIC / MOCK DATA. Not real results. Phase 0 skeleton only.",
}

st.set_page_config(page_title="AI Model Risk & Assurance Copilot", layout="centered")
st.title("AI Model Risk & Assurance Copilot")
st.caption("Phase 0 — Foundation skeleton. All data below is mock/synthetic.")

st.warning("SYNTHETIC / MOCK DATA — NOT REAL CREDIT DATA — NOT FOR PRODUCTION USE")

try:
    response = requests.get(API_URL, timeout=2)
    response.raise_for_status()
    result = response.json()
    st.success("Loaded mock result from the FastAPI backend.")
except requests.exceptions.RequestException:
    result = FALLBACK_MOCK_RESULT
    st.info("FastAPI backend not reachable — showing built-in fallback mock data instead.")

for section, data in result.items():
    if section == "note":
        continue
    st.subheader(section.replace("_", " ").title())
    st.json(data)

st.caption(result.get("note", ""))
