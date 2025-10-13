# app.py
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Map secrets => env (works on Streamlit Cloud)
for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "SERPAPI_API_KEY", "ADMIN_TOKEN", "CLINICIAN_TOKEN"):
    if k in st.secrets and st.secrets[k]:
        os.environ[k] = str(st.secrets[k]).strip()

st.set_page_config(page_title="Grothko Agentic Healthcare Assistant", layout="wide")
st.title("🩺 Grothko Agentic Healthcare Assistant")

st.markdown("""
**Welcome!** Choose your area from the left sidebar:
- **Patient Assistant** – friendly chat for high-level info and admin tasks (no medical advice).
- **Clinician Console** – view/update records, book appointments with explicit controls.
- **Developer Console** – seed DB, build FAISS index, run QA evals, debug.
""")

st.info("This landing page is public. The Clinician/Developer pages are gated by an access code.")
