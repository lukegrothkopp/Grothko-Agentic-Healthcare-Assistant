# pages/2_Clinician_Console.py
import os, json
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agents.graph_agent import build_graph
from utils.database_ops import get_patient_record, update_patient_record

load_dotenv()
for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "SERPAPI_API_KEY", "CLINICIAN_TOKEN"):
    if k in st.secrets and st.secrets[k]:
        os.environ[k] = str(st.secrets[k]).strip()

st.set_page_config(page_title="Clinician Console", layout="wide")
st.title("👩‍⚕️ Clinician Console")

# Gate with a clinician code
required = os.environ.get("CLINICIAN_TOKEN") or st.secrets.get("CLINICIAN_TOKEN", "")
code = st.sidebar.text_input("Access code", type="password")
if required and code.strip() != str(required).strip():
    st.warning("Enter a valid clinician access code to view this console.")
    st.stop()

st.caption("Admin & workflow tooling for clinicians/admins. No medical advice is generated.")

graph = build_graph(model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

# Patient selector
pid = st.text_input("Patient ID", "patient_001")
col1, col2 = st.columns(2)

with col1:
    st.subheader("View history")
    if st.button("Load history"):
        rec = get_patient_record(pid)
        if rec: st.json(rec)
        else: st.info("No record found.")

with col2:
    st.subheader("Add note")
    note = st.text_area("New note (will be merged into record)")
    if st.button("Save note"):
        if note.strip():
            update_patient_record(pid, {"latest_note": note.strip()})
            st.success("Note saved.")
        else:
            st.info("Type a note first.")

st.markdown("---")
st.subheader("Book appointment (natural language)")
appt_text = st.text_input(
    "Describe the appointment",
    value="Book a hypertension follow-up next Monday"
)
if st.button("Book"):
    with st.spinner("Booking…"):
        try:
            state = {
                "messages": [HumanMessage(content=appt_text)],
                "intent": None,
                "result": None,
                "patient_id": pid
            }
            result = graph.invoke(state)
            st.success(result["messages"][-1].content)
        except Exception as e:
            st.error(f"Failed: {e}")
