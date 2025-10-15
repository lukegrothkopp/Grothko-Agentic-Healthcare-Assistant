# pages/2_Clinician_Console.py
import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from utils.patient_memory import PatientMemory

load_dotenv()

st.set_page_config(page_title="Clinician Console", page_icon="👩🏼‍⚕️", layout="wide")
st.title("👩🏼‍⚕️ Clinician Console")
st.caption("Operational view of patient context, recent events, and quick actions.")

# --- Singleton memory ---
if "pmemory" not in st.session_state:
    st.session_state.pmemory = PatientMemory()
_mem: PatientMemory = st.session_state.pmemory

# --- Patient picker (blank by default) ---
patients = _mem.list_patients()
pretty = [f'{p["patient_id"]} — {p.get("name","") or "(no name)"}' for p in patients]

placeholder = "— Select a patient —"
options = [placeholder] + pretty if pretty else ["— No patients loaded —"]
sel = st.selectbox("Patient", options=options, index=0, disabled=(not pretty))

pid = None
if pretty and sel != placeholder:
    pid = patients[pretty.index(sel)].get("patient_id")

if not pid:
    st.info("Select a patient to view details.")
    st.stop()

# --- Summary header ---
data = _mem.get(pid) or {}
profile = data.get("profile") or {}
name = profile.get("full_name") or pid
st.subheader(name)
st.caption(f"Patient ID: {pid}")
if data.get("summary"):
    st.write(data["summary"])

# --- Columns: Problems / Meds / Latest Labs ---
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Key Problems**")
    probs = data.get("problems") or []
    if not probs:
        st.write("—")
    else:
        for p in probs[:6]:
            st.write(f"- {p.get('name','')}")
with c2:
    st.markdown("**Medications**")
    meds = data.get("medications") or []
    if not meds:
        st.write("—")
    else:
        for m in meds[:8]:
            st.write(f"- {m.get('name','')}: {m.get('dose','')} {m.get('frequency','')}".strip())
with c3:
    st.markdown("**Latest Labs**")
    labs = data.get("labs") or []
    if not labs:
        st.write("—")
    else:
        last = labs[-1]
        vals = last.get("values") or {}
        lines = []
        for k in ("creatinine_mg_dL","egfr_mL_min_1.73m2","a1c_percent","hemoglobin_g_dL","potassium_mmol_L","co2_bicarb_mmol_L","urine_acr_mg_g"):
            if k in vals:
                lines.append(f"- {k}: {vals[k]}")
        st.write("\n".join(lines) if lines else "—")

st.markdown("---")

# --- Recent window (last k entries/messages) ---
st.markdown("### Recent activity")
try:
    window = _mem.get_window(pid, k=8)
except AttributeError:
    st.error("PatientMemory.get_window is missing. Please update utils/patient_memory.py from the latest code.")
    st.stop()

if not window:
    st.write("No recent entries.")
else:
    for role, content, ts in window:
        # Simple role bubble
        with st.chat_message("user" if role.lower() == "user" else "assistant"):
            st.markdown(content)
        st.caption(ts or "")

# --- Footer ---
with st.expander("Technical details"):
    st.write({
        "OFFLINE_PATIENT_DIR": os.getenv("OFFLINE_PATIENT_DIR", "data/patient_memory"),
        "Loaded patients": len(_mem.patients),
    })
