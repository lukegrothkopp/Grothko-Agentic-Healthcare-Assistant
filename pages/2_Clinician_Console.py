# pages/2_Clinician_Console.py
from __future__ import annotations

import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.patient_memory import PatientMemory
from utils.database_ops import get_patient_record, update_patient_record

# ---------- boot ----------
load_dotenv()
for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "SERPAPI_API_KEY", "ADMIN_TOKEN"):
    if k in st.secrets and st.secrets[k]:
        os.environ[k] = str(st.secrets[k]).strip()

st.set_page_config(page_title="Clinician Console", page_icon="🩺", layout="wide")
st.title("🩺 Clinician Console")
st.caption("Demo — not medical advice. View patient context, recent activity, and plans.")

# ---------- session singletons ----------
if "pmemory" not in st.session_state:
    st.session_state.pmemory = PatientMemory()
_mem: PatientMemory = st.session_state.pmemory

# ---------- helpers ----------
def _latest_plan_for_patient(pid: str):
    """Return (query, steps) for the most recent plan in data/traces.jsonl for this patient."""
    traces_path = Path("data/traces.jsonl")
    if not traces_path.exists():
        return None, None
    try:
        lines = traces_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None, None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("type") == "plan" and row.get("patient_id") == pid:
            return row.get("query"), row.get("steps") or []
    return None, None

def _get_patient_dir_row(pid: str) -> dict | None:
    """Combine seed directory info with DB record for a richer summary."""
    sd = _mem.patients.get(pid).data if pid in _mem.patients else {}
    db = get_patient_record(pid) or {}
    merged = dict(sd)
    # merge DB on top (DB wins)
    for k, v in (db or {}).items():
        merged[k] = v
    merged["patient_id"] = pid
    return merged or None

# ---------- sidebar patient picker ----------
with st.sidebar:
    st.header("Patient")
    directory = _mem.list_patients()
    if not directory:
        st.warning("No patients loaded. Use the Developer Console to import seeds.")
        st.stop()

    label_map = {f"{r['name']} ({r['patient_id']})": r["patient_id"] for r in directory}
    sel_label = st.selectbox("Select patient", list(label_map.keys()))
    pid = label_map[sel_label]

# ---------- top section: patient overview ----------
info = _get_patient_dir_row(pid)
if info:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader(info.get("name", pid))
        st.write(f"**Patient ID:** {pid}")
        if info.get("age") is not None:
            st.write(f"**Age:** {info['age']}")
        if info.get("conditions"):
            st.write("**Conditions:**", ", ".join(map(str, info.get("conditions", []))) )
    with col2:
        st.subheader("Summary")
        st.write(_mem.get_summary(pid) or "—")
    with col3:
        st.subheader("Appointments")
        appts = info.get("appointments") or []
        if appts:
            df = pd.DataFrame([{k: a.get(k) for k in a.keys()} for a in appts])
            st.dataframe(df, use_container_width=True, height=180)
        else:
            st.write("—")
else:
    st.info("No summary available for this patient.")

# ---------- recent activity ----------
st.markdown("---")
st.subheader("Recent Activity")
try:
    window = _mem.get_window(pid, k=8)
except Exception as e:
    st.error(f"Could not load recent activity: {e}")
    window = []

if window:
    for row in window:
        ts = row.get("ts", "—")
        typ = row.get("type") or row.get("tag") or "event"
        txt = row.get("text") or row.get("notes") or row.get("diagnosis") or json.dumps(row)[:200]
        st.write(f"- [{typ} @ {ts}] {txt}")
else:
    st.write("No recent activity.")

# ---------- latest plan widget ----------
st.markdown("---")
st.subheader("Latest agent plan for this patient")
q, steps = _latest_plan_for_patient(pid)
if q or steps:
    if q:
        st.write("**User request that was planned:**", q)
    if steps:
        st.markdown("**Plan steps:**")
        for i, s in enumerate(steps, 1):
            st.write(f"{i}. {s}")
else:
    st.info("No plan logged for this patient yet. Trigger a request on the Patient page to generate one.")

# ---------- clinician note add ----------
st.markdown("---")
st.subheader("Add a clinician note")
note = st.text_area("Note", placeholder="e.g., Follow-up needed on BP logs; check labs next visit.")
if st.button("Save note"):
    try:
        # Save to DB
        ok = update_patient_record(pid, {"latest_note": note})
        # Record into memory log so it shows up under Recent Activity
        _mem.record_event(pid, text=f"[Clinician note] {note}", meta={"source": "clinician_console"})
        if ok:
            st.success("Note saved.")
        else:
            st.info("Record updated locally (DB may be read-only in this environment).")
    except Exception as e:
        st.error(f"Failed to save note: {e}")
