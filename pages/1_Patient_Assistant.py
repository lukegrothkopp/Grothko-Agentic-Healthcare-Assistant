# pages/1_Patient_Assistant.py
from __future__ import annotations

import os
import json
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage

from agents.graph_agent import build_graph
from utils.patient_memory import PatientMemory
from tools.booking_tool import book_appointment

# ----- env + secrets → env -----
load_dotenv()
for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "SERPAPI_API_KEY"):
    try:
        v = st.secrets.get(k)
        if v:
            os.environ[k] = str(v).strip()
    except Exception:
        pass

st.set_page_config(page_title="Patient Assistant", layout="wide")
st.title("🩺 Patient Assistant")
st.caption("Demo — not medical advice. Provides high-level info and admin logistics only.")

# ----- memory -----
if "pmemory" not in st.session_state:
    st.session_state.pmemory = PatientMemory()
mem: PatientMemory = st.session_state.pmemory

# ----- graph (cache once in session) -----
if "graph" not in st.session_state:
    st.session_state.graph = build_graph(model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
graph = st.session_state.graph

# ----- sidebar: pick patient + (restored) booking widget toggle -----
with st.sidebar:
    st.header("Your Info")
    directory = mem.list_patients()
    if directory:
        label_map = {f"{r['name']} ({r['patient_id']})": r["patient_id"] for r in directory}
        sel_label = st.selectbox("Select patient", list(label_map.keys()), index=0)
        patient_id = label_map[sel_label]
    else:
        patient_id = st.text_input("Patient ID", "patient_001")

    st.caption("Tip: You can use natural phrases like “book a cardiologist next Monday”.")
    st.markdown("---")
    show_booking = st.toggle("Show booking panel", value=True, help="Toggle the appointment panel below.")

# ========================
# Chat UI
# ========================
st.subheader("Ask for help")

if "messages" not in st.session_state:
    st.session_state.messages = []

# render prior messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# chat input
if prompt := st.chat_input("How can I help you today?"):
    # user bubble
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # resolve patient id from text; default to selection
    resolved_pid = mem.resolve_from_text(prompt, default=patient_id) or patient_id or "session"

    # call the graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                state = {
                    "messages": [HumanMessage(content=prompt)],
                    "intent": None,
                    "result": None,
                    "patient_id": resolved_pid,
                }
                result_state = graph.invoke(state)
                answer = result_state["messages"][-1].content
            except Exception as e:
                answer = f"Sorry—there was an error: {e}"

            st.markdown(answer)

    # log conversation into memory so clinician timeline sees it
    try:
        mem.record_event(
            resolved_pid,
            f"Patient asked: {prompt}\nAssistant: {answer}",
            meta={"kind": "chat", "by": "patient"},
        )
    except Exception:
        pass

    # save to session
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ========================
# (Restored) Booking Panel
# ========================
if show_booking:
    st.markdown("---")
    st.subheader("Book an appointment")

    tabs = st.tabs(["Quick Form", "Natural Language"])

    # --- Quick Form ---
    with tabs[0]:
        col1, col2, col3 = st.columns([1.2, 1, 1])
        with col1:
            target_pid = st.text_input("Patient ID", value=patient_id, help="Defaults to the patient you’re viewing.")
        with col2:
            doctor = st.text_input("Doctor / Specialty", value="Primary Care")
        with col3:
            date_sel = st.date_input("Date")

        if st.button("Book via Quick Form", use_container_width=True, key="book_qf_btn"):
            try:
                payload = {
                    "patient_id": target_pid.strip(),
                    "doctor_name": doctor.strip(),
                    "appointment_date": str(date_sel),  # YYYY-MM-DD
                }
                # booking tool understands JSON payload
                result_msg = book_appointment(json.dumps(payload))
                st.success(result_msg)

                # persist to runtime memory for instant UI reflection
                mem.record_event(
                    target_pid,
                    f"Booked appointment with Dr. {doctor} on {date_sel}.",
                    meta={"kind": "appointment", "doctor": doctor, "date": str(date_sel)},
                )

                # best-effort reflect into base appointments quickly
                try:
                    p = mem.patients.get(target_pid)
                    base = p.data if hasattr(p, "data") and isinstance(p.data, dict) else (p if isinstance(p, dict) else {})
                    appts = base.setdefault("appointments", [])
                    appts.append({"date": str(date_sel), "doctor": doctor, "status": "scheduled"})
                except Exception:
                    pass

                st.rerun()
            except Exception as e:
                st.error(f"Booking failed: {e}")

    # --- Natural Language ---
    with tabs[1]:
        nl_default = f"book {patient_id} with Dr. Lee next Monday"
        nl_text = st.text_input(
            "Describe the appointment",
            value=nl_default,
            help="Examples: 'book patient_001 with Dr. Lee tomorrow' or 'book patient_701 cardiology next Wednesday'",
        )
        if st.button("Book via Natural Language", use_container_width=True, key="book_nl_btn"):
            try:
                # booking tool also accepts free text (“today”, “tomorrow”, “next Monday”)
                result_msg = book_appointment(nl_text)
                st.success(result_msg)

                # log to selected patient for immediate visibility
                mem.record_event(
                    patient_id,
                    f"Booked appointment (NL): {nl_text}",
                    meta={"kind": "appointment_nl"},
                )
                st.rerun()
            except Exception as e:
                st.error(f"Booking failed: {e}")
