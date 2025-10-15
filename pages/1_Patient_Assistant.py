# pages/1_Patient_Assistant.py
import os
import json
from datetime import date
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from agents.graph_agent import build_graph
from utils.patient_memory import PatientMemory
from tools.booking_tool import get_booking_tool

load_dotenv()

st.set_page_config(page_title="Patient Assistant", layout="wide")
st.title("🩺 Patient Assistant")
st.caption("Ask for help with scheduling, records, and general info from trusted sources. (No medical advice.)")

# --- Singletons ---
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()  # works with/without OPENAI key
if "pmemory" not in st.session_state:
    st.session_state.pmemory = PatientMemory()
if "booking_tool" not in st.session_state:
    st.session_state.booking_tool = get_booking_tool()

graph = st.session_state.graph
mem = st.session_state.pmemory
booking_tool = st.session_state.booking_tool

# --- Safe logging wrapper (handles both old/new PatientMemory) ---
def _safe_log(mem_obj: PatientMemory, pid: str, role: str, content: str):
    try:
        fn = getattr(mem_obj, "add_message", None)
        if callable(fn):
            fn(pid or "session", role, content)
            return
        # Fallback to record_event if add_message doesn't exist
        fn2 = getattr(mem_obj, "record_event", None)
        if callable(fn2):
            fn2(pid or "session", f"[{role}] {content}", meta={"role": role})
    except Exception:
        # Don't let logging failures break UX
        pass

# --- Tabs ---
tab_general, tab_schedule = st.tabs(["General Assistant", "Quick Schedule"])

# =========================
# Tab 1: General Assistant
# =========================
with tab_general:
    st.subheader("How can we help you today?")
    prompt = st.text_input(
        "Type your request",
        placeholder="e.g., My 70-year-old father has chronic kidney disease — please book a nephrologist at North Clinic and summarize current treatments",
        key="general_prompt",
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        run_btn = st.button("Submit", key="general_submit")
    with c2:
        clear_btn = st.button("Clear", key="general_clear")

    if clear_btn:
        st.session_state.pop("last_response_general", None)
        st.session_state.pop("last_patient_general", None)
        st.rerun()

    if run_btn and prompt.strip():
        # Resolve patient id from free text (Hal/Marisol/Ethan if their seeds are present)
        pid = mem.resolve_from_text(prompt) or "session"

        # Log user message safely
        _safe_log(mem, pid, "user", prompt)

        # Run the agent graph
        try:
            state_in = {"messages": [HumanMessage(content=prompt)], "patient_id": pid}
            state_out = graph.invoke(state_in)

            # Prefer the last AI message; fall back to 'result'
            assistant_text = ""
            msgs = state_out.get("messages", [])
            for m in reversed(msgs):
                if isinstance(m, AIMessage):
                    assistant_text = m.content
                    break
            if not assistant_text:
                assistant_text = state_out.get("result", "") or "(no result returned)"

            # Log assistant message safely
            _safe_log(mem, pid, "assistant", assistant_text)

            # UI
            st.session_state.last_response_general = assistant_text
            st.session_state.last_patient_general = pid
            st.success(f"Resolved patient: {pid}")
            st.write(assistant_text)

        except Exception as e:
            st.error(f"Run failed: {e}")

    if st.session_state.get("last_response_general"):
        pid_echo = st.session_state.get("last_patient_general")
        if pid_echo:
            st.caption(f"(Last resolved patient: {pid_echo})")
        st.write(st.session_state["last_response_general"])

# =========================
# Tab 2: Quick Schedule
# =========================
with tab_schedule:
    st.subheader("Book an appointment (simple form)")

    # Load patients for dropdown
    patients = mem.list_patients()
    if not patients:
        st.info(
            "No patients loaded. Add seed files to OFFLINE_PATIENT_DIR (default: data/patient_memory) "
            "or use the Developer Console → Patient Seeds to import JSON."
        )

    pretty_labels = [
        f'{p["patient_id"]} — {p.get("name","") or "(no name)"}'
        for p in patients
    ] if patients else ["(none)"]

    sel_ix = 0 if patients else 0
    sel_label = st.selectbox(
        "Patient",
        options=pretty_labels,
        index=sel_ix,
        disabled=not patients,
        key="sched_patient_label",
    )
    pid = None
    if patients:
        pid = patients[pretty_labels.index(sel_label)]["patient_id"]

    # Pull booking hints from memory (clinic/modes/time window)
    hints = mem.booking_hints(pid) if pid else {}
    default_clinic = hints.get("clinic") or ""
    default_modes = hints.get("modes") or []
    default_tod = hints.get("preferred_time_of_day") or "morning"
    dr_placeholder = "e.g., Nephrologist or Dr. Jane Lee"

    cA, cB = st.columns([2, 1])
    with cA:
        doctor_name = st.text_input("Doctor or Specialty", value="", placeholder=dr_placeholder, key="sched_doc")
    with cB:
        clinic = st.text_input("Clinic (optional)", value=default_clinic, key="sched_clinic")

    modes = st.multiselect(
        "Appointment mode(s)",
        options=["in-person", "video", "phone"],
        default=default_modes if default_modes else ["in-person"],
        key="sched_modes",
    )
    tod = st.selectbox(
        "Preferred time of day (optional)",
        options=["morning", "afternoon", "evening"],
        index=["morning", "afternoon", "evening"].index(default_tod) if default_tod in ["morning","afternoon","evening"] else 0,
        key="sched_tod",
    )

    # Default date: start of suggested date_range if present, else today
    try:
        start_iso = (hints.get("date_range") or {}).get("start")
        default_date = date.fromisoformat(start_iso) if start_iso else date.today()
    except Exception:
        default_date = date.today()
    appt_date = st.date_input("Appointment date", value=default_date, key="sched_date")

    reason = st.text_area("Reason/Notes (optional)", placeholder="Short reason for visit…", key="sched_reason")

    book_clicked = st.button("Book Appointment", type="primary", disabled=not pid, key="sched_submit")

    if book_clicked and pid:
        payload = {
            "patient_id": pid,
            "doctor_name": (doctor_name or "Primary Care").strip(),
            "appointment_date": appt_date.isoformat(),
        }
        if clinic.strip():
            payload["clinic"] = clinic.strip()
        if modes:
            payload["modes"] = modes
        if tod:
            payload["preferred_time_of_day"] = tod
        if reason.strip():
            payload["reason"] = reason.strip()

        try:
            result = booking_tool.func(json.dumps(payload))
            result_str = result if isinstance(result, str) else json.dumps(result)
            st.success("Booking request sent.")
            st.code(result_str, language="json")
            _safe_log(mem, pid, "user", f"[QuickSchedule] {json.dumps(payload)}")
            _safe_log(mem, pid, "assistant", f"[QuickSchedule Result] {result_str}")
        except Exception as e:
            st.error(f"Booking failed: {e}")

# Footer diagnostics
with st.expander("Technical details"):
    st.write({
        "OFFLINE_PATIENT_DIR": os.getenv("OFFLINE_PATIENT_DIR", "data/patient_memory"),
        "Patients loaded": len(mem.patients),
    })
