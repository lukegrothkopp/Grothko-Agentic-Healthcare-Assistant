# pages/1_Patient_Assistant.py
from __future__ import annotations

import os
import re
import json
import uuid
from datetime import date, datetime
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
mem: PatientMemory = st.session_state.pmemory
booking_tool = st.session_state.booking_tool

# --- Safe logging wrapper (handles both old/new PatientMemory) ---
def _safe_log(mem_obj: PatientMemory, pid: str, role: str, content: str):
    try:
        fn = getattr(mem_obj, "add_message", None)
        if callable(fn):
            fn(pid or "session", role, content)
            return
        fn2 = getattr(mem_obj, "record_event", None)
        if callable(fn2):
            fn2(pid or "session", f"[{role}] {content}", meta={"role": role})
    except Exception:
        pass

# --- Helpers for Quick Schedule ---
def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _find_patient_id_by_name(mem_obj: PatientMemory, full_name: str) -> str | None:
    target = _norm_name(full_name)
    for pid, data in (mem_obj.patients or {}).items():
        name = _norm_name(((data.get("profile") or {}).get("full_name") or ""))
        if name and name == target:
            return pid
    return None

_PAT_ID_RE = re.compile(r"^patient_(\d+)$")
def _next_patient_id(mem_obj: PatientMemory) -> str:
    max_n = 700  # start near our examples
    for pid in (mem_obj.patients or {}):
        m = _PAT_ID_RE.match(pid)
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except Exception:
                continue
    return f"patient_{max_n + 1}"

def _maybe_int_age(dob: date | None, fallback: int | None) -> int | None:
    if isinstance(dob, date):
        today = date.today()
        years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return max(0, years)
    return fallback if (isinstance(fallback, int) and fallback >= 0) else None

def _ensure_patient_from_form(
    mem_obj: PatientMemory,
    full_name: str | None,
    dob: date | None,
    age_input: int | None,
    sex: str | None,
    phone: str | None,
    email: str | None,
    address: str | None,
    prefs: dict | None,
    booking_meta: dict | None,
    reason: str | None,
) -> str:
    """
    If name matches an existing patient (case-insensitive), update optional fields and return pid.
    Otherwise create a new patient JSON (saved into OFFLINE_PATIENT_DIR) and return the new pid.
    """
    full_name = (full_name or "").strip()
    pid = _find_patient_id_by_name(mem_obj, full_name) if full_name else None

    # Prepare profile fields
    age_val = _maybe_int_age(dob, age_input)
    profile = {
        "full_name": full_name or (pid or "New Patient"),
        "dob": dob.isoformat() if isinstance(dob, date) else None,
        "age": age_val,
        "sex": (sex or "").lower() if sex else None,
        "contact": {"phone": (phone or "").strip() or None, "email": (email or "").strip() or None},
        "address": (address or "").strip() or None,
    }
    # clean None keys in contact
    profile["contact"] = {k: v for k, v in profile["contact"].items() if v}
    profile = {k: v for k, v in profile.items() if v is not None}

    if pid:
        # Update existing (non-destructive merge)
        data = mem_obj.get(pid) or {"patient_id": pid}
        prof = data.get("profile") or {}
        prof.update(profile)
        data["profile"] = prof
        # preferences
        if prefs:
            data.setdefault("preferences", {}).update({k: v for k, v in prefs.items() if v})
        # optional summary tweak
        if reason and not data.get("summary"):
            data["summary"] = f"{prof.get('age','')} {prof.get('sex','')} — reason: {reason[:120]}"
        # add booking_request entry for traceability
        if booking_meta:
            entries = data.setdefault("entries", [])
            entries.append({
                "ts": datetime.utcnow().isoformat() + "Z",
                "type": "booking_request",
                "text": f"Booking request: {reason or '(no reason provided)'}",
                "meta": booking_meta,
            })
        mem_obj.save_patient_json(data)  # persists + updates in-memory
        return pid

    # Create new patient
    new_pid = _next_patient_id(mem_obj)
    new_record = {
        "patient_id": new_pid,
        "profile": profile,
        "preferences": prefs or {},
        "summary": (f"{profile.get('age','')} {profile.get('sex','')} — "
                    f"{(reason or 'new appointment').strip()}").strip(),
        "problems": [
            {"name": (reason or "Visit request"), "status": "active"}
        ] if reason else [],
        "entries": []
    }
    if booking_meta:
        new_record["entries"].append({
            "ts": datetime.utcnow().isoformat() + "Z",
            "type": "booking_request",
            "text": f"Booking request: {reason or '(no reason provided)'}",
            "meta": booking_meta,
        })

    mem_obj.save_patient_json(new_record)  # writes <OFFLINE_PATIENT_DIR>/<pid>.json and updates memory
    return new_pid

# --- Tabs ---
tab_general, tab_schedule = st.tabs(["General Assistant", "Quick Schedule"])

# =========================
# Tab 1: General Assistant
# =========================
with tab_general:
    st.subheader("How can we help you today?")
    prompt = st.text_input(
        "Type your request",
        placeholder=("e.g., My 70-year-old father has chronic kidney disease — please book a nephrologist "
                     "at North Clinic and summarize current treatments"),
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
        pid = mem.resolve_from_text(prompt) or "session"
        _safe_log(mem, pid, "user", prompt)

        try:
            state_in = {"messages": [HumanMessage(content=prompt)], "patient_id": pid}
            state_out = graph.invoke(state_in)

            # Prefer last AIMessage; fallback to 'result'
            assistant_text = ""
            for m in reversed(state_out.get("messages", [])):
                if isinstance(m, AIMessage):
                    assistant_text = m.content
                    break
            if not assistant_text:
                assistant_text = state_out.get("result", "") or "(no result returned)"

            _safe_log(mem, pid, "assistant", assistant_text)

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

    # --- Patient selection OR creation ---
    st.markdown("**Find or add a patient**")
    patients = mem.list_patients()
    pretty_labels = [
        f'{p["patient_id"]} — {p.get("name","") or "(no name)"}'
        for p in patients
    ] if patients else []

    cA, cB = st.columns([1, 1])
    with cA:
        # Build options with a blank placeholder first
        placeholder_option = "— Select a patient —"
        no_patients_option = "— No patients loaded —"
        options = ([placeholder_option] + pretty_labels) if pretty_labels else [no_patients_option]

        sel_label = st.selectbox(
            "Existing patient (optional)",
            options=options,
            index=0,                 # ← start on the placeholder (blank)
            disabled=(not pretty_labels),
            key="sched_patient_label",
        )

        # Only set selected_pid when a real patient was picked
        selected_pid = None
        if pretty_labels and sel_label != placeholder_option:
            selected_pid = patients[pretty_labels.index(sel_label)]["patient_id"]

    with cB:
        full_name = st.text_input("Full name (type to add or match)", value="", key="sched_fullname")

    # Optional demographics (used if creating or updating)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        know_dob = st.checkbox("I know the exact date of birth", key="sched_know_dob")
        dob = None
        if know_dob:
            dob = st.date_input(
                "Date of birth",
                value=date(1980, 1, 1),   # placeholder default
                min_value=date(1930, 1, 1),
                max_value=date.today(),
                key="sched_dob",
                format="YYYY-MM-DD",
            )
    with c2:
        age_input = st.number_input("Age (optional)", min_value=0, max_value=120, value=0, step=1, key="sched_age")
        age_val = int(age_input) if age_input else None
    with c3:
        sex = st.selectbox("Sex (optional)", options=["", "male", "female", "other"], index=0, key="sched_sex")

    c4, c5 = st.columns([1, 1])
    with c4:
        phone = st.text_input("Phone (optional)", value="", key="sched_phone")
    with c5:
        email = st.text_input("Email (optional)", value="", key="sched_email")
    address = st.text_input("Address (optional)", value="", key="sched_address")

    st.markdown("---")

    # --- Appointment details ---
    dr_placeholder = "e.g., Orthopedist (Sports Medicine) or Dr. Jane Lee"
    c6, c7 = st.columns([2, 1])
    with c6:
        doctor_name = st.text_input("Doctor or Specialty", value="", placeholder=dr_placeholder, key="sched_doc")
    with c7:
        clinic = st.text_input("Clinic (optional)", value="", key="sched_clinic")

    modes = st.multiselect(
        "Appointment mode(s)",
        options=["in-person", "video", "phone"],
        default=["in-person"],
        key="sched_modes",
    )
    tod = st.selectbox(
        "Preferred time of day (optional)",
        options=["", "morning", "afternoon", "evening"],
        index=0,
        key="sched_tod",
    )

    appt_date = st.date_input("Appointment date", value=date.today(), key="sched_date")
    reason = st.text_area("Reason/Notes", placeholder="Short reason for visit…", key="sched_reason")

    book_clicked = st.button("Book Appointment", type="primary", key="sched_submit")

    if book_clicked:
        # Determine preferences (used for persisting to patient record)
        prefs = {
            "appointment_modes": modes or [],
            "preferred_clinic": clinic.strip() or None,
            "preferred_time_of_day": (tod or None),
            "appointment_window_days": 14,
        }
        prefs = {k: v for k, v in prefs.items() if v}

        # Build booking meta for entry trail
        booking_meta = {
            "doctor_specialty": doctor_name.strip() or "Primary Care",
            "clinic": clinic.strip() or None,
            "modes": modes or [],
            "preferred_time_of_day": (tod or None),
            "date_range": {
                "start": appt_date.isoformat(),
                "end": appt_date.isoformat()
            }
        }

        # Decide which patient to use:
        pid = None
        typed_name = (full_name or "").strip()
        if typed_name:
            # Use typed name: match existing or create new
            pid = _ensure_patient_from_form(
                mem,
                full_name=typed_name,
                dob=(dob if isinstance(dob, date) else None),
                age_input=(age_val if age_val else None),
                sex=(sex if sex else None),
                phone=phone,
                email=email,
                address=address,
                prefs=prefs,
                booking_meta=booking_meta,
                reason=reason,
            )
        elif selected_pid:
            # Use selected existing patient (and optionally update details if provided)
            pid = _ensure_patient_from_form(
                mem,
                full_name=(mem.get(selected_pid) or {}).get("profile", {}).get("full_name") or selected_pid,
                dob=(dob if isinstance(dob, date) else None),
                age_input=(age_val if age_val else None),
                sex=(sex if sex else None),
                phone=phone,
                email=email,
                address=address,
                prefs=prefs,
                booking_meta=booking_meta,
                reason=reason,
            )
        else:
            st.error("Please select an existing patient or type a full name to create a new one.")
            st.stop()

        # Now perform the booking
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
            st.success(f"Booking request sent for **{pid}**.")
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
