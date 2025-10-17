# pages/1_Patient_Assistant.py
from __future__ import annotations

import os
import re
import json
from datetime import date, datetime
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from agents.graph_agent import build_graph
from utils.patient_memory import PatientMemory
from tools.booking_tool import get_booking_tool

from utils.secret_env import export_secrets_to_env
export_secrets_to_env()  # ensures OPENAI_API_KEY etc. are in os.environ
load_dotenv()

# -------------------------------
# Page + singletons (define EARLY)
# -------------------------------
st.set_page_config(page_title="Patient Assistant", page_icon="🧍🏽", layout="wide")

# Ensure memory object exists
if ("pmemory" not in st.session_state) or (not isinstance(st.session_state.get("pmemory"), PatientMemory)):
    st.session_state.pmemory = PatientMemory()
mem: PatientMemory = st.session_state.pmemory

# Keep a persistent "current patient" context that both chat & scheduler can use
if "current_patient_id" not in st.session_state:
    st.session_state.current_patient_id = "session"

# Build graph / tools once
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()  # works with/without OPENAI key
graph = st.session_state.graph

if "booking_tool" not in st.session_state:
    st.session_state.booking_tool = get_booking_tool()
booking_tool = st.session_state.booking_tool

# -------------------------------
# Helpers (define BEFORE any use)
# -------------------------------
def _resolve_pid_safe(mem: PatientMemory, text: str, default_id: str) -> str:
    """Resolve a patient id from free text; never throws; falls back to default."""
    fn = getattr(mem, "resolve_from_text", None)
    if callable(fn):
        try:
            val = fn(text, default=default_id)
            return val or default_id or "session"
        except Exception:
            return default_id or "session"
    return default_id or "session"

def _safe_log(mem_obj: PatientMemory, pid: str, role: str, content: str):
    """Log to memory regardless of PatientMemory version."""
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

def _extract_answer_from_state(state: dict) -> str:
    """Robustly pull an assistant reply from the returned graph state."""
    try:
        msgs = state.get("messages", []) or []
        for m in reversed(msgs):
            if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai" or (
                isinstance(m, dict) and m.get("role") == "assistant"
            ):
                content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
                if content and str(content).strip():
                    return str(content)
        if state.get("result"):
            return str(state["result"])
        if state.get("bullets"):
            bl = [f"- {b}" for b in state["bullets"] if b]
            if bl:
                return "Here’s what I found:\n" + "\n".join(bl[:8])
        if state.get("plan"):
            steps = [s for s in state["plan"] if s]
            if steps:
                return "Here’s a plan I can follow:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    except Exception:
        pass
    return "I’m here to help. I can summarize options, suggest next steps, or help book an appointment."

# ---------- Patient utilities for the Quick Schedule ----------
import uuid

def _norm_name(s: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", (s or "").strip().lower())

def _find_patient_id_by_name(mem_obj: PatientMemory, full_name: str) -> str | None:
    target = _norm_name(full_name)
    for pid, data in (mem_obj.patients or {}).items():
        name = _norm_name(((data.get("profile") or {}).get("full_name") or ""))
        if name and name == target:
            return pid
    return None

import re as _re
_PAT_ID_RE = _re.compile(r"^patient_(\d+)$")
def _next_patient_id(mem_obj: PatientMemory) -> str:
    max_n = 700
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
    """Find by name (case-insensitive) or create new patient, then persist/merge details."""
    full_name = (full_name or "").strip()
    pid = _find_patient_id_by_name(mem_obj, full_name) if full_name else None

    age_val = _maybe_int_age(dob, age_input)
    profile = {
        "full_name": full_name or (pid or "New Patient"),
        "dob": dob.isoformat() if isinstance(dob, date) else None,
        "age": age_val,
        "sex": (sex or "").lower() if sex else None,
        "contact": {"phone": (phone or "").strip() or None, "email": (email or "").strip() or None},
        "address": (address or "").strip() or None,
    }
    profile["contact"] = {k: v for k, v in profile["contact"].items() if v}
    profile = {k: v for k, v in profile.items() if v is not None}

    if pid:
        data = mem_obj.get(pid) or {"patient_id": pid}
        prof = data.get("profile") or {}
        prof.update(profile)
        data["profile"] = prof
        if prefs:
            data.setdefault("preferences", {}).update({k: v for k, v in prefs.items() if v})
        if reason and not data.get("summary"):
            data["summary"] = f"{prof.get('age','')} {prof.get('sex','')} — reason: {reason[:120]}"
        if booking_meta:
            entries = data.setdefault("entries", [])
            entries.append({
                "ts": datetime.utcnow().isoformat() + "Z",
                "type": "booking_request",
                "text": f"Booking request: {reason or '(no reason provided)'}",
                "meta": booking_meta,
            })
        mem_obj.save_patient_json(data)
        return pid

    new_pid = _next_patient_id(mem_obj)
    new_record = {
        "patient_id": new_pid,
        "profile": profile,
        "preferences": prefs or {},
        "summary": (f"{profile.get('age','')} {profile.get('sex','')} — "
                    f"{(reason or 'new appointment').strip()}").strip(),
        "problems": [{"name": (reason or "Visit request"), "status": "active"}] if reason else [],
        "entries": []
    }
    if booking_meta:
        new_record["entries"].append({
            "ts": datetime.utcnow().isoformat() + "Z",
            "type": "booking_request",
            "text": f"Booking request: {reason or '(no reason provided)'}",
            "meta": booking_meta,
        })
    mem_obj.save_patient_json(new_record)
    return new_pid

# -------------------------------
# UI
# -------------------------------
st.title("🧍🏽 Patient Assistant")
st.caption("Ask for help with scheduling, records, and general info from trusted sources. (Won't provide medical advice)")

# Render past chat
if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Top inline chat form
with st.form("ask_form", clear_on_submit=True):
    _ask_text = st.text_area("Type your question",
                             placeholder="e.g., I need help with severe headaches",
                             height=100)
    _ask_send = st.form_submit_button("Send")

if _ask_send and _ask_text and _ask_text.strip():
    # Show user bubble
    st.session_state.messages.append({"role": "user", "content": _ask_text})
    with st.chat_message("user"):
        st.markdown(_ask_text)

    # FIX: use session's current patient id as safe default (no undefined variable)
    resolved_pid = _resolve_pid_safe(
        st.session_state.pmemory,
        _ask_text,
        st.session_state.get("current_patient_id", "session"),
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                state = {
                    "messages": [HumanMessage(content=_ask_text)],
                    "intent": None,
                    "result": None,
                    "patient_id": resolved_pid,
                }
                result_state = graph.invoke(state)
                answer = _extract_answer_from_state(result_state)
            except Exception as e:
                answer = f"Sorry—there was an error: {e}"
            st.markdown(answer)

    # Persist to memory & session
    try:
        st.session_state.pmemory.record_event(
            resolved_pid,
            f"Patient asked: {_ask_text}\nAssistant: {answer}",
            meta={"kind": "chat", "by": "patient"},
        )
    except Exception:
        pass
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Remember which patient this chat used
    st.session_state.current_patient_id = resolved_pid

# -------------------------------
# Quick Schedule (single tab)
# -------------------------------
(tab_schedule,) = st.tabs(["Quick Schedule"])

# ================
# Tab: Quick Schedule
# ================
with tab_schedule:
    st.subheader("Book an appointment")

    # Find or add a patient
    st.markdown("**Find or add a patient**")
    patients = mem.list_patients()
    pretty_labels = [
        f'{p["patient_id"]} — {p.get("name","") or "(no name)"}'
        for p in patients
    ] if patients else []

    cA, cB = st.columns([1, 1])
    with cA:
        placeholder_option = "— Current Patient List —"
        no_patients_option = "— No patients loaded —"
        options = ([placeholder_option] + pretty_labels) if pretty_labels else [no_patients_option]

        sel_label = st.selectbox(
            "Existing Patients",
            options=options,
            index=0,
            disabled=(not pretty_labels),
            key="sched_patient_label",
        )
        selected_pid = None
        if pretty_labels and sel_label != placeholder_option:
            selected_pid = patients[pretty_labels.index(sel_label)]["patient_id"]

    with cB:
        full_name = st.text_input("New Patient - Full Name)", value="", key="sched_fullname")

    # Optional demographics
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        know_dob = st.checkbox("I know the exact date of birth", key="sched_know_dob")
        dob = None
        if know_dob:
            dob = st.date_input(
                "Date of birth",
                value=date(1980, 1, 1),
                min_value=date(1930, 1, 1),
                max_value=date.today(),
                key="sched_dob",
                format="YYYY-MM-DD",
            )
    with c2:
        age_input = st.number_input("Age)", min_value=0, max_value=120, value=0, step=1, key="sched_age")
        age_val = int(age_input) if age_input else None
    with c3:
        sex = st.selectbox("Sex", options=["", "male", "female", "other"], index=0, key="sched_sex")

    c4, c5 = st.columns([1, 1])
    with c4:
        phone = st.text_input("Phone", value="", key="sched_phone")
    with c5:
        email = st.text_input("Email", value="", key="sched_email")
    address = st.text_input("Address", value="", key="sched_address")

    st.markdown("---")

    # Appointment details
    dr_placeholder = "e.g., Orthopedist (Sports Medicine) or Dr. Jane Lee"
    c6, c7 = st.columns([2, 1])
    with c6:
        doctor_name = st.text_input("Doctor or Specialty", value="", placeholder=dr_placeholder, key="sched_doc")
    with c7:
        clinic = st.text_input("Clinic (optional)", value="", key="sched_clinic")

    modes = st.multiselect("Appointment mode(s)", options=["in-person", "video", "phone"],
                           default=["in-person"], key="sched_modes")
    tod = st.selectbox("Preferred time of day (optional)", options=["", "morning", "afternoon", "evening"],
                       index=0, key="sched_tod")

    appt_date = st.date_input("Appointment date", value=date.today(), key="sched_date")
    reason = st.text_area("Reason/Notes", placeholder="Short reason for visit…", key="sched_reason")

    book_clicked = st.button("Book Appointment", type="primary", key="sched_submit")

    if book_clicked:
        prefs = {
            "appointment_modes": modes or [],
            "preferred_clinic": clinic.strip() or None,
            "preferred_time_of_day": (tod or None),
            "appointment_window_days": 14,
        }
        prefs = {k: v for k, v in prefs.items() if v}

        booking_meta = {
            "doctor_specialty": doctor_name.strip() or "Primary Care",
            "clinic": clinic.strip() or None,
            "modes": modes or [],
            "preferred_time_of_day": (tod or None),
            "date_range": {"start": appt_date.isoformat(), "end": appt_date.isoformat()}
        }

        # Decide which patient to use
        typed_name = (full_name or "").strip()
        if typed_name:
            pid = _ensure_patient_from_form(
                mem, full_name=typed_name,
                dob=(dob if isinstance(dob, date) else None),
                age_input=(age_val if age_val else None),
                sex=(sex if sex else None),
                phone=phone, email=email, address=address,
                prefs=prefs, booking_meta=booking_meta, reason=reason,
            )
        elif selected_pid:
            pid = _ensure_patient_from_form(
                mem, full_name=(mem.get(selected_pid) or {}).get("profile", {}).get("full_name") or selected_pid,
                dob=(dob if isinstance(dob, date) else None),
                age_input=(age_val if age_val else None),
                sex=(sex if sex else None),
                phone=phone, email=email, address=address,
                prefs=prefs, booking_meta=booking_meta, reason=reason,
            )
        else:
            st.error("Please select an existing patient or type a full name to create a new one.")
            st.stop()

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

        # Call booking tool (tolerate dict or JSON string API)
        try:
            try:
                result = booking_tool.func(payload)  # preferred: dict
            except Exception:
                result = booking_tool.func(json.dumps(payload))  # fallback: JSON string
            result_str = result if isinstance(result, str) else json.dumps(result)
            st.success(f"Booking request sent for **{pid}**.")
            st.code(result_str, language="json")

            _safe_log(mem, pid, "user", f"[QuickSchedule] {json.dumps(payload)}")
            _safe_log(mem, pid, "assistant", f"[QuickSchedule Result] {result_str}")

            # Remember this patient globally so chat uses it as default
            st.session_state.current_patient_id = pid

        except Exception as e:
            st.error(f"Booking failed: {e}")

# Footer diagnostics
with st.expander("Technical details"):
    st.write({
        "OFFLINE_PATIENT_DIR": os.getenv("OFFLINE_PATIENT_DIR", "data/patient_memory"),
        "Patients loaded": len(mem.patients),
        "Current patient context": st.session_state.get("current_patient_id"),
    })
