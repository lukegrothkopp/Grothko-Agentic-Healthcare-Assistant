# tools/booking_tool.py
from __future__ import annotations

from utils.patient_memory import add_appointment   # <-- use the new one
from langchain.agents import Tool  # safer import across LangChain versions
import datetime, json, re
from datetime import date, timedelta

from typing import Tuple, Optional, Any

# --- Version-robust Tool import (with fallback shim) ---
Tool = None  # will become a class

try:
    # Newer LangChain releases
    from langchain_core.tools import Tool as _LC_Tool  # type: ignore
    Tool = _LC_Tool
except Exception:
    try:
        # Older LangChain releases
        from langchain.tools import Tool as _Old_Tool  # type: ignore
        Tool = _Old_Tool
    except Exception:
        # Local minimal shim with the same attributes your app uses
        class _ShimTool:
            def __init__(self, name: str, func, description: str = ""):
                self.name = name
                self.func = func
                self.description = description
        Tool = _ShimTool  # type: ignore

# -----------------------------
# Natural-language date parsing
# -----------------------------
WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
}

DATE_PAT = re.compile(
    r"(20\d{2}-\d{2}-\d{2})|((?:\d{1,2})/(?:\d{1,2})/(?:20\d{2}))",
    re.I
)

def _next_weekday(target_idx: int, today: date) -> date:
    days_ahead = (target_idx - today.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)

def _parse_relative_date_phrase(text: str, today: Optional[date] = None) -> Optional[str]:
    if not text:
        return None
    text_l = text.lower().strip()
    if today is None:
        today = date.today()

    if "today" in text_l:
        return today.isoformat()
    if "tomorrow" in text_l:
        return (today + timedelta(days=1)).isoformat()

    m = re.search(r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text_l)
    if m:
        wd = WEEKDAYS[m.group(1)]
        d = _next_weekday(wd, today)
        return d.isoformat()

    m2 = re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text_l)
    if m2:
        wd = WEEKDAYS[m2.group(1)]
        days_ahead = (wd - today.weekday() + 7) % 7
        if days_ahead == 0:
            days_ahead = 7
        d = today + timedelta(days=days_ahead)
        return d.isoformat()

    return None

def _parse_nl(s: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse free-text input -> (patient_id, doctor_name, YYYY-MM-DD)."""
    pid = None; doc = None; d_iso = None
    if not s:
        return pid, doc, d_iso

    # Patient id like "patient_001" or "patient 001"
    m = re.search(r"\bpatient[_\s-]?(\d{3,})\b", s, re.I)
    if m:
        pid = f"patient_{m.group(1)}"

    # "Dr. Jane Lee" or "Dr Lee"
    m = re.search(r"\bDr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", s)
    if m:
        doc = m.group(1)

    # explicit date (YYYY-MM-DD or MM/DD/YYYY)
    m = DATE_PAT.search(s)
    if m:
        d = m.group(0)
        if "/" in d:
            mm, dd, yyyy = d.split("/")
            d_iso = f"{yyyy}-{int(mm):02d}-{int(dd):02d}"
        else:
            d_iso = d

    # relative day words
    if not d_iso:
        d_iso = _parse_relative_date_phrase(s)

    return pid, doc, d_iso

def _coerce_payload(input_val: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Accept dict, JSON string, or NL string.
    Returns (patient_id, doctor_name, YYYY-MM-DD)
    """
    pid = None; doc = None; d_iso = None

    # Dict payload?
    if isinstance(input_val, dict):
        pid = input_val.get("patient_id") or pid
        doc = input_val.get("doctor_name") or input_val.get("specialty") or doc
        d_iso = input_val.get("appointment_date") or d_iso

    # JSON string?
    elif isinstance(input_val, str):
        s = input_val.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                data = json.loads(s)
                if isinstance(data, dict):
                    pid = data.get("patient_id") or pid
                    doc = data.get("doctor_name") or data.get("specialty") or doc
                    d_iso = data.get("appointment_date") or d_iso
            except Exception:
                # fall through to NL parse
                pass
        # Free text parse
        if not pid or not doc or not d_iso:
            p2, d2, di2 = _parse_nl(s)
            pid = pid or p2
            doc = doc or d2
            d_iso = d_iso or di2

    # Defaults if still missing
    if not d_iso:
        d_iso = (date.today() + timedelta(days=7)).isoformat()
    if not doc:
        doc = "Primary Care"

    return pid, doc, d_iso

# -----------------------------
# Booking tool
# -----------------------------
def book_appointment(input_payload: Any) -> str:
    """
    Create a booking confirmation string.
    NOTE: We do not mutate storage here; your Streamlit page logs to PatientMemory
    and updates the patient record immediately after calling this tool.
    """
    try:
        pid, doc, d_iso = _coerce_payload(input_payload)
        if not pid:
            return (
                "The booking input could not be parsed. Include a patient_id "
                "(e.g., 'patient_001') or provide JSON with patient_id, doctor_name, appointment_date."
            )
        booking_id = f"booking_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        appt = {
            "date": d_iso,
            "doctor": doc,
            "status": "scheduled",
            "booking_id": booking_id,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds")
        }

        # ✅ PERSIST the appointment to the shared patient store
        add_appointment(pid, appt)
        
        return (
            f"Appointment for patient {pid} with Dr. {doc} on {d_iso} booked. "
            f"Booking ID: {booking_id}."
        )
    except Exception as e:
        return f"Failed to book appointment: {e}"

def get_booking_tool() -> Tool:  # type: ignore[override]
    """
    Return a version-robust Tool (or shim) so the rest of your app works unchanged.
    """
    return Tool(
        name="Book Appointment",
        func=book_appointment,
        description=(
            "Book a medical appointment. Input may be plain text "
            "(e.g., 'book patient_001 with Dr. Lee next Monday' or 'book patient_002 tomorrow') "
            "or JSON with keys: patient_id, doctor_name, appointment_date (YYYY-MM-DD). "
            "Understands 'today', 'tomorrow', and weekday phrases like 'next Monday'."
        ),
    )
