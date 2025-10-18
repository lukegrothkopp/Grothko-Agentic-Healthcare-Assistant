# tools/booking_tool.py
import datetime, json, re
from datetime import date, timedelta
from typing import Union, Mapping

# Tool import (new + legacy compatibility)
try:
    from langchain.tools import Tool
except Exception:
    from langchain.agents import Tool  # legacy fallback

# Import global helper (no circular imports)
from utils.patient_memory import add_appointment

DATE_PAT = re.compile(r"(20\d{2}-\d{2}-\d{2})|((?:\d{1,2})/(?:\d{1,2})/(?:20\d{2}))", re.I)
WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
}

def _next_weekday(target_idx: int, today: date) -> date:
    days_ahead = (target_idx - today.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)

def _parse_relative_date_phrase(text: str, today: date = None) -> str | None:
    if not text:
        return None
    text_l = text.lower().strip()
    if today is None:
        today = date.today()
    if "today" in text_l:
        return today.isoformat()
    if "tomorrow" in text_l:
        return (today + timedelta(days=1)).isoformat()
    m = re.search(r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", text_l)
    if m:
        wd = WEEKDAYS[m.group(1)]
        d = _next_weekday(wd, today)
        return d.isoformat()
    m2 = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", text_l)
    if m2:
        wd = WEEKDAYS[m2.group(1)]
        days_ahead = (wd - today.weekday() + 7) % 7
        if days_ahead == 0:
            days_ahead = 7
        d = today + timedelta(days=days_ahead)
        return d.isoformat()
    return None

def _parse_nl(s: str):
    pid = None; doc = None; d_iso = None
    if not s:
        return pid, doc, d_iso
    m = re.search(r"patient[_\s-]?(\d{3,})", s, re.I)
    if m:
        pid = f"patient_{m.group(1)}"
    m = re.search(r"Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", s)
    if m:
        doc = m.group(1)
    m = DATE_PAT.search(s)
    if m:
        d = m.group(0)
        if "/" in d:
            mm, dd, yyyy = d.split("/")
            d_iso = f"{yyyy}-{int(mm):02d}-{int(dd):02d}"
        else:
            d_iso = d
    if not d_iso:
        d_iso = _parse_relative_date_phrase(s)
    return pid, doc, d_iso

def _coerce_payload(inp: Union[str, Mapping]) -> tuple[str | None, str | None, str | None]:
    pid = None; doc = None; d_iso = None
    if isinstance(inp, Mapping):
        data = dict(inp)
    else:
        try:
            data = json.loads(inp or "")
        except Exception:
            data = None
    if isinstance(data, dict):
        pid = data.get("patient_id")
        doc = data.get("doctor_name")
        d_iso = data.get("appointment_date")
    # NL fallback
    if not (pid and doc and d_iso):
        text = inp if isinstance(inp, str) else json.dumps(data or {})
        p2, d2, di2 = _parse_nl(text or "")
        pid = pid or p2
        doc = doc or d2
        d_iso = d_iso or di2
    if not d_iso:
        d_iso = (date.today() + timedelta(days=7)).isoformat()
    if not doc:
        doc = "Primary Care"
    return pid, doc, d_iso

def book_appointment(input_obj: Union[str, Mapping]) -> str:
    """
    Book a medical appointment. Input may be:
      • Natural language: "book patient_001 with Dr. Lee next Monday"
      • JSON string:      '{"patient_id":"patient_001","doctor_name":"Dr. Lee","appointment_date":"2025-10-30"}'
      • Dict with same keys
    """
    try:
        pid, doc, d_iso = _coerce_payload(input_obj)
        if not pid:
            return ("The booking input could not be parsed. Include a patient_id "
                    "(e.g., 'patient_001') or provide JSON with patient_id, doctor_name, appointment_date.")
        booking_id = f"booking_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        appt = {
            "date": d_iso,
            "doctor": doc,
            "status": "scheduled",
            "booking_id": booking_id,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        add_appointment(pid, appt)  # persist globally
        return (f"Appointment for {pid} with {doc} on {d_iso} booked. "
                f"Booking ID: {booking_id}.")
    except Exception as e:
        return f"Failed to book appointment: {e}"

def get_booking_tool() -> Tool:
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
