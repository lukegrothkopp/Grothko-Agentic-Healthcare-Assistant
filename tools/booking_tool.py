import datetime, json, re, os
from datetime import date, timedelta
from langchain.tools import Tool
from utils.metrics import log_tool
from utils.database_ops import add_appointment

DATE_PAT = re.compile(r"(20\d{2}-\d{2}-\d{2})|((?:\d{1,2})/(?:\d{1,2})/(?:20\d{2}))", re.I)
WEEKDAYS = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}

def _next_weekday(target_idx: int, today: date) -> date:
    days_ahead = (target_idx - today.weekday() + 7) % 7
    if days_ahead == 0: days_ahead = 7
    return today + timedelta(days=days_ahead)

def _parse_relative_date_phrase(text: str, today: date = None) -> str | None:
    if not text: return None
    t = text.lower().strip()
    today = today or date.today()
    if "today" in t: return today.isoformat()
    if "tomorrow" in t: return (today + timedelta(days=1)).isoformat()
    m = re.search(r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", t)
    if m: return _next_weekday(WEEKDAYS[m.group(1)], today).isoformat()
    m2 = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", t)
    if m2:
        wd = WEEKDAYS[m2.group(1)]
        days_ahead = (wd - today.weekday() + 7) % 7
        if days_ahead == 0: days_ahead = 7
        return (today + timedelta(days=days_ahead)).isoformat()
    return None

def _parse_nl(s: str):
    pid = None; doc = None; d_iso = None
    if not s: return pid, doc, d_iso
    m = re.search(r"patient[_\s-]?(\d{3,})", s, re.I)
    if m: pid = f"patient_{m.group(1)}"
    m = re.search(r"Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", s)
    if m: doc = m.group(1)
    m = DATE_PAT.search(s)
    if m:
        d = m.group(0)
        if "/" in d:
            mm, dd, yyyy = d.split("/")
            d_iso = f"{yyyy}-{int(mm):02d}-{int(dd):02d}"
        else:
            d_iso = d
    if not d_iso: d_iso = _parse_relative_date_phrase(s)
    if not doc:
        t = (s or "").lower()
        if "hypertension" in t: doc = "Hypertension Specialist (Cardiologist)"
        elif "kidney" in t or "nephro" in t: doc = "Nephrologist"
        elif "foot" in t or "ankle" in t: doc = "Podiatrist"
        else: doc = "Primary Care"
    return pid, doc, d_iso

def _coerce_payload(input_str: str):
    pid = None; doc = None; d_iso = None
    try:
        data = json.loads(input_str)
        pid = data.get("patient_id") or pid
        doc = data.get("doctor_name") or doc
        d_iso = data.get("appointment_date") or d_iso
    except Exception:
        p2, d2, di2 = _parse_nl(input_str or "")
        pid = pid or p2; doc = doc or d2; d_iso = d_iso or di2
    if not d_iso: d_iso = (date.today() + timedelta(days=7)).isoformat()
    if not doc: doc = "Primary Care"
    return pid, doc, d_iso

def book_appointment(input_str: str) -> str:
    try:
        pid, doc, d_iso = _coerce_payload(input_str)
        if not pid:
            return ("The booking input could not be parsed. Include a patient_id "
                    "(e.g., 'patient_001') or provide JSON with patient_id, doctor_name, appointment_date.")
        booking_id = f"booking_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        appt = {
            "date": d_iso, "doctor": doc, "status": "scheduled",
            "booking_id": booking_id, "created_at": datetime.datetime.now().isoformat(timespec="seconds")
        }
        if os.getenv("EVAL_MODE") == "1":  # sandbox for batch evals
            return (f"[SANDBOX] Would book {pid} with {doc} on {d_iso} (no DB write). "
                    f"Booking ID: {booking_id}.")
        add_appointment(pid, appt)
        msg = (f"Appointment for patient {pid} with Dr. {doc} on {d_iso} booked. "
                f"Booking ID: {booking_id}.")
        try:
            log_tool('Book Appointment', 'success', {'patient_id': pid, 'doctor': doc, 'date': d_iso})
        except Exception:
            pass
        return msg
    except Exception as e:
        try:
            log_tool('Book Appointment', 'failure', {'error': str(e)})
        except Exception:
            pass
        return f"Failed to book appointment: {e}"

def get_booking_tool() -> Tool:
    return Tool(
        name="Book Appointment",
        func=book_appointment,
        description=("Book a medical appointment. Input may be plain text "
                     "or JSON with keys: patient_id, doctor_name, appointment_date (YYYY-MM-DD). "
                     "Understands 'today', 'tomorrow', and weekday phrases like 'next Monday'."),
    )
