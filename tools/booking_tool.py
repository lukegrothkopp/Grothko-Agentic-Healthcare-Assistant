import datetime
from langchain.tools import Tool

def book_appointment(input_str: str) -> str:
    """Input format: JSON string with keys patient_id, doctor_name, appointment_date (YYYY-MM-DD)."""
    import json
    try:
        data = json.loads(input_str)
        pid = data.get("patient_id") or "unknown_patient"
        doc = data.get("doctor_name") or "Unknown Doctor"
        date = data.get("appointment_date") or datetime.date.today().isoformat()
        booking_id = f"booking_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        return (f"Appointment for patient {pid} with Dr. {doc} on {date} booked. "
                f"Booking ID: {booking_id}.")
    except Exception as e:
        return f"Failed to parse booking input. Provide JSON with patient_id, doctor_name, appointment_date. Error: {e}"

def get_booking_tool() -> Tool:
    return Tool(
        name="Book Appointment",
        func=book_appointment,
        description=(
            "Book a medical appointment. Input must be a JSON string with keys: "
            "patient_id, doctor_name, appointment_date (YYYY-MM-DD)."
        ),
    )
