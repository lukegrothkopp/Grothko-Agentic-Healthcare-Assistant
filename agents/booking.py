from __future__ import annotations
from typing import List, Tuple
from core import db
from core.logging import get_logger

log = get_logger("BookingAgent")

class BookingAgent:
    def __init__(self):
        db.init_db(seed=True)

    def search_doctors(self, specialty: str, location: str = ""):
        return db.find_doctors(specialty, location)

    def get_slots(self, doctor_id: int) -> List[Tuple[str, str]]:
        return db.available_slots(doctor_id)

    def book(self, patient_id: int, doctor_id: int, start_ts: str, end_ts: str) -> int:
        appt_id = db.book_appointment(patient_id, doctor_id, start_ts, end_ts)
        log.info(f"Booked appt #{appt_id} for patient {patient_id} with doctor {doctor_id} at {start_ts}")
        return appt_id

    def list_appointments(self, patient_id: int = None):
        return db.list_appointments(patient_id)
