# agents/booking.py
from __future__ import annotations

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from core.db import list_doctors, create_appointment, patient_exists, doctor_exists


class BookingAgent:
    """
    Finds a doctor by specialty and books an appointment.
    Expects:
      - doctors table with columns: id, name, specialty, contact
      - patients table with id
      - create_appointment(patient_id, doctor_id, appt_time, reason)
    """

    def find_doctor(self, specialty: str) -> Optional[Dict[str, Any]]:
        """Return the first doctor matching the given specialty (case-insensitive)."""
        docs: List[Dict[str, Any]] = list_doctors()
        if not docs:
            return None
        sp = (specialty or "").strip().lower()
        # Prefer exact/substring specialty matches, else any doctor
        for d in docs:
            if sp and sp in (d.get("specialty") or "").lower():
                return d
        # fallback: first doctor if no specialty match
        return docs[0] if docs else None

    def _default_next_slot(self) -> str:
        """Default to next weekday at 10:00 local (ISO 8601)."""
        now = datetime.now()
        # next day
        dt = now + timedelta(days=1)
        # if weekend, push to Monday
        while dt.weekday() >= 5:  # 5=Sat, 6=Sun
            dt += timedelta(days=1)
        dt = dt.replace(hour=10, minute=0, second=0, microsecond=0)
        return dt.isoformat(timespec="minutes")

    def book(self, patient_id: Optional[int], specialty: str, preferred_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Book an appointment for the given patient and specialty.
        - If preferred_time is None/empty, choose a reasonable default slot.
        - Validates patient/doctor existence. Returns success flag and details.
        """
        if not patient_id:
            return {"success": False, "error": "Missing patient_id"}

        if not patient_exists(patient_id):
            return {"success": False, "error": f"Patient #{patient_id} not found"}

        doc = self.find_doctor(specialty)
        if not doc:
            return {"success": False, "error": f"No doctors available for specialty '{specialty}'"}

        doctor_id = doc["id"]
        if not doctor_exists(doctor_id):
            return {"success": False, "error": f"Doctor #{doctor_id} not found"}

        appt_time = (preferred_time or "").strip() or self._default_next_slot()

        try:
            # Will raise ValueError if invalid ISO
            datetime.fromisoformat(appt_time)
        except Exception:
            return {"success": False, "error": f"Invalid ISO datetime: {appt_time}"}

        # You can pass a richer reason if desired; keeping simple for demo
        return create_appointment(patient_id=patient_id, doctor_id=doctor_id, appt_time=appt_time, reason=f"Consult: {specialty or 'General'}")
