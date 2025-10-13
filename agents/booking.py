from __future__ import annotations
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from core.db import list_doctors, create_appointment

class BookingAgent:
    def find_doctor(self, specialty: str) -> Optional[Dict[str, Any]]:
        docs = list_doctors(specialty)
        return docs[0] if docs else None

    def book(self, patient_id: int, specialty: str, when: Optional[str]=None) -> Dict[str, Any]:
        doc = self.find_doctor(specialty)
        if not doc:
            return {"success": False, "error": f"No doctors available for {specialty}"}
        appt_time = when or (datetime.utcnow() + timedelta(days=2)).isoformat(timespec="minutes")
        appt_id = create_appointment(patient_id, doc["id"], appt_time)
        return {"success": True, "appointment_id": appt_id, "doctor": doc, "time": appt_time}
