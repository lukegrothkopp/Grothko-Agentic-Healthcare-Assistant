from __future__ import annotations
from core import db
from core.logging import get_logger

log = get_logger("HistoryAgent")

class HistoryAgent:
    def __init__(self):
        db.init_db(seed=True)

    def add(self, patient_id: int, text: str, tags: str = "") -> int:
        hid = db.add_history(patient_id, text, tags)
        log.info(f"Added history #{hid} for patient {patient_id}")
        return hid

    def get(self, patient_id: int):
        return db.get_history(patient_id)
