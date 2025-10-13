import json
from typing import Optional, Dict, Any

PATIENT_DB_PATH = "data/patient_db.json"

def _load_db() -> Dict[str, Any]:
    try:
        with open(PATIENT_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_db(db: Dict[str, Any]) -> None:
    with open(PATIENT_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

def get_patient_record(patient_id: str) -> Optional[Dict[str, Any]]:
    return _load_db().get(patient_id)

def add_patient_record(patient_id: str, data: Dict[str, Any]) -> bool:
    db = _load_db()
    if patient_id in db:
        return False
    db[patient_id] = data
    _save_db(db)
    return True

def update_patient_record(patient_id: str, data: Dict[str, Any]) -> bool:
    db = _load_db()
    if patient_id not in db:
        db[patient_id] = data
    else:
        # shallow merge for dict fields
        db[patient_id].update(data)
    _save_db(db)
    return True

# NEW: ensure a minimal record exists
def _ensure_patient(patient_id: str) -> Dict[str, Any]:
    db = _load_db()
    rec = db.get(patient_id)
    if not rec:
        rec = {
            "name": patient_id,           # or set elsewhere
            "age": None,
            "conditions": [],
            "history": [],
            "appointments": []
        }
        db[patient_id] = rec
        _save_db(db)
    # normalize list fields
    rec.setdefault("history", [])
    rec.setdefault("appointments", [])
    return rec

# NEW: append an appointment and persist
def add_appointment(patient_id: str, appt: Dict[str, Any]) -> None:
    db = _load_db()
    rec = db.get(patient_id)
    if not rec:
        rec = _ensure_patient(patient_id)
        db = _load_db()  # reload after ensure
    rec = db.get(patient_id)
    rec.setdefault("appointments", [])
    rec["appointments"].append(appt)
    db[patient_id] = rec
    _save_db(db)
