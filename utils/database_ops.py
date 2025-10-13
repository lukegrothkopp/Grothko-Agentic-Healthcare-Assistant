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
        # shallow merge
        db[patient_id].update(data)
    _save_db(db)
    return True
