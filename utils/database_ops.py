import json, os

PATIENT_DB_PATH = "data/patient_db.json"

def _load_db():
    try:
        with open(PATIENT_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_db(db):
    os.makedirs("data", exist_ok=True)
    with open(PATIENT_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

def get_patient_record(patient_id: str) -> dict | None:
    db = _load_db()
    return db.get(patient_id)

def add_patient_record(patient_id: str, data: dict):
    db = _load_db()
    if patient_id not in db:
        db[patient_id] = data
        _save_db(db)
        return True
    return False

def update_patient_record(patient_id: str, data: dict):
    db = _load_db()
    if patient_id in db:
        db[patient_id].update(data)
        _save_db(db)
        return True
    return False

def add_appointment(patient_id: str, appt: dict) -> None:
    db = _load_db()
    rec = db.get(patient_id) or {
        "name": patient_id, "age": None, "conditions": [], "history": [], "appointments": []
    }
    rec.setdefault("appointments", []).append(appt)
    db[patient_id] = rec
    _save_db(db)
