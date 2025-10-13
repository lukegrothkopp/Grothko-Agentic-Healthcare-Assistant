# core/db.py
import os, sqlite3
from contextlib import contextmanager

DB_PATH = os.getenv("DB_PATH", "data/healthcare.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    dob TEXT,
    contact TEXT
);

CREATE TABLE IF NOT EXISTS doctors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    specialty TEXT
);

CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    doctor_id INTEGER NOT NULL,
    appt_time TEXT NOT NULL,
    reason TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(id),
    FOREIGN KEY(doctor_id) REFERENCES doctors(id)
);
"""

@contextmanager
def get_conn():
    # Ensure directory exists
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def ensure_schema(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)

def init_db():
    with get_conn() as conn:
        ensure_schema(conn)

def seed_demo():
    with get_conn() as conn:
        ensure_schema(conn)
        # Only seed if empty
        cur = conn.execute("SELECT COUNT(1) AS c FROM patients")
        if cur.fetchone()["c"] == 0:
            conn.executemany(
                "INSERT INTO patients (first_name,last_name,dob,contact) VALUES (?,?,?,?)",
                [
                    ("Ava","Lopez","1985-03-14","ava@example.com"),
                    ("Noah","Kim","1990-07-02","noah@example.com"),
                    ("Mia","Singh","1978-11-22","mia@example.com"),
                ],
            )
        cur = conn.execute("SELECT COUNT(1) AS c FROM doctors")
        if cur.fetchone()["c"] == 0:
            conn.executemany(
                "INSERT INTO doctors (name,specialty) VALUES (?,?)",
                [
                    ("Dr. Patel","Nephrology"),
                    ("Dr. Chen","Cardiology"),
                    ("Dr. Rivera","Primary Care"),
                ],
            )
        cur = conn.execute("SELECT COUNT(1) AS c FROM appointments")
        if cur.fetchone()["c"] == 0:
            conn.executemany(
                "INSERT INTO appointments (patient_id,doctor_id,appt_time,reason) VALUES (?,?,?,?)",
                [
                    (1,1,"2025-10-15 10:00","CKD follow-up"),
                    (2,3,"2025-10-17 14:30","Annual physical"),
                ],
            )

# --- Safe query helpers -------------------------------------------------------

def _safe_select_all(table: str):
    with get_conn() as conn:
        try:
            return list(conn.execute(f"SELECT * FROM {table}"))
        except sqlite3.OperationalError:
            # If table missing for any reason, create schema and retry once
            ensure_schema(conn)
            return list(conn.execute(f"SELECT * FROM {table}"))

def list_patients():
    return _safe_select_all("patients")

def list_doctors():
    return _safe_select_all("doctors")

def list_appointments():
    return _safe_select_all("appointments")

def add_patient(first_name: str, last_name: str, dob: str = None, contact: str = None):
    with get_conn() as conn:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO patients (first_name,last_name,dob,contact) VALUES (?,?,?,?)",
            (first_name, last_name, dob, contact),
        )
