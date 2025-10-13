python
"""
SQLite database layer for Patients, Doctors, Appointments, and Medical Histories.
"""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime

DB_PATH = "data/healthcare.db"

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

def init_db():
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS patients(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            sex TEXT,
            contact TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS doctors(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            specialty TEXT NOT NULL
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS appointments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            appt_time TEXT NOT NULL,
            status TEXT DEFAULT 'scheduled',
            FOREIGN KEY(patient_id) REFERENCES patients(id),
            FOREIGN KEY(doctor_id) REFERENCES doctors(id)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS histories(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            note TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
        """)

def seed_demo():
    """
    Seeds demo doctors/patients if tables are empty.
    """
    with get_conn() as c:
        cur = c.cursor()
        # Doctors
        cur.execute("SELECT COUNT(*) AS n FROM doctors")
        if cur.fetchone()["n"] == 0:
            cur.executemany(
                "INSERT INTO doctors(name, specialty) VALUES(?, ?)",
                [
                    ("Dr. Priya Nayar", "Nephrology"),
                    ("Dr. Ken Ito", "Cardiology"),
                    ("Dr. Sofia Alvarez", "Endocrinology"),
                    ("Dr. Ming Zhao", "Primary Care"),
                ]
            )
        # Patients
        cur.execute("SELECT COUNT(*) AS n FROM patients")
        if cur.fetchone()["n"] == 0:
            cur.executemany(
                "INSERT INTO patients(name, age, sex, contact) VALUES(?, ?, ?, ?)",
                [
                    ("John Smith", 70, "M", "john@example.com"),
                    ("Ava Johnson", 45, "F", "ava@example.com"),
                ]
            )
        # Histories
        cur.execute("SELECT COUNT(*) AS n FROM histories")
        if cur.fetchone()["n"] == 0:
            # Add a CKD note for John Smith (patient_id=1)
            cur.execute(
                "INSERT INTO histories(patient_id, note, created_at) VALUES(?, ?, ?)",
                (1, "Chronic Kidney Disease (Stage 3). On ACE inhibitors. Monitor eGFR quarterly.", datetime.utcnow().isoformat())
            )

def list_doctors(specialty: Optional[str]=None) -> List[Dict[str, Any]]:
    with get_conn() as c:
        cur = c.cursor()
        if specialty:
            cur.execute("SELECT * FROM doctors WHERE specialty LIKE ?", (f"%{specialty}%",))
        else:
            cur.execute("SELECT * FROM doctors")
        return [dict(r) for r in cur.fetchall()]

def list_patients() -> List[Dict[str, Any]]:
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("SELECT * FROM patients")
        return [dict(r) for r in cur.fetchall()]

def add_patient(name: str, age: int, sex: str, contact: str) -> int:
    with get_conn() as c:
        cur = c.cursor()
        cur.execute(
            "INSERT INTO patients(name, age, sex, contact) VALUES(?, ?, ?, ?)",
            (name, age, sex, contact)
        )
        return cur.lastrowid

def add_history(patient_id: int, note: str):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute(
            "INSERT INTO histories(patient_id, note, created_at) VALUES(?, ?, ?)",
            (patient_id, note, datetime.utcnow().isoformat())
        )

def get_history(patient_id: int) -> List[Dict[str, Any]]:
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("SELECT * FROM histories WHERE patient_id=? ORDER BY created_at DESC", (patient_id,))
        return [dict(r) for r in cur.fetchall()]

def create_appointment(patient_id: int, doctor_id: int, appt_time: str) -> int:
    with get_conn() as c:
        cur = c.cursor()
        cur.execute(
            "INSERT INTO appointments(patient_id, doctor_id, appt_time) VALUES(?, ?, ?)",
            (patient_id, doctor_id, appt_time),
        )
        return cur.lastrowid

def list_appointments(status: Optional[str]=None) -> List[Dict[str, Any]]:
    with get_conn() as c:
        cur = c.cursor()
        if status:
            cur.execute("SELECT * FROM appointments WHERE status=?", (status,))
        else:
            cur.execute("SELECT * FROM appointments")
        return [dict(r) for r in cur.fetchall()]
