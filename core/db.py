from __future__ import annotations
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.getenv("DB_PATH", "./data/gha.sqlite")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(seed: bool = True) -> None:
    with connect() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            notes TEXT
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            specialty TEXT NOT NULL,
            location TEXT
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            start_ts TEXT NOT NULL,
            end_ts TEXT NOT NULL,
            status TEXT DEFAULT 'booked',
            FOREIGN KEY(patient_id) REFERENCES patients(id),
            FOREIGN KEY(doctor_id) REFERENCES doctors(id)
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS histories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            entry_ts TEXT NOT NULL,
            text TEXT NOT NULL,
            tags TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
        """)
    if seed:
        seed_db()


def seed_db():
    with connect() as c:
        # Seed doctors if empty
        cur = c.execute("SELECT COUNT(*) as n FROM doctors")
        if cur.fetchone()[0] == 0:
            c.executemany(
                "INSERT INTO doctors (name, specialty, location) VALUES (?,?,?)",
                [
                    ("Dr. Alice Ren", "Nephrology", "Seattle, WA"),
                    ("Dr. Brian Cho", "Cardiology", "Bellevue, WA"),
                    ("Dr. Carla Mehta", "Endocrinology", "Seattle, WA"),
                    ("Dr. Diego Ruiz", "Primary Care", "Kirkland, WA"),
                ],
            )
        # Seed a demo patient
        cur = c.execute("SELECT COUNT(*) as n FROM patients")
        if cur.fetchone()[0] == 0:
            c.execute(
                "INSERT INTO patients (name, age, notes) VALUES (?,?,?)",
                ("John Doe", 70, "Demo patient for testing"),
            )


def list_patients() -> List[sqlite3.Row]:
    with connect() as c:
        return list(c.execute("SELECT * FROM patients ORDER BY id"))


def upsert_patient(name: str, age: Optional[int] = None, notes: str = "") -> int:
    with connect() as c:
        c.execute("INSERT INTO patients (name, age, notes) VALUES (?,?,?)", (name, age, notes))
        return int(c.execute("SELECT last_insert_rowid()").fetchone()[0])


def find_doctors(specialty: str, location_like: str = "") -> List[sqlite3.Row]:
    q = "SELECT * FROM doctors WHERE specialty LIKE ? AND location LIKE ? ORDER BY name"
    with connect() as c:
        return list(c.execute(q, (f"%{specialty}%", f"%{location_like}%")))


def doctor_by_id(doc_id: int) -> Optional[sqlite3.Row]:
    with connect() as c:
        cur = c.execute("SELECT * FROM doctors WHERE id=?", (doc_id,))
        return cur.fetchone()


def patient_by_id(pid: int) -> Optional[sqlite3.Row]:
    with connect() as c:
        cur = c.execute("SELECT * FROM patients WHERE id=?", (pid,))
        return cur.fetchone()


def list_appointments(patient_id: Optional[int] = None) -> List[sqlite3.Row]:
    with connect() as c:
        if patient_id is None:
            cur = c.execute(
                "SELECT a.*, p.name as patient_name, d.name as doctor_name FROM appointments a\n"
                "JOIN patients p ON p.id=a.patient_id JOIN doctors d ON d.id=a.doctor_id\n"
                "ORDER BY start_ts DESC"
            )
        else:
            cur = c.execute(
                "SELECT a.*, p.name as patient_name, d.name as doctor_name FROM appointments a\n"
                "JOIN patients p ON p.id=a.patient_id JOIN doctors d ON d.id=a.doctor_id\n"
                "WHERE a.patient_id=? ORDER BY start_ts DESC",
                (patient_id,),
            )
        return list(cur)


def available_slots(doctor_id: int, days_ahead: int = 14) -> List[Tuple[str, str]]:
    """Return (start_ts, end_ts) pairs for 30-min slots 9am-5pm next N days, excluding booked."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    slots: List[Tuple[str, str]] = []
    with connect() as c:
        booked = {
            r["start_ts"] for r in c.execute(
                "SELECT start_ts FROM appointments WHERE doctor_id=? AND date(start_ts) >= date('now')",
                (doctor_id,),
            )
        }
    for d in range(days_ahead):
        day = today + timedelta(days=d)
        for h in range(9, 17):
            for m in (0, 30):
                start = day.replace(hour=h, minute=m)
                end = start + timedelta(minutes=30)
                s = start.isoformat(sep=" ")
                if s not in booked:
                    slots.append((s, end.isoformat(sep=" ")))
    return slots


def book_appointment(patient_id: int, doctor_id: int, start_ts: str, end_ts: str) -> int:
    with connect() as c:
        # Ensure not double-booked
        cur = c.execute(
            "SELECT 1 FROM appointments WHERE doctor_id=? AND start_ts=?",
            (doctor_id, start_ts),
        )
        if cur.fetchone():
            raise ValueError("Slot already booked.")
        c.execute(
            "INSERT INTO appointments (patient_id, doctor_id, start_ts, end_ts) VALUES (?,?,?,?)",
            (patient_id, doctor_id, start_ts, end_ts),
        )
        return int(c.execute("SELECT last_insert_rowid()").fetchone()[0])


def add_history(patient_id: int, text: str, tags: str = "") -> int:
    with connect() as c:
        ts = datetime.now().isoformat(sep=" ")
        c.execute(
            "INSERT INTO histories (patient_id, entry_ts, text, tags) VALUES (?,?,?,?)",
            (patient_id, ts, text, tags),
        )
        return int(c.execute("SELECT last_insert_rowid()").fetchone()[0])


def get_history(patient_id: int) -> List[sqlite3.Row]:
    with connect() as c:
        cur = c.execute(
            "SELECT * FROM histories WHERE patient_id=? ORDER BY entry_ts DESC",
            (patient_id,),
        )
        return list(cur)
