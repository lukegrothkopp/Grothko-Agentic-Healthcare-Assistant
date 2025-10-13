# core/db.py
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime

# Resolve DB path from env or default
DB_PATH = os.getenv("DB_PATH", "data/healthcare.db")


@contextmanager
def get_conn():
    # Ensure parent directory exists
    db_dir = os.path.dirname(DB_PATH) or "."
    os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Helpful pragmas for small apps
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                name     TEXT NOT NULL,
                age      INTEGER,
                sex      TEXT,
                contact  TEXT
            );

            CREATE TABLE IF NOT EXISTS doctors (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                name     TEXT NOT NULL,
                specialty TEXT,
                contact   TEXT
            );

            CREATE TABLE IF NOT EXISTS appointments (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                doctor_id  INTEGER NOT NULL,
                appt_time  TEXT NOT NULL,
                reason     TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(id),
                FOREIGN KEY(doctor_id)  REFERENCES doctors(id)
            );

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )


def _get_meta(conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, value),
    )


def seed_demo() -> None:
    """Insert a small set of demo rows once."""
    with get_conn() as conn:
        if _get_meta(conn, "seeded", "0") == "1":
            return

        # Patients
        conn.executemany(
            "INSERT INTO patients(name, age, sex, contact) VALUES (?, ?, ?, ?)",
            [
                ("Ava Nguyen", 44, "F", "ava@example.com"),
                ("Marcus Grothkopp", 14, "M", "marcus@example.com"),
                ("Mia Grothkopp", 12, "F", "mia@example.com"),
            ],
        )

        # Doctors
        conn.executemany(
            "INSERT INTO doctors(name, specialty, contact) VALUES (?, ?, ?)",
            [
                ("Dr. Priya Raman", "Nephrology", "priya.raman@clinic.org"),
                ("Dr. Diego Silva", "Primary Care", "diego.silva@clinic.org"),
                ("Dr. Elena Park", "Cardiology", "elena.park@clinic.org"),
            ],
        )

        # Sample appointments (optional)
        conn.executemany(
            "INSERT INTO appointments(patient_id, doctor_id, appt_time, reason) VALUES (?, ?, ?, ?)",
            [
                (1, 1, "2025-10-15T10:00", "Follow-up"),
                (2, 2, "2025-10-16T14:30", "Annual physical"),
            ],
        )

        _set_meta(conn, "seeded", "1")


# -----------------------
# Existence helpers
# -----------------------
def patient_exists(patient_id: int) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM patients WHERE id = ?", (patient_id,)).fetchone()
        return bool(row)


def doctor_exists(doctor_id: int) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM doctors WHERE id = ?", (doctor_id,)).fetchone()
        return bool(row)


# -----------------------
# CRUD & query helpers
# -----------------------
def add_patient(name: str, age: int, sex: str, contact: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO patients(name, age, sex, contact) VALUES (?, ?, ?, ?)",
            (name, age, sex, contact),
        )
        return cur.lastrowid


def list_patients() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, age, sex, contact FROM patients ORDER BY name COLLATE NOCASE ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def list_doctors() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, specialty, contact FROM doctors ORDER BY specialty COLLATE NOCASE, name COLLATE NOCASE"
        ).fetchall()
        return [dict(r) for r in rows]


def list_appointments() -> List[Dict[str, Any]]:
    """
    Return appointments with joined patient/doctor names for a nicer DataFrame.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                a.id,
                a.patient_id,
                p.name AS patient_name,
                a.doctor_id,
                d.name AS doctor_name,
                a.appt_time,
                a.reason
            FROM appointments a
            LEFT JOIN patients p ON p.id = a.patient_id
            LEFT JOIN doctors  d ON d.id = a.doctor_id
            ORDER BY a.appt_time DESC, a.id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def create_appointment(patient_id: int, doctor_id: int, appt_time: str, reason: str = "") -> Dict[str, Any]:
    """
    Insert a new appointment. appt_time must be ISO-8601 (e.g., '2025-10-15T10:00').
    Returns a dict with success flag and inserted row fields.
    """
    # Validate timestamp
    try:
        datetime.fromisoformat(appt_time)
    except Exception:
        return {"success": False, "error": f"Invalid ISO datetime: {appt_time}"}

    # Optional sanity checks
    if not patient_exists(patient_id):
        return {"success": False, "error": f"Patient #{patient_id} not found"}
    if not doctor_exists(doctor_id):
        return {"success": False, "error": f"Doctor #{doctor_id} not found"}

    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO appointments (patient_id, doctor_id, appt_time, reason)
            VALUES (?, ?, ?, ?)
            """,
            (patient_id, doctor_id, appt_time, reason or ""),
        )
        appt_id = cur.lastrowid
        row = conn.execute(
            """
            SELECT
                a.id,
                a.patient_id,
                p.name AS patient_name,
                a.doctor_id,
                d.name AS doctor_name,
                a.appt_time,
                a.reason
            FROM appointments a
            LEFT JOIN patients p ON p.id = a.patient_id
            LEFT JOIN doctors  d ON d.id = a.doctor_id
            WHERE a.id = ?
            """,
            (appt_id,),
        ).fetchone()

    return {"success": True, **(dict(row) if row else {"id": appt_id})}
