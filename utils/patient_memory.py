# utils/patient_memory.py
from __future__ import annotations

import os
import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, date, timezone

# Defaults (override with OFFLINE_PATIENT_DIR in env if needed)
DEFAULT_SEED_DIR = os.environ.get("OFFLINE_PATIENT_DIR", "data/patient_memory")
DEFAULT_DB_PATH = Path("data/patient_db.json")
DEFAULT_DB_GLOB = "data/patient_db/*.json"

MEMORY_LOG_DIR = Path("data/memory")
Path(MEMORY_LOG_DIR).mkdir(parents=True, exist_ok=True)


# -----------------------------
# Time helpers
# -----------------------------
def _now_iso() -> str:
    # UTC ISO (Z), stable for logging & sorting
    return (
        datetime.utcnow()
        .replace(tzinfo=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

def _to_epoch(ts) -> float:
    """Convert many timestamp forms to UTC epoch seconds. Returns -inf on failure."""
    try:
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str) and ts.strip():
            s = ts.strip().rstrip("Z")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.timestamp()
    except Exception:
        pass
    return float("-inf")


# -----------------------------
# Model
# -----------------------------
@dataclass
class _Patient:
    patient_id: str
    data: Dict[str, Any]


class PatientMemory:
    """
    Lightweight file-backed memory:
      • Loads patients from a seed dir, falls back to data/patient_db{.json,/*.json}
      • Persists runtime events in data/memory/<patient_id>.jsonl
      • Exposes add_appointment(), get_appointments(), get_window(), list_patients(), get_summary()
    """

    def __init__(self, seed_dir: Optional[str] = None):
        self.seed_dir: str = seed_dir or DEFAULT_SEED_DIR
        Path(self.seed_dir).mkdir(parents=True, exist_ok=True)

        self.patients: Dict[str, _Patient | Dict[str, Any]] = {}
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}

        self.reload_from_dir(self.seed_dir)

    # -----------------------------
    # Loading / persistence
    # -----------------------------
    def reload_from_dir(self, seed_dir: str) -> None:
        self.seed_dir = seed_dir
        Path(self.seed_dir).mkdir(parents=True, exist_ok=True)
        self.patients.clear()
        self.history.clear()
        self.sessions.clear()

        self._load_seed_folder(seed_dir)
        if not self.patients:
            self._load_from_patient_db_fallback()

        # Merge runtime memory logs
        for fp in sorted(glob.glob(str(MEMORY_LOG_DIR / "*.jsonl"))):
            pid = Path(fp).stem
            rows: List[Dict[str, Any]] = []
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            continue
            except Exception:
                rows = []
            if rows:
                self.history.setdefault(pid, []).extend(rows)

    def _load_seed_folder(self, folder: str) -> None:
        for fp in sorted(glob.glob(os.path.join(folder, "*.json"))):
            try:
                data = json.loads(Path(fp).read_text(encoding="utf-8"))
            except Exception:
                continue

            if isinstance(data, list):
                for rec in data:
                    self._ingest_patient_record(rec)
            elif isinstance(data, dict):
                if "patients" in data and isinstance(data["patients"], list):
                    for rec in data["patients"]:
                        self._ingest_patient_record(rec)
                else:
                    self._ingest_patient_record(data)

    def _load_from_patient_db_fallback(self) -> None:
        # A) data/patient_db.json (dict keyed by patient_id)
        if DEFAULT_DB_PATH.exists():
            try:
                data = json.loads(DEFAULT_DB_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._ingest_patient_db_dict(data)
            except Exception:
                pass

        # B) any loose JSONs under data/patient_db/*.json
        for fp in sorted(glob.glob(DEFAULT_DB_GLOB)):
            try:
                data = json.loads(Path(fp).read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict) and any(str(k).startswith("patient_") for k in data.keys()):
                self._ingest_patient_db_dict(data)
            elif isinstance(data, dict):
                self._ingest_patient_record(data)
            elif isinstance(data, list):
                for rec in data:
                    self._ingest_patient_record(rec)

    def _ingest_patient_db_dict(self, db: Dict[str, Any]) -> None:
        for pid, rec in db.items():
            if isinstance(rec, dict):
                rec = dict(rec)
                rec.setdefault("patient_id", pid)
                self._ingest_patient_record(rec)

    def _ingest_patient_record(self, rec: Dict[str, Any]) -> None:
        if not isinstance(rec, dict):
            return
        pid = rec.get("patient_id") or rec.get("id") or rec.get("pid")
        if not pid:
            name = (
                rec.get("name")
                or rec.get("profile", {}).get("full_name")
                or "patient"
            ).lower().replace(" ", "_")
            pid = f"{name}_{len(self.patients) + 1}"

        self.patients[pid] = _Patient(patient_id=pid, data=rec)

        # normalize history & appts into history list; ensure timestamps
        def _ensure_ts(x):
            if isinstance(x, dict) and not x.get("ts"):
                x["ts"] = x.get("date") or x.get("timestamp") or _now_iso()
            return x

        hist = rec.get("history") or []
        if isinstance(hist, dict):
            hist = [hist]
        hist = [_ensure_ts(h) for h in hist if isinstance(h, dict)]

        appts = rec.get("appointments") or []
        if isinstance(appts, dict):
            appts = [appts]
        appts = [_ensure_ts({"type": "appointment", **a}) for a in appts if isinstance(a, dict)]

        if hist or appts:
            self.history.setdefault(pid, []).extend(hist + appts)

    def get(self, patient_id: str) -> Dict[str, Any] | None:
        p = self.patients.get(patient_id)
        if not p:
            return None
        return p.data if hasattr(p, "data") else (p if isinstance(p, dict) else None)

    def save_patient_json(self, rec: Dict[str, Any], dir_path: Optional[str] = None) -> str:
        d = dir_path or self.seed_dir
        Path(d).mkdir(parents=True, exist_ok=True)
        pid = rec.get("patient_id") or rec.get("id") or rec.get("pid") or "patient_unnamed"
        out = Path(d) / f"{pid}.json"
        out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return str(out)

    # -----------------------------
    # Memory operations
    # -----------------------------
    def record_event(self, patient_id: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not patient_id:
            return
        row = {
            "ts": _now_iso(),
            "type": "event",
            "text": text,
            "meta": meta or {},
        }
        self.history.setdefault(patient_id, []).append(row)
        log_path = MEMORY_LOG_DIR / f"{patient_id}.jsonl"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass

    def add_appointment(self, patient_id: str, appt: Dict[str, Any]) -> Dict[str, Any]:
        """Persist appointment into the patient’s record and log a memory event."""
        if not patient_id or not isinstance(appt, dict):
            raise ValueError("add_appointment requires a patient_id and an appointment dict")

        base = self.get(patient_id) or {"patient_id": patient_id}
        appts = base.get("appointments") or []
        if isinstance(appts, dict):
            appts = [appts]
        appts.append(appt)
        base["appointments"] = appts

        # save to seed dir
        self.save_patient_json(base, self.seed_dir)

        # log memory event for timeline
        self.record_event(
            patient_id,
            f"Booked appointment with {appt.get('doctor','(unknown)')} on {appt.get('date','(unknown date)')}.",
            meta={"kind": "appointment", **{k: v for k, v in appt.items() if k != "created_at"}}
        )

        # refresh in-memory copy
        self.patients[patient_id] = _Patient(patient_id=patient_id, data=base)
        return appt

    def get_appointments(self, patient_id: str, include_past: bool = False) -> List[Dict[str, Any]]:
        base = self.get(patient_id) or {}
        appts = base.get("appointments") or []
        if isinstance(appts, dict):
            appts = [appts]
        rows = [a for a in appts if isinstance(a, dict)]
        rows.sort(key=lambda a: _to_epoch(a.get("date")))
        if not include_past:
            today = date.today().isoformat()
            rows = [a for a in rows if (a.get("date") or "") >= today]
        return rows

    def get_window(self, patient_id: str, k: int = 10) -> List[Dict[str, Any]]:
        raw = self.history.get(patient_id, []) + self.sessions.get(patient_id, [])
        entries = [e for e in raw if isinstance(e, dict)]
        now_iso = _now_iso()
        for e in entries:
            if not e.get("ts"):
                e["ts"] = e.get("timestamp") or e.get("time") or e.get("date") or now_iso
        entries.sort(key=lambda e: _to_epoch(e.get("ts")))
        return list(reversed(entries[-k:]))

    def search(self, patient_id: str, query: str, k: int = 3) -> List[str]:
        if not query:
            return []
        q = query.lower()
        hits: List[Tuple[float, str]] = []

        for row in self.history.get(patient_id, []):
            txt = (row.get("text") or row.get("notes") or row.get("diagnosis") or "")
            if not isinstance(txt, str):
                continue
            t = txt.lower()
            if q in t:
                hits.append((1.0, txt))
        if hits:
            hits.sort(key=lambda x: x[0], reverse=True)
            return [h[1] for h in hits[:k]]

        base = self.get(patient_id) or {}
        for c in base.get("conditions") or []:
            t = str(c)
            if q in t.lower():
                hits.append((0.5, f"Condition match: {t}"))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [h[1] for h in hits[:k]]

    def get_summary(self, patient_id: str) -> str:
        base = self.get(patient_id) or {}
        name = base.get("name") or base.get("profile", {}).get("full_name") or patient_id
        age = base.get("age") or base.get("profile", {}).get("age")
        conds = base.get("conditions") or []

        last_appt = None
        appts = base.get("appointments") or []
        if isinstance(appts, dict):
            appts = [appts]
        if appts:
            try:
                appts_sorted = sorted(appts, key=lambda a: _to_epoch(a.get("date")), reverse=True)
                last_appt = appts_sorted[0]
            except Exception:
                last_appt = appts[-1]

        recent = self.get_window(patient_id, k=3)
        recent_blurbs = []
        for r in recent:
            txt = r.get("text") or r.get("notes") or r.get("diagnosis") or ""
            typ = r.get("type") or r.get("tag") or "event"
            ts = r.get("ts")
            if txt:
                recent_blurbs.append(f"[{typ} @ {ts}] {txt}")

        parts = [
            f"{name}" + (f", {age}" if age is not None else ""),
            f"Conditions: {', '.join(map(str, conds)) or '—'}",
        ]
        if last_appt:
            parts.append(
                f"Last appointment: {last_appt.get('date')} — "
                f"{last_appt.get('doctor','(unknown)')} ({last_appt.get('status','')})"
            )
        if recent_blurbs:
            parts.append("Recent: " + " | ".join(recent_blurbs))
        return " | ".join(parts)

    def list_patients(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for pid, p in self.patients.items():
            base = p.data if hasattr(p, "data") and isinstance(p.data, dict) else (p if isinstance(p, dict) else {})
            name = base.get("name") or base.get("profile", {}).get("full_name") or pid
            age = base.get("age") or base.get("profile", {}).get("age")
            rows.append({
                "patient_id": pid,
                "name": name,
                "age": age,
                "conditions": ", ".join(map(str, base.get("conditions") or [])),
            })
        rows.sort(key=lambda r: (str(r.get("name") or ""), str(r.get("patient_id") or "")))
        return rows

    # Optional: resolve a patient id from free text
    def resolve_from_text(self, text: str, default: Optional[str] = None) -> str:
        if not text:
            return default or "session"
        t = text.lower()
        # detect patient_### tokens
        import re as _re
        m = _re.search(r"patient[_\s-]?(\d{3,})", t)
        if m:
            return f"patient_{m.group(1)}"
        # try name match
        def _norm(s: str) -> str:
            import re as _re2
            return _re2.sub(r"\s+", " ", (s or "").strip().lower())
        target = _norm(text)
        for pid, data in (self.patients or {}).items():
            base = data.data if hasattr(data, "data") else (data if isinstance(data, dict) else {})
            full = _norm(base.get("profile", {}).get("full_name") or base.get("name") or "")
            if full and full in target:
                return pid
        return default or "session"


# -----------------------------
# Global store helpers (so tools can write without Streamlit session)
# -----------------------------
_GLOBAL_PM: Optional[PatientMemory] = None

def get_store() -> PatientMemory:
    global _GLOBAL_PM
    if _GLOBAL_PM is None:
        _GLOBAL_PM = PatientMemory()
    return _GLOBAL_PM

def add_appointment(patient_id: str, appt: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level helper (imported by tools/booking_tool.py)."""
    return get_store().add_appointment(patient_id, appt)

def list_appointments(patient_id: str, include_past: bool = False) -> List[Dict[str, Any]]:
    return get_store().get_appointments(patient_id, include_past=include_past)

