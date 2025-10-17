# utils/patient_memory.py
from __future__ import annotations

import os
import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

DEFAULT_SEED_DIR = "data/patient_seeds"
MEMORY_LOG_DIR = Path("data/memory")
MEMORY_LOG_DIR.mkdir(parents=True, exist_ok=True)

def _safe_parse_ts(ts) -> datetime:
    """Parse assorted timestamp forms into datetime; fallback to datetime.min on failure."""
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts)
        if isinstance(ts, str) and ts.strip():
            s = ts.strip().rstrip("Z")  # tolerate trailing 'Z'
            # fromisoformat is quite tolerant of "YYYY-MM-DDTHH:MM[:SS]" formats
            return datetime.fromisoformat(s)
    except Exception:
        pass
    return datetime.min

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

@dataclass
class _Patient:
    patient_id: str
    data: Dict[str, Any]

class PatientMemory:
    """
    Lightweight, file-backed patient memory:
    - Loads 'seed' patient JSON files from OFFLINE_PATIENT_DIR (or default).
    - Persists per-patient events into data/memory/<patient_id>.jsonl
    - Provides summary, retrieval, and recent-window helpers.
    """

    def __init__(self, seed_dir: Optional[str] = None):
        self.seed_dir: str = seed_dir or os.environ.get("OFFLINE_PATIENT_DIR", DEFAULT_SEED_DIR)
        Path(self.seed_dir).mkdir(parents=True, exist_ok=True)

        self.patients: Dict[str, _Patient] = {}
        self.history: Dict[str, List[Dict[str, Any]]] = {}   # unified events (notes, bookings, etc.)
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}  # optional session logs

        self.reload_from_dir(self.seed_dir)

    # -----------------------------
    # Seed loading / persistence
    # -----------------------------
    def reload_from_dir(self, seed_dir: str) -> None:
        """Load patient JSON files from a folder; rebuild in-memory index."""
        self.seed_dir = seed_dir
        self.patients.clear()
        self.history.clear()
        self.sessions.clear()

        # Load seed patients (files can be a list of records or a single dict)
        for fp in sorted(glob.glob(os.path.join(seed_dir, "*.json"))):
            try:
                data = json.loads(Path(fp).read_text(encoding="utf-8"))
            except Exception:
                continue

            if isinstance(data, list):
                for rec in data:
                    self._ingest_patient_record(rec)
            elif isinstance(data, dict):
                # one record or container
                if "patients" in data and isinstance(data["patients"], list):
                    for rec in data["patients"]:
                        self._ingest_patient_record(rec)
                else:
                    self._ingest_patient_record(data)

        # Load per-patient memory logs
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

    def _ingest_patient_record(self, rec: Dict[str, Any]) -> None:
        """Normalize and store a patient record."""
        pid = rec.get("patient_id")
        if not pid:
            # try common shapes
            pid = rec.get("id") or rec.get("pid")
        if not pid:
            # generate a simple deterministic ID from name if present
            name = (rec.get("name") or "patient").lower().replace(" ", "_")
            pid = f"{name}_{len(self.patients) + 1}"

        self.patients[pid] = _Patient(patient_id=pid, data=rec)

        # If the record has embedded "history" or "appointments", index them as events
        # Ensure they have timestamps so recent-window sort works
        def _ensure_ts(x):
            if isinstance(x, dict):
                if not x.get("ts"):
                    x["ts"] = x.get("date") or x.get("timestamp") or _now_iso()
            return x

        hist = rec.get("history") or []
        hist = [_ensure_ts(h) for h in hist if isinstance(h, dict)]
        appts = rec.get("appointments") or []
        appts = [_ensure_ts({"type": "appointment", **a}) for a in appts if isinstance(a, dict)]

        if hist or appts:
            self.history.setdefault(pid, []).extend(hist + appts)

    def save_patient_json(self, rec: Dict[str, Any], dir_path: Optional[str] = None) -> str:
        """Write a patient JSON file into the seed directory and return path."""
        d = dir_path or self.seed_dir
        Path(d).mkdir(parents=True, exist_ok=True)
        pid = rec.get("patient_id") or rec.get("id") or rec.get("pid") or \
              (rec.get("name", "patient").lower().replace(" ", "_"))
        out = Path(d) / f"{pid}.json"
        out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return str(out)

    # -----------------------------
    # Memory operations
    # -----------------------------
    def record_event(self, patient_id: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Append a memory event and persist to data/memory/<patient_id>.jsonl"""
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

    def get_window(self, patient_id: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Safe recent activity window:
        - Backfills ts when missing,
        - Tolerates various ts formats,
        - Always sorts by a valid key.
        """
        raw = self.history.get(patient_id, []) + self.sessions.get(patient_id, [])
        entries = [e for e in raw if isinstance(e, dict)]

        # Backfill ts from common aliases; fallback to now if still missing
        now_iso = _now_iso()
        for e in entries:
            if not e.get("ts"):
                e["ts"] = e.get("timestamp") or e.get("time") or e.get("date") or now_iso

        # Sort safely; unparseable timestamps drop to the beginning
        entries.sort(key=lambda e: _safe_parse_ts(e.get("ts")))
        return list(reversed(entries[-k:]))

    def search(self, patient_id: str, query: str, k: int = 3) -> List[str]:
        """Simple keyword search over recent memory texts + seed history notes."""
        if not query:
            return []
        q = query.lower()
        hits: List[Tuple[float, str]] = []

        for row in self.history.get(patient_id, []):
            txt = (row.get("text") or
                   row.get("notes") or
                   row.get("diagnosis") or "")
            if not isinstance(txt, str):
                continue
            t = txt.lower()
            if q in t:
                hits.append((1.0, txt))
            else:
                # tiny heuristic: overlap of words
                qw = set(w for w in q.split() if len(w) > 2)
                tw = set(w for w in t.split() if len(w) > 2)
                score = len(qw & tw) / max(1, len(qw))
                if score > 0:
                    hits.append((score, txt))

        # fall back to seed patient top-level fields (like conditions)
        p = self.patients.get(patient_id)
        if p and not hits:
            conds = p.data.get("conditions") or []
            for c in conds:
                t = str(c)
                if q in t.lower():
                    hits.append((0.5, f"Condition match: {t}"))

        hits.sort(key=lambda x: x[0], reverse=True)
        return [h[1] for h in hits[:k]]

    def get_summary(self, patient_id: str) -> str:
        """Compact one-paragraph summary from seed record + last activity."""
        p = self.patients.get(patient_id)
        if not p:
            return ""
        d = p.data
        name = d.get("name", patient_id)
        age = d.get("age")
        conds = d.get("conditions") or []
        last_appt = None
        appts = d.get("appointments") or []
        if appts:
            # sort by date string, tolerate missing
            try:
                appts_sorted = sorted(appts, key=lambda a: _safe_parse_ts(a.get("date")), reverse=True)
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
                f"Last appointment: {last_appt.get('date')} — {last_appt.get('doctor','(unknown)')} ({last_appt.get('status','')})"
            )
        if recent_blurbs:
            parts.append("Recent: " + " | ".join(recent_blurbs))

        return " | ".join(parts)

    def list_patients(self) -> List[Dict[str, Any]]:
        """Lightweight patient directory for UI select boxes."""
        rows: List[Dict[str, Any]] = []
        for pid, p in self.patients.items():
            d = p.data
            rows.append({
                "patient_id": pid,
                "name": d.get("name", pid),
                "age": d.get("age"),
                "conditions": ", ".join(map(str, d.get("conditions") or [])),
            })
        rows.sort(key=lambda r: (str(r.get("name") or ""), str(r.get("patient_id") or "")))
        return rows
