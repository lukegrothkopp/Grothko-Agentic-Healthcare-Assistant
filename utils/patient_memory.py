# utils/patient_memory.py
from __future__ import annotations

import os
import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

DEFAULT_SEED_DIR = "data/patient_seeds"
DEFAULT_DB_PATH = Path("data/patient_db.json")
DEFAULT_DB_GLOB = "data/patient_db/*.json"

MEMORY_LOG_DIR = Path("data/memory")
MEMORY_LOG_DIR.mkdir(parents=True, exist_ok=True)

def _now_iso() -> str:
    # Naive ISO is fine; we normalize to UTC epoch when sorting
    return datetime.now().isoformat(timespec="seconds")

def _to_epoch(ts) -> float:
    """
    Convert assorted timestamp forms to a single sortable number (UTC epoch seconds).
    Returns -inf on failure so bad timestamps land at the beginning.
    """
    try:
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str) and ts.strip():
            s = ts.strip().rstrip("Z")  # tolerate trailing Z
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                # treat naive as UTC to avoid local-tz surprises
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.timestamp()
    except Exception:
        pass
    return float("-inf")

@dataclass
class _Patient:
    patient_id: str
    data: Dict[str, Any]

class PatientMemory:
    """
    Lightweight, file-backed patient memory:
    - Loads 'seed' patient JSON files from OFFLINE_PATIENT_DIR (or default).
    - If seeds are empty/missing, FALLS BACK to data/patient_db.json and data/patient_db/*.json.
    - Persists per-patient events into data/memory/<patient_id>.jsonl
    - Provides summary, retrieval, and recent-window helpers.
    """

    def __init__(self, seed_dir: Optional[str] = None):
        self.seed_dir: str = seed_dir or os.environ.get("OFFLINE_PATIENT_DIR", DEFAULT_SEED_DIR)
        Path(self.seed_dir).mkdir(parents=True, exist_ok=True)

        self.patients: Dict[str, _Patient | Dict[str, Any]] = {}
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

        # 1) Try seeds in the configured folder
        self._load_seed_folder(seed_dir)

        # 2) If still empty, FALLBACK to patient_db.json and data/patient_db/*.json
        if not self.patients:
            self._load_from_patient_db_fallback()

        # 3) Load per-patient memory logs
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
        """Load patients from a seed folder (expects *.json files)."""
        for fp in sorted(glob.glob(os.path.join(folder, "*.json"))):
            try:
                data = json.loads(Path(fp).read_text(encoding="utf-8"))
            except Exception:
                continue

            if isinstance(data, list):
                for rec in data:
                    self._ingest_patient_record(rec)
            elif isinstance(data, dict):
                # container or single record
                if "patients" in data and isinstance(data["patients"], list):
                    for rec in data["patients"]:
                        self._ingest_patient_record(rec)
                else:
                    self._ingest_patient_record(data)

    def _load_from_patient_db_fallback(self) -> None:
        """Fallback loader from the legacy DB files if seed folder is empty."""
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
            # If it's a dict of patients keyed by id
            if isinstance(data, dict) and "patient_001" in data or any(k.startswith("patient_") for k in data.keys()):
                self._ingest_patient_db_dict(data)
            # Or a single record / list of records
            elif isinstance(data, dict):
                self._ingest_patient_record(data)
            elif isinstance(data, list):
                for rec in data:
                    self._ingest_patient_record(rec)

    def _ingest_patient_db_dict(self, db: Dict[str, Any]) -> None:
        """db is a dict keyed by patient_id → record dict."""
        for pid, rec in db.items():
            if not isinstance(rec, dict):
                continue
            rec = dict(rec)
            rec.setdefault("patient_id", pid)
            self._ingest_patient_record(rec)

    def _ingest_patient_record(self, rec: Dict[str, Any]) -> None:
        """Normalize and store a patient record."""
        if not isinstance(rec, dict):
            return
        pid = rec.get("patient_id") or rec.get("id") or rec.get("pid")
        if not pid:
            # generate a simple deterministic ID from name if present
            name = (rec.get("name") or "patient").lower().replace(" ", "_")
            pid = f"{name}_{len(self.patients) + 1}"

        # allow downstream to handle either dict or dataclass
        self.patients[pid] = _Patient(patient_id=pid, data=rec)

        # Normalize embedded history + appointments as events with timestamps
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
        - Normalizes to UTC epoch seconds for sorting (no tz errors),
        - Returns most-recent first.
        """
        raw = self.history.get(patient_id, []) + self.sessions.get(patient_id, [])
        entries = [e for e in raw if isinstance(e, dict)]

        # Backfill ts from common aliases; fallback to now if still missing
        now_iso = _now_iso()
        for e in entries:
            if not e.get("ts"):
                e["ts"] = e.get("timestamp") or e.get("time") or e.get("date") or now_iso

        # Sort safely by epoch seconds (floats), avoiding naive/aware datetime comparisons
        entries.sort(key=lambda e: _to_epoch(e.get("ts")))
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
        base = p.data if hasattr(p, "data") and isinstance(p.data, dict) else (p if isinstance(p, dict) else {})
        if p and not hits:
            conds = base.get("conditions") or []
            for c in conds:
                t = str(c)
                if q in t.lower():
                    hits.append((0.5, f"Condition match: {t}"))

        hits.sort(key=lambda x: x[0], reverse=True)
        return [h[1] for h in hits[:k]]

    def get_summary(self, patient_id: str) -> str:
        """Compact one-paragraph summary from seed/DB record + last activity."""
        p = self.patients.get(patient_id)
        base = p.data if hasattr(p, "data") and isinstance(p.data, dict) else (p if isinstance(p, dict) else {})
        if not base:
            return ""
        name = base.get("name", patient_id)
        age = base.get("age")
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
                f"Last appointment: {last_appt.get('date')} — {last_appt.get('doctor','(unknown)')} ({last_appt.get('status','')})"
            )
        if recent_blurbs:
            parts.append("Recent: " + " | ".join(recent_blurbs))

        return " | ".join(parts)

    def list_patients(self) -> List[Dict[str, Any]]:
        """Lightweight patient directory for UI select boxes."""
        rows: List[Dict[str, Any]] = []
        for pid, p in self.patients.items():
            base = p.data if hasattr(p, "data") and isinstance(p.data, dict) else (p if isinstance(p, dict) else {})
            rows.append({
                "patient_id": pid,
                "name": base.get("name", pid),
                "age": base.get("age"),
                "conditions": ", ".join(map(str, base.get("conditions") or [])),
            })
        rows.sort(key=lambda r: (str(r.get("name") or ""), str(r.get("patient_id") or "")))
        return rows
