# utils/patient_memory.py
from __future__ import annotations

import os
import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, date, timezone

DEFAULT_SEED_DIR = "data/patient_seeds"
EXTRA_SEED_DIR = "data/patient_memory"          # extra seed dir used in your project
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


def _coerce_int(x: Any) -> Optional[int]:
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int,)):
            return int(x)
        if isinstance(x, float):
            return int(x)
        if isinstance(x, str) and x.strip():
            return int(float(x.strip()))
    except Exception:
        return None
    return None


def _parse_dob_to_age(dob_str: str) -> Optional[int]:
    """Support YYYY-MM-DD and MM/DD/YYYY. Returns age in years."""
    if not isinstance(dob_str, str) or not dob_str.strip():
        return None
    s = dob_str.strip()
    fmt_candidates = ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"]
    for fmt in fmt_candidates:
        try:
            dt = datetime.strptime(s, fmt).date()
            today = date.today()
            age = today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day))
            return age if age >= 0 and age < 140 else None
        except Exception:
            continue
    return None


def _normalize_conditions(raw: Any) -> List[str]:
    """
    Accepts:
      - list[str]
      - list[dict] with keys like 'name'/'condition'/'dx'
      - comma/semicolon separated string
    Returns: list[str]
    """
    out: List[str] = []
    if raw is None:
        return out
    if isinstance(raw, list):
        for it in raw:
            if isinstance(it, str):
                s = it.strip()
                if s:
                    out.append(s)
            elif isinstance(it, dict):
                val = it.get("name") or it.get("condition") or it.get("dx") or it.get("value")
                if isinstance(val, str) and val.strip():
                    out.append(val.strip())
    elif isinstance(raw, str):
        # split by comma/semicolon
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        out.extend(parts)
    return out


@dataclass
class _Patient:
    patient_id: str
    data: Dict[str, Any]


class PatientMemory:
    """
    Lightweight, file-backed patient memory:
    - Loads 'seed' patient JSON files from OFFLINE_PATIENT_DIR (or default).
    - Also loads from data/patient_memory/ (project's extra seed dir).
    - Merges in data from data/patient_db.json and data/patient_db/*.json.
    - Registers stub patients found only in data/memory/*.jsonl logs.
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
    # Seed/DB loading (merged) + logs
    # -----------------------------
    def reload_from_dir(self, seed_dir: str) -> None:
        """Load/merge patients from seeds, extra dir, DB, then register any missing from memory logs."""
        self.seed_dir = seed_dir
        self.patients.clear()
        self.history.clear()
        self.sessions.clear()

        # A) Always load configured seed folder
        self._load_seed_folder(seed_dir)

        # B) Also load project-specific extra seed folder
        if os.path.isdir(EXTRA_SEED_DIR):
            self._load_seed_folder(EXTRA_SEED_DIR)

        # C) Merge DB fallback (adds any not already present)
        self._load_from_patient_db_fallback_merge()

        # D) Load per-patient memory logs; auto-register stubs for log-only patients
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
            if pid not in self.patients:
                self._register_stub_patient(pid)

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

    def _is_db_patient_map(self, data: Dict[str, Any]) -> bool:
        """Heuristic: dict keyed by patient IDs (e.g., 'patient_001')."""
        try:
            return any(str(k).startswith("patient_") for k in data.keys())
        except Exception:
            return False

    def _load_from_patient_db_fallback_merge(self) -> None:
        """Add patients from patient_db files if they aren't already loaded from seeds."""
        # A) data/patient_db.json
        if DEFAULT_DB_PATH.exists():
            try:
                data = json.loads(DEFAULT_DB_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    if self._is_db_patient_map(data):
                        for pid, rec in data.items():
                            if pid not in self.patients and isinstance(rec, dict):
                                rec = dict(rec); rec.setdefault("patient_id", pid)
                                self._ingest_patient_record(rec)
                    else:
                        pid = data.get("patient_id") or data.get("id") or data.get("pid")
                        if pid and pid not in self.patients:
                            self._ingest_patient_record(data)
                elif isinstance(data, list):
                    for rec in data:
                        if isinstance(rec, dict):
                            pid = rec.get("patient_id") or rec.get("id") or rec.get("pid")
                            if pid and pid not in self.patients:
                                self._ingest_patient_record(rec)
            except Exception:
                pass

        # B) data/patient_db/*.json
        for fp in sorted(glob.glob(DEFAULT_DB_GLOB)):
            try:
                data = json.loads(Path(fp).read_text(encoding="utf-8"))
            except Exception:
                continue

            if isinstance(data, dict):
                if self._is_db_patient_map(data):
                    for pid, rec in data.items():
                        if pid not in self.patients and isinstance(rec, dict):
                            rec = dict(rec); rec.setdefault("patient_id", pid)
                            self._ingest_patient_record(rec)
                else:
                    pid = data.get("patient_id") or data.get("id") or data.get("pid")
                    if pid and pid not in self.patients:
                        self._ingest_patient_record(data)
            elif isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        pid = rec.get("patient_id") or rec.get("id") or rec.get("pid")
                        if pid and pid not in self.patients:
                            self._ingest_patient_record(rec)

    def _register_stub_patient(self, pid: str) -> None:
        """Create a minimal patient entry so log-only patients show up in UIs."""
        stub = {
            "patient_id": pid,
            "name": pid.replace("_", " ").title(),
            "conditions": [],
            "appointments": [],
            "history": [],
        }
        self.patients[pid] = _Patient(patient_id=pid, data=stub)

    # -----------------------------
    # Ingest + persistence
    # -----------------------------
    def _ingest_patient_db_dict(self, db: Dict[str, Any]) -> None:
        """db is a dict keyed by patient_id → record dict."""
        for pid, rec in db.items():
            if not isinstance(rec, dict):
                continue
            rec = dict(rec)
            rec.setdefault("patient_id", pid)
            self._ingest_patient_record(rec)

    def _ingest_patient_record(self, rec: Dict[str, Any]) -> None:
        """
        Normalize and store a patient record.

        Adds:
          - name from profile.full_name if missing
          - age from age/profile.age/demographics.age or derived from dob/date_of_birth/profile.dob/demographics.dob
          - conditions normalized from various shapes
          - entries[] folded into history[]
        """
        if not isinstance(rec, dict):
            return

        # 1) Name normalization from profile.full_name
        if "name" not in rec and isinstance(rec.get("profile"), dict):
            fn = rec["profile"].get("full_name")
            if isinstance(fn, str) and fn.strip():
                rec["name"] = fn.strip()

        # 2) Determine patient_id (fall back to name-derived)
        pid = rec.get("patient_id") or rec.get("id") or rec.get("pid")
        if not pid:
            name_for_id = (rec.get("name") or rec.get("profile", {}).get("full_name") or "patient")
            pid = name_for_id.lower().replace(" ", "_")

        # 3) Age normalization
        age_val = rec.get("age")
        if age_val is None:
            # fallback to profile/demographics
            prof = rec.get("profile") or {}
            demo = rec.get("demographics") or {}
            age_val = prof.get("age", demo.get("age"))
        age_int = _coerce_int(age_val)

        if age_int is None:
            # derive from DOB if possible
            dob = rec.get("dob") or rec.get("date_of_birth")
            if not dob and isinstance(rec.get("profile"), dict):
                dob = rec["profile"].get("dob")
            if not dob and isinstance(rec.get("demographics"), dict):
                dob = rec["demographics"].get("dob")
            if isinstance(dob, str):
                age_int = _parse_dob_to_age(dob)

        if age_int is not None:
            rec["age"] = age_int  # store normalized age for downstream UI

        # 4) Conditions normalization
        conds = rec.get("conditions")
        if conds is None:
            # alternate keys sometimes used
            conds = rec.get("diagnoses") or rec.get("problems") or rec.get("dx")
            if conds is None and isinstance(rec.get("profile"), dict):
                conds = rec["profile"].get("conditions")
            if conds is None and isinstance(rec.get("demographics"), dict):
                conds = rec["demographics"].get("conditions")
        rec["conditions"] = _normalize_conditions(conds)

        # Persist the patient (dataclass wrapper)
        self.patients[pid] = _Patient(patient_id=pid, data=rec)

        # 5) Normalize embedded history + appointments + entries as events with timestamps
        def _ensure_ts(x):
            if isinstance(x, dict) and not x.get("ts"):
                x["ts"] = x.get("date") or x.get("timestamp") or _now_iso()
            return x

        hist = rec.get("history") or []
        if isinstance(hist, dict):
            hist = [hist]
        hist = [_ensure_ts(h) for h in hist if isinstance(h, dict)]

        # entries → history (your custom schema)
        entries = rec.get("entries") or []
        if isinstance(entries, dict):
            entries = [entries]
        entries = [_ensure_ts(e) for e in entries if isinstance(e, dict)]

        appts = rec.get("appointments") or []
        if isinstance(appts, dict):
            appts = [appts]
        appts = [_ensure_ts({"type": "appointment", **a}) for a in appts if isinstance(a, dict)]

        merged = []
        if hist:     merged.extend(hist)
        if entries:  merged.extend(entries)
        if appts:    merged.extend(appts)

        if merged:
            self.history.setdefault(pid, []).extend(merged)

    def save_patient_json(self, rec: Dict[str, Any], dir_path: Optional[str] = None) -> str:
        """Write a patient JSON file into the seed directory and return path."""
        d = dir_path or self.seed_dir
        Path(d).mkdir(parents=True, exist_ok=True)
        pid = rec.get("patient_id") or rec.get("id") or rec.get("pid") or \
              (rec.get("name", rec.get("profile", {}).get("full_name", "patient")).lower().replace(" ", "_"))
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
        # ensure age present if we can derive it now
        if base.get("age") is None:
            # try to derive from any known DOB aliases
            dob = base.get("dob") or base.get("date_of_birth")
            if not dob and isinstance(base.get("profile"), dict):
                dob = base["profile"].get("dob")
            if not dob and isinstance(base.get("demographics"), dict):
                dob = base["demographics"].get("dob")
            if isinstance(dob, str):
                age_int = _parse_dob_to_age(dob)
                if age_int is not None:
                    base["age"] = age_int
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
            # ensure normalized fields
            name = base.get("name", pid)
            age = base.get("age")
            if age is None:
                dob = base.get("dob") or base.get("date_of_birth")
                if not dob and isinstance(base.get("profile"), dict):
                    dob = base["profile"].get("dob")
                if not dob and isinstance(base.get("demographics"), dict):
                    dob = base["demographics"].get("dob")
                if isinstance(dob, str):
                    age = _parse_dob_to_age(dob)
            conds = base.get("conditions") or []
            rows.append({
                "patient_id": pid,
                "name": name,
                "age": age,
                "conditions": ", ".join(map(str, conds)),
            })
        rows.sort(key=lambda r: (str(r.get("name") or ""), str(r.get("patient_id") or "")))
        return rows
