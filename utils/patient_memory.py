# utils/patient_memory.py
# Lightweight in-memory patient store with seed auto-load, simple search, and text-based id resolution.

import os, json, re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

SEED_DIR_ENV = "OFFLINE_PATIENT_DIR"
DEFAULT_SEED_DIR = "data/patient_memory"

class PatientMemory:
    def __init__(self, seed_dir: Optional[str] = None):
        self.seed_dir = seed_dir or os.getenv(SEED_DIR_ENV, DEFAULT_SEED_DIR)
        self.patients: Dict[str, Dict[str, Any]] = {}
        self._load_seed_records(self.seed_dir)

    # ---------- Seed loading ----------
    def _load_seed_records(self, path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
        for name in sorted(os.listdir(path)) if os.path.isdir(path) else []:
            p = os.path.join(path, name)
            if not os.path.isfile(p) or not name.lower().endswith(".json"):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pid = str(data.get("patient_id") or "").strip()
                if pid:
                    self.patients[pid] = data
            except Exception:
                continue

    # ---------- Core API ----------
    def upsert_patient_json(self, data: Dict[str, Any]) -> None:
        pid = str(data.get("patient_id") or "").strip()
        if not pid:
            raise ValueError("patient_id is required")
        self.patients[pid] = data

    def get(self, patient_id: str) -> Optional[Dict[str, Any]]:
        return self.patients.get(patient_id)

    def record_event(self, patient_id: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        now = datetime.utcnow().isoformat() + "Z"
        p = self.patients.setdefault(patient_id, {"patient_id": patient_id})
        entries = p.setdefault("entries", [])
        entries.append({"ts": now, "type": "note", "text": text, "meta": meta or {}})
        p["last_updated"] = now

    # ---------- Summaries & search ----------
    def get_summary(self, patient_id: Optional[str]) -> str:
        if not patient_id:
            return ""
        p = self.patients.get(patient_id)
        if not p:
            return ""
        if isinstance(p.get("summary"), str) and p["summary"].strip():
            return p["summary"]
        # fallback brief synthetic summary
        name = (p.get("profile") or {}).get("full_name") or patient_id
        probs = ", ".join(q.get("name","") for q in p.get("problems", [])[:3] if q.get("name"))
        return f"{name}: {probs}" if probs else name

    def search(self, patient_id: Optional[str], query: str, k: int = 3) -> List[str]:
        """Very simple relevance: look in entries, problems, meds, labs text fields."""
        if not patient_id or not query or not isinstance(query, str):
            return []
        p = self.patients.get(patient_id)
        if not p:
            return []
        q = query.lower()
        hits: List[Tuple[int, str]] = []

        def add_hit(weight: int, text: str):
            if text and isinstance(text, str):
                hits.append((weight, text))

        # prioritize entries
        for e in p.get("entries", []):
            t = str(e.get("text") or "")
            if q in t.lower():
                add_hit(10, t)
        # problems / meds
        for prob in p.get("problems", []):
            t = str(prob.get("name") or "")
            if q in t.lower():
                add_hit(7, f"Problem: {t}")
        for med in p.get("medications", []):
            t = str(med.get("name") or "")
            if q in t.lower():
                add_hit(4, f"Medication: {t}")
        # labs summary lines
        for lab in p.get("labs", []):
            vals = lab.get("values", {})
            line = []
            for key in ("creatinine_mg_dL","egfr_mL_min_1.73m2","urine_acr_mg_g","a1c_percent","hemoglobin_g_dL","potassium_mmol_L","co2_bicarb_mmol_L"):
                if key in vals:
                    line.append(f"{key}={vals[key]}")
            if line:
                add_hit(5, "Labs: " + ", ".join(line))

        # Keep top-k by weight & recency-ish order
        hits.sort(key=lambda x: x[0], reverse=True)
        out = [t for _, t in hits[:max(1,int(k))]]
        return out

    # ---------- Patient resolution from free text ----------
    _AGE_RE = re.compile(r"\b(\d{2})-?year-?old\b|\bage\s+(\d{2})\b", re.I)
    def resolve_from_text(self, text: str) -> Optional[str]:
        """Heuristics to pick a patient from user text."""
        if not text:
            return None
        t = text.lower()
        # Direct ID mention
        m = re.search(r"patient[_\s-]?(\d{3,})", t)
        if m:
            pid = f"patient_{m.group(1)}"
            if pid in self.patients:
                return pid
        # Name mention
        for pid, data in self.patients.items():
            name = ((data.get("profile") or {}).get("full_name") or "").lower()
            last = name.split()[-1] if name else ""
            if name and (name in t or last and last in t):
                return pid
        # CKD father/70 heuristic (fits Hal)
        if ("father" in t or "dad" in t) and ("kidney" in t or "ckd" in t):
            # check for ~70
            m2 = self._AGE_RE.search(t)
            if m2:
                age = m2.group(1) or m2.group(2)
                try:
                    if 68 <= int(age) <= 74:
                        # prefer a CKD patient in store if present
                        for pid, data in self.patients.items():
                            probs = [p.get("name","").lower() for p in data.get("problems",[])]
                            if any("chronic kidney disease" in x for x in probs):
                                return pid
                except Exception:
                    pass
            # fallback to any CKD
            for pid, data in self.patients.items():
                probs = [p.get("name","").lower() for p in data.get("problems",[])]
                if any("chronic kidney disease" in x for x in probs):
                    return pid
        return None

    # ---------- Preferences helpers ----------
    def booking_hints(self, patient_id: Optional[str]) -> Dict[str, Any]:
        if not patient_id:
            return {}
        p = self.patients.get(patient_id) or {}
        prefs = p.get("preferences") or {}
        # derive a suggested date within window (default 14 days)
        days = int(prefs.get("appointment_window_days", 14) or 14)
        start = (datetime.utcnow() + timedelta(days=1)).date().isoformat()
        end = (datetime.utcnow() + timedelta(days=days)).date().isoformat()
        return {
            "clinic": prefs.get("preferred_clinic"),
            "modes": prefs.get("appointment_modes", []),
            "preferred_time_of_day": prefs.get("preferred_time_of_day"),
            "date_range": {"start": start, "end": end}
        }
