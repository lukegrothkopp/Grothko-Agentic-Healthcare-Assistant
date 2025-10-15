# utils/patient_memory.py
# Lightweight in-memory patient store with seed auto-load, simple search,
# seed-dir knob OFFLINE_PATIENT_DIR, and small utilities used by the Console.

import os, json, re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

SEED_DIR_ENV = "OFFLINE_PATIENT_DIR"
DEFAULT_SEED_DIR = "data/patient_memory"

class PatientMemory:
    def __init__(self, seed_dir: Optional[str] = None):
        self.seed_dir = seed_dir or os.getenv(SEED_DIR_ENV, DEFAULT_SEED_DIR)
        self.patients: Dict[str, Dict[str, Any]] = {}
        self.reload_from_dir(self.seed_dir)

    # ---------- Seed loading / saving ----------
    def reload_from_dir(self, path: Optional[str] = None) -> None:
        """Clear and (re)load all patient JSON records from a folder."""
        if path:
            self.seed_dir = path
        try:
            os.makedirs(self.seed_dir, exist_ok=True)
        except Exception:
            pass
        self.patients.clear()
        if not os.path.isdir(self.seed_dir):
            return
        for name in sorted(os.listdir(self.seed_dir)):
            p = os.path.join(self.seed_dir, name)
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

    def save_patient_json(self, data: Dict[str, Any], dir_path: Optional[str] = None) -> str:
        """Write a single patient JSON to <dir>/<patient_id>.json and (re)load it."""
        target_dir = dir_path or self.seed_dir
        os.makedirs(target_dir, exist_ok=True)
        pid = str(data.get("patient_id") or "").strip()
        if not pid:
            raise ValueError("patient_id is required")
        out_path = os.path.join(target_dir, f"{pid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # refresh just this record
        self.patients[pid] = data
        return out_path

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

        # entries
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
        # labs summary
        for lab in p.get("labs", []):
            vals = lab.get("values", {})
            line = []
            for key in ("creatinine_mg_dL","egfr_mL_min_1.73m2","urine_acr_mg_g","a1c_percent","hemoglobin_g_dL","potassium_mmol_L","co2_bicarb_mmol_L"):
                if key in vals:
                    line.append(f"{key}={vals[key]}")
            if line:
                add_hit(5, "Labs: " + ", ".join(line))

        hits.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in hits[:max(1,int(k))]]

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
        # CKD father/70 heuristic
        if ("father" in t or "dad" in t) and ("kidney" in t or "ckd" in t):
            m2 = self._AGE_RE.search(t)
            if m2:
                age = m2.group(1) or m2.group(2)
                try:
                    if 68 <= int(age) <= 74:
                        for pid, data in self.patients.items():
                            probs = [p.get("name","").lower() for p in data.get("problems",[])]
                            if any("chronic kidney disease" in x for x in probs):
                                return pid
                except Exception:
                    pass
            for pid, data in self.patients.items():
                probs = [p.get("name","").lower() for p in data.get("problems",[])]
                if any("chronic kidney disease" in x for x in probs):
                    return pid
        return None

    # ---------- Convenience for the Console ----------
    def list_patients(self) -> List[Dict[str, Any]]:
        """Return a compact list for preview: id, name, age, key problems, last_updated."""
        out: List[Dict[str, Any]] = []
        for pid, p in sorted(self.patients.items(), key=lambda kv: kv[0]):
            prof = p.get("profile") or {}
            probs = ", ".join(q.get("name","") for q in p.get("problems", [])[:3] if q.get("name"))
            out.append({
                "patient_id": pid,
                "name": prof.get("full_name") or "",
                "age": prof.get("age"),
                "key_problems": probs,
                "last_updated": p.get("last_updated", "")
            })
        return out

    def booking_hints(self, patient_id: Optional[str]) -> Dict[str, Any]:
        if not patient_id:
            return {}
        p = self.patients.get(patient_id) or {}
        prefs = p.get("preferences") or {}
        days = int(prefs.get("appointment_window_days", 14) or 14)
        start = (datetime.utcnow() + timedelta(days=1)).date().isoformat()
        end = (datetime.utcnow() + timedelta(days=days)).date().isoformat()
        return {
            "clinic": prefs.get("preferred_clinic"),
            "modes": prefs.get("appointment_modes", []),
            "preferred_time_of_day": prefs.get("preferred_time_of_day"),
            "date_range": {"start": start, "end": end}
        }

