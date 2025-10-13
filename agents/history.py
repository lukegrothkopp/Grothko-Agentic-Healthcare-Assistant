python
from __future__ import annotations
from typing import List, Dict, Any

from core.db import get_history, add_history

class HistoryAgent:
    def retrieve(self, patient_id: int) -> List[Dict[str, Any]]:
        return get_history(patient_id)

    def add(self, patient_id: int, note: str):
        add_history(patient_id, note)
        return {"success": True}
