from __future__ import annotations
from typing import List, Dict
from prompts import PLANNER_PROMPT

class Planner:
    def plan(self, user_input: str) -> List[Dict]:
        """Very simple pattern-based planner for demo purposes."""
        steps: List[Dict] = []
        q = user_input.lower()
        if any(w in q for w in ["book", "appointment", "schedule"]):
            steps.append({"action": "booking.search_doctors", "inputs": {}})
            steps.append({"action": "booking.pick_slot", "inputs": {}})
        if any(w in q for w in ["history", "record", "past"]):
            steps.append({"action": "history.get", "inputs": {}})
        if any(w in q for w in ["latest", "treatment", "info", "disease", "guideline", "research"]):
            steps.append({"action": "info_search.query", "inputs": {}})
        if not steps:
            steps.append({"action": "memory.search", "inputs": {}})
        return steps
