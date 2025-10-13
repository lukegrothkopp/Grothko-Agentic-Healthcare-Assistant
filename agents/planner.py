python
from __future__ import annotations
from pydantic import BaseModel
from typing import List, Dict, Any
import json

from prompts import SYSTEM_PLANNER

class PlanStep(BaseModel):
    action: str
    inputs: Dict[str, Any] = {}

class Planner:
    """
    Very lightweight rule-based planner (can be replaced by an LLM call if desired).
    """
    def plan(self, user_input: str) -> List[PlanStep]:
        text = user_input.lower()
        steps: List[PlanStep] = []
        steps.append(PlanStep(action="identify_patient"))
        steps.append(PlanStep(action="retrieve_history"))
        if any(k in text for k in ["book", "schedule", "appointment", "nephrologist", "doctor"]):
            steps.append(PlanStep(action="book_appointment"))
        if any(k in text for k in ["latest", "treatment", "options", "what is", "tell me about", "disease", "ckd", "cancer", "diabetes"]):
            steps.append(PlanStep(action="medical_info_search"))
        steps.append(PlanStep(action="summarize"))
        return steps
