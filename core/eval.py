"""
Lightweight evaluation utilities (heuristic Q/A scoring).
"""
from __future__ import annotations
from typing import Dict

def simple_precision(expected_keywords, text: str) -> float:
    if not expected_keywords:
        return 0.0
    found = sum(1 for k in expected_keywords if k.lower() in (text or "").lower())
    return found / len(expected_keywords)

def eval_summary(summary: str, task_hints: Dict[str, list]) -> Dict[str, float]:
    """
    task_hints example: {"ckd": ["ckd", "eGFR", "ACE", "dialysis"], "booking": ["appointment", "doctor"]}
    """
    results = {}
    for name, keys in task_hints.items():
        results[name] = round(simple_precision(keys, summary), 3)
    results["overall"] = round(sum(results.values())/max(1, len(results)), 3)
    return results
