from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class LogRecord:
    step: str
    detail: str
    success: bool = True
    meta: Dict[str, Any] = None

class RunLogger:
    def __init__(self):
        self.records: List[LogRecord] = []

    def log(self, step: str, detail: str, success: bool=True, meta: Dict[str, Any]=None):
        self.records.append(LogRecord(step=step, detail=detail, success=success, meta=meta or {}))

    def to_dicts(self):
        return [asdict(r) for r in self.records]
