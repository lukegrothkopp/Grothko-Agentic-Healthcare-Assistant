import os, json, datetime, threading
from pathlib import Path
from typing import Any, Dict

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
_METRICS_PATH = DATA_DIR / "metrics.json"
_TRACES_PATH = DATA_DIR / "traces.jsonl"

_lock = threading.Lock()

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

def inc_counter(name: str, by: int = 1):
    with _lock:
        data = _read_json(_METRICS_PATH)
        data.setdefault("counters", {})
        data["counters"][name] = int(data["counters"].get(name, 0)) + by
        _write_json(_METRICS_PATH, data)

def log_tool(tool_name: str, status: str, meta: Dict[str, Any] | None = None):
    row = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "type": "tool_usage",
        "tool": tool_name,
        "status": status,
        "meta": meta or {},
    }
    _append_jsonl(_TRACES_PATH, row)
    inc_counter(f"tool.{tool_name}.{status}", 1)

def log_plan(patient_id: str | None, user_query: str, steps: list[str]):
    row = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "type": "plan",
        "patient_id": patient_id,
        "query": user_query,
        "steps": steps,
    }
    _append_jsonl(_TRACES_PATH, row)
    inc_counter("plans", 1)

def log_result(stage: str, text: str, meta: Dict[str, Any] | None = None):
    row = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "type": "result",
        "stage": stage,
        "text": text[:2000],
        "meta": meta or {},
    }
    _append_jsonl(_TRACES_PATH, row)

def get_metrics_summary() -> Dict[str, Any]:
    data = _read_json(_METRICS_PATH)
    counters = data.get("counters", {})
    succ = counters.get("tool.Book Appointment.success", 0)
    fail = counters.get("tool.Book Appointment.failure", 0)
    total = succ + fail
    rate = (succ / total) if total else 0.0
    return {
        "counters": counters,
        "booking_success_rate": round(rate, 3),
        "total_plans": counters.get("plans", 0),
    }

def iter_traces(limit: int = 500):
    if not _TRACES_PATH.exists():
        return []
    rows = []
    with _TRACES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows[-limit:]
