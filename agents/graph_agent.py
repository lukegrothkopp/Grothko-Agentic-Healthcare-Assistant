# agents/graph_agent.py
# Compiles the Agent Workflow graph for the Patient Assistant page.
# - Safe system prompt
# - Auto-identify patient from free text (via PatientMemory.resolve_from_text)
# - Booking preferences pulled from PatientMemory
# - Offline KB (RAG) + trusted online search
# - Returns only plain strings (no numpy objects) to avoid truthiness errors

import os
import json
import re
from datetime import date, timedelta
from operator import add as list_concat
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tools.search_tool import get_medical_search_tool
from tools.record_tool import get_record_tools
from tools.booking_tool import get_booking_tool
from tools.offline_kb_tool import get_offline_kb_tool
from utils.rag_pipeline import RAGPipeline
from utils.patient_memory import PatientMemory

__all__ = ["build_graph"]

# =========================
# State
# =========================
class AgentState(dict):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: Optional[str]
    result: Optional[str]
    patient_id: Optional[str]
    plan: Annotated[List[str], list_concat]

# =========================
# Prompts
# =========================
SYSTEM_SAFETY = (
    "You are a cautious healthcare admin/info assistant.\n"
    "- NEVER provide medical advice, diagnosis, or treatment instructions.\n"
    "- You summarize reputable sources and help with logistics (records, appointments).\n"
    "- If a request is clinical, gently defer to a licensed clinician."
)

SAFETY_CORE = (
    "You ONLY help with general information and admin tasks. "
    "You never provide medical advice, diagnosis, or treatment."
)

PLAN_TEMPLATE = (
    "Produce a minimal ordered list of actions the assistant should take given the user's message. "
    "Valid steps are a subset of: ['search', 'records', 'booking', 'rag']. "
    "Return only steps needed and nothing else."
)

class PlanOut(BaseModel):
    steps: List[str] = Field(..., description="Ordered list of minimal steps")

DISCLAIMER = (
    "\n\n— This assistant shares general information only (not medical advice). "
    "For severe, sudden, or worsening symptoms—or red flags like head injury, fever + stiff neck, "
    "confusion, weakness, vision changes—seek licensed care or emergency services."
)

# =========================
# Memory singleton
# =========================
_MEM_SINGLETON: Optional[PatientMemory] = None
def _mem() -> PatientMemory:
    global _MEM_SINGLETON
    if _MEM_SINGLETON is None:
        _MEM_SINGLETON = PatientMemory()  # auto-loads OFFLINE_PATIENT_DIR (default data/patient_memory)
    return _MEM_SINGLETON

def memory_context_provider(patient_id: Optional[str], user_query: str, k: int = 3):
    summary = _mem().get_summary(patient_id) if patient_id else ""
    recalls = _mem().search(patient_id, user_query, k=k) if patient_id else []
    pre_lines: List[str] = [f"Patient context summary:\n{summary or '(none)'}"]
    if recalls:
        pre_lines.append("Relevant past entries:")
        pre_lines.extend([f"- {r}" for r in recalls])
    return summary, recalls, "\n\n".join(pre_lines).strip()

# =========================
# Helpers
# =========================
def _classify_intent(text: str) -> str:
    t = (text or "").lower()
    if any(w in t for w in ["book", "schedule", "appointment"]): return "booking"
    if any(w in t for w in ["record", "history", "ehr", "note"]): return "records"
    if any(w in t for w in ["kb", "knowledge base", "rag", "offline"]): return "rag"
    if any(w in t for w in ["info", "guideline", "treatment", "what is", "symptom", "disease", "condition"]): return "search"
    return "search"

DATE_PAT = re.compile(
    r"(20\d{2}-\d{2}-\d{2})|((?:\d{1,2})/(?:\d{1,2})/(?:20\d{2}))",
    re.I,
)

def _parse_relative_date_phrase(text: str, today: Optional[date] = None) -> Optional[str]:
    if not text:
        return None
    t = text.lower().strip()
    today = today or date.today()
    if t == "today":
        return today.isoformat()
    if t == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    m2 = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", t)
    if m2:
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        wd = weekdays.index(m2.group(1))
        days_ahead = (wd - today.weekday() + 7) % 7
        if days_ahead == 0:
            days_ahead = 7
        return (today + timedelta(days=days_ahead)).isoformat()
    return None

def _extract_date(text: str) -> Optional[str]:
    m = DATE_PAT.search(text or "")
    if m:
        d = m.group(0)
        if "/" in d:
            mm, dd, yyyy = d.split("/")
            return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"
        return d
    return _parse_relative_date_phrase(text)

def _extract_doctor(text: str) -> Optional[str]:
    m = re.search(r"Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text or "")
    if m:
        return m.group(1)
    t = (text or "").lower()
    if "nephro" in t or "kidney" in t:
        return "Nephrologist"
    if "hypertension" in t:
        return "Cardiologist"
    return "Primary Care"

def _make_booking_payload(user_text: str, fallback_pid: Optional[str], hints: Dict[str, Any]) -> str:
    pid = fallback_pid or "patient_001"
    doc = _extract_doctor(user_text) or "Primary Care"
    d_iso = _extract_date(user_text)
    if not d_iso:
        d_iso = (hints.get("date_range") or {}).get("start")
    if not d_iso:
        d_iso = (date.today() + timedelta(days=7)).isoformat()
    payload: Dict[str, Any] = {
        "patient_id": pid,
        "doctor_name": doc,
        "appointment_date": d_iso,
    }
    # optional fields if your tool supports them
    clinic = hints.get("clinic")
    modes = hints.get("modes") or []
    tod = hints.get("preferred_time_of_day")
    if clinic:
        payload["clinic"] = clinic
    if modes:
        payload["modes"] = modes
    if tod:
        payload["preferred_time_of_day"] = tod
    return json.dumps(payload)

def _nonempty_str(s: Any) -> bool:
    return isinstance(s, str) and len(s.strip()) > 0

# =========================
# Build graph
# =========================
def build_graph(model_name: str = "gpt-4o-mini"):
    key = os.getenv("OPENAI_API_KEY", "").strip()
    llm = ChatOpenAI(model=model_name, temperature=0.1, api_key=key) if key else None
    planner = llm.with_structured_output(PlanOut) if llm else None

    tools: List[Tool] = []
    tools.append(get_medical_search_tool())
    tools.extend(get_record_tools())
    tools.append(get_booking_tool())
    tools.append(get_offline_kb_tool())

    rag = RAGPipeline()

    def rag_retrieve(q: str) -> str:
        try:
            pairs = rag.retrieve(q, k=3)
        except Exception as e:
            return f"(RAG error) {e}"
        if not pairs:
            return "No local KB results."
        lines = [f"- {str(txt)[:400]} (score={float(score):.3f})" for txt, score in pairs]
        return "\n".join(lines) or "No local KB results."

    graph = StateGraph(AgentState)

    # ---- Nodes ----
    def start(state: AgentState):
        # latest user text
        user_text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                user_text = m.content
                break

        # auto-identify patient from text (if not already set)
        pid = state.get("patient_id")
        if not pid:
            pid = _mem().resolve_from_text(user_text)

        out: Dict[str, Any] = {}
        if pid and pid != state.get("patient_id"):
            out["patient_id"] = pid

        # inject safety + memory context
        _, _, pre = memory_context_provider(pid, user_text, k=3)
        already = any(
            isinstance(m, SystemMessage)
            and isinstance(getattr(m, "content", None), str)
            and SYSTEM_SAFETY.splitlines()[0] in m.content
            for m in state.get("messages", [])
        )
        if not already:
            out.setdefault("messages", []).append(SystemMessage(content=f"{SYSTEM_SAFETY}\n\n{pre}".strip()))
        return out

    def plan(state: AgentState):
        user_text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                user_text = m.content
                break
        steps: List[str] = ["search", "booking"]  # default likely flow
        if planner and _nonempty_str(user_text):
            prompt = [
                SystemMessage(content=SAFETY_CORE),
                HumanMessage(content=PLAN_TEMPLATE + "\n\nUser: " + user_text),
            ]
            try:
                out = planner.invoke(prompt)
                if out and isinstance(out.steps, list) and len(out.steps) > 0:
                    steps = [s for s in out.steps if s in {"search", "records", "booking", "rag"}]
            except Exception:
                pass
        return {"plan": steps}

    def classify(state: AgentState):
        text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                text = m.content
                break
        return {"intent": _classify_intent(text)}

    def do_booking(state: AgentState):
        # build payload with patient prefs
        text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                text = m.content
                break
        pid = state.get("patient_id")
        hints = _mem().booking_hints(pid)
        payload = _make_booking_payload(text, pid, hints)

        res: Any = "Booking tool unavailable."
        for t in tools:
            if t.name == "Book Appointment":
                res = t.func(payload)
                break

        res_str = res if isinstance(res, str) else json.dumps(res)

        # log booking event
        if pid:
            _mem().record_event(pid, f"[Booking] {res_str}", meta={"stage": "booking"})

        # human-friendly echo of preferences
        prefs_note = ""
        if hints:
            clinic = hints.get("clinic")
            modes = ", ".join(hints.get("modes", [])) if hints.get("modes") else ""
            tod = hints.get("preferred_time_of_day")
            extras = " | ".join(
                x
                for x in [
                    f"clinic={clinic}" if clinic else None,
                    f"modes={modes}" if modes else None,
                    f"time={tod}" if tod else None,
                ]
                if x
            )
            if extras:
                prefs_note = f"\nPreferences: {extras}"

        msg = res_str + prefs_note if prefs_note else res_str
        return {"result": res_str, "messages": [AIMessage(content=msg)]}

    def do_records(state: AgentState):
        pid = state.get("patient_id")
        if not pid:
            return {"messages": [AIMessage(content="No patient identified for records retrieval.")]}

        summary = _mem().get_summary(pid)
        recalls = _mem().search(pid, "ckd", k=3)
        text = f"Patient {pid} history:\n- {summary}\n"
        if recalls:
            text += "\n".join([f"- {r}" for r in recalls])
        return {"result": text, "messages": [AIMessage(content=text)]}

    def do_search(state: AgentState):
        text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                text = m.content
                break

        parts: List[str] = []

        rag_text = rag_retrieve(text)
        if _nonempty_str(rag_text) and "No local KB results." not in rag_text:
            parts.append("**Offline KB**\n" + rag_text)

        online_text = ""
        search_tools = [t for t in tools if t.name == "Clinical Search (SERP/DDG)"] or [
            t for t in tools if "search" in t.name.lower()
        ]
        if search_tools:
            try:
                res2 = search_tools[0].func(text)
                online_text = res2 if isinstance(res2, str) else json.dumps(res2)
            except Exception as e:
                online_text = f"(online search error) {e}"

        if _nonempty_str(online_text):
            parts.append("**Trusted online sources**\n" + online_text)

        out = "\n\n".join(parts) if parts else "No results."
        out += DISCLAIMER
        return {"result": out, "messages": [AIMessage(content=out)]}

    def do_rag(state: AgentState):
        text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                text = m.content
                break
        out = rag_retrieve(text) or "No local KB results."
        out += DISCLAIMER
        return {"result": out, "messages": [AIMessage(content=out)]}

    def finalize(state: AgentState):
        return {}

    def route(state: AgentState) -> str:
        intent = (state.get("intent") or "search").lower()
        if intent not in {"booking", "records", "rag", "search"}:
            return "search"
        return intent

    # ---- Wire graph ----
    graph.add_node("start", start)
    graph.add_node("plan", plan)
    graph.add_node("classify", classify)
    graph.add_node("booking", do_booking)
    graph.add_node("records", do_records)
    graph.add_node("search", do_search)
    graph.add_node("rag", do_rag)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("start")
    graph.add_edge("start", "plan")
    graph.add_edge("plan", "classify")
    graph.add_conditional_edges(
        "classify",
        route,
        {"booking": "booking", "records": "records", "rag": "rag", "search": "search"},
    )
    graph.add_edge("booking", "finalize")
    graph.add_edge("records", "finalize")
    graph.add_edge("search", "finalize")
    graph.add_edge("rag", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()

