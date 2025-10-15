import os, json, re, datetime
from datetime import date, timedelta
from operator import add as list_concat
from typing import TypedDict, List, Optional
from typing_extensions import Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, AnyMessage
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tools.search_tool import get_medical_search_tool
from tools.record_tool import get_record_tools
from tools.booking_tool import get_booking_tool
from tools.offline_kb_tool import get_offline_kb_tool
from utils.rag_pipeline import RAGPipeline
from utils.patient_memory import PatientMemory


# =========================
# State (with safe reducers)
# =========================

class AgentState(TypedDict):
    # Multiple nodes can write in the same step; messages are appended safely.
    messages: Annotated[List[AnyMessage], add_messages]
    intent: Optional[str]
    result: Optional[str]
    patient_id: Optional[str]
    # Optional planning field (concatenates if written by >1 node)
    plan: Annotated[List[str], list_concat]


# =========================
# Safety & planning helpers
# =========================

SYSTEM_SAFETY = (
    "You are a cautious healthcare admin/info assistant.\n"
    "- NEVER provide medical advice, diagnosis, or treatment instructions.\n"
    "- You summarize reputable sources and help with logistics (records, appointments).\n"
    "- Be concise and clear. If a request is clinical, gently defer to a licensed clinician."
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


# =========================
# Memory accessor
# =========================

_MEM_SINGLETON: Optional[PatientMemory] = None
def _mem() -> PatientMemory:
    global _MEM_SINGLETON
    if _MEM_SINGLETON is None:
        _MEM_SINGLETON = PatientMemory()
    return _MEM_SINGLETON

def memory_context_provider(patient_id: Optional[str], user_query: str, k: int = 3):
    """Return (summary, recalls, preamble_text) for system injection."""
    summary = _mem().get_summary(patient_id) if patient_id else ""
    recalls = _mem().search(patient_id, user_query, k=k) if patient_id else []
    pre = (
        f"Patient context summary:\n{summary or '(none)'}\n\n"
        + ("Relevant past entries:\n" + "\n".join(f"- {r}" for r in recalls) if recalls else "")
    ).strip()
    return summary, recalls, pre


# =========================
# Intent & parsing helpers
# =========================

def _classify_intent(text: str) -> str:
    t = (text or "").lower()
    if any(w in t for w in ["book", "schedule", "appointment"]): return "booking"
    if any(w in t for w in ["record", "history", "ehr", "note"]): return "records"
    if any(w in t for w in ["kb", "knowledge base", "rag", "offline"]): return "rag"
    if any(w in t for w in ["info", "guideline", "treatment", "what is", "symptom", "disease", "condition"]): return "search"
    return "search"

DATE_PAT = re.compile(r"(20\d{2}-\d{2}-\d{2})|((?:\d{1,2})/(?:\d{1,2})/(?:20\d{2}))", re.I)
WEEKDAYS = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}

def _next_weekday(target_idx: int, today: date) -> date:
    days_ahead = (target_idx - today.weekday() + 7) % 7
    if days_ahead == 0: days_ahead = 7
    return today + timedelta(days=days_ahead)

def _parse_relative_date_phrase(text: str, today: Optional[date] = None) -> Optional[str]:
    if not text: return None
    t = text.lower().strip()
    today = today or date.today()
    if t in {"today"}: return today.isoformat()
    if t in {"tomorrow"}: return (today + timedelta(days=1)).isoformat()
    m2 = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", t)
    if m2:
        wd = WEEKDAYS[m2.group(1)]
        days_ahead = (wd - today.weekday() + 7) % 7
        if days_ahead == 0: days_ahead = 7
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
    if m: return m.group(1)
    t = (text or "").lower()
    if "hypertension" in t: return "Hypertension Specialist (Cardiologist)"
    if "kidney" in t or "nephro" in t: return "Nephrologist"
    if any(x in t for x in ["foot","ankle"]): return "Podiatrist"
    return "Primary Care"

def _make_booking_payload(user_text: str, fallback_pid: Optional[str]) -> str:
    pid = None
    m = re.search(r"patient[_\s-]?(\d{3,})", user_text or "", re.I)
    if m:
        pid = f"patient_{m.group(1)}"
    pid = pid or fallback_pid or "patient_001"
    doc = _extract_doctor(user_text) or "Primary Care"
    d_iso = _extract_date(user_text) or (date.today() + timedelta(days=7)).isoformat()
    payload = {"patient_id": pid, "doctor_name": doc, "appointment_date": d_iso}
    return json.dumps(payload)


# =========================
# Graph
# =========================

def build_graph(model_name: str = "gpt-4o-mini"):
    key = os.getenv("OPENAI_API_KEY", "").strip()
    has_key = key.startswith("sk-")
    llm = ChatOpenAI(model=model_name, temperature=0.1, api_key=key) if has_key else None
    planner = llm.with_structured_output(PlanOut) if llm else None

    tools: List[Tool] = []
    tools.append(get_medical_search_tool())
    tools.extend(get_record_tools())
    tools.append(get_booking_tool())
    tools.append(get_offline_kb_tool())

    rag = RAGPipeline()
    def rag_retrieve(q: str) -> str:
        pairs = rag.retrieve(q, k=3)
        if not pairs: return "No local KB results."
        return "\n".join([f"- {t[:400]}" for t, _ in pairs])
    tools.append(Tool(name="RAG Retrieve", func=rag_retrieve,
                      description="Retrieve top chunks from offline KB (no LLM required)."))

    graph = StateGraph(AgentState)

    # ---- Nodes ----
    def start(state: AgentState):
        # Inject safety + memory context once as a SystemMessage
        user_text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                user_text = m.content
                break
        _, _, pre = memory_context_provider(state.get("patient_id"), user_text, k=3)
        sys_msg = SystemMessage(content=(SYSTEM_SAFETY + "\n\n" + pre).strip())
        if any(isinstance(m, SystemMessage) and SYSTEM_SAFETY.splitlines()[0] in getattr(m, "content", "") for m in state.get("messages", [])):
            return {}
        return {"messages": [sys_msg]}

    def plan(state: AgentState):
        user_text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                user_text = m.content; break
        steps: List[str] = ["search"]
        if planner and user_text:
            prompt = [
                SystemMessage(content=SAFETY_CORE),
                HumanMessage(content=PLAN_TEMPLATE + "\n\nUser: " + user_text)
            ]
            try:
                out = planner.invoke(prompt)
                if out and out.steps:
                    steps = [s for s in out.steps if s in {"search","records","booking","rag"}]
            except Exception:
                pass
        return {"plan": steps}

    def classify(state: AgentState):
        text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                text = m.content; break
        return {"intent": _classify_intent(text)}

    def do_booking(state: AgentState):
        text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                text = m.content; break
        payload = _make_booking_payload(text, state.get("patient_id"))
        res = "Booking tool unavailable."
        for t in tools:
            if t.name == "Book Appointment":
                res = t.func(payload); break
        pid = state.get("patient_id") or json.loads(payload).get("patient_id")
        if pid:
            _mem().record_event(pid, f"[Booking] {res}", meta={"stage": "booking"})
        return {"result": res, "messages": [AIMessage(content=res)]}

    def do_records(state: AgentState):
        text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                text = m.content; break
        if any(w in text.lower() for w in ["load","retrieve","show"]):
            tool_name = "Retrieve Medical History"
            input_payload = (state.get("patient_id") or "")
        else:
            tool_name = "Manage Medical Records"
            pid = state.get("patient_id") or "patient_001"
            input_payload = json.dumps({"patient_id": pid, "data": {"latest_note": text}})
        res = "Records tool unavailable."
        for t in tools:
            if t.name == tool_name:
                res = t.func(input_payload); break
        return {"result": res, "messages": [AIMessage(content=res)]}

    def do_search(state: AgentState):
        text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                text = m.content; break
        res = ""
        for t in tools:
            if t.name == "RAG Retrieve":
                res = t.func(text); break
        res = res or "No local KB results."
        for t in tools:
            if t.name == "Clinical Search (SERP/DDG)":
                try:
                    res2 = t.func(text)
                    if res2 and isinstance(res2, str):
                        res = res + "\n\n" + res2 if res else res2
                except Exception:
                    pass
                break
        if not res:
            res = "No results."
        return {"result": res, "messages": [AIMessage(content=res)]}

    def do_rag(state: AgentState):
        text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                text = m.content; break
        res = "No local KB results."
        for t in tools:
            if t.name == "RAG Retrieve":
                res = t.func(text); break
        return {"result": res, "messages": [AIMessage(content=res)]}

    def finalize(state: AgentState):
        return {}

    def route(state: AgentState) -> str:
        intent = (state.get("intent") or "search").lower()
        if intent not in {"booking","records","rag","search"}:
            return "search"
        return intent

    # ---- Wire the graph (sequential; no unintended parallel branches) ----
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
    graph.add_conditional_edges("classify", route, {
        "booking": "booking",
        "records": "records",
        "rag": "rag",
        "search": "search",
    })
    graph.add_edge("booking", "finalize")
    graph.add_edge("records", "finalize")
    graph.add_edge("search", "finalize")
    graph.add_edge("rag", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
