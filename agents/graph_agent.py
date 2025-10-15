import os, json, re
from datetime import date, timedelta
from operator import add as list_concat
from typing import TypedDict, List, Optional, Tuple, Any
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
# Array-safe helpers
# =========================

def _to_list(obj: Any) -> List[Any]:
    """Normalize arrays/iterables into a plain Python list."""
    if obj is None:
        return []
    try:
        import numpy as np  # optional
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, (list, tuple)):
        return list(obj)
    # Fallback: wrap singletons
    return [obj]

def _nonempty_str(s: Any) -> bool:
    """True iff s is a non-empty string after stripping."""
    return isinstance(s, str) and len(s.strip()) > 0


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
        """
        Always return a STRING. Safely normalizes any array-like results to avoid
        'truth value of an array is ambiguous' errors.
        """
        try:
            pairs = rag.retrieve(q, k=3)
        except Exception as e:
            return f"(RAG error) {e}"

        items = _to_list(pairs)
        if len(items) == 0:
            return "No local KB results."

        lines: List[str] = []
        for it in items:
            # Expect (text, score) or just text
            text: str
            if isinstance(it, (list, tuple)) and len(it) >= 1:
                text = str(it[0])
            else:
                text = str(it)
            lines.append(f"- {text[:400]}")
        out = "\n".join(lines).strip()
        return out if _nonempty_str(out) else "No local KB results."

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
        sys_payload = (SYSTEM_SAFETY + "\n\n" + pre).strip()

        already = any(
            isinstance(m, SystemMessage)
            and isinstance(getattr(m, "content", None), str)
            and SYSTEM_SAFETY.splitlines()[0] in m.content
            for m in state.get("messages", [])
        )
        if already:
            return {}
        return {"messages": [SystemMessage(content=sys_payload)]}

    def plan(state: AgentState):
        user_text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                user_text = m.content; break
        steps: List[str] = ["search"]
        if planner and _nonempty_str(user_text):
            prompt = [
                SystemMessage(content=SAFETY_CORE),
                HumanMessage(content=PLAN_TEMPLATE + "\n\nUser: " + user_text)
            ]
            try:
                out = planner.invoke(prompt)
                if out and isinstance(out.steps, list) and len(out.steps) > 0:
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
        res: Any = "Booking tool unavailable."
        for t in tools:
            if t.name == "Book Appointment":
                res = t.func(payload); break
        # Normalize to string
        res_str = res if isinstance(res, str) else json.dumps(res)
        pid = state.get("patient_id") or json.loads(payload).get("patient_id")
        if pid:
            _mem().record_event(pid, f"[Booking] {res_str}", meta={"stage": "booking"})
        return {"result": res_str, "messages": [AIMessage(content=res_str)]}

    def do_records(state: AgentState):
        text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                text = m.content; break
        if any(w in (text or "").lower() for w in ["load","retrieve","show"]):
            tool_name = "Retrieve Medical History"
            input_payload = (state.get("patient_id") or "")
        else:
            tool_name = "Manage Medical Records"
            pid = state.get("patient_id") or "patient_001"
            input_payload = json.dumps({"patient_id": pid, "data": {"latest_note": text}})
        res: Any = "Records tool unavailable."
        for t in tools:
            if t.name == tool_name:
                res = t.func(input_payload); break
        res_str = res if isinstance(res, str) else json.dumps(res)
        return {"result": res_str, "messages": [AIMessage(content=res_str)]}

    def do_search(state: AgentState):
        text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                text = m.content; break

        # First: local RAG (always returns a string)
        rag_text = rag_retrieve(text)
        out = rag_text if _nonempty_str(rag_text) else "No local KB results."

        # Second: clinical search tool (guard outputs to string)
        for t in tools:
            if t.name == "Clinical Search (SERP/DDG)":
                try:
                    res2 = t.func(text)
                    res2_str = res2 if isinstance(res2, str) else json.dumps(res2)
                    if _nonempty_str(res2_str):
                        out = (out + "\n\n" + res2_str) if _nonempty_str(out) else res2_str
                except Exception:
                    pass
                break

        if not _nonempty_str(out):
            out = "No results."
        return {"result": out, "messages": [AIMessage(content=out)]}

    def do_rag(state: AgentState):
        text = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                text = m.content; break
        out = rag_retrieve(text)
        out = out if _nonempty_str(out) else "No local KB results."
        return {"result": out, "messages": [AIMessage(content=out)]}

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
