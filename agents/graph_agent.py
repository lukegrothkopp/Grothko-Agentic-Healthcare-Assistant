# agents/graph_agent.py
from __future__ import annotations

from typing import List, Optional, TypedDict, Any, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
import re

from utils.rag_pipeline import RAGPipeline
try:
    import streamlit as st  # optional for session PatientMemory reuse
except Exception:  # pragma: no cover
    st = None
from utils.patient_memory import PatientMemory


class AgentState(TypedDict, total=False):
    messages: List[Any]
    intent: Optional[str]
    result: Optional[str]
    patient_id: Optional[str]
    plan: Optional[List[str]]
    bullets: Optional[List[str]]
    urgent: Optional[bool]            # <- new: internal flag (not shown to user)


# --------- Patient memory context ---------
def _safe_pm() -> PatientMemory:
    if st is not None:
        try:
            pm = st.session_state.get("pmemory")
            if isinstance(pm, PatientMemory):
                return pm
        except Exception:
            pass
    return PatientMemory()

def memory_context_provider(patient_id: Optional[str], user_query: str) -> Tuple[str, List[str]]:
    pid = (patient_id or "").strip() or "unknown"
    pm = _safe_pm()

    try:
        summary = (pm.get_summary(pid) or "").strip()
    except Exception:
        summary = ""

    recalls: List[str] = []
    try:
        hits = pm.search(pid, user_query or "", k=3) or []
        if not hits:
            win = pm.get_window(pid, k=3)
            for r in win:
                txt = r.get("text") or r.get("notes") or r.get("diagnosis")
                if txt:
                    recalls.append(str(txt)[:200])
        else:
            recalls.extend([str(h)[:200] for h in hits])
    except Exception:
        pass

    return summary[:600], [r[:200] for r in recalls[:3]]


# --------- Small helpers ---------
def _msg_text(m: Any) -> str:
    if hasattr(m, "content"):
        return getattr(m, "content") or ""
    if isinstance(m, dict):
        return str(m.get("content", "") or "")
    return str(m or "")

def _last_user_text(state: AgentState) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage) or (isinstance(m, dict) and m.get("role") == "user"):
            return _msg_text(m)
    return ""

def _ensure_final_ai(state: AgentState, text: str) -> AgentState:
    msgs = list(state.get("messages") or [])
    msgs.append(AIMessage(content=text))
    state["messages"] = msgs
    return state


# --------- Urgency detection + admin reply ---------
_URGENT_PHRASES = [
    r"\bchest pain\b",
    r"\btrouble breathing\b|\bshort(ness)? of breath\b",
    r"\bsevere headache\b|\bworst headache\b",
    r"\bconfusion\b|\bfaint(ing)?\b",
    r"\bweakness on one side\b|\bslurred speech\b",
    r"\bstiff neck\b",
    r"\bseizure\b",
    r"\buncontrolled bleeding\b",
]

def _is_very_high_fever(text: str) -> bool:
    # match temperatures like 103, 103.5, 104F, 40C etc. (rough, admin-level)
    for n in re.findall(r"(\d{2,3}(?:\.\d)?)", text):
        try:
            v = float(n)
            if v >= 103:  # Fahrenheit — admin-level “very high”
                return True
        except Exception:
            continue
    # also catch phrases
    if re.search(r"\b(103|104|105)\b", text):
        return True
    return False

def _urgent_admin_reply(user_text: str) -> str:
    # High-level, non-clinical guidance + logistics; safe for admin assistant.
    lines = [
        "That sounds important. I’m a logistics assistant (not a clinician), but here’s how I can help right now:",
        "• If you feel severely unwell or unsafe, **call emergency services or go to the nearest ER.**",
        "• I can book the **earliest available appointment** or help message your clinic.",
        "• I can also pull trusted, high-level information while you get care arranged.",
        "",
        "Tell me if you want me to **book urgent care** or **contact your clinician**. If you can, include a preferred time or location.",
    ]
    # Add a tiny echo for empathy without re-stating medical advice
    ut = user_text.strip()
    if ut:
        lines.insert(0, f"You said: “{ut}”.")
    return "\n".join(lines)


# --------- Nodes ---------
def start_node(state: AgentState) -> AgentState:
    pid = state.get("patient_id") or "unknown"
    user_q = _last_user_text(state)
    try:
        summary, recalls = memory_context_provider(pid, user_q)
    except Exception:
        summary, recalls = "", []

    sys_txt = (
        f"You are assisting patient_id={pid}. "
        f"Provide high-level, non-diagnostic guidance. Be concise and practical."
    )
    if summary:
        sys_txt += f"\nPatient summary: {summary}"
    if recalls:
        sys_txt += "\nRecent memory snippets:\n" + "\n".join(f"- {r}" for r in recalls)

    msgs = list(state.get("messages") or [])
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs.insert(0, SystemMessage(content=sys_txt))
    state["messages"] = msgs
    return state

def classify_node(state: AgentState) -> AgentState:
    text = _last_user_text(state).lower()
    intent = "info"
    if any(w in text for w in ("book", "appointment", "schedule")):
        intent = "book"
    elif any(w in text for w in ("history", "records", "what happened")):
        intent = "history"

    # urgent flag (does not change routing; affects response content)
    urgent = _is_very_high_fever(text) or any(re.search(p, text) for p in _URGENT_PHRASES)
    state["urgent"] = bool(urgent)
    state["intent"] = intent
    return state

def plan_node(state: AgentState) -> AgentState:
    intent = (state.get("intent") or "info").lower()
    plan: List[str] = []
    if intent == "info":
        plan = [
            "Retrieve trusted information from the medical KB",
            "Extract concise, patient-friendly bullets with sources",
            "Offer next administrative steps (e.g., booking, questions for clinician)",
        ]
    elif intent == "book":
        plan = [
            "Parse patient id, doctor/specialty, and date (supports ‘today’, ‘tomorrow’, ‘next Monday’)",
            "Create an appointment record if parsed successfully",
            "Confirm to the patient and log to their memory",
        ]
    else:
        plan = ["Answer clearly and log to memory"]
    state["plan"] = plan
    return state

def info_node(state: AgentState) -> AgentState:
    """
    Pulls concise snippets from the offline KB. If urgent flag is set OR
    no usable bullets were found, produce a patient-facing admin reply instead
    of exposing internal plan steps.
    """
    user_q = _last_user_text(state)
    urgent = bool(state.get("urgent"))

    bullets: List[str] = []
    try:
        rag = RAGPipeline()
        pairs = rag.retrieve(user_q, k=4) or []  # e.g., List[(text, score)]
        for txt, _score in pairs:
            if not txt:
                continue
            snippet = txt.strip().replace("\n", " ").strip()
            if snippet:
                bullets.append(snippet[:320])
    except Exception:
        pass

    state["bullets"] = bullets

    # If urgent or no snippets, produce admin-safe, helpful text
    if urgent or not bullets:
        state["result"] = _urgent_admin_reply(user_q)
    else:
        state["result"] = "\n".join(f"- {b}" for b in bullets[:5])
    return state

def book_node(state: AgentState) -> AgentState:
    # Booking remains UI/tool-driven; keep node for symmetry/future use.
    # If you later move booking here, set state["result"] accordingly.
    if state.get("urgent"):
        # Even in a booking flow, if flagged urgent, remind of admin steps
        user_q = _last_user_text(state)
        state["result"] = _urgent_admin_reply(user_q)
    return state

def finalize_node(state: AgentState) -> AgentState:
    """
    Always append a final assistant reply. Do NOT show internal plan steps.
    """
    # If tool or info provided a result, prefer that
    if state.get("result"):
        return _ensure_final_ai(state, str(state["result"]))

    # Second choice: trusted bullets
    bl = [f"- {b}" for b in (state.get("bullets") or []) if b]
    if bl:
        return _ensure_final_ai(state, "Here’s what I found:\n" + "\n".join(bl[:8]))

    # Last resort: helpful admin reply (not the raw plan)
    user_q = _last_user_text(state)
    return _ensure_final_ai(state, _urgent_admin_reply(user_q))


# --------- Graph wiring ---------
def build_graph(model_name: str = "gpt-4o-mini"):
    """
    start → classify → plan → (info | book) → finalize
    """
    graph = StateGraph(AgentState)

    graph.add_node("start", start_node)
    graph.add_node("classify", classify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("info", info_node)
    graph.add_node("book", book_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("start")
    graph.add_edge("start", "classify")
    graph.add_edge("classify", "plan")

    def _route(state: AgentState) -> str:
        intent = (state.get("intent") or "").lower()
        return "book" if intent == "book" else "info"

    graph.add_conditional_edges("plan", _route, {"info": "info", "book": "book"})
    graph.add_edge("info", "finalize")
    graph.add_edge("book", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
