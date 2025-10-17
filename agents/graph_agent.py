# agents/graph_agent.py
from __future__ import annotations

from typing import List, Optional, TypedDict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from utils.rag_pipeline import RAGPipeline

# --------------------------
# State
# --------------------------
class AgentState(TypedDict, total=False):
    messages: List[Any]             # LangChain messages or dicts with "content"
    intent: Optional[str]
    result: Optional[str]
    patient_id: Optional[str]
    plan: Optional[List[str]]       # optional, plan steps
    bullets: Optional[List[str]]    # optional, info bullets

# --------------------------
# Helpers
# --------------------------
def _msg_text(m: Any) -> str:
    if hasattr(m, "content"):
        return getattr(m, "content") or ""
    if isinstance(m, dict):
        return str(m.get("content", "") or "")
    return str(m or "")

def _ensure_final_ai(state: AgentState, text: str) -> AgentState:
    msgs = list(state.get("messages") or [])
    msgs.append(AIMessage(content=text))
    state["messages"] = msgs
    return state

def _compose_fallback(state: AgentState) -> str:
    if state.get("result"):
        return str(state["result"])

    if state.get("bullets"):
        bl = [f"- {b}" for b in state["bullets"] if b]
        if bl:
            return "Here’s what I found:\n" + "\n".join(bl[:8])

    if state.get("plan"):
        steps = [s for s in state["plan"] if s]
        if steps:
            return "Here’s a plan I can follow:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

    # mirror last user message if available
    user_text = ""
    for m in reversed(state.get("messages") or []):
        # HumanMessage or dict role=user
        if isinstance(m, HumanMessage) or (isinstance(m, dict) and m.get("role") == "user"):
            user_text = _msg_text(m)
            break
    base = "I’m here to help."
    if user_text:
        return f"{base} You said: “{user_text}”. I can summarize options, suggest next steps, or help book an appointment."
    return f"{base} I can summarize options, suggest next steps, or help book an appointment."

# --------------------------
# Nodes
# --------------------------
def start_node(state: AgentState) -> AgentState:
    # Optionally inject a tiny system primer; keep it short to avoid token bloat.
    pid = state.get("patient_id") or "unknown"
    primer = SystemMessage(
        content=f"You are assisting patient_id={pid}. Provide high-level guidance only (no medical advice)."
    )
    msgs = list(state.get("messages") or [])
    # Only insert once
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs.insert(0, primer)
    state["messages"] = msgs
    return state

def classify_node(state: AgentState) -> AgentState:
    text = ""
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage) or (isinstance(m, dict) and m.get("role") == "user"):
            text = _msg_text(m).lower()
            break
    intent = "info"
    if any(w in text for w in ("book", "appointment", "schedule")):
        intent = "book"
    elif any(w in text for w in ("history", "records", "what happened")):
        intent = "history"
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
            "Parse patient id, doctor/specialty, and date (supports words like ‘tomorrow’, ‘next Monday’)",
            "Create an appointment record if parsed successfully",
            "Confirm to the patient and log to their memory",
        ]
    else:
        plan = ["Answer clearly and log to memory"]
    state["plan"] = plan
    return state

def info_node(state: AgentState) -> AgentState:
    try:
        rag = RAGPipeline()
        query = ""
        for m in reversed(state.get("messages") or []):
            if isinstance(m, HumanMessage) or (isinstance(m, dict) and m.get("role") == "user"):
                query = _msg_text(m)
                break
        pairs = rag.retrieve(query, k=4) or []  # List[Tuple[text, score]] or similar
        bullets: List[str] = []
        for txt, _score in pairs:
            if not txt:
                continue
            snippet = txt.strip().replace("\n", " ").strip()
            if snippet:
                bullets.append(snippet[:320])
        state["bullets"] = bullets
        if bullets:
            state["result"] = "\n".join(f"- {b}" for b in bullets[:5])
    except Exception:
        pass
    return state

def book_node(state: AgentState) -> AgentState:
    # Booking is usually handled by your Streamlit UI + booking tool.
    # If you implement booking here, set state["result"] with the confirmation text.
    return state

def finalize_node(state: AgentState) -> AgentState:
    # If the last message is already assistant text, keep it; otherwise synthesize a safe reply.
    msgs = state.get("messages") or []
    if msgs:
        last = msgs[-1]
        if isinstance(last, AIMessage) or (isinstance(last, dict) and last.get("role") == "assistant"):
            content = _msg_text(last)
            if content and content.strip():
                return state
    reply = _compose_fallback(state)
    return _ensure_final_ai(state, reply)

# --------------------------
# Graph
# --------------------------
def build_graph(model_name: str = "gpt-4o-mini"):
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
