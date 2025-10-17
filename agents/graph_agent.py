# agents/graph_agent.py
from __future__ import annotations

from utils.web_search import search_trusted
from utils.summarize import llm_bullets_with_citations, have_openai

from typing import List, Optional, TypedDict, Any, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
import re

from utils.rag_pipeline import RAGPipeline
try:
    import streamlit as st  # optional for reusing session PatientMemory
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
    urgent: Optional[bool]            # internal flag only


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


# --------- Urgency & intent detection ---------
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
_CONTACT_PATTERNS = [
    r"\bcontact (my )?(clinician|doctor|provider|gp|pcp)\b",
    r"\bmessage (my )?(clinician|doctor|provider)\b",
    r"\breach out to (the )?(clinic|doctor|provider)\b",
    r"\bcall (my )?(doctor|clinic|provider)\b",
]

def _is_very_high_fever(text: str) -> bool:
    for n in re.findall(r"(\d{2,3}(?:\.\d)?)", text):
        try:
            if float(n) >= 103:  # Fahrenheit, admin-level threshold
                return True
        except Exception:
            continue
    if re.search(r"\b(103|104|105)\b", text):
        return True
    return False

def _is_contact_request(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _CONTACT_PATTERNS)


# --------- Admin-safe reply templates ---------
def _urgent_admin_reply(user_text: str) -> str:
    lines = [
        f'You said: “{user_text.strip()}”.' if user_text.strip() else "",
        "That sounds important. I’m a logistics assistant (not a clinician), but here’s how I can help right now:",
        "• If you feel severely unwell or unsafe, **call emergency services or go to the nearest ER.**",
        "• I can book the **earliest available appointment** or help message your clinic.",
        "• I can also pull trusted, high-level information while you get care arranged.",
        "",
        "Tell me if you want me to **book urgent care** or **contact your clinician**. If you can, include a preferred time or location.",
    ]
    return "\n".join([l for l in lines if l])

def _nonurgent_admin_reply(user_text: str) -> str:
    lines = [
        f'You said: “{user_text.strip()}”.' if user_text.strip() else "",
        "Here’s how I can help:",
        "• I can connect you with your clinic/clinician.",
        "• I can book an appointment (tell me a date or say “tomorrow morning”).",
        "• I can fetch high-level, trusted info while we sort logistics.",
        "",
        "Would you like me to **contact your clinician** or **book an appointment**? Include a date/time or location if you can.",
    ]
    return "\n".join([l for l in lines if l])


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
    # default
    intent = "info"
    if any(w in text for w in ("book", "appointment", "schedule", "book urgent care")):
        intent = "book"
    elif _is_contact_request(text):
        intent = "contact"
    elif any(w in text for w in ("history", "records", "what happened")):
        intent = "history"

    # urgent flag (does not force routing; shapes reply)
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
    elif intent == "contact":
        plan = [
            "Capture a brief message for the clinician and any preference (time/location)",
            "Log a contact_request for the patient in memory so the care team can see it",
            "Offer to also book a visit if appropriate",
        ]
    else:
        plan = ["Answer clearly and log to memory"]
    state["plan"] = plan
    return state

def info_node(state: AgentState) -> AgentState:
    """
    Retrieve KB snippets; then (if needed) augment with trusted web sources and produce
    a single patient-facing message that:
    - tells them to book below,
    - shows high-level bullets with source labels,
    - adds a short safety line if urgent.
    """
    user_q = _last_user_text(state)
    urgent = bool(state.get("urgent"))

    # 1) Offline KB first (fast)
    kb_bullets: List[str] = []
    try:
        rag = RAGPipeline()
        pairs = rag.retrieve(user_q, k=4) or []  # List[(text, score)]
        for txt, _score in pairs:
            if not txt:
                continue
            snippet = txt.strip().replace("\n", " ").strip()
            if snippet:
                kb_bullets.append(snippet[:320])
    except Exception:
        pass

    # 2) If we have <3 bullets, try trusted web (Tavily/SerpAPI) + LLM summary
    web_text = ""
    if len(kb_bullets) < 3:
        try:
            docs = search_trusted(user_q, k=5)
            if docs:
                web_text, web_bullets = llm_bullets_with_citations(user_q, docs)
                # If we failed to call OpenAI, web_text will still contain simple bullets+Sources
            # else: keep kb_bullets only
        except Exception:
            pass

    # 3) Compose unified patient-first reply
    header = "You can book an appointment below in **Quick Schedule**."
    if urgent:
        header += "\nIf you feel severely unwell or unsafe, **call emergency services or go to the nearest ER.**"

    body = ""
    if web_text:
        body = "\n\n**Trusted info (high-level):**\n" + web_text.strip()
    else:
        # fall back to KB bullets (if any) or topic fallbacks (from previous code)
        if kb_bullets:
            body = "\n\n**Trusted info (high-level):**\n" + "\n".join([f"• {b}" for b in kb_bullets[:5]])
        else:
            # keep the previous topic fallback behavior (ensure you still have `_topic_fallback_bullets`)
            fb = _topic_fallback_bullets(user_q)
            if fb:
                body = "\n\n**Trusted info (high-level):**\n" + "\n".join([f"• {b}" for b in fb])
            else:
                body = "\n\nI can also pull high-level info from trusted sources; try a more specific question."

    state["bullets"] = kb_bullets  # keep for any downstream widgets
    state["result"] = (header + body).strip()
    return state

def contact_node(state: AgentState) -> AgentState:
    """Log a contact request so clinicians see it in Recent Activity, and confirm to patient."""
    pid = state.get("patient_id") or "unknown"
    user_q = _last_user_text(state)

    # Very light “message” extraction: remove the trigger phrase if present.
    cleaned = re.sub("|".join(_CONTACT_PATTERNS), "", user_q, flags=re.IGNORECASE).strip()
    note = cleaned if cleaned else user_q

    try:
        pm = _safe_pm()
        pm.record_event(
            pid,
            f"Patient requested clinician contact: {note}",
            meta={"kind": "contact_request", "source": "patient_assistant"},
        )
    except Exception:
        pass

    # Patient-facing confirmation
    state["result"] = (
        "Okay — I’ll flag your clinician team with this message:\n"
        f"“{note}”\n\n"
        "If you’d like, I can also **book an appointment**. "
        "Tell me the date/time window (e.g., “tomorrow morning” or “next Monday afternoon”) and preferred clinic."
    )
    return state

def book_node(state: AgentState) -> AgentState:
    # Booking remains UI/tool-driven; keep node for symmetry/future use.
    if state.get("urgent"):
        user_q = _last_user_text(state)
        state["result"] = _urgent_admin_reply(user_q)
    else:
        # Non-urgent booking guidance if user typed "book" without details
        state["result"] = (
            "Sure — I can help book a visit. Please include:\n"
            "• Doctor or specialty (e.g., Primary Care, Orthopedics)\n"
            "• Date or phrase (e.g., “next Tuesday morning”)\n"
            "• Optional: clinic or telehealth preference\n"
            "Or use the **Quick Schedule** tab below."
        )
    return state

def finalize_node(state: AgentState) -> AgentState:
    # Prefer node result if any
    if state.get("result"):
        return _ensure_final_ai(state, str(state["result"]))

    # Next: trusted bullets
    bl = [f"- {b}" for b in (state.get("bullets") or []) if b]
    if bl:
        return _ensure_final_ai(state, "Here’s what I found:\n" + "\n".join(bl[:8]))

    # Last resort: non-urgent reply (don’t show internal plan)
    user_q = _last_user_text(state)
    return _ensure_final_ai(state, _nonurgent_admin_reply(user_q))


# --------- Graph wiring ---------
def build_graph(model_name: str = "gpt-4o-mini"):
    """
    start → classify → plan → (info | book | contact) → finalize
    """
    graph = StateGraph(AgentState)

    graph.add_node("start", start_node)
    graph.add_node("classify", classify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("info", info_node)
    graph.add_node("contact", contact_node)
    graph.add_node("book", book_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("start")
    graph.add_edge("start", "classify")
    graph.add_edge("classify", "plan")

    def _route(state: AgentState) -> str:
        intent = (state.get("intent") or "").lower()
        if intent == "book":
            return "book"
        if intent == "contact":
            return "contact"
        return "info"

    graph.add_conditional_edges("plan", _route, {"info": "info", "book": "book", "contact": "contact"})
    graph.add_edge("info", "finalize")
    graph.add_edge("contact", "finalize")
    graph.add_edge("book", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
