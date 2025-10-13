
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from tools.search_tool import get_medical_search_tool
from tools.record_tool import get_record_tools
from tools.booking_tool import get_booking_tool
from tools.offline_kb_tool import get_offline_kb_tool
from utils.rag_pipeline import RAGPipeline
import json, re, datetime
from datetime import date, timedelta

SYSTEM_SAFETY = (
    "You are a cautious healthcare admin/info assistant. "
    "You NEVER provide medical advice, diagnosis, or treatment instructions. "
    "You summarize reputable sources and help with logistics (records, appointments)."
)

class AgentState(TypedDict):
    messages: List
    intent: Optional[str]
    result: Optional[str]
    patient_id: Optional[str]

def _classify_intent(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["book", "schedule", "appointment"]):
        return "booking"
    if any(w in t for w in ["record", "history", "ehr"]):
        return "records"
    if any(w in t for w in ["kb", "knowledge base", "rag"]):
        return "rag"
    if any(w in t for w in ["info", "guideline", "treatment", "what is", "symptom", "disease", "condition"]):
        return "search"
    return "search"

DATE_PAT = re.compile(r"(20\d{2}-\d{2}-\d{2})|((?:\d{1,2})/(?:\d{1,2})/(?:20\d{2}))", re.I)
WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}

def _next_weekday(target_idx: int, today: date) -> date:
    days_ahead = (target_idx - today.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)

def _parse_relative_date_phrase(text: str, today: date = None) -> str | None:
    if not text:
        return None
    text_l = text.lower().strip()
    if today is None:
        today = date.today()

    if "today" in text_l:
        return today.isoformat()
    if "tomorrow" in text_l:
        return (today + timedelta(days=1)).isoformat()

    m = re.search(r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", text_l)
    if m:
        wd = WEEKDAYS[m.group(1)]
        d = _next_weekday(wd, today)
        return d.isoformat()

    m2 = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", text_l)
    if m2:
        wd = WEEKDAYS[m2.group(1)]
        days_ahead = (wd - today.weekday() + 7) % 7
        if days_ahead == 0:
            days_ahead = 7
        d = today + timedelta(days=days_ahead)
        return d.isoformat()

    return None

def _extract_date(text: str) -> str | None:
    m = DATE_PAT.search(text or "")
    if m:
        d = m.group(0)
        if "/" in d:
            mm, dd, yyyy = d.split("/")
            return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"
        return d
    return _parse_relative_date_phrase(text)

def _extract_doctor(text: str) -> str | None:
    m = re.search(r"Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text or "")
    if m:
        return m.group(1)
    if "hypertension" in (text or "").lower():
        return "Hypertension Specialist (Cardiologist)"
    if "kidney" in (text or "").lower() or "nephro" in (text or "").lower():
        return "Nephrologist"
    return None

def _make_booking_payload(user_text: str, fallback_pid: Optional[str]) -> str:
    pid = None
    m = re.search(r"patient[_\s-]?(\d{3,})", user_text or "", re.I)
    if m:
        pid = f"patient_{m.group(1)}"
    pid = pid or fallback_pid
    date_iso = _extract_date(user_text) or (date.today() + timedelta(days=7)).isoformat()
    doc = _extract_doctor(user_text) or "Primary Care"
    payload = {"patient_id": pid, "doctor_name": doc, "appointment_date": date_iso}
    return json.dumps(payload)

import os
from langchain_openai import ChatOpenAI

def build_graph(model_name: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model_name, temperature=0.1, api_key=key) if has_key else none
    tools: List[Tool] = []
    search_tool = get_medical_search_tool()
    tools.append(search_tool)
    rec_tools = get_record_tools()
    tools.extend(rec_tools)
    booking_tool = get_booking_tool()
    tools.append(booking_tool)
    offline_kb = get_offline_kb_tool()
    tools.append(offline_kb)
    rag = RAGPipeline()
    def rag_retrieve(q: str) -> str:
        pairs = rag.retrieve(q, k=3)
        if not pairs:
            return "No local KB results."
        return "\n".join([f"- {t[:400]}" for t, _ in pairs])
    tools.append(Tool(name="RAG Retrieve", func=rag_retrieve,
                      description="Retrieve high-level snippets from local medical KB (no internet)."))

    def start(state: AgentState) -> AgentState:
        return state

    def classify(state: AgentState) -> AgentState:
        last = state["messages"][-1]
        user_text = last.content if hasattr(last, "content") else str(last)
        state["intent"] = _classify_intent(user_text)
        return state

    def route(state: AgentState) -> str:
        return state.get("intent") or "search"

    def do_booking(state: AgentState) -> AgentState:
        user_text = state["messages"][-1].content
        payload = _make_booking_payload(user_text, state.get("patient_id"))
        result = booking_tool.run(payload)
        state["result"] = result
        return state

    def do_records(state: AgentState) -> AgentState:
        user_text = state["messages"][-1].content.lower()
        manage_tool, retrieve_tool = rec_tools[0], rec_tools[1]
        if any(w in user_text for w in ["get", "show", "retrieve"]):
            pid = state.get("patient_id") or "patient_001"
            result = retrieve_tool.run(pid)
        else:
            pid = state.get("patient_id") or "patient_001"
            payload = json.dumps({"patient_id": pid, "data": {"notes": "Updated by agent."}})
            result = manage_tool.run(payload)
        state["result"] = result
        return state

    def do_search(state: AgentState) -> AgentState:
        user_text = state["messages"][-1].content
        offline = offline_kb.run(user_text)
        if not offline.startswith("No offline KB"):
            state["result"] = offline
            return state
        result = search_tool.run(user_text)
        state["result"] = result
        return state

    def do_rag(state: AgentState) -> AgentState:
        user_text = state["messages"][-1].content
        result = tools[-1].run(user_text)
        state["result"] = result
        return state

    def finalize(state: AgentState) -> AgentState:
        msgs = [
            SystemMessage(content=SYSTEM_SAFETY + " Rewrite the following tool output into 3–6 concise bullets. Keep sources if present."),
            state["messages"][-1],
            HumanMessage(content=state.get("result") or "No result."),
        ]
        try:
            if llm is None:
                raise RuntimeError("No valid OPENAI_API_KEY")
            answer = llm.invoke(msgs).content
        except Exception:
            # Graceful fallback: just bulletize the tool output
            raw = state.get("result") or "No result."
            lines = [ln.strip() for ln in (raw.splitlines() or [raw]) if ln.strip()]
            bullets = [('- ' + ln) if not ln.startswith('-') else ln for ln in lines[:6]]
            answer = "\n".join(bullets) if bullets else "- No result."
    
        state["messages"].append(AIMessage(content=answer))
        return state

    graph = StateGraph(AgentState)
    graph.add_node("start", start)
    graph.add_node("classify", classify)
    graph.add_node("booking", do_booking)
    graph.add_node("records", do_records)
    graph.add_node("search", do_search)
    graph.add_node("rag", do_rag)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("start")
    graph.add_edge("start", "classify")
    graph.add_conditional_edges("classify", route, {
        "booking": "booking",
        "records": "records",
        "rag": "rag",
        "search": "search"
    })
    graph.add_edge("booking", "finalize")
    graph.add_edge("records", "finalize")
    graph.add_edge("search", "finalize")
    graph.add_edge("rag", "finalize")
    graph.add_edge("finalize", END)

    app = graph.compile()
    return app

