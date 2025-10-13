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

SYSTEM_SAFETY = (
    "You are a cautious healthcare admin/info assistant. "
    "You NEVER provide medical advice, diagnosis, or treatment instructions. "
    "You summarize reputable sources and help with logistics (records, appointments)."
)

class AgentState(TypedDict):
    messages: List
    intent: Optional[str]
    result: Optional[str]

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

def build_graph(model_name: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model_name, temperature=0.1)
    # Tools
    tools: List[Tool] = []
    search_tool = get_medical_search_tool()
    tools.append(search_tool)
    tools.extend(get_record_tools())
    tools.append(get_booking_tool())
    tools.append(get_offline_kb_tool())
    rag = RAGPipeline()
    # lightweight RAG tool inline to avoid circular import in graph
    def rag_retrieve(q: str) -> str:
        pairs = rag.retrieve(q, k=3)
        if not pairs:
            return "No local KB results."
        return "\n".join([f"- {t[:400]}" for t, _ in pairs])
    tools.append(Tool(name="RAG Retrieve", func=rag_retrieve,
                      description="Retrieve high-level snippets from local medical KB (no internet)."))

    # Nodes
    def start(state: AgentState) -> AgentState:
        return state

    def classify(state: AgentState) -> AgentState:
        last = state["messages"][-1]
        user_text = last.content if hasattr(last, "content") else str(last)
        state["intent"] = _classify_intent(user_text)
        return state

    def route(state: AgentState) -> str:
        intent = state.get("intent") or "search"
        if intent == "booking":
            return "booking"
        if intent == "records":
            return "records"
        if intent == "rag":
            return "rag"
        return "search"

    def do_booking(state: AgentState) -> AgentState:
        # Ask model to produce the JSON payload for booking tool
        msgs = [
            SystemMessage(content=SYSTEM_SAFETY + " Produce a JSON payload with keys patient_id, doctor_name, appointment_date."),
            *state["messages"]
        ]
        j = llm.invoke(msgs).content
        result = tools[-3].run(j)  # Book Appointment tool position kept stable
        state["result"] = result
        return state

    def do_records(state: AgentState) -> AgentState:
        # Try retrieval first if user asks to see history; else update
        user_text = state["messages"][-1].content.lower()
        if any(w in user_text for w in ["get", "show", "retrieve"]):
            # expects patient_id string
            result = tools[-2].run(user_text.split()[-1])  # naive: last token as patient_id
        else:
            # ask model to craft JSON payload {patient_id, data:{...}}
            msgs = [
                SystemMessage(content=SYSTEM_SAFETY + " Create JSON with keys patient_id and data (object to merge)."),
                *state["messages"]
            ]
            payload = llm.invoke(msgs).content
            result = tools[-3].run(payload)  # Manage Medical Records
        state["result"] = result
        return state

    def do_search(state: AgentState) -> AgentState:
        # First try offline KB to ensure something helpful even if web is blocked
        user_text = state["messages"][-1].content
        offline = tools[-4].run(user_text)  # Offline Medical KB
        if not offline.startswith("No offline"):
            state["result"] = offline
            return state
        # Fallback to web search
        result = tools[0].run(user_text)
        state["result"] = result
        return state

    def do_rag(state: AgentState) -> AgentState:
        user_text = state["messages"][-1].content
        result = tools[-1].run(user_text)  # RAG Retrieve
        state["result"] = result
        return state

    def finalize(state: AgentState) -> AgentState:
        # Wrap the tool result into a safe, tidy answer
        msgs = [
            SystemMessage(content=SYSTEM_SAFETY + " Rewrite the following tool output into 3–6 concise bullets. Keep sources if present."),
            HumanMessage(content=state.get("result") or "No result.")
        ]
        answer = llm.invoke(msgs).content
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
