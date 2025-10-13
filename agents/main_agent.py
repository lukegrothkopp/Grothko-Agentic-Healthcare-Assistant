from typing import List
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from tools.booking_tool import get_booking_tool
from tools.record_tool import get_record_tools
from tools.search_tool import get_medical_search_tool
from utils.rag_pipeline import RAGPipeline

SYSTEM_SAFETY = (
    "You are a cautious healthcare admin/info assistant. "
    "You NEVER provide medical advice, diagnosis, or treatment instructions. "
    "You can summarize reputable sources and handle logistics like records and appointments. "
    "Prefer WHO/CDC/NIH/MedlinePlus/Mayo/NICE. If unsure, say so."
)

def make_rag_tool(rag: RAGPipeline) -> Tool:
    def _retrieve(q: str) -> str:
        pairs = rag.retrieve(q, k=3)
        if not pairs:
            return "No local KB results."
        lines = [f"- {text[:400]} (score={dist:.3f})" for text, dist in pairs]
        return "\n".join(lines)
    return Tool(
        name="RAG Retrieve",
        func=_retrieve,
        description=(
            "Retrieve high-level medical knowledge from the local KB (no web). "
            "Input is a plain text query; returns top-matching snippets."
        ),
    )

def create_healthcare_agent(model_name: str = "gpt-4o-mini", temperature: float = 0.1) -> tuple:
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)
    tools: List[Tool] = []
    tools.append(get_medical_search_tool())
    tools.extend(get_record_tools())
    tools.append(get_booking_tool())
    rag = RAGPipeline()
    tools.append(make_rag_tool(rag))

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        system_message=SYSTEM_SAFETY,
    )
    return agent, rag
