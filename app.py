import os, json
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agents.graph_agent import build_graph

load_dotenv()

st.set_page_config(page_title="Agentic Healthcare Assistant", layout="wide")
st.title("👨‍⚕️ Agentic Healthcare Assistant — LangGraph Orchestration")
st.caption("Not medical advice. High-level info & logistics only.")

graph = build_graph(model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

# Sidebar
with st.sidebar:
    st.header("Context")
    st.info("The agent routes: booking / records / search / RAG. Offline Mini-KB ensures results even if web is blocked.")
    st.markdown("---")
    st.caption("Set OPENAI_API_KEY in .env for LLM polishing.")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about logistics or high-level info…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                state = {"messages": [HumanMessage(content=prompt)], "intent": None, "result": None}
                result_state = graph.invoke(state)
                answer = result_state["messages"][-1].content
            except Exception as e:
                answer = f"There was an error: {e}"
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.sidebar.markdown("---")
st.sidebar.header("LLMOps (placeholder)")
st.sidebar.write("Add traces, token usage, and evals here.")

