# pages/1_Patient_Assistant.py
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from agents.graph_agent import build_graph
from utils.patient_memory import PatientMemory

load_dotenv()

st.set_page_config(page_title="Patient Assistant", layout="wide")
st.title("🩺 Patient Assistant")
st.caption("Ask for help with scheduling, records, and general info from trusted sources. (No medical advice.)")

# Singleton-ish
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "pmemory" not in st.session_state:
    st.session_state.pmemory = PatientMemory()

graph = st.session_state.graph
mem = st.session_state.pmemory

# Query box
prompt = st.text_input("How can I help you today?", placeholder="e.g., I need to get an appointment for my 50 year old mother's knee issue")

col1, col2 = st.columns([1,1])
with col1:
    run_btn = st.button("Submit")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.session_state.pop("last_response", None)
    st.rerun()

if run_btn and prompt.strip():
    # Resolve patient id from free text (will pick patient_702 for 50-yo mother with knee issue)
    pid = mem.resolve_from_text(prompt) or "session"

    # Log user message (legacy call now supported)
    mem.add_message(pid, "user", prompt)

    # Run the agent graph
    try:
        state_in = {"messages": [HumanMessage(content=prompt)], "patient_id": pid}
        state_out = graph.invoke(state_in)

        # Prefer the last AI message; fall back to 'result'
        assistant_text = ""
        msgs = state_out.get("messages", [])
        for m in reversed(msgs):
            if isinstance(m, AIMessage):
                assistant_text = m.content
                break
        if not assistant_text:
            assistant_text = state_out.get("result", "") or "(no result returned)"

        # Log assistant message
        mem.add_message(pid, "assistant", assistant_text)

        # Show result
        st.session_state.last_response = assistant_text
        st.success(f"Patient: {pid}")
        st.write(assistant_text)

    except Exception as e:
        st.error(f"Run failed: {e}")

# Show the last answer (if any) on reload
if st.session_state.get("last_response"):
    st.write(st.session_state.last_response)
