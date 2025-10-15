import os, json
import streamlit as st
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from agents.graph_agent import build_graph
from utils.patient_memory import PatientMemory
...
if "patient_memory" not in st.session_state:
    st.session_state.patient_memory = PatientMemory()
mem = st.session_state.patient_memory

load_dotenv()
for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "SERPAPI_API_KEY"):
    if k in st.secrets and st.secrets[k]:
        os.environ[k] = str(st.secrets[k]).strip()

st.set_page_config(page_title="Patient Assistant", layout="wide")
st.title("👩🏽‍⚕️ Patient Assistant")
st.caption("Demo — not medical advice. Provides high-level info and admin logistics only.")

# Build the agent graph once
graph = build_graph(model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

# Sidebar: just patient context
with st.sidebar:
    st.header("Your Info")
    patient_id = st.text_input("Patient ID", "patient_001")
    st.caption("Tip: You can use natural phrases like 'book a cardiologist next Monday'.")

# Chat UI (patient only)
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                state = {
                    "messages": [HumanMessage(content=prompt)],
                    "intent": None,
                    "result": None,
                    "patient_id": patient_id,  # <-- use sidebar value
                }
                result_state = graph.invoke(state)
                answer = result_state["messages"][-1].content
            except Exception as e:
                answer = f"Sorry—there was an error: {e}"
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# --- Patient-side quick booking (main page section) ---
st.markdown("---")
st.subheader("Book an appointment")
nl_booking = st.text_input(
    "Describe the appointment",
    value="Book a hypertension follow-up next Monday",
    key="patient_booking_text",
)
if st.button("Book appointment", key="patient_book_btn"):
    with st.spinner("Booking…"):
        try:
            state = {
                "messages": [HumanMessage(content=nl_booking)],
                "intent": None,
                "result": None,
                "patient_id": patient_id,   # <-- correct variable
            }
            result_state = graph.invoke(state)
            st.success(result_state["messages"][-1].content)
        except Exception as e:
            st.error(f"Failed: {e}")
