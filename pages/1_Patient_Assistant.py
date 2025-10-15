import os, json
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agents.graph_agent import build_graph
from utils.patient_memory import PatientMemory

load_dotenv()
# Map secrets -> env for Streamlit Cloud
for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "SERPAPI_API_KEY"):
    try:
        if k in st.secrets and st.secrets[k]:
            os.environ[k] = str(st.secrets[k]).strip()
    except Exception:
        pass

st.set_page_config(page_title="Patient Assistant", layout="wide")
st.title("🩺 Patient Assistant")
st.caption("Demo — not medical advice. Provides high-level info and admin logistics only.")

# Build agent graph once
if "graph" not in st.session_state:
    st.session_state.graph = build_graph(model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
graph = st.session_state.graph

# Patient memory (singleton)
if "patient_memory" not in st.session_state:
    st.session_state.patient_memory = PatientMemory()
mem = st.session_state.patient_memory

# Sidebar: patient info + memory summary
with st.sidebar:
    st.header("Your Info")
    patient_id = st.text_input("Patient ID", "patient_001")
    st.caption("Tip: Try phrases like “book a cardiologist next Monday”.")
    st.markdown("---")
    st.subheader("Context summary")
    st.write(mem.get_summary(patient_id) or "_No summary yet_")

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("How can I help you today?"):
    # log user turn
    mem.add_message(patient_id, "user", prompt)
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
                    "patient_id": patient_id,
                }
                result_state = graph.invoke(state)
                answer = result_state["messages"][-1].content
            except Exception as e:
                answer = f"Sorry—there was an error: {e}"
            st.markdown(answer)

    # log assistant turn + maybe summarize
    mem.add_message(patient_id, "assistant", answer)
    mem.maybe_autosummarize(patient_id)
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
st.subheader("Book an appointment (natural language)")
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
                "patient_id": patient_id,
            }
            result_state = graph.invoke(state)
            msg = result_state["messages"][-1].content
            st.success(msg)
            # record booking event & assistant reply
            mem.record_event(patient_id, f"[Booking] {msg}", meta={"source": "patient_page"})
            mem.add_message(patient_id, "assistant", msg)
            mem.maybe_autosummarize(patient_id)
        except Exception as e:
            st.error(f"Failed: {e}")
