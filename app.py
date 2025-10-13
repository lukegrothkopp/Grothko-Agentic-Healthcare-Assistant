import os
import json
from datetime import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.db import init_db, seed_demo, list_patients, list_doctors, list_appointments, add_patient
from core.logging import RunLogger
from core.memory import MemoryStore
from core.eval import eval_summary
from agents.planner import Planner
from agents.booking import BookingAgent
from agents.history import HistoryAgent
from agents.info_search import InfoSearchAgent
from prompts import SUMMARY_PROMPT

# ---------- Setup ----------
load_dotenv(override=True)
st.set_page_config(page_title="Agentic Healthcare Assistant", layout="wide")
st.title("🩺 Agentic Healthcare Assistant")

with st.sidebar:
    st.header("Admin / Setup")
    if st.button("Initialize Database"):
        init_db()
        seed_demo()
        st.success("Database initialized with demo data.")
    st.divider()
    st.subheader("Environment")
    st.caption("Set your keys in `.env` (OPENAI_API_KEY, optional BING_API_KEY).")

if "logger" not in st.session_state:
    st.session_state.logger = RunLogger()
if "memory" not in st.session_state:
    st.session_state.memory = MemoryStore()
if "planner" not in st.session_state:
    st.session_state.planner = Planner()
if "history_agent" not in st.session_state:
    st.session_state.history_agent = HistoryAgent()
if "booking_agent" not in st.session_state:
    st.session_state.booking_agent = BookingAgent()
if "info_agent" not in st.session_state:
    st.session_state.info_agent = InfoSearchAgent(st.session_state.memory)

logger: RunLogger = st.session_state.logger
memory: MemoryStore = st.session_state.memory
planner: Planner = st.session_state.planner
history_agent: HistoryAgent = st.session_state.history_agent
booking_agent: BookingAgent = st.session_state.booking_agent
info_agent: InfoSearchAgent = st.session_state.info_agent

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Patient & Doctor Views", "Agent Playground", "Medical Info Search", "Memory & Logs", "Evaluation"]
)

# ---------- Patient & Doctor Views ----------
with tab1:
    st.subheader("Patients")
    pts = list_patients()
    st.dataframe(pd.DataFrame(pts))
    with st.expander("Add a patient"):
        col1, col2, col3, col4 = st.columns(4)
        name = col1.text_input("Name")
        age = col2.number_input("Age", 0, 120, 40)
        sex = col3.selectbox("Sex", ["M", "F", "Other"])
        contact = col4.text_input("Contact (email/phone)")
        if st.button("Save Patient"):
            if name:
                pid = add_patient(name, int(age), sex, contact)
                st.success(f"Added patient #{pid}")
            else:
                st.error("Name required")
    st.divider()

    st.subheader("Doctors")
    docs = list_doctors()
    st.dataframe(pd.DataFrame(docs))

    st.divider()
    st.subheader("Appointments")
    appts = list_appointments()
    st.dataframe(pd.DataFrame(appts))

# ---------- Agent Playground (end-to-end) ----------
with tab2:
    st.subheader("Plan & Execute")
    st.caption("Try multi-step queries like: “My 70-year-old father has chronic kidney disease. Book a nephrologist and summarize latest treatment methods.”")
    user_query = st.text_area("User input", height=100,
        value="My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him next Tuesday at 10am. Also, can you summarize latest treatment methods?")
    selected_patient = st.selectbox("Which patient is this about?", options=[(p['id'], p['name']) for p in list_patients()], format_func=lambda x: f"#{x[0]} {x[1]}", index=0 if list_patients() else None)
    specialty = st.text_input("Desired specialty (e.g., Nephrology)", value="Nephrology")
    preferred_time = st.text_input("Preferred time (ISO, e.g., 2025-10-15T10:00)", value="")

    if st.button("Plan & Run"):
        logger.log("input", f"User: {user_query}")
        plan = planner.plan(user_query)
        st.json([s.model_dump() for s in plan])
        logger.log("plan", f"Steps: {[s.action for s in plan]}")
        pid = selected_patient[0] if selected_patient else 1

        summary_chunks = []
        search_docs = []

        for step in plan:
            if step.action == "identify_patient":
                logger.log("identify_patient", f"Selected patient_id={pid}")
            elif step.action == "retrieve_history":
                hist = history_agent.retrieve(pid)
                logger.log("retrieve_history", f"Retrieved {len(hist)} entries", meta={"count": len(hist)})
                # store a concise summary into memory
                if hist:
                    text = "\n".join([h["note"] for h in hist])
                    memory.add([text], [{"type": "history", "patient_id": pid}])
                    summary_chunks.append(text)
            elif step.action == "book_appointment":
                res = booking_agent.book(pid, specialty, preferred_time or None)
                logger.log("book_appointment", json.dumps(res), success=res.get("success", False), meta=res)
            elif step.action == "medical_info_search":
                docs = info_agent.search(user_query, k=5)
                logger.log("medical_info_search", f"{len(docs)} docs")
                search_docs = docs
                # add a brief combined text for summarization
                combined = "\n\n".join([f"{d['title']}\n{d['snippet']}" for d in docs])
                memory.add([combined], [{"type": "medical_info", "query": user_query}])
                summary_chunks.append(combined)
            elif step.action == "summarize":
                # Simple concatenation + (optionally) send to LLM if key is present
                full_context = "\n\n".join(summary_chunks) if summary_chunks else "No prior history or info found."
                try:
                    from openai import OpenAI
                    api = os.getenv("OPENAI_API_KEY")
                    if api:
                        client = OpenAI(api_key=api)
                        prompt = f"{SUMMARY_PROMPT}\n\nContext:\n{full_context}"
                        chat = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are a helpful healthcare assistant (no medical advice)."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.2,
                        )
                        summary = chat.choices[0].message.content
                    else:
                        summary = "Summary (demo):\n- Patient Context: CKD with ACE inhibitor. Monitor eGFR.\n- Latest Options: RAAS control, SGLT2 inhibitors, lifestyle; consult nephrology."
                except Exception as e:
                    summary = f"(LLM unavailable) Heuristic summary:\n{full_context[:800]}"
                logger.log("summarize", "Generated summary.")
                st.markdown("#### Assistant Summary")
                st.write(summary)
                st.session_state["last_summary"] = summary
                if search_docs:
                    st.markdown("#### Sources")
                    for d in search_docs:
                        st.write(f"- [{d['title']}]({d['url']}) — {d['snippet']}")

# ---------- Medical Info Search ----------
with tab3:
    st.subheader("Trusted Medical Info Search")
    query = st.text_input("Query", value="latest CKD treatment options 2025")
    if st.button("Search"):
        docs = info_agent.search(query, k=5)
        st.write("Results:")
        for d in docs:
            st.write(f"- [{d['title']}]({d['url']}) — {d['snippet']}")

# ---------- Memory & Logs ----------
with tab4:
    st.subheader("Vector Memory Lookup")
    q = st.text_input("Search memory", value="kidney disease")
    if st.button("Search memory"):
        hits = memory.search(q, k=5)
        for h in hits:
            st.code(h["document"][:500] + ("..." if len(h["document"])>500 else ""))
            st.json(h["metadata"])
    st.divider()
    st.subheader("Agent Run Logs")
    st.dataframe(pd.DataFrame(logger.to_dicts()))

# ---------- Evaluation ----------
with tab5:
    st.subheader("Heuristic Evaluation")
    last = st.session_state.get("last_summary", "")
    st.text_area("Generated summary (read-only)", last, height=160, disabled=True)
    expected = st.text_input("Expected keywords (comma-separated)", value="CKD,eGFR,ACE,SGLT2,appointment,nephrologist")
    if st.button("Evaluate"):
        hints = {"ckd": [k.strip() for k in expected.split(",") if k.strip()]}
        scores = eval_summary(last, hints)
        st.json(scores)

st.caption("Demo app for educational purposes only. Not medical advice.")
