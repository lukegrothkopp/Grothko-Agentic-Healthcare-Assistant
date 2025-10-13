import os
import json
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
import pandas as pd

# ---------------------------------------------------------------------
# Load env early & promote Streamlit secrets to os.environ
# ---------------------------------------------------------------------
load_dotenv()  # local dev via .env

# Promote keys/paths so downstream libs using os.getenv(...) can see them
if hasattr(st, "secrets"):
    for key in ("OPENAI_API_KEY", "SERPAPI_API_KEY", "BING_API_KEY", "DB_PATH"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])

# ---------------------------------------------------------------------
# Safe to import modules that may read env at import time
# ---------------------------------------------------------------------
from core.db import (
    init_db, seed_demo,
    list_patients, list_doctors, list_appointments,
    add_patient,
)
from core.logging import RunLogger
from core.memory import MemoryStore
from core.eval import eval_summary
from agents.planner import Planner
from agents.booking import BookingAgent
from agents.history import HistoryAgent
from agents.info_search import InfoSearchAgent
from prompts import SUMMARY_PROMPT

# ---------------------------------------------------------------------
# Streamlit page config (do this early)
# ---------------------------------------------------------------------
st.set_page_config(page_title="Agentic Healthcare Assistant", layout="wide")
st.title("🩺 Agentic Healthcare Assistant")
st.caption(f"DB path: {os.getenv('DB_PATH', 'data/healthcare.db')}")

# ---------------------------------------------------------------------
# Ensure DB exists before any queries (also resilient to server restarts)
# ---------------------------------------------------------------------
@st.cache_resource
def _ensure_db_once() -> bool:
    init_db()
    try:
        # Seed is idempotent if seed_demo uses a meta flag; safe in demos
        seed_demo()
    except Exception:
        pass
    return True

_ = _ensure_db_once()

# ---------------------------------------------------------------------
# Sidebar: Admin
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Admin / Setup")
    if st.button("Reinitialize Database (demo)"):
        # Explicit re-init/seed for demo purposes
        init_db()
        try:
            seed_demo()
            st.success("Database initialized with demo data.")
        except Exception as e:
            st.warning(f"Init ok; seeding skipped: {e}")
    st.divider()
    st.subheader("Environment")
    st.caption("Set keys in Streamlit **Secrets** or a local `.env`:\n"
               "- OPENAI_API_KEY\n- SERPAPI_API_KEY (or BING_API_KEY)\n- DB_PATH (optional)")

# ---------------------------------------------------------------------
# Singletons in session (logger, memory, agents)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Load initial data AFTER DB is ensured
# ---------------------------------------------------------------------
try:
    patients = list_patients()
    doctors = list_doctors()
    appointments = list_appointments()
except Exception as e:
    st.error("Database not ready. Click **Reinitialize Database (demo)** in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Patient & Doctor Views", "Agent Playground", "Medical Info Search", "Memory & Logs", "Evaluation"]
)

# =========================
# Tab 1: Patient & Doctor Views
# =========================
with tab1:
    st.subheader("Patients")
    st.dataframe(pd.DataFrame(patients))

    with st.expander("Add a patient"):
        col1, col2, col3, col4 = st.columns(4)
        name = col1.text_input("Name")
        age = col2.number_input("Age", 0, 120, 40)
        sex = col3.selectbox("Sex", ["M", "F", "Other"])
        contact = col4.text_input("Contact (email/phone)")
        if st.button("Save Patient"):
            if name:
                try:
                    pid = add_patient(name, int(age), sex, contact)
                    st.success(f"Added patient #{pid}")
                except Exception as e:
                    st.error(f"Failed to add patient: {e}")
            else:
                st.error("Name required")

    st.divider()
    st.subheader("Doctors")
    st.dataframe(pd.DataFrame(doctors))

    st.divider()
    st.subheader("Appointments")
    st.dataframe(pd.DataFrame(appointments))

# =========================
# Tab 2: Agent Playground
# =========================
with tab2:
    st.subheader("Plan & Execute")
    st.caption(
        "Try multi-step queries like: "
        "“My 70-year-old father has chronic kidney disease. Book a nephrologist and summarize latest treatment methods.”"
    )

    user_query = st.text_area(
        "User input",
        height=100,
        value=(
            "My 70-year-old father has chronic kidney disease. "
            "I want to book a nephrologist for him next Tuesday at 10am. "
            "Also, can you summarize latest treatment methods?"
        ),
    )

    # Build selection safely even if empty
    patient_opts = [(p.get("id"), p.get("name", f"Patient {p.get('id')}")) for p in patients]
    selected_patient = st.selectbox(
        "Which patient is this about?",
        options=patient_opts if patient_opts else [(-1, "No patients found")],
        format_func=lambda x: f"#{x[0]} {x[1]}",
        index=0,
    )

    specialty = st.text_input("Desired specialty (e.g., Nephrology)", value="Nephrology")
    preferred_time = st.text_input("Preferred time (ISO, e.g., 2025-10-15T10:00)", value="")

    if st.button("Plan & Run"):
        logger.log("input", f"User: {user_query}")

        # Plan steps
        try:
            plan = planner.plan(user_query)
            st.json([s.model_dump() for s in plan])
            logger.log("plan", f"Steps: {[s.action for s in plan]}")
        except Exception as e:
            st.error(f"Planning failed: {e}")
            st.stop()

        pid = selected_patient[0] if selected_patient and selected_patient[0] != -1 else None
        summary_chunks = []
        search_docs = []

        for step in plan:
            try:
                if step.action == "identify_patient":
                    logger.log("identify_patient", f"Selected patient_id={pid}")
                elif step.action == "retrieve_history":
                    hist = history_agent.retrieve(pid) if pid else []
                    logger.log("retrieve_history", f"Retrieved {len(hist)} entries", meta={"count": len(hist)})
                    if hist:
                        text = "\n".join([str(h.get("note", "")) for h in hist])
                        memory.add([text], [{"type": "history", "patient_id": pid}])
                        summary_chunks.append(text)
                elif step.action == "book_appointment":
                    res = booking_agent.book(pid, specialty, preferred_time or None)
                    ok = bool(res.get("success"))
                    logger.log("book_appointment", json.dumps(res), success=ok, meta=res)
                elif step.action == "medical_info_search":
                    docs = info_agent.search(user_query, k=5)
                    logger.log("medical_info_search", f"{len(docs)} docs")
                    search_docs = docs
                    combined = "\n\n".join([f"{d['title']}\n{d.get('snippet','')}" for d in docs])
                    memory.add([combined], [{"type": "medical_info", "query": user_query}])
                    summary_chunks.append(combined)
                elif step.action == "summarize":
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
                                    {"role": "user", "content": prompt},
                                ],
                                temperature=0.2,
                            )
                            summary = chat.choices[0].message.content
                        else:
                            summary = (
                                "Summary (demo):\n"
                                "- Patient Context: CKD with ACE inhibitor. Monitor eGFR.\n"
                                "- Latest Options: RAAS control, SGLT2 inhibitors, lifestyle; consult nephrology."
                            )
                    except Exception:
                        summary = f"(LLM unavailable) Heuristic summary:\n{full_context[:1000]}"
                    logger.log("summarize", "Generated summary.")
                    st.markdown("#### Assistant Summary")
                    st.write(summary)
                    st.session_state["last_summary"] = summary
                    if search_docs:
                        st.markdown("#### Sources")
                        for d in search_docs:
                            title = d.get("title") or d.get("url")
                            url = d.get("url", "")
                            snippet = d.get("snippet", "")
                            st.write(f"- [{title}]({url}) — {snippet}")
            except Exception as e:
                logger.log("error", f"{step.action} failed: {e}", success=False, meta={"error": str(e)})
                st.warning(f"Step '{step.action}' encountered an issue: {e}")

# =========================
# Tab 3: Medical Info Search
# =========================
with tab3:
    st.subheader("Trusted Medical Info Search")
    query = st.text_input("Query", value="latest CKD treatment options 2025")
    if st.button("Search"):
        try:
            docs = info_agent.search(query, k=5)
            st.write("Results:")
            for d in docs:
                title = d.get("title") or d.get("url")
                url = d.get("url", "")
                snippet = d.get("snippet", "")
                st.write(f"- [{title}]({url}) — {snippet}")
        except Exception as e:
            st.error(f"Search failed: {e}")

# =========================
# Tab 4: Memory & Logs
# =========================
with tab4:
    st.subheader("Vector Memory Lookup")
    q = st.text_input("Search memory", value="kidney disease")
    if st.button("Search memory"):
        try:
            hits = memory.search(q, k=5)
            for h in hits:
                doc = h.get("document", "")
                st.code(doc[:500] + ("..." if len(doc) > 500 else ""))
                st.json(h.get("metadata", {}))
        except Exception as e:
            st.error(f"Memory search failed: {e}")

    st.divider()
    st.subheader("Agent Run Logs")
    try:
        st.dataframe(pd.DataFrame(logger.to_dicts()))
    except Exception as e:
        st.error(f"Unable to render logs: {e}")

# =========================
# Tab 5: Evaluation
# =========================
with tab5:
    st.subheader("Heuristic Evaluation")
    last = st.session_state.get("last_summary", "")
    st.text_area("Generated summary (read-only)", last, height=160, disabled=True)
    expected = st.text_input("Expected keywords (comma-separated)", value="CKD,eGFR,ACE,SGLT2,appointment,nephrologist")
    if st.button("Evaluate"):
        try:
            hints = {"ckd": [k.strip() for k in expected.split(",") if k.strip()]}
            scores = eval_summary(last, hints)
            st.json(scores)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

st.caption("Demo app for educational purposes only. Not medical advice.")

