from __future__ import annotations
import os
import streamlit as st
import pandas as pd
from datetime import datetime

from core import db
from core.memory import MemoryStore
from core.eval import eval_summary
from core.logging import get_logger

from agents.planner import Planner
from agents.booking import BookingAgent
from agents.history import HistoryAgent
from agents.info_search import InfoSearchAgent

st.set_page_config(page_title="Grothko Agentic Healthcare Assistant", layout="wide")
log = get_logger("App")

# Initialize subsystems
if "memory" not in st.session_state:
    st.session_state.memory = MemoryStore()

db.init_db(seed=True)
planner = Planner()
booking = BookingAgent()
history = HistoryAgent()
info = InfoSearchAgent()

st.title("🩺 Grothko Agentic Healthcare Assistant")
st.caption("Demo — not medical advice. For admin workflows & high level info only.")

with st.sidebar:
    st.header("Quick Actions")
    seed = st.button("Seed data")
    if seed:
        db.init_db(seed=True)
        st.toast("Database seeded.")
    st.markdown("---")
    st.subheader("Memory")
    q = st.text_input("Search memory")
    if st.button("Search") and q:
        hits = st.session_state.memory.search(q)
        for h in hits:
            st.write("•", h.text)

# Tabs
main, booking_tab, hist_tab, info_tab, eval_tab = st.tabs([
    "Plan", "Book Appointment", "Patient History", "Medical Info Search", "Eval & Logs",
])

with main:
    st.subheader("Planner")
    user_q = st.text_input("What do you need?", placeholder="e.g., 'Book a nephrologist for my father and summarize latest CKD treatments.'")
    if st.button("Make plan") and user_q:
        steps = planner.plan(user_q)
        st.json(steps)
        st.session_state.memory.add(f"Plan for: {user_q}", type="plan", steps=steps)

with booking_tab:
    st.subheader("Find a doctor & book")
    patients = db.list_patients()
    patient_map = {f"{p['name']} (#{p['id']})": int(p["id"]) for p in patients}
    sel_patient = st.selectbox("Patient", list(patient_map.keys()))
    pid = patient_map[sel_patient]

    col1, col2 = st.columns(2)
    with col1:
        spec = st.text_input("Specialty", value="Nephrology")
        loc = st.text_input("Location contains", value="Seattle")
        if st.button("Search doctors"):
            docs = booking.search_doctors(spec, loc)
            st.session_state.docs = docs
    with col2:
        docs = st.session_state.get("docs", [])
        if docs:
            df = pd.DataFrame([{k: d[k] for k in d.keys()} for d in docs])
            st.dataframe(df)

    if docs := st.session_state.get("docs"):
        doc_names = {f"{d['name']} — {d['specialty']} ({d['location']})": int(d["id"]) for d in docs}
        pick = st.selectbox("Choose doctor", list(doc_names.keys()))
        doc_id = doc_names[pick]
        st.session_state.doc_id = doc_id
        if st.button("Load slots"):
            st.session_state.slots = booking.get_slots(doc_id)

    slots = st.session_state.get("slots", [])
    if slots:
        slot_strs = [f"{s} → {e}" for s, e in slots[:30]]
        chosen = st.selectbox("Available slots", slot_strs)
        if st.button("Book this slot"):
            start_ts, end_ts = chosen.split(" → ")
            appt_id = booking.book(pid, doc_id, start_ts, end_ts)
            st.success(f"Booked appointment #{appt_id}")
            st.session_state.memory.add(
                f"Booked appt {appt_id} for patient {pid} with doctor {doc_id} at {start_ts}",
                type="booking",
            )

    st.markdown("### Appointments")
    appts = booking.list_appointments(pid)
    if appts:
        st.dataframe(pd.DataFrame([{k: a[k] for k in a.keys()} for a in appts]))

with hist_tab:
    st.subheader("Patient history")
    patients = db.list_patients()
    patient_map = {f"{p['name']} (#{p['id']})": int(p["id"]) for p in patients}
    sel_patient = st.selectbox("Patient", list(patient_map.keys()), key="hist_patient")
    pid = patient_map[sel_patient]

    st.text_area("New entry", key="hist_text")
    if st.button("Add entry") and st.session_state.get("hist_text"):
        hid = history.add(pid, st.session_state["hist_text"], tags="manual")
        st.success(f"Added history #{hid}")
        st.session_state.memory.add(st.session_state["hist_text"], type="history", patient_id=pid)

    rows = history.get(pid)
    if rows:
        st.dataframe(pd.DataFrame([{k: r[k] for k in r.keys()} for r in rows]))

with info_tab:
    st.subheader("Medical information search (high level)")
    q = st.text_input("Query", value="chronic kidney disease latest treatments")
    use_llm = st.checkbox("Use LLM summarization (requires OPENAI_API_KEY)", value=True)

    if st.button("Search info"):
        out = info.query(q, use_llm=use_llm)
        st.session_state.last_info_out = out

    out = st.session_state.get("last_info_out")
    if out:
        st.write("**Top sources (filtered / trusted-first):**")
        st.json(out.get("sources", []))
        st.markdown("**Bullets:**")
        bullets = out.get("bullets") or []
        if bullets:
            for b in bullets:
                st.write(b)
            st.caption(f"LLM summarization used: {out.get('used_llm', False)}")
            st.session_state.memory.add("\n".join(bullets), type="info", query=q)
        else:
            st.info("No concise snippets found. Try refining the query.")
    else:
        st.caption("Enter a query and click **Search info**.")

with eval_tab:
    st.subheader("Evaluation & Logs")
    memo = st.text_area("Paste a response to score", value="")
    if st.button("Score it") and memo:
        st.json(eval_summary(memo))

    st.markdown("### Memory dump")
    for it in st.session_state.memory.dump()[-20:]:
        st.write(f"• {it.text}")
