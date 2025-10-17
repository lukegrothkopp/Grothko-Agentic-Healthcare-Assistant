# pages/2_Clinician_Console.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional
from datetime import date
from pathlib import Path

from tools.booking_tool import book_appointment
import json  # if not already imported

import pandas as pd
import streamlit as st

# Local utilities
from utils.patient_memory import PatientMemory, _to_epoch  # uses your updated, robust loader/sorter

# -------------------------
# Small CSS polish
# -------------------------
CSS = """
<style>
:root {
  --border: #374151;
  --card-bg: #111827;
  --muted: #9ca3af;
  --text: #e5e7eb;
  --chip-bg: #1f2937;
  --chip-border: #374151;
}

.badge {
  display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
  background: var(--chip-bg); border:1px solid var(--chip-border); margin-right:8px; color: var(--text);
}
.card {
  border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; background: var(--card-bg); color: var(--text);
}
.subtitle { color: var(--muted); font-size: 0.9rem; margin-bottom: 8px; }
.kv { color: var(--text); }
.kv b { color: #fff; }
.timeline-item { padding:8px 0; border-bottom:1px dashed var(--border); }
.timeline-item:last-child { border-bottom: none; }
.small-muted { color: var(--muted); font-size:0.85rem; }
.section-title { font-weight: 600; margin-top: 4px; }
</style>
"""
st.set_page_config(page_title="Clinician Console", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.title("👩🏽‍⚕️ Clinician Console")
st.caption("Clinical summary derived from seeds/DB + runtime memory")

# -------------------------
# Session-scoped memory
# -------------------------
if "pmemory" not in st.session_state:
    st.session_state.pmemory = PatientMemory()
_mem: PatientMemory = st.session_state.pmemory

# -------------------------
# Sidebar: patient picker
# -------------------------
with st.sidebar:
    st.header("Patient")
    rows = _mem.list_patients()
    if not rows:
        st.info("No patients loaded. Use the Developer Console to import seeds or check data paths.")
        st.stop()

    label_map = {f"{r['name']} ({r['patient_id']})": r["patient_id"] for r in rows}
    default_label = list(label_map.keys())[0]
    sel_label = st.selectbox("Select patient", options=list(label_map.keys()), index=0)
    pid = label_map.get(sel_label)

    # Quick actions
    colA, colB = st.columns(2)
    with colA:
        if st.button("Refresh", use_container_width=True):
            _mem.reload_from_dir(_mem.seed_dir)
            st.rerun()
    with colB:
        if st.button("Add demo note", type="secondary", use_container_width=True):
            _mem.record_event(pid, "Demo: clinician viewed chart and added a note.", meta={"kind": "note", "by": "clinician"})
            st.toast("Demo note added.")
            st.rerun()

# -------------------------
# Helpers
# -------------------------
def _get_base(pid: str) -> Dict[str, Any]:
    p = _mem.patients.get(pid)
    return p.data if hasattr(p, "data") and isinstance(p.data, dict) else (p if isinstance(p, dict) else {})

def _find_latest_plan(pid: str) -> Optional[List[str]]:
    """Scan memory events for last plan-like entry; supports various shapes."""
    events = _mem.history.get(pid, [])
    for ev in reversed(events):
        t = (ev.get("type") or "").lower()
        meta = ev.get("meta") or {}
        if t in {"plan", "planning", "care_plan"}:
            steps = ev.get("steps") or meta.get("steps")
            if isinstance(steps, list) and steps:
                return [str(s) for s in steps]
            text = ev.get("text") or meta.get("text")
            if isinstance(text, str) and text.strip():
                # naive bulletization
                parts = [p.strip(" •-") for p in text.split("\n") if p.strip()]
                if parts:
                    return parts
        # fallbacks: detect thing that "looks like" plan
        if isinstance(meta.get("plan"), list) and meta["plan"]:
            return [str(s) for s in meta["plan"]]
    return None

def _next_upcoming(appts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    today = date.today().isoformat()
    # appts might be missing/strings; we normalize with _to_epoch
    try:
        fut = [a for a in appts if isinstance(a, dict) and str(a.get("date","")) >= "1900-01-01"]
        fut.sort(key=lambda a: _to_epoch(a.get("date")))
        # return first appointment from today forward
        out = [a for a in fut if (a.get("date") or "") >= today]
        return out[0] if out else (fut[-1] if fut else None)
    except Exception:
        return None

def _appointments_df(base: Dict[str, Any]) -> pd.DataFrame:
    appts = base.get("appointments") or []
    if isinstance(appts, dict):
        appts = [appts]
    # normalize & sort
    rows = []
    for a in appts:
        if not isinstance(a, dict): 
            continue
        rows.append({
            "date": a.get("date"),
            "doctor": a.get("doctor") or a.get("provider") or "",
            "status": a.get("status") or "",
            "booking_id": a.get("booking_id") or "",
        })
    if rows:
        rows.sort(key=lambda r: _to_epoch(r.get("date")))
    return pd.DataFrame(rows)

def _recent_activity(pid: str, k: int = 12) -> List[Dict[str, Any]]:
    try:
        return _mem.get_window(pid, k=k)
    except Exception:
        return []

def _activity_icon(typ: str) -> str:
    t = (typ or "").lower()
    if "appoint" in t: return "📅"
    if "note" in t: return "📝"
    if "lab" in t: return "🧪"
    if "image" in t or "imaging" in t: return "🩻"
    if "plan" in t: return "🧭"
    if "event" in t: return "📌"
    return "•"

# -------------------------
# Header + key metrics
# -------------------------
base = _get_base(pid)
name = base.get("name", pid)
st.markdown(f"### {name}  <span class='small-muted'>({pid})</span>", unsafe_allow_html=True)

summary = _mem.get_summary(pid)  # concise line with conditions/last appt/recent
with st.container():
    st.markdown(f"<div class='card'><div class='subtitle'>Summary</div>{summary}</div>", unsafe_allow_html=True)

# Metrics
age = base.get("age", "—")
conditions = base.get("conditions") or []
appts = base.get("appointments") or []
if isinstance(appts, dict):
    appts = [appts]
upcoming = _next_upcoming(appts)
upc_text = f"{upcoming.get('date')} — {upcoming.get('doctor','')}" if upcoming else "—"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Age", age if age is not None else "—")
m2.metric("Conditions", len(conditions))
m3.metric("Last/Upcoming Appt", upc_text)
m4.metric("Notes (recent)", len([e for e in _recent_activity(pid, k=20) if (e.get('type') or '').lower() in {'note','event'}]))

st.markdown("---")

# -------------------------
# Two columns: Plan + Appointments / Activity
# -------------------------
left, right = st.columns([1, 1])

with left:
    # Latest Plan
    st.markdown("#### Latest Plan")
    steps = _find_latest_plan(pid)
    if steps:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        for i, s in enumerate(steps, 1):
            st.markdown(f"- {s}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No plan recorded yet.")

    # Upcoming Appointments
    st.markdown("#### Upcoming Appointments")
    df_appts = _appointments_df(base)
    if not df_appts.empty:
        # mark upcoming (>= today)
        today = date.today().isoformat()
        df_show = df_appts.copy()
        df_show["is_upcoming"] = df_show["date"].fillna("").apply(lambda d: (d >= today))
        # Show upcoming first
        df_show = pd.concat([df_show[df_show["is_upcoming"]], df_show[~df_show["is_upcoming"]]], ignore_index=True)
        st.dataframe(df_show.drop(columns=["is_upcoming"]), use_container_width=True, height=220)
    else:
        st.info("No appointments on file.")

with right:
    # Recent Activity (timeline)
    st.markdown("#### Recent Activity")
    try:
        window = _recent_activity(pid, k=12)
        if not window:
            st.info("No recent activity.")
        else:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            for e in window:
                typ = e.get("type") or e.get("tag") or "event"
                icon = _activity_icon(typ)
                ts = e.get("ts") or e.get("date") or e.get("timestamp") or ""
                txt = e.get("text") or e.get("notes") or e.get("diagnosis") or ""
                # compact line
                st.markdown(
                    f"<div class='timeline-item'><span class='small-muted'>{ts}</span> &nbsp; {icon} "
                    f"<span class='kv'><b>{typ.capitalize()}</b></span> — {txt}</div>",
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as ex:
        st.error(f"Could not load recent activity: {ex}")

st.markdown("---")

# -------------------------
# Full History (filterable)
# -------------------------
st.markdown("### Full History")
flt_col1, flt_col2 = st.columns([2,1])
with flt_col1:
    q = st.text_input("Filter (matches in text/notes/diagnosis)", value="")
with flt_col2:
    k = st.slider("Window size", min_value=10, max_value=200, value=50, step=10)

rows: List[Dict[str, Any]] = []
for e in _mem.get_window(pid, k=k):
    rows.append({
        "ts": e.get("ts") or e.get("date") or e.get("timestamp"),
        "type": e.get("type") or e.get("tag") or "event",
        "text": e.get("text") or e.get("notes") or e.get("diagnosis") or "",
        "meta": json.dumps(e.get("meta") or {}, ensure_ascii=False),
    })
df_hist = pd.DataFrame(rows)
if q.strip():
    ql = q.lower()
    mask = df_hist["text"].astype(str).str.lower().str.contains(ql) | \
           df_hist["meta"].astype(str).str.lower().str.contains(ql)
    df_hist = df_hist[mask]
st.dataframe(df_hist, use_container_width=True, height=320)

# -------------------------
# Add Clinical Note (persists to memory)
# -------------------------
st.markdown("### Add Clinical Note")
with st.form("add_note_form", clear_on_submit=True):
    note = st.text_area("Note (stored in runtime memory log)", height=120, placeholder="e.g., Discussed BP home-monitoring; schedule BMP labs next visit.")
    submitted = st.form_submit_button("Save note")
    if submitted:
        if not note.strip():
            st.warning("Please enter a note.")
        else:
            _mem.record_event(pid, note.strip(), meta={"kind": "note", "by": "clinician"})
            st.success("Note saved to memory.")
            st.rerun()
