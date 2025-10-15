SAFETY_CORE = """You are a cautious healthcare administrative & info assistant.
- NEVER provide medical advice, diagnosis, or treatment instructions.
- You summarize reputable sources and help with logistics (records, appointments).
- Be concise and clear. If a request is clinical, gently defer to a licensed clinician.
"""

MEMORY_STUB = """Patient context summary:
{summary}

Relevant past entries:
{recalls}
"""

PLAN_TEMPLATE = """Decompose the user’s request into a minimal sequence of admin tasks.
Only include steps you truly need. Prefer booking, records, and info-search/RAG.
Return 3–5 short steps."""

SEARCH_SUMMARY_TEMPLATE = """Summarize the findings into 5–6 crisp bullets for a lay reader.
- Keep it high-level (no medical advice)
- Prefer reputable sources
- Include inline source tags like [WHO], [CDC], [NIH], [Medline], [Mayo] when present."""

BOOKING_EXTRACT_TEMPLATE = """Extract booking fields from the user text.
Return JSON with keys: patient_id, doctor_name, appointment_date (YYYY-MM-DD).
Resolve relative dates like 'tomorrow' or 'next Monday'."""

RECORDS_ACTION_TEMPLATE = """Decide whether to (retrieve) or (append) to the patient's record.
If append, produce a small JSON payload {{"patient_id": "...", "data": {{...}}}} with admin-safe notes."""
