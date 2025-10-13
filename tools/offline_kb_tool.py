import json, os
from typing import List, Dict
from langchain.tools import Tool

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _llm_polish(bullets: List[Dict[str, str]]) -> List[str]:
    if not OPENAI_KEY:
        return [f"- {b['text']} ({b['source']})" for b in bullets][:6]
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        raw = "\n".join([f"- {b['text']} ({b['source']})" for b in bullets][:8])
        sys = (
            "You polish high-level healthcare info for laypeople. "
            "No medical advice; keep each bullet concise and include source in parentheses."
        )
        user = (
            "Rewrite the bullets in clear, neutral language (3–6 bullets). Keep the source in parentheses.\n\n"
            f"{raw}"
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.2, max_tokens=400
        )
        text = resp.choices[0].message.content.strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("- ")]
        return lines[:6] if lines else [f"- {b['text']} ({b['source']})" for b in bullets][:6]
    except Exception:
        return [f"- {b['text']} ({b['source']})" for b in bullets][:6]

def _lookup_offline(topic: str) -> List[Dict[str, str]]:
    from difflib import get_close_matches
    path = "data/mini_kb.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            kb = json.load(f)
    except Exception:
        kb = {}
    key = (topic or "").strip().lower()
    if key in kb:
        return kb[key]
    keys = list(kb.keys())
    match = get_close_matches(key, keys, n=1, cutoff=0.6)
    if match:
        return kb.get(match[0], [])
    for k in keys:
        if k in key or key in k:
            return kb[k]
    return []

def offline_kb_query(topic: str) -> str:
    bullets = _lookup_offline(topic)
    if not bullets:
        return "No offline KB match found. Try a broader or related topic (e.g., 'hypertension', 'diabetes')."
    lines = _llm_polish(bullets)
    return "\n".join(lines)

def get_offline_kb_tool() -> Tool:
    return Tool(
        name="Offline Medical KB",
        func=offline_kb_query,
        description=(
            "Retrieve high-level, source-cited bullets from an offline mini knowledge base. "
            "Input is a disease/condition/topic string (e.g., 'lung cancer')."
        ),
    )
