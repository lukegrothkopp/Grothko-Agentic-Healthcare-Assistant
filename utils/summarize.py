# utils/summarize.py
from __future__ import annotations
import os
from typing import List, Dict, Tuple

def have_openai() -> bool:
    v = os.getenv("OPENAI_API_KEY", "") or ""
    return bool(v.strip() and not v.strip().startswith("${"))

def llm_bullets_with_citations(query: str, docs: List[Dict]) -> Tuple[str, List[str]]:
    """
    Return (markdown_text, bullets_list) with [1]-style citations and a Sources section.
    Falls back to simple formatting if no OpenAI key.
    """
    sources = [{"title": d["title"], "url": d["url"]} for d in docs]
    if not have_openai() or not docs:
        bullets = []
        for i, d in enumerate(docs[:5], 1):
            bullets.append(f"{d['title']} — {d['snippet']} [{i}]")
        md = "• " + "\n• ".join(bullets) if bullets else ""
        if sources:
            md += "\n\n**Sources:**\n" + "\n".join([f"[{i+1}] {s['title']} — {s['url']}" for i, s in enumerate(sources)])
        return md.strip(), bullets

    # OpenAI via langchain-openai (already in your repo)
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    sys = SystemMessage(content=(
        "You are a medical information summarizer for an admin assistant. "
        "Create 4–7 concise, patient-friendly bullets. No diagnosis or treatment instructions; "
        "stick to high-level guidance and definitions. Add bracketed numeric citations like [1], [2] "
        "that map to the provided source list. Avoid speculation."
    ))
    src_lines = "\n".join([f"[{i+1}] {d['title']} — {d['url']}" for i, d in enumerate(docs)])
    hum = HumanMessage(content=(
        f"User query: {query}\n\n"
        f"Candidate source snippets:\n" +
        "\n".join([f"[{i+1}] {d['title']} — {d['snippet']}" for i, d in enumerate(docs)]) +
        "\n\nReturn ONLY:\n- 4–7 bullets with citations like [1], [2]\n- Then a 'Sources:' list mapping those numbers to the URLs provided."
    ))
    out = llm.invoke([sys, hum])
    text = (out.content or "").strip()

    # extract bullets for optional downstream use (simple split)
    bullets = [line.lstrip("-• ").strip() for line in text.splitlines() if line.strip().startswith(("-", "•"))]
    return text, bullets[:7]
