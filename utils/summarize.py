# utils/summarize.py
from __future__ import annotations
import os
from typing import List, Dict, Tuple

def have_openai() -> bool:
    v = os.getenv("OPENAI_API_KEY", "") or ""
    return bool(v.strip() and not v.strip().startswith("${"))

def _format_sources_md(docs: List[Dict]) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        title = d.get("title") or d.get("url") or f"Source {i}"
        url = d.get("url") or ""
        # Always output a real markdown link when we have a URL
        if url:
            lines.append(f"[{i}] [{title}]({url})")
        else:
            lines.append(f"[{i}] {title}")
    return "\n".join(lines)

def llm_bullets_with_citations(query: str, docs: List[Dict]) -> Tuple[str, str]:
    """
    Returns (bullets_md, sources_md).
    - bullets_md: 4–7 bullets with [1]-style numeric citations (no 'Sources:' section inside).
    - sources_md: enumerated [i] links (we build deterministically from docs).
    Falls back to simple formatting if no OpenAI key.
    """
    sources_md = _format_sources_md(docs) if docs else ""

    if not have_openai() or not docs:
        # Fallback: simple bullets that already include [i], plus our sources list
        bullets = []
        for i, d in enumerate(docs[:7], 1):
            title = d.get("title") or ""
            snippet = (d.get("snippet") or "").strip()
            if snippet:
                bullets.append(f"• {snippet} [{i}]")
            elif title:
                bullets.append(f"• {title} [{i}]")
        return ("\n".join(bullets).strip(), sources_md)

    # OpenAI via langchain-openai
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    sys = SystemMessage(content=(
        "You are a medical information summarizer for an admin/logistics assistant. "
        "Write 4–7 concise, patient-friendly bullets for a lay audience. "
        "No diagnosis or prescriptive treatment; only high-level guidance and definitions. "
        "Use bracketed numeric citations [1], [2], … referring to source indices provided. "
        "IMPORTANT: Output ONLY the bullets. DO NOT include a 'Sources' section; the caller will append it."
    ))
    src_snippets = "\n".join([
        f"[{i+1}] {d.get('title') or d.get('url')}: {(d.get('snippet') or '')[:400]}"
        for i, d in enumerate(docs)
    ])
    hum = HumanMessage(content=(
        f"User query: {query}\n\n"
        f"Source snippets:\n{src_snippets}\n\n"
        "Return ONLY bullets (each starting with '-' or '•'), with [1]-style citations. "
        "Do not include a 'Sources' section."
    ))
    out = llm.invoke([sys, hum])
    raw = (out.content or "").strip()

    # Normalize bullets to markdown bullets (• or - accepted)
    bullets = []
    for line in raw.splitlines():
        L = line.strip()
        if not L:
            continue
        if L.startswith(("•", "-")):
            bullets.append(L)
        else:
            # Be forgiving: convert stray lines to bullets
            bullets.append("• " + L)

    bullets_md = "\n".join(bullets[:7]).strip()
    return (bullets_md, sources_md)
