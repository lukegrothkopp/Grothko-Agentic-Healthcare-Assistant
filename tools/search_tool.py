# tools/search_tool.py
# Clinical search tool constrained to trusted sources (NIH, CDC, WHO, Mayo, Cleveland Clinic, MedlinePlus).
# Returns plain text; no numpy objects. Works with duckduckgo_search if available.

from typing import List, Dict, Any
from langchain.tools import Tool

TRUSTED_SITES = [
    "nih.gov",
    "cdc.gov",
    "who.int",
    "mayoclinic.org",
    "clevelandclinic.org",
    "medlineplus.gov",
]

def _ddg_text(query: str, max_results: int = 6) -> List[Dict[str, Any]]:
    try:
        # pip install duckduckgo_search>=6
        from duckduckgo_search import DDGS  # type: ignore
        # Try "api" backend first, fallback to "lite"
        try:
            return list(DDGS().text(query, max_results=max_results, backend="api"))
        except Exception:
            return list(DDGS().text(query, max_results=max_results, backend="lite"))
    except Exception:
        # No dependency or network blocked
        return []

def _clinical_search(user_query: str) -> str:
    q = (user_query or "").strip()
    if not q:
        return "No query provided."
    # Bias results to trusted domains
    domain_filter = " OR ".join(f"site:{d}" for d in TRUSTED_SITES)
    query = f"{q} {domain_filter}"

    results = _ddg_text(query, max_results=8)
    if not results:
        return "(online search unavailable or returned no results; if running locally, add 'duckduckgo_search' to requirements.txt)"

    # Format as a simple, safe list
    lines: List[str] = []
    for r in results[:6]:
        title = str(r.get("title") or "").strip()
        href = str(r.get("href") or r.get("url") or "").strip()
        if not href:
            continue
        if not title:
            title = href
        lines.append(f"- {title} — {href}")

    out = "\n".join(lines).strip()
    return out if out else "(no results from trusted sources)"

def get_medical_search_tool() -> Tool:
    # IMPORTANT: Name matches what the graph looks for.
    return Tool(
        name="Clinical Search (SERP/DDG)",
        func=_clinical_search,
        description="Search trusted medical sources (NIH, CDC, WHO, Mayo Clinic, Cleveland Clinic, MedlinePlus). Returns a plain-text bulleted list with links."
    )
