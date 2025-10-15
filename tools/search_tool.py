# tools/search_tool.py
# Robust clinical search with multiple fallbacks and trusted-domain filtering.
# Order: (1) SerpAPI if key present -> (2) duckduckgo_search -> (3) raw HTML fallback.
# Returns plain text; never returns numpy objects.

import os
import re
import html
import json
import urllib.parse
from typing import List, Dict, Any, Tuple, Optional

import requests
from langchain.tools import Tool

TRUSTED_SITES = [
    "nih.gov",
    "cdc.gov",
    "who.int",
    "mayoclinic.org",
    "clevelandclinic.org",
    "medlineplus.gov",
]

def _mk_query(user_query: str) -> str:
    user_query = (user_query or "").strip()
    domain_filter = " OR ".join(f"site:{d}" for d in TRUSTED_SITES)
    return f"{user_query} {domain_filter}".strip()

# ---------- (1) SerpAPI ----------
def _serpapi_search(query: str, max_results: int = 8) -> List[Tuple[str, str]]:
    key = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not key:
        return []
    try:
        params = {
            "engine": "google",
            "q": query,
            "num": max_results,
            "hl": "en",
            "output": "json",
            "api_key": key,
        }
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        items = []
        for item in data.get("organic_results", [])[:max_results]:
            title = str(item.get("title") or "").strip()
            link = str(item.get("link") or item.get("url") or "").strip()
            if link:
                items.append((title or link, link))
        return items
    except Exception:
        return []

# ---------- (2) duckduckgo_search ----------
def _ddg_search(query: str, max_results: int = 8) -> List[Tuple[str, str]]:
    """
    Try multiple backends to maximize success.
    """
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except Exception:
        return []

    backends = [
        {"backend": "api"},
        {"backend": None},        # library default
        {"backend": "lite"},      # lightweight HTML backend in ddg lib
    ]
    for cfg in backends:
        try:
            ddgs = DDGS()
            gen = ddgs.text(query, max_results=max_results, backend=cfg["backend"]) \
                  if cfg["backend"] is not None else ddgs.text(query, max_results=max_results)
            rows = list(gen)
            items = []
            for r in rows[:max_results]:
                title = str(r.get("title") or "").strip()
                href = str(r.get("href") or r.get("url") or "").strip()
                if href:
                    items.append((title or href, href))
            if items:
                return items
        except Exception:
            continue
    return []

# ---------- (3) Raw HTML fallback to DuckDuckGo ----------
_A_TAG = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    flags=re.I | re.S
)

def _raw_ddg_html_search(query: str, max_results: int = 8) -> List[Tuple[str, str]]:
    """
    Parse DuckDuckGo 'html' results page without external parsers.
    This is a last-resort fallback; may return fewer items if layout changes.
    """
    try:
        q = urllib.parse.quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={q}"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; GrothkoHealthcareAssistant/1.0; +https://example.com)"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200 or not resp.text:
            return []
        html_text = resp.text

        # Extract <a class="result__a" ...>Title</a>
        matches = _A_TAG.findall(html_text)
        items: List[Tuple[str, str]] = []
        for href, title_html in matches:
            href = html.unescape(href).strip()
            title = html.unescape(re.sub(r"<.*?>", "", title_html)).strip()
            if not href:
                continue
            # Basic domain filter to trusted sites only
            if not any(d in href for d in TRUSTED_SITES):
                continue
            items.append((title or href, href))
            if len(items) >= max_results:
                break
        return items
    except Exception:
        return []

# ---------- Compose + public tool ----------
def _format_results(pairs: List[Tuple[str, str]]) -> str:
    lines: List[str] = []
    for title, url in pairs:
        t = (title or "").strip()
        u = (url or "").strip()
        if not u:
            continue
        if not t:
            t = u
        lines.append(f"- {t} — {u}")
    return "\n".join(lines).strip()

def _clinical_search(user_query: str) -> str:
    """
    Robust clinical search through trusted sources.
    Returns plain text, or a diagnostic message if nothing worked.
    """
    q = _mk_query(user_query)
    diagnostics: List[str] = []

    # 1) SerpAPI (if key present)
    try:
        pairs = _serpapi_search(q, max_results=8)
        if pairs:
            return _format_results(pairs)
        else:
            diagnostics.append("SerpAPI: no results or key missing")
    except Exception as e:
        diagnostics.append(f"SerpAPI error: {e}")

    # 2) duckduckgo_search (multiple backends)
    try:
        pairs = _ddg_search(q, max_results=8)
        if pairs:
            return _format_results(pairs)
        else:
            diagnostics.append("duckduckgo_search: no results")
    except Exception as e:
        diagnostics.append(f"duckduckgo_search error: {e}")

    # 3) Raw HTML fallback
    try:
        pairs = _raw_ddg_html_search(q, max_results=8)
        if pairs:
            return _format_results(pairs)
        else:
            diagnostics.append("raw DDG HTML: no results")
    except Exception as e:
        diagnostics.append(f"raw DDG HTML error: {e}")

    # Nothing worked
    return "(online search unavailable or returned no results)\n" + "\n".join(diagnostics)

def get_medical_search_tool() -> Tool:
    # IMPORTANT: Keep this exact name; the graph looks for it.
    return Tool(
        name="Clinical Search (SERP/DDG)",
        func=_clinical_search,
        description="Search trusted medical sources (NIH, CDC, WHO, Mayo Clinic, Cleveland Clinic, MedlinePlus). Returns a plain-text bulleted list with links."
    )
