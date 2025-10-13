from langchain.tools import Tool
from typing import Optional
import os

def _ddg_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        hits = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=6):
                url = r.get("href") or r.get("link") or ""
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()
                if url and title:
                    hits.append(f"- {title} ({url}): {body[:200]}")
        return "\n".join(hits) if hits else "No results found via DuckDuckGo."
    except Exception as e:
        return f"DuckDuckGo search failed: {e}"

def _serp_search(query: str) -> str:
    try:
        from langchain.utilities import SerpAPIWrapper
        search = SerpAPIWrapper()
        return search.run(query)
    except Exception as e:
        return f"SerpAPI search failed: {e}"

def medical_search(query: str) -> str:
    if os.getenv("SERPAPI_API_KEY"):
        out = _serp_search(query)
        if out and not out.lower().startswith("serpapi search failed"):
            return out
    return _ddg_search(query)

def get_medical_search_tool() -> Tool:
    return Tool(
        name="Medical Search",
        func=medical_search,
        description=(
            "Search external medical info (WHO/NIH/CDC etc.). Input is a plain text query; "
            "returns short snippets with URLs."
        ),
    )
