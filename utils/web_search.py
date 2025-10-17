# utils/web_search.py
from __future__ import annotations
import os, re
from typing import List, Dict

TRUSTED_DOMAINS = [
    "who.int", "cdc.gov", "nih.gov", "medlineplus.gov", "mayoclinic.org",
    "cancer.gov", "ncbi.nlm.nih.gov", "nhs.uk"
]

def _has_key(name: str) -> bool:
    v = os.getenv(name, "") or ""
    return bool(v.strip() and not v.strip().startswith("${"))

def _domain_ok(url: str) -> bool:
    try:
        host = re.sub(r"^https?://", "", url).split("/")[0].lower()
        return any(host.endswith(d) or (d in host) for d in TRUSTED_DOMAINS)
    except Exception:
        return False

def search_trusted(query: str, k: int = 5) -> List[Dict]:
    """
    Return up to k trusted results: [{title, url, snippet}]
    Prefers Tavily if available, then falls back to SerpAPI.
    """
    out: List[Dict] = []

    # --- Tavily (preferred) ---
    if _has_key("TAVILY_API_KEY"):
        try:
            from tavily import TavilyClient
            tv = TavilyClient(api_key=os.getenv("TAVILY_API_KEY").strip())
            res = tv.search(query, max_results=max(k, 5), include_domains=TRUSTED_DOMAINS)
            for it in res.get("results", []):
                url = it.get("url") or ""
                if not _domain_ok(url):
                    continue
                title = it.get("title") or ""
                snippet = (it.get("content") or it.get("snippet") or "")[:400]
                if title and url:
                    out.append({"title": title, "url": url, "snippet": snippet})
            if out:
                return out[:k]
        except Exception:
            pass

    # --- SerpAPI fallback ---
    if _has_key("SERPAPI_API_KEY"):
        try:
            from serpapi import GoogleSearch
            params = {"q": query, "api_key": os.getenv("SERPAPI_API_KEY").strip(), "num": max(k, 10)}
            search = GoogleSearch(params)
            res = search.get_dict()
            for it in (res.get("organic_results") or []):
                url = it.get("link") or ""
                if not _domain_ok(url):
                    continue
                title = it.get("title") or ""
                snippet = (it.get("snippet") or "")[:400]
                if title and url:
                    out.append({"title": title, "url": url, "snippet": snippet})
            if out:
                return out[:k]
        except Exception:
            pass

    return out[:k]
