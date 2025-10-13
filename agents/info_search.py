from __future__ import annotations
import os
import re
import requests
from duckduckgo_search import DDGS
from typing import List, Dict
from core.logging import get_logger

log = get_logger("InfoSearchAgent")

TRUSTED = ["who.int", "medlineplus.gov", "cdc.gov", "nih.gov", "mayoclinic.org"]

class InfoSearchAgent:
    def _search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Use DuckDuckGo to find reputable sources; filter for TRUSTED domains."""
        hits = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results*3):
                url = r.get("href") or r.get("link") or ""
                if any(dom in url for dom in TRUSTED):
                    hits.append({"title": r.get("title"), "snippet": r.get("body"), "url": url})
                if len(hits) >= max_results:
                    break
        return hits

    def _fetch(self, url: str, timeout: int = 10) -> str:
        try:
            html = requests.get(url, timeout=timeout).text
        except Exception as e:
            log.warning(f"Fetch failed: {e}")
            return ""
        # crude text extraction
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()[:12000]

    def query(self, q: str) -> Dict:
        results = self._search(q)
        pages = []
        for r in results:
            body = self._fetch(r["url"])[:4000]
            pages.append({"title": r["title"], "url": r["url"], "snippet": r.get("snippet"), "body": body})
        bullets = []
        for p in pages[:3]:
            # naive extractive summary: first 2 sentences containing the query term
            sents = re.split(r"(?<=[.!?])\s+", p["body"])[:50]
            picks = [s for s in sents if any(w in s.lower() for w in q.lower().split())][:2]
            if picks:
                bullets.append(f"- {p['title']} ({p['url'].split('/')[2]}): " + " ".join(picks)[:300])
        return {"query": q, "sources": results, "bullets": bullets}
