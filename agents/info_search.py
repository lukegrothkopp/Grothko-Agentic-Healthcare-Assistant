# agents/info_search.py
from __future__ import annotations
import re
import requests
from typing import List, Dict, Optional
from duckduckgo_search import DDGS
from core.logging import get_logger

log = get_logger("InfoSearchAgent")

TRUSTED = [
    "who.int",
    "cdc.gov",
    "nih.gov",
    "medlineplus.gov",
    "mayoclinic.org",
    "cancer.gov",
    "nice.org.uk",
]

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

class InfoSearchAgent:
    def _ddg(self, query: str, max_results: int = 8) -> List[Dict]:
        """DuckDuckGo search with a trusted-first strategy and a fallback."""
        hits: List[Dict] = []

        def _pack(r: Dict) -> Optional[Dict]:
            url = r.get("href") or r.get("link") or ""
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()
            if not url or not title:
                return None
            return {"title": title, "snippet": body, "url": url}

        # 1) Try with trusted sites bias (site: filters)
        trusted_query = (
            f"{query} (site:who.int OR site:cdc.gov OR site:nih.gov OR "
            f"site:medlineplus.gov OR site:mayoclinic.org OR site:cancer.gov OR site:nice.org.uk)"
        )

        with DDGS() as ddgs:
            for r in ddgs.text(trusted_query, max_results=max_results):
                p = _pack(r)
                if p:
                    hits.append(p)

        # 2) If nothing, do a normal search then filter to trusted; if still nothing, keep top general results
        if not hits:
            generic: List[Dict] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results * 3):
                    p = _pack(r)
                    if p:
                        generic.append(p)
            trusted_generic = [g for g in generic if any(dom in g["url"] for dom in TRUSTED)]
            hits = (trusted_generic[:max_results]) if trusted_generic else (generic[:max_results])

        return hits[:max_results]

    def _fetch(self, url: str, timeout: int = 10) -> str:
        """Fetch page and extract readable text; return empty string on failure."""
        try:
            resp = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
            resp.raise_for_status()
            html = resp.text or ""
        except Exception as e:
            log.warning(f"Fetch failed for {url}: {e}")
            return ""

        # Remove scripts/styles
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
        # Strip tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Keep it manageable
        return text[:20000]

    def _split_sents(self, text: str) -> List[str]:
        # Simple sentence splitter
        sents = re.split(r"(?<=[.!?])\s+", text)
        # Clean short/noisy ones
        return [s.strip() for s in sents if len(s.strip()) > 40]

    def _extract_bullets(self, text: str, query: str, fallback_snippet: str, title: str, url: str) -> Optional[str]:
        """
        Try to produce one concise bullet from page text that loosely matches the query;
        fall back to the search snippet if page text is unusable.
        """
        host = url.split("/")[2] if "://" in url else url
        if not text:
            if fallback_snippet:
                return f"- {title} ({host}): {fallback_snippet[:300]}"
            return None

        sents = self._split_sents(text)[:80]  # first ~80 sentences is plenty
        if not sents:
            if fallback_snippet:
                return f"- {title} ({host}): {fallback_snippet[:300]}"
            return None

        # Score sentences by presence of query keywords (loose match); prefer earlier sentences
        q_tokens = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t and len(t) > 2]
        def score(sent: str) -> int:
            s = sent.lower()
            return sum(1 for t in q_tokens if t in s)

        scored = sorted(((score(s), i, s) for i, s in enumerate(sents)), key=lambda x: (-x[0], x[1]))
        best_score, _, best_sent = scored[0]

        # If keyword match is too weak, prefer snippet fallback to avoid irrelevant pull
        if best_score == 0 and fallback_snippet:
            return f"- {title} ({host}): {fallback_snippet[:300]}"

        return f"- {title} ({host}): {best_sent[:300]}"

    def query(self, q: str) -> Dict:
        results = self._ddg(q, max_results=8)
        bullets: List[str] = []
        pages: List[Dict] = []

        for r in results:
            url = r["url"]
            title = r["title"]
            snippet = r.get("snippet", "")
            body = self._fetch(url)
            pages.append({"title": title, "url": url, "snippet": snippet, "body": body})

        for p in pages:
            bullet = self._extract_bullets(
                text=p["body"],
                query=q,
                fallback_snippet=p.get("snippet", ""),
                title=p["title"],
                url=p["url"],
            )
            if bullet:
                bullets.append(bullet)
            if len(bullets) >= 5:
                break

        # If still nothing, at least surface titles so the UI has something meaningful
        if not bullets and results:
            for r in results[:3]:
                host = r["url"].split("/")[2] if "://" in r["url"] else r["url"]
                bullets.append(f"- {r['title']} ({host}): {r.get('snippet', '')[:300]}")

        return {"query": q, "sources": results, "bullets": bullets}
