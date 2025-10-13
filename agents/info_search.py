# agents/info_search.py
from __future__ import annotations
import os
import re
import requests
from typing import List, Dict, Optional
from duckduckgo_search import DDGS
from core.logging import get_logger

# Optional OpenAI import; handled gracefully if not installed/keys missing
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
try:
    if OPENAI_KEY:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_KEY)
    else:
        _openai_client = None
except Exception:
    _openai_client = None

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
        """DuckDuckGo search with a trusted-first strategy and a graceful fallback."""
        hits: List[Dict] = []

        def _pack(r: Dict) -> Optional[Dict]:
            url = r.get("href") or r.get("link") or ""
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()
            if not url or not title:
                return None
            return {"title": title, "snippet": body, "url": url}

        # 1) Bias to trusted domains using site: filters
        trusted_query = (
            f"{query} (site:who.int OR site:cdc.gov OR site:nih.gov OR "
            f"site:medlineplus.gov OR site:mayoclinic.org OR site:cancer.gov OR site:nice.org.uk)"
        )
        with DDGS() as ddgs:
            for r in ddgs.text(trusted_query, max_results=max_results):
                p = _pack(r)
                if p:
                    hits.append(p)

        # 2) If nothing: general search → filter to trusted; else keep top general
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
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:20000]

    def _split_sents(self, text: str) -> List[str]:
        sents = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sents if len(s.strip()) > 40]

    def _extract_bullets(self, text: str, query: str, fallback_snippet: str, title: str, url: str) -> Optional[str]:
        """Make one concise bullet from page text; fallback to search snippet."""
        host = url.split("/")[2] if "://" in url else url
        if not text:
            return f"- {title} ({host}): {fallback_snippet[:300]}" if fallback_snippet else None

        sents = self._split_sents(text)[:80]
        if not sents:
            return f"- {title} ({host}): {fallback_snippet[:300]}" if fallback_snippet else None

        q_tokens = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t and len(t) > 2]
        def score(sent: str) -> int:
            s = sent.lower()
            return sum(1 for t in q_tokens if t in s)

        scored = sorted(((score(s), i, s) for i, s in enumerate(sents)), key=lambda x: (-x[0], x[1]))
        best_score, _, best_sent = scored[0]
        if best_score == 0 and fallback_snippet:
            return f"- {title} ({host}): {fallback_snippet[:300]}"
        return f"- {title} ({host}): {best_sent[:300]}"

    # ---------- LLM summarization ----------
    def _llm_summarize(self, query: str, pages: List[Dict], max_bullets: int = 5) -> List[str]:
        """
        Use OpenAI to synthesize 3–6 polished bullets with clear, source-cited statements.
        Each bullet must name the source host in parentheses, e.g., (NIH), (CDC), (Mayo Clinic).
        """
        if not _openai_client:
            return []

        # Build compact context for the model
        # Include per-source: title, host, short snippet of body OR the search snippet
        items = []
        for p in pages[:8]:
            host = p["url"].split("/")[2] if "://" in p["url"] else p["url"]
            body = p.get("body") or ""
            snippet = (body[:1200] if body else (p.get("snippet") or ""))[:1200]
            items.append(f"- {p['title']} ({host}) :: {snippet}")

        context = "\n".join(items) if items else "No sources."
        sys = (
            "You are a cautious healthcare information summarizer for laypeople. "
            "You DO NOT provide medical advice or instructions. "
            "Summaries must be factual, neutral, and name sources."
        )
        user = (
            "Task: Create concise, polished bullets (3–6) answering the user's query from the provided sources.\n"
            "Rules:\n"
            "• No medical advice or directives. High-level info only.\n"
            "• Each bullet must include a source in parentheses, using the site/organization name, e.g., (WHO), (CDC), (NIH), (MedlinePlus), (Mayo Clinic), (NICE).\n"
            "• Prefer reputable sources and avoid speculation.\n"
            "• If evidence disagrees, note that briefly.\n"
            f"User query: {query}\n\n"
            f"Sources:\n{context}\n\n"
            "Output format:\n"
            "- Bullet 1 (Source)\n"
            "- Bullet 2 (Source)\n"
            "- Bullet N (Source)\n"
        )

        try:
            resp = _openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"LLM summarization failed: {e}")
            return []

        # Parse bullets (lines starting with dash)
        bullets = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("- ")]
        bullets = bullets[:max_bullets] if bullets else []
        # Safety filter: keep reasonably short bullets
        return [b[:350] for b in bullets if len(b) > 10]

    # ---------- Public API ----------
    def query(self, q: str, use_llm: bool = False) -> Dict:
        results = self._ddg(q, max_results=8)

        # Fetch pages
        pages: List[Dict] = []
        for r in results:
            url = r["url"]
            title = r["title"]
            snippet = r.get("snippet", "")
            body = self._fetch(url)
            pages.append({"title": title, "url": url, "snippet": snippet, "body": body})

        # Extractive bullets first (robust fallback)
        bullets: List[str] = []
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

        # If enabled and key present, ask LLM to polish into source-cited bullets
        if use_llm and _openai_client:
            llm_bullets = self._llm_summarize(q, pages, max_bullets=5)
            if llm_bullets:
                bullets = llm_bullets

        # Last-resort: surface titles/snippets
        if not bullets and results:
            for r in results[:3]:
                host = r["url"].split("/")[2] if "://" in r["url"] else r["url"]
                bullets.append(f"- {r['title']} ({host}): {r.get('snippet', '')[:300]}")

        return {"query": q, "sources": results, "bullets": bullets, "used_llm": bool(use_llm and _openai_client)}

