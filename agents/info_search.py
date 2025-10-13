# agents/info_search.py
from __future__ import annotations
import os
import re
import requests
from typing import List, Dict, Optional, Any
from duckduckgo_search import DDGS
from core.logging import get_logger

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

def _host(u: str) -> str:
    return u.split("/")[2] if "://" in u else u

class InfoSearchAgent:
    def __init__(self):
        self._last_debug: Dict[str, Any] = {}

    def get_debug(self) -> Dict[str, Any]:
        return dict(self._last_debug)

    # ---------- Search ----------
    def _pack(self, r: Dict) -> Optional[Dict]:
        url = r.get("href") or r.get("link") or ""
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        if not url or not title:
            return None
        return {"title": title, "snippet": body, "url": url}

    def _ddg_text(self, query: str, max_results: int) -> List[Dict]:
        out: List[Dict] = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    p = self._pack(r)
                    if p:
                        out.append(p)
        except Exception as e:
            log.warning(f"DDG text failed: {e}")
        return out

    def _ddg_news(self, query: str, max_results: int) -> List[Dict]:
        out: List[Dict] = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.news(query, max_results=max_results):
                    p = self._pack(r)
                    if p:
                        out.append(p)
        except Exception as e:
            log.warning(f"DDG news failed: {e}")
        return out

    def _ddg_answers(self, query: str, max_results: int) -> List[Dict]:
        out: List[Dict] = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.answers(query, max_results=max_results):
                    p = self._pack(r)
                    if p:
                        out.append(p)
        except Exception as e:
            log.warning(f"DDG answers failed: {e}")
        return out

    def _search(self, query: str, max_results: int = 8) -> List[Dict]:
        # Pass 1: trusted-first
        trusted_q = (
            f"{query} (site:who.int OR site:cdc.gov OR site:nih.gov OR "
            f"site:medlineplus.gov OR site:mayoclinic.org OR site:cancer.gov OR site:nice.org.uk)"
        )
        hits = self._ddg_text(trusted_q, max_results=max_results)

        # Pass 2: general web → filter trusted, else keep general
        if not hits:
            generic = self._ddg_text(query, max_results=max_results * 3)
            trusted_generic = [g for g in generic if any(dom in g["url"] for dom in TRUSTED)]
            hits = (trusted_generic[:max_results]) if trusted_generic else (generic[:max_results])

        # Pass 3: news (sometimes better for guidelines/announcements)
        if not hits:
            hits = self._ddg_news(query, max_results=max_results)

        # Pass 4: answers (last resort)
        if not hits:
            hits = self._ddg_answers(query, max_results=max_results)

        return hits[:max_results]

    # ---------- Fetch ----------
    def _fetch(self, url: str, timeout: int = 10) -> str:
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

    # ---------- Summarization helpers ----------
    def _extract_bullets(self, text: str, query: str, fallback_snippet: str, title: str, url: str) -> Optional[str]:
        host = _host(url)
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

    def _llm_summarize(self, query: str, pages: List[Dict], max_bullets: int = 5) -> List[str]:
        if not _openai_client:
            return []
        items = []
        for p in pages[:8]:
            host = _host(p["url"])
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
            "• No medical advice or prescriptive instructions.\n"
            "• Each bullet must include a source in parentheses, e.g., (WHO), (CDC), (NIH), (MedlinePlus), (Mayo Clinic), (NICE).\n"
            "• Prefer reputable sources and avoid speculation.\n"
            f"User query: {query}\n\n"
            f"Sources:\n{context}\n\n"
            "Output format:\n- Bullet 1 (Source)\n- Bullet 2 (Source)\n- Bullet N (Source)\n"
        )
        try:
            resp = _openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=600,
            )
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"LLM summarization failed: {e}")
            return []

        bullets = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("- ")]
        bullets = bullets[:max_bullets] if bullets else []
        return [b[:350] for b in bullets if len(b) > 10]

    # ---------- Public API ----------
    def query(self, q: str, use_llm: bool = False) -> Dict:
        debug: Dict[str, Any] = {"query": q, "stages": []}

        # Search
        results = self._search(q, max_results=8)
        debug["stages"].append({"stage": "search", "results": len(results), "urls": [r["url"] for r in results]})

        # Fetch pages
        pages: List[Dict] = []
        fetch_errors = 0
        for r in results:
            url = r["url"]
            title = r["title"]
            snippet = r.get("snippet", "")
            body = self._fetch(url)
            if body == "":
                fetch_errors += 1
            pages.append({"title": title, "url": url, "snippet": snippet, "body": body})
        debug["stages"].append({"stage": "fetch", "pages": len(pages), "fetch_errors": fetch_errors})

        # Extractive bullets
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
        debug["stages"].append({"stage": "extractive", "bullets": len(bullets)})

        # LLM polishing
        used_llm = False
        if use_llm and _openai_client:
            llm_bullets = self._llm_summarize(q, pages, max_bullets=5)
            if llm_bullets:
                bullets = llm_bullets
                used_llm = True
        debug["stages"].append({"stage": "llm", "used": used_llm, "final_bullets": len(bullets)})

        # Last resort: titles/snippets
        if not bullets and results:
            for r in results[:3]:
                bullets.append(f"- {r['title']} ({_host(r['url'])}): {r.get('snippet', '')[:300]}")
        debug["stages"].append({"stage": "fallback", "final_bullets_after_fallback": len(bullets)})

        # If still nothing, produce a useful diagnostic bullet
        if not bullets:
            bullets = [
                "- No sources could be retrieved. This may be due to network restrictions in the runtime. "
                "Try a simpler query or a different environment."
            ]

        out = {"query": q, "sources": results, "bullets": bullets, "used_llm": used_llm}
        self._last_debug = debug
        return out
