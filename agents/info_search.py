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

# ---------------------- OFFLINE MINI-KB (last resort) ----------------------
# Note: These are concise, high-level bullets with source attributions.
# Links are included for when network becomes available.
LAST_RESORT_KB: Dict[str, List[str]] = {
    "lung cancer": [
        "- Lung cancer is broadly categorized into non–small cell lung cancer (NSCLC, ~80–85%) and small cell lung cancer (SCLC, ~15%), which differ in growth patterns and treatment approaches. (NIH / NCI: cancer.gov)",
        "- Major risk factors include tobacco smoking, secondhand smoke, radon exposure, occupational carcinogens (e.g., asbestos), and air pollution. (CDC: cdc.gov)",
        "- Diagnosis typically involves imaging (chest CT), followed by tissue confirmation (biopsy) and staging (TNM for NSCLC; limited vs. extensive for SCLC). (NCI / NCCN overview: cancer.gov, cdc.gov)",
        "- Treatment options depend on stage and subtype: surgery, radiation, chemotherapy, targeted therapies (e.g., EGFR, ALK, ROS1, BRAF, MET, KRAS G12C), and immunotherapy (e.g., PD-1/PD-L1 inhibitors). (NCI / NICE / Mayo Clinic: cancer.gov, nice.org.uk, mayoclinic.org)",
        "- Supportive care includes smoking cessation, symptom management, and multidisciplinary coordination; screening with low-dose CT is recommended for certain high-risk adults. (USPSTF/CDC summary: cdc.gov, mayoclinic.org)",
    ],
    "chronic kidney disease": [
        "- CKD is defined by persistent kidney damage or reduced eGFR (<60) over ≥3 months; common causes include diabetes and hypertension. (NIH / NIDDK: niddk.nih.gov)",
        "- Management prioritizes BP and glucose control, ACEi/ARB, SGLT2 inhibitors, and risk-based referral to nephrology; monitor albuminuria and eGFR. (NICE / KDIGO overview: nice.org.uk, nih.gov)",
        "- Lifestyle measures (salt restriction, exercise, smoking cessation) and vaccine updates reduce complications. (CDC / NIDDK: cdc.gov, niddk.nih.gov)",
    ],
    "diabetes": [
        "- Diabetes mellitus includes type 1 (autoimmune β-cell loss) and type 2 (insulin resistance/relative deficiency); diagnosis uses A1c, fasting plasma glucose, or OGTT thresholds. (CDC / NIH: cdc.gov, nih.gov)",
        "- Management includes lifestyle modification, metformin first-line for T2D, and additional agents (GLP-1 RAs, SGLT2 inhibitors, insulin) based on comorbidities and A1c goals. (NICE / NIH: nice.org.uk, nih.gov)",
        "- Complication screening: eyes, kidneys, feet, BP and lipid control; individualized targets. (CDC / NIH: cdc.gov, nih.gov)",
    ],
    "hypertension": [
        "- Hypertension is persistently elevated BP; confirm with repeated or ambulatory readings; lifestyle changes are first-line. (CDC / NICE: cdc.gov, nice.org.uk)",
        "- First-line meds often include thiazide-like diuretics, ACEi/ARB, or calcium-channel blockers; targets vary by risk/comorbidity. (NICE / NIH: nice.org.uk, nih.gov)",
    ],
    "asthma": [
        "- Asthma is a chronic inflammatory airway disease with variable airflow obstruction and symptoms like wheeze, cough, and dyspnea. (NIH / NHLBI: nhlbi.nih.gov)",
        "- Controller therapy (inhaled corticosteroids ± LABA) reduces exacerbations; trigger avoidance and action plans are key. (NIH / CDC: nhlbi.nih.gov, cdc.gov)",
    ],
    "influenza": [
        "- Influenza is a contagious respiratory illness; annual vaccination is recommended for most people ≥6 months old. (CDC: cdc.gov)",
        "- Antivirals (e.g., oseltamivir) may reduce severity/duration when started early in high-risk or severe cases. (CDC / NIH: cdc.gov, nih.gov)",
    ],
    "covid-19": [
        "- COVID-19 is caused by SARS-CoV-2; vaccination and boosters reduce severe disease. (CDC / WHO: cdc.gov, who.int)",
        "- High-risk patients may benefit from antivirals per current guidance; recommendations evolve with variants. (CDC / NIH: cdc.gov, nih.gov)",
    ],
    "heart failure": [
        "- HF is a clinical syndrome from structural/functional cardiac disorders; assess ejection fraction (HFrEF vs HFpEF). (NIH: nih.gov)",
        "- Guideline-directed therapy (ACEi/ARB/ARNI, β-blockers, MRA, SGLT2 inhibitors) improves outcomes in HFrEF; comorbidity management is essential. (NICE / NIH: nice.org.uk, nih.gov)",
    ],
}

class InfoSearchAgent:
    def __init__(self):
        self._last_debug: Dict[str, Any] = {}

    def get_debug(self) -> Dict[str, Any]:
        return dict(self._last_debug)

    # ---------------------- Search ----------------------
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
        trusted_q = (
            f"{query} (site:who.int OR site:cdc.gov OR site:nih.gov OR "
            f"site:medlineplus.gov OR site:mayoclinic.org OR site:cancer.gov OR site:nice.org.uk)"
        )
        hits = self._ddg_text(trusted_q, max_results=max_results)
        if not hits:
            generic = self._ddg_text(query, max_results=max_results * 3)
            trusted_generic = [g for g in generic if any(dom in g["url"] for dom in TRUSTED)]
            hits = (trusted_generic[:max_results]) if trusted_generic else (generic[:max_results])
        if not hits:
            hits = self._ddg_news(query, max_results=max_results)
        if not hits:
            hits = self._ddg_answers(query, max_results=max_results)
        return hits[:max_results]

    # ---------------------- Fetch & slice ----------------------
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

    # ---------------------- Summaries ----------------------
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

    # ---------------------- Public API ----------------------
    def query(self, q: str, use_llm: bool = False) -> Dict:
        debug: Dict[str, Any] = {"query": q, "stages": []}

        # 1) Live search (may be blocked in your environment)
        results = self._search(q, max_results=8)
        debug["stages"].append({"stage": "search", "results": len(results), "urls": [r["url"] for r in results]})

        # 2) Fetch and extract
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

        bullets: List[str] = []
        for p in pages:
            bullet = self._extract_bullets(
                text=p["body"], query=q, fallback_snippet=p.get("snippet", ""), title=p["title"], url=p["url"]
            )
            if bullet:
                bullets.append(bullet)
            if len(bullets) >= 5:
                break
        debug["stages"].append({"stage": "extractive", "bullets": len(bullets)})

        # 3) LLM polishing (if key available)
        used_llm = False
        if use_llm and _openai_client and pages:
            llm_bullets = self._llm_summarize(q, pages, max_bullets=5)
            if llm_bullets:
                bullets = llm_bullets
                used_llm = True
        debug["stages"].append({"stage": "llm", "used": used_llm, "final_bullets": len(bullets)})

        # 4) If still nothing and live search failed, use OFFLINE KB
        if not bullets:
            key = q.strip().lower()
            # try exact match, then simple normalization
            aliases = {
                "lung carcinoma": "lung cancer",
                "nsclc": "lung cancer",
                "sclc": "lung cancer",
                "ckd": "chronic kidney disease",
                "covid": "covid-19",
                "covid 19": "covid-19",
            }
            if key in aliases:
                key = aliases[key]
            if key in LAST_RESORT_KB:
                bullets = LAST_RESORT_KB[key][:5]
                debug["stages"].append({"stage": "offline_kb", "used": True, "topic": key, "bullets": len(bullets)})
            else:
                # simple fuzzy pick: find any KB topic contained in query
                match = next((k for k in LAST_RESORT_KB.keys() if k in key), None)
                if match:
                    bullets = LAST_RESORT_KB[match][:5]
                    debug["stages"].append({"stage": "offline_kb", "used": True, "topic": match, "bullets": len(bullets)})

        # 5) Absolute last resort (shouldn’t happen now)
        if not bullets:
            bullets = [
                "- No sources could be retrieved (network likely blocked), and the query didn't match the offline mini-KB. "
                "Try a different term (e.g., 'diabetes', 'hypertension', 'asthma', 'heart failure')."
            ]

        out = {"query": q, "sources": results, "bullets": bullets, "used_llm": used_llm}
        self._last_debug = debug
        return out
