from __future__ import annotations
import os, requests
from typing import Dict, Any, List

class InfoSearchAgent:
    def __init__(self, memory):
        self.memory = memory
        self.serp_key = os.getenv("SERPAPI_API_KEY")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        try:
            if self.serp_key:
                r = requests.get(
                    "https://serpapi.com/search.json",
                    params={"engine": "google", "q": query, "num": k, "api_key": self.serp_key},
                    timeout=15,
                )
                r.raise_for_status()
                js = r.json()
                for it in js.get("organic_results", [])[:k]:
                    docs.append({
                        "title": it.get("title"),
                        "url": it.get("link"),
                        "snippet": it.get("snippet")
                    })
            elif self.bing_key:
                headers = {"Ocp-Apim-Subscription-Key": self.bing_key}
                r = requests.get(self.BING_URL, params={"q": query, "count": k}, headers=headers, timeout=15)
                r.raise_for_status()
                js = r.json()
                for it in js.get("webPages", {}).get("value", []):
                    docs.append({
                        "title": it.get("name"),
                        "url": it.get("url"),
                        "snippet": it.get("snippet")
                    })
            else:
                docs = [{
                    "title": "CKD overview (demo)",
                    "url": "https://medlineplus.gov/kidneydiseases.html",
                    "snippet": "Chronic kidney disease overview and treatment basics."
                }]
        except Exception:
            pass

        # Store results in vector memory
        if docs:
            texts = [f"{d['title']}\n{d['snippet']}\n{d['url']}" for d in docs]
            metas = [{"source": d["url"], "type": "medical_info"} for d in docs]
            self.memory.add(texts, metas)
        return docs
