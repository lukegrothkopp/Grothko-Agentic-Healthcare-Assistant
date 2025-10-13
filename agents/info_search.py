from __future__ import annotations
from typing import Dict, Any, List
import os, requests

from core.memory import MemoryStore

class InfoSearchAgent:
    """
    Searches trusted medical info using Bing Web Search (optional) and stores results in vector memory.
    """
    def __init__(self, memory: MemoryStore):
        self.memory = memory
        self.bing_key = os.getenv("BING_API_KEY")
        self.BING_URL = "https://api.bing.microsoft.com/v7.0/search"

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        docs = []
        if self.bing_key:
            headers = {"Ocp-Apim-Subscription-Key": self.bing_key}
            params = {"q": query + " site:who.int OR site:nih.gov OR site:medlineplus.gov", "count": k}
            resp = requests.get(self.BING_URL, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            js = resp.json()
            for it in js.get("webPages", {}).get("value", []):
                docs.append({"title": it.get("name"), "url": it.get("url"), "snippet": it.get("snippet")})
        else:
            # Fallback demo content
            docs = [
                {"title": "CKD overview (demo)", "url": "https://medlineplus.gov/kidneydiseases.html",
                 "snippet": "Chronic kidney disease involves gradual loss of kidney function. Management includes BP control, RAAS inhibitors, lifestyle changes."}
            ]
        # push to memory
        texts = [f"{d['title']}\n{d['snippet']}\n{d['url']}" for d in docs]
        metas = [{"source": d["url"], "type": "medical_info"} for d in docs]
        if texts:
            self.memory.add(texts, metas)
        return docs
