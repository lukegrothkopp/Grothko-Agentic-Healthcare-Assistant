from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading

@dataclass
class MemoryItem:
    text: str
    meta: Dict[str, Any]

class MemoryStore:
    """Simple, ephemeral TF IDF memory. Persist externally if needed."""

    def __init__(self):
        self._items: List[MemoryItem] = []
        self._lock = threading.Lock()
        self._vectorizer = TfidfVectorizer(max_features=4096)
        self._matrix = None

    def add(self, text: str, **meta):
        with self._lock:
            self._items.append(MemoryItem(text=text, meta=meta))
            self._rebuild()

    def _rebuild(self):
        corpus = [it.text for it in self._items] or [""]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int = 5) -> List[MemoryItem]:
        if not self._items:
            return []
        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        idxs = sims.argsort()[::-1][:k]
        return [self._items[i] for i in idxs]

    def dump(self) -> List[MemoryItem]:
        return list(self._items)
