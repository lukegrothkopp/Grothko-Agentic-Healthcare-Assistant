# core/memory.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading

@dataclass
class MemoryItem:
    text: str
    meta: Dict[str, Any]

class MemoryStore:
    """Simple, ephemeral TF-IDF memory. Safe on empty input."""
    def __init__(self):
        self._items: List[MemoryItem] = []
        self._lock = threading.Lock()
        self._vectorizer = TfidfVectorizer(max_features=4096)
        self._matrix = None  # type: Optional[any]

    def add(self, text: str, **meta):
        """Ignore empty/whitespace-only entries to avoid empty vocabulary errors."""
        if not text or not str(text).strip():
            return
        with self._lock:
            self._items.append(MemoryItem(text=str(text).strip(), meta=meta))
            self._rebuild()

    def _rebuild(self):
        # Keep only non-empty strings
        corpus = [it.text.strip() for it in self._items if it.text and it.text.strip()]
        if not corpus:
            # Nothing meaningful to index
            self._matrix = None
            return
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int = 5) -> List[MemoryItem]:
        if not self._items or self._matrix is None or not query or not str(query).strip():
            return []
        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        idxs = sims.argsort()[::-1][:k]
        # Map indices back to the compacted corpus; since _items may include empties filtered out,
        # rebuild the mapping once here:
        non_empty_items = [it for it in self._items if it.text and it.text.strip()]
        return [non_empty_items[i] for i in idxs] if non_empty_items else []

    def dump(self) -> List[MemoryItem]:
        return list(self._items)
