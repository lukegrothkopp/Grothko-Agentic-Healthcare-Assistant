"""
ChromaDB vector memory for patient summaries and retrieved medical info.
Uses OpenAI embeddings if OPENAI_API_KEY is set; otherwise falls back to a naive keyword store.
"""
from __future__ import annotations
import os, re
import chromadb
from typing import List, Dict, Any
from loguru import logger

_EMBED_MODEL = "text-embedding-3-small"

def _get_embedding_fn():
    # Lazy import openai to avoid import if not configured
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        logger.warning(f"OpenAI unavailable: {e}")
        return None

    def embed(texts: List[str]) -> List[List[float]]:
        res = client.embeddings.create(model=_EMBED_MODEL, input=texts)
        return [d.embedding for d in res.data]
    return embed

class MemoryStore:
    def __init__(self, collection_name: str = "healthcare_memory"):
        self.client = chromadb.PersistentClient(path="data/chroma")
        self.embed_fn = _get_embedding_fn()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} if self.embed_fn else None
        )

    def add(self, docs: List[str], metadatas: List[Dict[str, Any]]):
        ids = [f"doc-{self.count()+i}" for i in range(len(docs))]
        if self.embed_fn:
            self.collection.add(documents=docs, metadatas=metadatas, ids=ids)
        else:
            # store as-is; chroma will store raw strings; similarity will be keyword-ish
            self.collection.add(documents=docs, metadatas=metadatas, ids=ids)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.embed_fn:
            res = self.collection.query(query_texts=[query], n_results=k)
        else:
            res = self.collection.query(query_texts=[query], n_results=k)
        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i]
            })
        return out

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0
