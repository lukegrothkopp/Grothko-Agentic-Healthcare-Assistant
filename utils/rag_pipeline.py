# utils/rag_pipeline.py
# Simple, dependency-light TF-IDF RAG with strictly Python lists/floats (no NumPy truthiness)

import os
import re
from math import log, sqrt
from typing import List, Tuple, Optional, Dict
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _read_pdf_file(path: str) -> str:
    # Optional PDF support without introducing new deps:
    # If PyPDF2 is present, use it; otherwise skip PDFs gracefully.
    try:
        import PyPDF2  # type: ignore
    except Exception:
        return ""
    try:
        text_parts: List[str] = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    pass
        return "\n".join([t for t in text_parts if t]).strip()
    except Exception:
        return ""


class RAGPipeline:
    """
    Minimal offline RAG:
    - Indexes text-like files under OFFLINE_KB_DIR (default: data/offline_kb).
    - Pure Python (no sklearn/numpy required) to avoid ambiguous truthiness.
    - retrieve(query, k) returns List[Tuple[str, float]] -> (snippet, score)
    """

    def __init__(self, kb_dir: Optional[str] = None):
        self.kb_dir: str = kb_dir or os.getenv("OFFLINE_KB_DIR", "data/offline_kb")
        self.docs: List[str] = []          # raw doc texts
        self.paths: List[str] = []         # file paths (parallel to docs)
        self.doc_weights: List[Dict[str, float]] = []  # tf-idf weight dict per doc
        self.doc_norms: List[float] = []   # L2 norm per doc
        self.idf: Dict[str, float] = {}    # idf per term
        self._build_index()

    # ---------- Indexing ----------
    def _iter_files(self) -> List[str]:
        if not os.path.isdir(self.kb_dir):
            return []
        out: List[str] = []
        for root, _, files in os.walk(self.kb_dir):
            for name in files:
                p = os.path.join(root, name)
                if p.lower().endswith((".txt", ".md", ".markdown", ".rst", ".log", ".cfg", ".ini", ".json", ".csv", ".tsv", ".yml", ".yaml")):
                    out.append(p)
                elif p.lower().endswith(".pdf"):
                    out.append(p)
        return out

    def _load_text(self, path: str) -> str:
        if path.lower().endswith(".pdf"):
            return _read_pdf_file(path)
        return _read_text_file(path)

    def _build_index(self) -> None:
        files = self._iter_files()
        self.docs = []
        self.paths = []
        tokenized_docs: List[List[str]] = []

        for p in files:
            text = self._load_text(p)
            if text and isinstance(text, str):
                self.docs.append(text)
                self.paths.append(p)
                tokenized_docs.append(_tokenize(text))
        N = len(tokenized_docs)

        # Early out: empty index
        if N == 0:
            self.idf = {}
            self.doc_weights = []
            self.doc_norms = []
            return

        # Compute DF and IDF
        df: Counter = Counter()
        for toks in tokenized_docs:
            for term in set(toks):
                df[term] += 1

        self.idf = {t: (log((N + 1.0) / (df[t] + 1.0)) + 1.0) for t in df}

        # Compute doc TF-IDF weights + norms
        self.doc_weights = []
        self.doc_norms = []
        for toks in tokenized_docs:
            tf = Counter(toks)
            w: Dict[str, float] = {}
            for term, cnt in tf.items():
                idf_val = self.idf.get(term, 0.0)
                if idf_val <= 0.0:
                    continue
                w[term] = float(cnt) * float(idf_val)
            norm = sqrt(sum((val * val) for val in w.values())) if w else 0.0
            self.doc_weights.append(w)
            self.doc_norms.append(float(norm))

    # ---------- Retrieval ----------
    def _query_weights(self, query: str) -> Dict[str, float]:
        toks = _tokenize(query)
        if not toks:
            return {}
        tf = Counter(toks)
        w: Dict[str, float] = {}
        for term, cnt in tf.items():
            idf_val = self.idf.get(term, 0.0)
            if idf_val <= 0.0:
                continue
            w[term] = float(cnt) * float(idf_val)
        return w

    @staticmethod
    def _dot(a: Dict[str, float], b: Dict[str, float]) -> float:
        # Iterate over the smaller dict for speed
        if len(a) > len(b):
            a, b = b, a
        s = 0.0
        for term, av in a.items():
            bv = b.get(term)
            if bv is not None:
                s += av * bv
        return float(s)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Return top-k (snippet, score) pairs. No NumPy objects; only Python lists/floats.
        """
        # Guard: empty index or query
        if not isinstance(query, str) or not query.strip() or len(self.docs) == 0:
            return []

        q_w = self._query_weights(query)
        if not q_w:
            return []

        q_norm = sqrt(sum((val * val) for val in q_w.values()))
        if q_norm == 0.0:
            return []

        results: List[Tuple[int, float]] = []
        for i, (dw, dnorm) in enumerate(zip(self.doc_weights, self.doc_norms)):
            if not dw or dnorm == 0.0:
                continue
            dot = self._dot(q_w, dw)
            sim = float(dot) / float(q_norm * dnorm) if (q_norm * dnorm) > 0.0 else 0.0
            if sim > 0.0:
                results.append((i, float(sim)))

        if not results:
            return []

        # Sort by similarity desc
        results.sort(key=lambda x: x[1], reverse=True)
        top = results[: max(1, int(k))]

        # Make short snippets for readability (first ~500 chars)
        out: List[Tuple[str, float]] = []
        for idx, score in top:
            text = self.docs[idx]
            snippet = (text[:500] + "…") if len(text) > 500 else text
            out.append((snippet, float(score)))
        return out
