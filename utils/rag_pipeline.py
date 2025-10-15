# utils/rag_pipeline.py
# Pure-Python TF-IDF RAG with diagnostics, backend label, PDF/DOCX optional support,
# and simple medical abbreviation expansions to improve recall (e.g., "CKD").

import os
import re
from math import log, sqrt
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter

# ---------------------------
# Tokenization & expansions
# ---------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# Common medical abbreviations -> expanded tokens (very light-weight)
_ABBREV_EXPANSIONS: Dict[str, List[str]] = {
    "ckd": ["chronic", "kidney", "disease"],
    "htn": ["hypertension"],
    "dm": ["diabetes", "mellitus"],
    "copd": ["chronic", "obstructive", "pulmonary", "disease"],
    "cad": ["coronary", "artery", "disease"],
    "mi": ["myocardial", "infarction"],
    "afib": ["atrial", "fibrillation"],
}

def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]

def _expand_tokens(tokens: List[str]) -> List[str]:
    """
    If a token is a known abbreviation (e.g., 'ckd'), append its expansions
    so doc/query vocabularies overlap more often.
    """
    out: List[str] = []
    for t in tokens:
        out.append(t)
        if t in _ABBREV_EXPANSIONS:
            out.extend(_ABBREV_EXPANSIONS[t])
    return out


# ---------------------------
# Light file readers
# ---------------------------

def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def _read_pdf_file(path: str) -> str:
    """Optional PDF support if PyPDF2 is available; otherwise returns empty."""
    try:
        import PyPDF2  # type: ignore
    except Exception:
        return ""
    try:
        parts: List[str] = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    pass
        return "\n".join([p for p in parts if p]).strip()
    except Exception:
        return ""

def _read_docx_file(path: str) -> str:
    """Optional DOCX support if python-docx OR docx2txt is available."""
    # Try python-docx first
    try:
        import docx  # type: ignore
        try:
            doc = docx.Document(path)
            return "\n".join([p.text for p in doc.paragraphs if p.text]).strip()
        except Exception:
            pass
    except Exception:
        pass
    # Try docx2txt as fallback
    try:
        import docx2txt  # type: ignore
        try:
            return (docx2txt.process(path) or "").strip()
        except Exception:
            pass
    except Exception:
        pass
    return ""


class RAGPipeline:
    """
    Minimal offline RAG:
    - Indexes text-like files under OFFLINE_KB_DIR (default: data/offline_kb).
    - Pure Python (no sklearn/numpy required) to avoid ambiguous truthiness.
    - retrieve(query, k) returns List[Tuple[str, float]] -> (snippet, score)
    - Provides diagnostics and backend label for the Developer Console.
    """

    def __init__(self, kb_dir: Optional[str] = None):
        # Public labels read by the Developer Console
        self.backend: str = "tfidf-python"
        self.backend_name: str = "tfidf-python"
        self.backend_label: str = "tfidf-python"

        self.kb_dir: str = kb_dir or os.getenv("OFFLINE_KB_DIR", "data/offline_kb")
        self.docs: List[str] = []           # raw doc texts
        self.paths: List[str] = []          # file paths (parallel to docs)
        self.doc_weights: List[Dict[str, float]] = []  # tf-idf weight dict per doc
        self.doc_norms: List[float] = []    # L2 norm per doc
        self.idf: Dict[str, float] = {}     # idf per term
        self._build_index()

    # ---------- Public utilities ----------
    def set_kb_dir(self, kb_dir: str) -> None:
        """Change KB directory and rebuild index."""
        self.kb_dir = kb_dir
        self._build_index()

    def rebuild_index(self) -> None:
        """Rebuild the index from current kb_dir."""
        self._build_index()

    def status(self) -> Dict[str, Any]:
        """Return diagnostics for UI/console."""
        exts: Counter = Counter()
        if os.path.isdir(self.kb_dir):
            for root, _, files in os.walk(self.kb_dir):
                for name in files:
                    _, ext = os.path.splitext(name)
                    exts[ext.lower()] += 1
        return {
            "backend": self.backend_label,
            "kb_dir": self.kb_dir,
            "kb_exists": os.path.isdir(self.kb_dir),
            "num_docs": len(self.docs),
            "file_type_counts": dict(exts),
        }

    # ---------- Indexing ----------
    def _iter_files(self) -> List[str]:
        if not os.path.isdir(self.kb_dir):
            return []
        out: List[str] = []
        for root, _, files in os.walk(self.kb_dir):
            for name in files:
                p = os.path.join(root, name)
                low = p.lower()
                if low.endswith((".txt", ".md", ".markdown", ".rst", ".log", ".cfg", ".ini",
                                 ".json", ".csv", ".tsv", ".yml", ".yaml")):
                    out.append(p)
                elif low.endswith(".pdf"):
                    out.append(p)
                elif low.endswith(".docx"):
                    out.append(p)
        return out

    def _load_text(self, path: str) -> str:
        low = path.lower()
        if low.endswith(".pdf"):
            return _read_pdf_file(path)
        if low.endswith(".docx"):
            return _read_docx_file(path)
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
                toks = _tokenize(text)
                toks = _expand_tokens(toks)
                tokenized_docs.append(toks)

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
        toks = _expand_tokens(toks)
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
        Return top-k (snippet, score) pairs. Uses only Python lists/floats.
        """
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

        # Make short snippets for readability (first ~600 chars)
        out: List[Tuple[str, float]] = []
        for idx, score in top:
            text = self.docs[idx]
            snippet = (text[:600] + "…") if len(text) > 600 else text
            out.append((snippet, float(score)))
        return out

