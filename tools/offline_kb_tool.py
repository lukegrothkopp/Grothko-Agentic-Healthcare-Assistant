# tools/offline_kb_tool.py
# Offline KB tools: (a) query, (b) diagnostics for the Developer Console.

from typing import List, Tuple
from langchain.tools import Tool
from utils.rag_pipeline import RAGPipeline

_PIPE = RAGPipeline()

def _offline_kb_query(q: str) -> str:
    try:
        pairs: List[Tuple[str, float]] = _PIPE.retrieve(q, k=3)
        if len(pairs) == 0:
            st = _PIPE.status()
            hint = f"(KB dir: {st.get('kb_dir')}, docs: {st.get('num_docs')})"
            return "No local KB results.\n" + hint
        lines = []
        for txt, score in pairs:
            txt = str(txt) if txt is not None else ""
            lines.append(f"- {txt[:400]} (score={float(score):.3f})")
        return "\n".join(lines).strip() or "No local KB results."
    except Exception as e:
        return f"(Offline KB error) {e}"

def _offline_kb_diagnostics(_: str = "") -> str:
    st = _PIPE.status()
    lines = [
        f"Backend: {st.get('backend')}",
        f"KB Dir: {st.get('kb_dir')}",
        f"Exists: {st.get('kb_exists')}",
        f"Docs Indexed: {st.get('num_docs')}",
        f"File Types: {st.get('file_type_counts')}",
    ]
    return "\n".join(lines)

def get_offline_kb_tool() -> Tool:
    return Tool(
        name="Offline KB Query",
        func=_offline_kb_query,
        description="Query the local knowledge base (no network). Returns a plain-text bulleted list."
    )

def get_offline_kb_diag_tool() -> Tool:
    return Tool(
        name="Offline KB Diagnostics",
        func=_offline_kb_diagnostics,
        description="Show backend, KB dir, existence, doc count, and file-type counts."
    )
