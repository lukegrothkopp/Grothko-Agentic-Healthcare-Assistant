# tools/offline_kb_tool.py
# Offline KB tool wrapper that always returns a STRING (never arrays)

from typing import List, Tuple
from langchain.tools import Tool
from utils.rag_pipeline import RAGPipeline

_PIPE = RAGPipeline()

def _offline_kb_query(q: str) -> str:
    try:
        pairs: List[Tuple[str, float]] = _PIPE.retrieve(q, k=3)
        if len(pairs) == 0:
            return "No local KB results."
        lines = []
        for txt, score in pairs:
            # Ensure pure str/float formatting
            txt = str(txt) if txt is not None else ""
            try:
                sc = float(score)
            except Exception:
                sc = 0.0
            lines.append(f"- {txt[:400]} (score={sc:.3f})")
        return "\n".join(lines).strip() or "No local KB results."
    except Exception as e:
        # Ensure stringified error; never raise to the graph
        return f"(Offline KB error) {e}"

def get_offline_kb_tool() -> Tool:
    return Tool(
        name="Offline KB Query",
        func=_offline_kb_query,
        description="Query the local knowledge base (no network). Returns a plain-text bulleted list."
    )

