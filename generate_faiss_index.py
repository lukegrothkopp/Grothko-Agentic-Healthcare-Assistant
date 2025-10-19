# generate_faiss_index.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Union
from dotenv import load_dotenv

# --- Vector store (robust import) ---
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    # legacy fallback
    from langchain.vectorstores import FAISS  # type: ignore

# --- OpenAI embeddings (robust import) ---
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    from langchain.embeddings.openai import OpenAIEmbeddings  # type: ignore

# --- Text splitters (robust imports across versions) ---
Splitter = None
try:
    # Preferred new package
    from langchain_text_splitters import RecursiveCharacterTextSplitter as Splitter  # type: ignore
except Exception:
    try:
        # Newer LangChain core exports
        from langchain.text_splitter import RecursiveCharacterTextSplitter as Splitter  # type: ignore
    except Exception:
        try:
            # Very old name
            from langchain.text_splitter import CharacterTextSplitter as Splitter  # type: ignore
        except Exception:
            Splitter = None


def _iter_text_files(kb_dir: str) -> Iterable[str]:
    """
    Yield raw text from simple, common file types. Keep it lightweight to avoid extra deps.
    """
    kb = Path(kb_dir)
    if not kb.exists():
        return
    patterns = ("*.txt", "*.md", "*.json")
    for pat in patterns:
        for fp in kb.rglob(pat):
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if fp.suffix.lower() == ".json":
                # Try to pretty-print JSON so it’s semantically chunkable
                try:
                    obj = json.loads(txt)
                    if isinstance(obj, (dict, list)):
                        yield json.dumps(obj, ensure_ascii=False, indent=2)
                        continue
                except Exception:
                    pass
            yield txt


def generate_index(api_key: str | None = None,
                   kb_dir: str | None = None,
                   out_dir: str = "vector_store") -> str:
    """
    Build a FAISS index from OFFLINE_KB_DIR (or provided kb_dir) using OpenAI embeddings.
    Returns the path to the created index, preferring a single .bin file (for your console).
    """
    load_dotenv()

    # Ensure key is in env for OpenAIEmbeddings
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY not set or invalid.")

    # Decide KB dir
    kb_dir = kb_dir or os.getenv("OFFLINE_KB_DIR", "data/offline_kb")
    if not Path(kb_dir).exists():
        raise FileNotFoundError(f"KB directory not found: {kb_dir}")

    # Splitter availability
    if Splitter is None:
        raise ImportError(
            "No text splitter available. Install `langchain-text-splitters` or use a LangChain version that "
            "exports RecursiveCharacterTextSplitter/CharacterTextSplitter."
        )

    # Load and chunk text
    raw_texts = list(_iter_text_files(kb_dir))
    if not raw_texts:
        raise RuntimeError(f"No .txt/.md/.json files found under {kb_dir}")

    splitter = Splitter(chunk_size=1000, chunk_overlap=150)
    chunks: List[str] = []
    for t in raw_texts:
        pieces = splitter.split_text(t)
        chunks.extend([p for p in pieces if p.strip()])

    if not chunks:
        raise RuntimeError("No chunks generated from KB (check file encodings/contents).")

    # Build vector store
    embeddings = OpenAIEmbeddings()  # reads key from env
    vs = FAISS.from_texts(chunks, embeddings)

    # Save as both LangChain-local dir and a single .bin (to match your console diagnostics)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # LangChain’s native format (directory with index/faiss + pkl)
    vs.save_local(str(out_path / "faiss_index"))

    # Also write a single .bin for your existing check
    try:
        import faiss, pickle  # type: ignore
        faiss.write_index(vs.index, str(out_path / "faiss_index.bin"))
        with open(out_path / "faiss_store.pkl", "wb") as f:
            # Persist docstore (ids -> Document) so you can reload manually if ever needed
            pickle.dump(vs.docstore._dict, f)
        return str(out_path / "faiss_index.bin")
    except Exception:
        # If faiss python bindings aren’t available, fall back to the directory path
        return str(out_path / "faiss_index")


if __name__ == "__main__":
    print(generate_index())
