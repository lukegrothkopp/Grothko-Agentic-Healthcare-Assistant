# generate_faiss_index.py
import os
import faiss
import numpy as np
from pathlib import Path

# Embeddings (new API first, legacy fallback)
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    from langchain.embeddings.openai import OpenAIEmbeddings  # legacy

# Text splitters (new package first, legacy fallback)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter as TextSplitter
except Exception:
    try:
        from langchain.text_splitter import CharacterTextSplitter as TextSplitter
    except Exception:
        TextSplitter = None

from langchain.docstore.document import Document

KB_DIR = Path(os.environ.get("OFFLINE_KB_DIR", "data/medical_kb"))
OUT_PATH = Path("vector_store/faiss_index.bin")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _iter_kb_texts(dir_path: Path):
    if not dir_path.exists():
        return []
    texts = []
    for p in dir_path.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".txt", ".md"}:
            try:
                texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
    return texts


def generate_index(api_key: str | None = None) -> str:
    """Build a FAISS index from OFFLINE_KB_DIR and save to vector_store/faiss_index.bin."""
    if TextSplitter is None:
        raise RuntimeError(
            "No text splitter available. Install `langchain-text-splitters` "
            "or use a LangChain version that exports CharacterTextSplitter."
        )

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    texts = _iter_kb_texts(KB_DIR)
    if not texts:
        raise RuntimeError(f"No plaintext files found under {KB_DIR.resolve()}")

    splitter = TextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = []
    for t in texts:
        for chunk in splitter.split_text(t):
            docs.append(Document(page_content=chunk))

    embedder = OpenAIEmbeddings()
    vecs = embedder.embed_documents([d.page_content for d in docs])

    arr = np.array(vecs, dtype=np.float32)
    dim = arr.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(arr)

    faiss.write_index(index, str(OUT_PATH))
    return str(OUT_PATH)
