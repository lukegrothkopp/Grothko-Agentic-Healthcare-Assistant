# generate_faiss_index.py
import os
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

def generate_index(
    api_key: str | None = None,
    kb_dir: str = "data/medical_kb",
    out_path: str = "vector_store/faiss_index.bin",
):
    """Generates and saves a FAISS index from the medical knowledge base."""
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key.startswith("sk-"):
        raise RuntimeError("No valid OPENAI_API_KEY provided to generate_index().")

    # Ensure output dir exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load and chunk all .txt docs from the KB
    texts: list[str] = []
    for fn in os.listdir(kb_dir):
        if not fn.endswith(".txt"):
            continue
        fp = os.path.join(kb_dir, fn)
        with open(fp, "r", encoding="utf-8") as f:
            raw = f.read()
        chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_text(raw)
        texts.extend(chunks)

    if not texts:
        raise RuntimeError(f"No .txt documents found in {kb_dir}")

    # Embed and build FAISS index
    emb = OpenAIEmbeddings(api_key=key)
    vecs = emb.embed_documents(texts)
    mat = np.array(vecs, dtype=np.float32)
    index = faiss.IndexFlatL2(mat.shape[1])
    index.add(mat)

    # Persist
    faiss.write_index(index, out_path)
    with open(out_path + ".docs.txt", "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")

    return out_path
