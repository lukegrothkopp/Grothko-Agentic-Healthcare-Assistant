import os
import glob
import numpy as np
import faiss
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings

class RAGPipeline:
    def __init__(self, kb_dir: str = "data/medical_kb", index_path: str = "vector_store/faiss_index.bin"):
        self.kb_dir = kb_dir
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings()
        self.index = None
        self.docs: List[str] = []
        self._load_or_build()

    def _load_or_build(self):
        if os.path.exists(self.index_path) and os.path.getsize(self.index_path) > 0:
            try:
                self.index = faiss.read_index(self.index_path)
                # Load docs alongside index
                docs_path = self.index_path + ".docs.txt"
                if os.path.exists(docs_path):
                    with open(docs_path, "r", encoding="utf-8") as f:
                        self.docs = [line.rstrip("\n") for line in f]
                return
            except Exception:
                pass
        self._build_from_kb()

    def _build_from_kb(self):
        # Ingest all .txt files from KB
        texts: List[str] = []
        for fp in glob.glob(os.path.join(self.kb_dir, "*.txt")):
            with open(fp, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    texts.append(txt)
        if not texts:
            self.index = None
            self.docs = []
            return
        # Embed full docs (simple for demo; chunking optional)
        vecs = self.embeddings.embed_documents(texts)
        mat = np.array(vecs, dtype=np.float32)
        dim = mat.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(mat)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(index, self.index_path)
        with open(self.index_path + ".docs.txt", "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t.replace("\n", " ") + "\n")
        self.index = index
        self.docs = [t.replace("\n", " ") for t in texts]

    def is_ready(self) -> bool:
        return self.index is not None and len(self.docs) > 0

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if not self.is_ready():
            return []
        qv = np.array([self.embeddings.embed_query(query)], dtype=np.float32)
        D, I = self.index.search(qv, min(k, len(self.docs)))
        idxs = I[0].tolist()
        ds = D[0].tolist()
        return [(self.docs[i], float(ds[j])) for j, i in enumerate(idxs)]
