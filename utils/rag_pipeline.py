import os, glob
from typing import List, Tuple
import numpy as np

class RAGPipeline:
    """
    Prefers OpenAI embeddings + FAISS if OPENAI_API_KEY is present.
    Falls back to a lightweight TF-IDF retriever (no network) when no key is set.
    """
    def __init__(self, kb_dir: str = "data/medical_kb", index_path: str = "vector_store/faiss_index.bin"):
        self.kb_dir = kb_dir
        self.index_path = index_path
        self.backend = "openai" if os.getenv("OPENAI_API_KEY") else "tfidf"
        self.docs: List[str] = []

        if self.backend == "openai":
            try:
                import faiss  # type: ignore
                from langchain_openai import OpenAIEmbeddings
                self.faiss = faiss
                self.emb = OpenAIEmbeddings()
                self.index = None
                self._load_or_build_faiss()
            except Exception:
                # If anything fails (e.g., no key), drop to TF-IDF
                self.backend = "tfidf"
                self._build_tfidf()
        else:
            self._build_tfidf()

    # ---------- OpenAI + FAISS path ----------
    def _load_or_build_faiss(self):
        if os.path.exists(self.index_path) and os.path.getsize(self.index_path) > 0:
            try:
                self.index = self.faiss.read_index(self.index_path)
                docs_path = self.index_path + ".docs.txt"
                if os.path.exists(docs_path):
                    with open(docs_path, "r", encoding="utf-8") as f:
                        self.docs = [line.rstrip("\n") for line in f]
                return
            except Exception:
                pass
        self._build_from_kb_faiss()

    def _build_from_kb_faiss(self):
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
        vecs = self.emb.embed_documents(texts)
        mat = np.array(vecs, dtype=np.float32)
        dim = mat.shape[1]
        index = self.faiss.IndexFlatL2(dim)
        index.add(mat)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.faiss.write_index(index, self.index_path)
        with open(self.index_path + ".docs.txt", "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t.replace("\n", " ") + "\n")
        self.index = index
        self.docs = [t.replace("\n", " ") for t in texts]

    # ---------- TF-IDF fallback ----------
    def _build_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self.TfidfVectorizer = TfidfVectorizer
        self.cosine_similarity = cosine_similarity
        texts: List[str] = []
        for fp in glob.glob(os.path.join(self.kb_dir, "*.txt")):
            with open(fp, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    texts.append(txt)
        self.docs = [t.replace("\n", " ") for t in texts]
        if not self.docs:
            self.vec = None
            self.mat = None
            return
        self.vec = self.TfidfVectorizer(stop_words="english").fit(self.docs)
        self.mat = self.vec.transform(self.docs)

    def is_ready(self) -> bool:
        if self.backend == "openai":
            return self.index is not None and len(self.docs) > 0
        return self.mat is not None and len(self.docs) > 0

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if not self.is_ready():
            return []
        if self.backend == "openai":
            qv = np.array([self.emb.embed_query(query)], dtype=np.float32)
            D, I = self.index.search(qv, min(k, len(self.docs)))
            idxs = I[0].tolist()
            ds = D[0].tolist()
            return [(self.docs[i], float(ds[j])) for j, i in enumerate(idxs)]
        # TF-IDF
        qv = self.vec.transform([query])
        sims = self.cosine_similarity(qv, self.mat)[0]
        idxs = sims.argsort()[::-1][:min(k, len(self.docs))]
        return [(self.docs[i], float(sims[i])) for i in idxs]
