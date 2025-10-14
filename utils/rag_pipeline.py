import os, glob
import numpy as np

# If FAISS index is present + OpenAI key, use that; else TF-IDF fallback.
class RAGPipeline:
    def __init__(self, medical_kb_path: str = "data/medical_kb"):
        self.kb_path = medical_kb_path
        self.backend = "tfidf"
        self._init_backend()

    def _init_backend(self):
        # Prefer FAISS if index and key exist
        key = os.getenv("OPENAI_API_KEY", "").strip()
        has_faiss = os.path.exists("vector_store/faiss_index.bin")
        if key.startswith("sk-") and has_faiss:
            try:
                import faiss
                from langchain_openai import OpenAIEmbeddings
                self.emb = OpenAIEmbeddings(api_key=key)
                self.index = faiss.read_index("vector_store/faiss_index.bin")
                self.docs = self._load_docs_for_faiss_labels()
                self.backend = "openai"  # signals FAISS+embeddings
                return
            except Exception:
                pass
        # Fallback: TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.texts = self._load_all_texts()
        self.mat = self.vectorizer.fit_transform(self.texts) if self.texts else None
        self.backend = "tfidf"

    def _load_all_texts(self):
        texts = []
        for fp in glob.glob(os.path.join(self.kb_path, "*.txt")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            except Exception:
                continue
        return texts

    def _load_docs_for_faiss_labels(self):
        # optional: show doc lines alongside FAISS output (for developer probe UI)
        docs_txt = "vector_store/faiss_index.bin.docs.txt"
        if os.path.exists(docs_txt):
            with open(docs_txt, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines()]
        # fallback to KB texts
        return self._load_all_texts()

    def retrieve(self, query: str, k: int = 3):
        if self.backend == "openai":
            # FAISS distance: smaller is better
            qv = self.emb.embed_query(query)
            D, I = self.index.search(np.array([qv], dtype=np.float32), k)
            out = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self.docs): 
                    continue
                out.append((self.docs[idx], float(dist)))
            return out

        # TF-IDF cosine similarity: larger is better
        if not getattr(self, "mat", None) or self.mat.shape[0] == 0:
            return []
        qv = self.vectorizer.transform([query])
        sims = (self.mat @ qv.T).toarray().ravel()
        idxs = sims.argsort()[::-1][:k]
        return [(self.texts[i], float(sims[i])) for i in idxs if sims[i] > 0.0]
