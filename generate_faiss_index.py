import os, glob
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings

def generate_index(kb_dir="data/medical_kb", out_path="vector_store/faiss_index.bin"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    texts = []
    for fp in glob.glob(os.path.join(kb_dir, "*.txt")):
        with open(fp, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if txt:
                texts.append(txt)
    if not texts:
        print("No KB docs found.")
        return
    emb = OpenAIEmbeddings()
    vecs = emb.embed_documents(texts)
    mat = np.array(vecs, dtype=np.float32)
    index = faiss.IndexFlatL2(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, out_path)
    with open(out_path + ".docs.txt", "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")
    print("FAISS index saved to", out_path)

if __name__ == "__main__":
    generate_index()
