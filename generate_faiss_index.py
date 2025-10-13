import os, faiss, numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

def generate_index(api_key: str | None = None):
    kb_dir = "data/medical_kb"
    out_path = "vector_store/faiss_index.bin"
    os.makedirs("vector_store", exist_ok=True)

    key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not key.startswith("sk-"):
        raise RuntimeError("No valid OPENAI_API_KEY provided to generate_index().")

    # load docs
    import os
    docs = []
    for fn in os.listdir(kb_dir):
        if fn.endswith(".txt"):
            text = open(os.path.join(kb_dir, fn), "r", encoding="utf-8").read()
            chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_text(text)
            docs.extend([Document(page_content=c) for c in chunks])

    if not docs:
        raise RuntimeError("No documents found in data/medical_kb/")

    emb = OpenAIEmbeddings(api_key=key)
    vecs = emb.embed_documents([d.page_content for d in docs])
    mat = np.array(vecs, dtype=np.float32)
    index = faiss.IndexFlatL2(mat.shape[1])
    index.add(mat)

    faiss.write_index(index, out_path)
    with open(out_path + ".docs.txt", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d.page_content.replace("\n", " ") + "\n")

    return out_path
