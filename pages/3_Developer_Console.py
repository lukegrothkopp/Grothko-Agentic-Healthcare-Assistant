# pages/3_Developer_Console.py
import os
import json
import pandas as pd
import streamlit as st
import io, zipfile
from dotenv import load_dotenv

from generate_faiss_index import generate_index
from utils.rag_pipeline import RAGPipeline
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
from agents.graph_agent import build_graph
from langchain_core.messages import HumanMessage

load_dotenv()  # for local .env

# Map Streamlit secrets -> environment (works on Cloud)
for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "SERPAPI_API_KEY", "ADMIN_TOKEN"):
    if hasattr(st, "secrets") and k in st.secrets and st.secrets[k]:
        os.environ[k] = str(st.secrets[k]).strip()

st.set_page_config(page_title="Developer Console", layout="wide")
st.title("⚕️ Developer Console")
st.caption("For ops, QA, indexing, and diagnostics. Not visible to patients/clinicians.")

# ---- Simple access gating (only if configured) ----
required = os.environ.get("ADMIN_TOKEN") or (st.secrets.get("ADMIN_TOKEN") if hasattr(st, "secrets") else "")
if required:
    code = st.sidebar.text_input("Access code", type="password")
    if code.strip() != str(required).strip():
        st.warning("Enter a valid admin access code to view this console.")
        st.stop()
else:
    st.sidebar.info("No admin token configured — this page is open.")

# ---- RAG index builder ----
st.subheader("RAG Index")
if st.button("Build FAISS index now"):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key.startswith("sk-"):
        st.warning("No valid OPENAI_API_KEY found; the app will use TF-IDF fallback.")
    else:
        with st.spinner("Building FAISS index…"):
            try:
                out_path = generate_index(api_key=key)
                st.success(f"FAISS index built at {out_path}")
            except Exception as e:
                st.error(f"Failed to build: {e}")

# ---- Probe the local KB (FAISS/TF-IDF) ----
st.markdown("---")
st.subheader("Probe the local KB")

probe_q = st.text_input(
    "Test a query against the local medical KB",
    "latest CKD treatments",
    key="kb_probe_q",
)
if st.button("Retrieve top-3", key="kb_probe_btn"):
    try:
        rag = RAGPipeline()
        backend = getattr(rag, "backend", "unknown")
        is_faiss = backend == "openai"  # our RAG uses 'openai' when FAISS+embeddings is active

        pairs = rag.retrieve(probe_q, k=3)
        label = "distance (lower=better)" if is_faiss else "similarity (higher=better)"
        st.caption(f"Backend in use: {backend} — showing {label}")

        for i, (text, score) in enumerate(pairs, 1):
            st.write(f"**{i}.** {label.split()[0]}={score:.4f}")
            st.write(text[:1000] + ("…" if len(text) > 1000 else ""))
    except Exception as e:
        st.error(f"Probe failed: {e}")

# ---- Q&A Eval (LLM-as-judge) ----
st.markdown("---")
st.subheader("Q&A Eval (LLM-as-judge)")

def _parse_jsonl(s: str):
    import json
    examples, predictions = [], []
    for i, line in enumerate(s.splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        q = row.get("query", "")
        gold = row.get("answer", "")
        pred = row.get("result", "")
        if not q:
            raise ValueError(f"Line {i+1}: missing 'query'")
        examples.append({"query": q, "answer": gold})
        predictions.append({"query": q, "result": pred})
    return examples, predictions

def _f1_local(pred: str, gold: str) -> float:
    import re
    tok = lambda s: [w for w in re.findall(r"\w+", s.lower()) if w]
    p, g = set(tok(pred)), set(tok(gold))
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec  = tp / len(g) if g else 0.0
    return (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0

def _local_eval_table(examples, predictions):
    rows = []
    for ex, pr in zip(examples, predictions):
        q, gold, pred = ex["query"], ex["answer"], pr["result"]
        score = _f1_local(pred or "", gold or "")
        rows.append({
            "query": q, "prediction": pred, "gold": gold,
            "score": round(score, 3), "why": "Local token F1 overlap (no LLM key)."
        })
    return rows

def _parse_many(uploaded_files):
    """Return merged examples/preds with a __dataset label, plus per-dataset lists."""
    merged_ex, merged_pr = [], []
    per_dataset = []  # list of (name, examples, predictions)
    for f in uploaded_files:
        content = f.read().decode("utf-8", errors="ignore")
        examples, predictions = _parse_jsonl(content)
        per_dataset.append((f.name, examples, predictions))
        merged_ex.extend([dict(item, __dataset=f.name) for item in examples])
        merged_pr.extend([dict(item, __dataset=f.name) for item in predictions])
    return merged_ex, merged_pr, per_dataset

uploaded_files = st.file_uploader(
    "Upload one or more JSONL files (fields: query, answer, result)",
    type=["jsonl"],
    accept_multiple_files=True,
)

has_key = os.environ.get("OPENAI_API_KEY", "").strip().startswith("sk-")
use_llm = st.toggle("Use LLM judge (QAEvalChain)", value=has_key)
model_name = st.text_input("LLM model", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

if uploaded_files and st.button("Run Evaluation"):
    try:
        ex_all, pr_all, per_ds = _parse_many(uploaded_files)
    except Exception as e:
        st.error(f"Parse error: {e}")
        st.stop()

    if not ex_all:
        st.info("No items found in uploaded files.")
        st.stop()

    rows = []
    with st.spinner("Scoring…"):
        if use_llm and has_key:
            from langchain_openai import ChatOpenAI
            from langchain.evaluation import load_evaluator
            try:
                llm = ChatOpenAI(model=model_name, temperature=0)
                evaluator = load_evaluator("qa", llm=llm)
                results = evaluator.evaluate(ex_all, pr_all)
                for ex, pr, r in zip(ex_all, pr_all, results):
                    rows.append({
                        "dataset": ex.get("__dataset", ""),
                        "query": ex["query"],
                        "prediction": pr["result"],
                        "gold": ex["answer"],
                        "judgment": r.get("score") or r.get("label") or "",
                        "why": (r.get("text") or "").strip(),
                    })
            except Exception as e:
                st.error(f"LLM eval failed: {e}")
                use_llm = False  # fall back
        if not use_llm:
            # local fallback (F1)
            rows = []
            for ex, pr in zip(ex_all, pr_all):
                q, gold, pred = ex["query"], ex["answer"], pr["result"]
                score = _f1_local(pred or "", gold or "")
                rows.append({
                    "dataset": ex.get("__dataset", ""),
                    "query": q, "prediction": pred, "gold": gold,
                    "score": round(score, 3), "why": "Local token F1 overlap (no LLM key)."
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=420)

    # ---- Summaries: overall + per-dataset
    st.markdown("#### Summary")
    if "judgment" in df.columns:
        overall_correct = df["judgment"].astype(str).str.contains("CORRECT", case=False, na=False).mean()
        st.write(f"**Overall CORRECT rate:** {overall_correct:.3f}")
        for name, sub in df.groupby("dataset"):
            rate = sub["judgment"].astype(str).str.contains("CORRECT", case=False, na=False).mean()
            st.write(f"- {name}: {rate:.3f}")
    elif "score" in df.columns:
        st.write(f"**Overall Avg F1:** {df['score'].mean():.3f}")
        for name, sub in df.groupby("dataset"):
            st.write(f"- {name}: {sub['score'].mean():.3f}")

    # ---- Downloads: combined CSV and ZIP (combined + per-dataset)
    st.download_button(
        "Download combined results (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="qa_eval_results_combined.csv",
        mime="text/csv",
    )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("combined.csv", df.to_csv(index=False))
        for name, sub in df.groupby("dataset"):
            safe = name.replace("/", "_")
            z.writestr(f"{safe}.csv", sub.to_csv(index=False))
    st.download_button(
        "Download all results (ZIP)",
        data=zip_buf.getvalue(),
        file_name="qa_eval_results_all.zip",
        mime="application/zip",
    )

# (Optional) Offer a curated pack download if you keep JSONLs in data/eval_sets
curated_dir = "data/eval_sets"
if os.path.isdir(curated_dir):
    st.markdown("##### Curated test pack")
    pack = io.BytesIO()
    with zipfile.ZipFile(pack, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(curated_dir)):
            if fn.endswith(".jsonl"):
                with open(os.path.join(curated_dir, fn), "rb") as f:
                    z.writestr(fn, f.read())
    st.download_button(
        "Download curated test pack (.zip)",
        data=pack.getvalue(),
        file_name="eval_sets.zip",
        mime="application/zip",
    )

# ---- Diagnostics ----
st.markdown("---")
st.subheader("Diagnostics")
st.write({
    "OPENAI key detected": has_key,
    "Model": os.environ.get("OPENAI_MODEL"),
    "FAISS index exists": os.path.exists("vector_store/faiss_index.bin"),
})
