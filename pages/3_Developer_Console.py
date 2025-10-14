# pages/3_Developer_Console.py
import os, io, json, zipfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from generate_faiss_index import generate_index
from utils.rag_pipeline import RAGPipeline
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator

load_dotenv()  # local .env

# Map Streamlit secrets -> env (Cloud)
for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "SERPAPI_API_KEY", "ADMIN_TOKEN"):
    try:
        v = st.secrets.get(k)
        if v:
            os.environ[k] = str(v).strip()
    except Exception:
        pass

def _get_openai_key() -> str:
    """Prefer Streamlit secrets; fall back to environment."""
    try:
        v = st.secrets.get("OPENAI_API_KEY")
        if v:
            return str(v).strip()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "").strip()

st.set_page_config(page_title="Developer Console", layout="wide")
st.title("⚕️ Developer Console")
st.caption("For ops, QA, indexing, and diagnostics. Not visible to patients/clinicians.")

# Optional access gating
required = os.environ.get("ADMIN_TOKEN") or (getattr(st, "secrets", {}).get("ADMIN_TOKEN") if hasattr(st, "secrets") else "")
if required:
    code = st.sidebar.text_input("Access code", type="password")
    if code.strip() != str(required).strip():
        st.warning("Enter a valid admin access code to view this console.")
        st.stop()

# ---- RAG index builder ----
st.subheader("RAG Index")
if st.button("Build FAISS index now"):
    key = _get_openai_key()
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

probe_q = st.text_input("Test a query against the local medical KB", "latest CKD treatments", key="kb_probe_q")
if st.button("Retrieve top-3", key="kb_probe_btn"):
    try:
        rag = RAGPipeline()
        backend = getattr(rag, "backend", "unknown")
        is_faiss = (backend == "openai")
        label = "distance (lower=better)" if is_faiss else "similarity (higher=better)"
        st.caption(f"Backend in use: {backend} — showing {label}")

        pairs = rag.retrieve(probe_q, k=3)
        if not pairs:
            st.info("No results from KB.")
        else:
            for i, (text, score) in enumerate(pairs, 1):
                st.write(f"**{i}.** {label.split()[0]}={score:.4f}")
                st.write(text[:1000] + ("…" if len(text) > 1000 else ""))
    except Exception as e:
        st.error(f"Probe failed: {e}")

# ---- Q&A Eval (LLM-as-judge) with multi-file upload ----
st.markdown("---")
st.subheader("Q&A Eval (LLM-as-judge)")

def _parse_jsonl(s: str):
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
    merged_ex, merged_pr = [], []
    for f in uploaded_files:
        content = f.read().decode("utf-8", errors="ignore")
        examples, predictions = _parse_jsonl(content)
        merged_ex.extend([dict(item, __dataset=f.name) for item in examples])
        merged_pr.extend([dict(item, __dataset=f.name) for item in predictions])
        f.seek(0)  # reset for reuse, if needed
    return merged_ex, merged_pr

uploaded_files = st.file_uploader(
    "Upload one or more JSONL files (fields: query, answer, result)",
    type=["jsonl"], accept_multiple_files=True,
)

has_key = _get_openai_key().startswith("sk-")
use_llm = st.toggle("Use LLM judge (QAEvalChain)", value=has_key)
model_name = st.text_input("LLM model", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

if uploaded_files and st.button("Run Evaluation"):
    try:
        ex_all, pr_all = _parse_many(uploaded_files)
    except Exception as e:
        st.error(f"Parse error: {e}")
        st.stop()

    rows = []
    with st.spinner("Scoring…"):
        if use_llm and has_key:
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
                rows = _local_eval_table(ex_all, pr_all)
        else:
            rows = _local_eval_table(ex_all, pr_all)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=420)

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

    st.download_button(
        "Download combined results (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="qa_eval_results_combined.csv",
        mime="text/csv",
    )

# ---- Diagnostics ----
st.markdown("---")
st.subheader("Diagnostics")
st.write({
    "OPENAI key detected": has_key,
    "Model": os.environ.get("OPENAI_MODEL"),
    "FAISS index exists": os.path.exists("vector_store/faiss_index.bin"),
})

