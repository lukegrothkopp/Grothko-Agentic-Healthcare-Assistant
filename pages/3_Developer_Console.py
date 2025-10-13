# pages/3_Developer_Console.py
import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from generate_faiss_index import generate_index
from utils.rag_pipeline import RAGPipeline
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator

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
        st.caption(
            f"Backend in use: {getattr(rag, 'backend', 'unknown')} "
            "(FAISS if OpenAI key + index present; TF-IDF otherwise)"
        )
        pairs = rag.retrieve(probe_q, k=3)
        if not pairs:
            st.info("No results from KB.")
        else:
            for i, (text, score) in enumerate(pairs, 1):
                st.write(f"**{i}.** score={score:.4f}")
                st.write(text[:1000] + ("…" if len(text) > 1000 else ""))
    except Exception as e:
        st.error(f"Probe failed: {e}")

# ---- Q&A Eval (QAEvalChain) ----
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
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

def _local_eval_table(examples, predictions):
    rows = []
    for ex, pr in zip(examples, predictions):
        q, gold, pred = ex["query"], ex["answer"], pr["result"]
        score = _f1_local(pred or "", gold or "")
        rows.append({
            "query": q,
            "prediction": pred,
            "gold": gold,
            "score": round(score, 3),
            "why": "Local token F1 overlap (no LLM key)."
        })
    return rows

uploaded = st.file_uploader("Upload JSONL (query, answer, result)", type=["jsonl"])
has_key = os.environ.get("OPENAI_API_KEY", "").strip().startswith("sk-")
use_llm = st.toggle("Use LLM judge (QAEvalChain)", value=has_key)
model_name = st.text_input("LLM model", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

if uploaded is not None and st.button("Run Evaluation"):
    content = uploaded.read().decode("utf-8", errors="ignore")
    try:
        examples, predictions = _parse_jsonl(content)
    except Exception as e:
        st.error(f"Parse error: {e}")
        st.stop()

    rows = []
    with st.spinner("Scoring…"):
        if use_llm and has_key:
            try:
                llm = ChatOpenAI(model=model_name, temperature=0)
                evaluator = load_evaluator("qa", llm=llm)
                results = evaluator.evaluate(examples, predictions)
                for ex, pr, r in zip(examples, predictions, results):
                    verdict = r.get("score") or r.get("label") or ""
                    why = (r.get("text") or "").strip()
                    rows.append({
                        "query": ex["query"],
                        "prediction": pr["result"],
                        "gold": ex["answer"],
                        "judgment": verdict,
                        "why": why
                    })
            except Exception as e:
                st.error(f"LLM eval failed: {e}")
                rows = _local_eval_table(examples, predictions)
        else:
            rows = _local_eval_table(examples, predictions)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=400)
    if "judgment" in df.columns:
        correct = df["judgment"].astype(str).str.contains("CORRECT", case=False, na=False).sum()
        st.write(f"**Judged CORRECT:** {correct}/{len(df)}")
    elif "score" in df.columns:
        st.write(f"**Avg F1:** {df['score'].mean():.3f}")

    st.download_button(
        "Download results (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="qa_eval_results.csv",
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
