# pages/3_Developer_Console.py  (merged superset: seeds + KB mgmt + FAISS + probe + eval + metrics + KB summary)
import os, io, json, zipfile
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# App features
from utils.metrics import get_metrics_summary, iter_traces
from generate_faiss_index import generate_index
from utils.rag_pipeline import RAGPipeline
from utils.patient_memory import PatientMemory

# Optional evaluation (LLM-as-judge)
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator

load_dotenv()  # local .env

# Map Streamlit secrets -> env (Cloud)
for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "SERPAPI_API_KEY", "ADMIN_TOKEN", "OFFLINE_KB_DIR", "OFFLINE_PATIENT_DIR"):
    try:
        v = st.secrets.get(k)
        if v:
            os.environ[k] = str(v).strip()
    except Exception:
        pass

def _get_openai_key() -> str:
    try:
        v = st.secrets.get("OPENAI_API_KEY")
        if v:
            return str(v).strip()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "").strip()

def _get_token(name: str) -> str:
    try:
        v = st.secrets.get(name)
        if v:
            return str(v).strip()
    except Exception:
        pass
    return str(os.getenv(name, "")).strip()

st.set_page_config(page_title="Developer Console", page_icon="🧑🏽‍💻", layout="wide")
st.title("🧑🏽‍💻 Developer Console")
st.caption("For ops, QA, indexing, diagnostics. Not visible to patients/clinicians.")

# ---- Access gate (Developer) ----
ADMIN_REQUIRED = _get_token("ADMIN_TOKEN")
if ADMIN_REQUIRED:
    with st.sidebar:
        admin_code = st.text_input("Developer access code", type="password")
    if (admin_code or "").strip() != ADMIN_REQUIRED:
        st.warning("Enter a valid admin access code to view this console.")
        st.stop()
# ---- end access gate ----

# ---------------------------
# Latest retrieved medical info (Offline KB summary)
# ---------------------------
st.markdown("### Latest retrieved medical information (offline KB)")
LAST = Path("data/last_retrieved.json")
if LAST.exists():
    data = json.loads(LAST.read_text(encoding="utf-8"))
    st.write("**Query:**", data.get("query"))
    st.markdown("**Result:**")
    st.code(data.get("result", "")[:4000])
else:
    st.info("No KB retrievals logged yet. Use the Patient page or Offline KB tool.")

st.markdown("---")

# ---------------------------
# Patient Seeds (OFFLINE_PATIENT_DIR)
# ---------------------------
st.markdown("### Patient Seeds")

if "pmemory" not in st.session_state:
    st.session_state.pmemory = PatientMemory()

pmemory: PatientMemory = st.session_state.pmemory

seed_default = os.environ.get("OFFLINE_PATIENT_DIR", pmemory.seed_dir)
seed_dir = st.text_input("Seed directory (OFFLINE_PATIENT_DIR)", value=seed_default, key="seed_dir_input")

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("Apply & Reload Seeds"):
        try:
            os.environ["OFFLINE_PATIENT_DIR"] = seed_dir
            pmemory.reload_from_dir(seed_dir)
            st.success(f"Loaded {len(pmemory.patients)} patient(s) from {seed_dir}")
        except Exception as e:
            st.error(f"Reload failed: {e}")
with c2:
    if st.button("Create seed folder"):
        try:
            os.makedirs(seed_dir, exist_ok=True)
            st.success(f"Created: {seed_dir}")
        except Exception as e:
            st.error(f"Create failed: {e}")
with c3:
    if st.button("Show loaded patients"):
        data = pmemory.list_patients()
        if data:
            st.dataframe(pd.DataFrame(data), use_container_width=True, height=240)
        else:
            st.info("No patients loaded.")
with c4:
    if st.button("Show seed dir status"):
        exists = os.path.isdir(seed_dir)
        files = sorted([n for n in os.listdir(seed_dir)]) if exists else []
        st.write({"exists": exists, "file_count": len(files)})
        if files:
            st.write(files[:20] + (["…"] if len(files) > 20 else []))

st.markdown("**Import patient JSON (.json or .zip of many)** — records will be written into the seed directory above.")
seed_file = st.file_uploader("Choose patient JSON or a .zip of JSONs", type=["json", "zip"], key="seed_uploader")
if seed_file is not None and st.button("Import into Seeds"):
    try:
        os.makedirs(seed_dir, exist_ok=True)
        imported = 0
        if seed_file.type.endswith("zip"):
            with zipfile.ZipFile(io.BytesIO(seed_file.read())) as z:
                for name in z.namelist():
                    if not name.lower().endswith(".json"):
                        continue
                    data = json.loads(z.read(name).decode("utf-8", errors="ignore"))
                    # accept list, dict-with-patients, or single record
                    records = []
                    if isinstance(data, list):
                        records = data
                    elif isinstance(data, dict) and "patients" in data and isinstance(data["patients"], list):
                        records = data["patients"]
                    else:
                        records = [data]
                    for rec in records:
                        pmemory.save_patient_json(rec, dir_path=seed_dir)
                        imported += 1
        else:
            data = json.loads(seed_file.read().decode("utf-8", errors="ignore"))
            records = []
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "patients" in data and isinstance(data["patients"], list):
                records = data["patients"]
            else:
                records = [data]
            for rec in records:
                pmemory.save_patient_json(rec, dir_path=seed_dir)
                imported += 1
        # reload index
        pmemory.reload_from_dir(seed_dir)
        st.success(f"Imported {imported} patient record(s) into {seed_dir}")
        st.dataframe(pd.DataFrame(pmemory.list_patients()), use_container_width=True, height=240)
    except Exception as e:
        st.error(f"Import failed: {e}")

st.caption({
    "OFFLINE_PATIENT_DIR": os.environ.get("OFFLINE_PATIENT_DIR", pmemory.seed_dir),
    "loaded_patients": len(pmemory.patients),
})

st.markdown("---")

# ---------------------------
# Offline KB (TF-IDF)
# ---------------------------
st.markdown("### Offline KB (TF-IDF)")

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()

rag: RAGPipeline = st.session_state.rag

default_kb = os.environ.get("OFFLINE_KB_DIR", rag.kb_dir)
kb_dir = st.text_input("KB directory (OFFLINE_KB_DIR)", value=default_kb, key="kb_dir_input")

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("Apply & Rebuild (TF-IDF)"):
        try:
            os.environ["OFFLINE_KB_DIR"] = kb_dir
            rag.set_kb_dir(kb_dir)   # also rebuilds index
            st.success(f"Rebuilt index for: {rag.kb_dir}")
        except Exception as e:
            st.error(f"Rebuild failed: {e}")
with c2:
    if st.button("Rebuild KB"):
        try:
            rag.rebuild_index()
            st.success("Index rebuilt.")
        except Exception as e:
            st.error(f"Rebuild failed: {e}")
with c3:
    if st.button("Create KB folder"):
        try:
            os.makedirs(kb_dir, exist_ok=True)
            st.success(f"Created: {kb_dir}")
        except Exception as e:
            st.error(f"Create failed: {e}")
with c4:
    if st.button("Show KB status"):
        st.json(rag.status())

st.markdown("**Upload KB (.zip)** — contents will be extracted into the KB directory above and indexed.")
uploaded_zip = st.file_uploader("Choose a .zip with .txt/.md/.pdf/.docx/.json files", type=["zip"])
if uploaded_zip is not None and st.button("Import ZIP into KB"):
    try:
        os.makedirs(kb_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as z:
            z.extractall(kb_dir)
        rag.set_kb_dir(kb_dir)  # rebuilds index
        st.success(f"Imported ZIP into {kb_dir}. Docs indexed: {rag.status().get('num_docs', 0)}")
        st.json(rag.status())
    except Exception as e:
        st.error(f"Import failed: {e}")

st.caption({
    "backend": getattr(rag, "backend", getattr(rag, "backend_label", "unknown")),
    "kb_dir": rag.kb_dir,
    "docs_indexed": rag.status().get("num_docs", 0),
    "file_types": rag.status().get("file_type_counts", {}),
})

st.markdown("---")

# ---------------------------
# Optional: Build FAISS index (OpenAI embeddings)
# ---------------------------
st.markdown("### Optional: Build FAISS index (OpenAI embeddings)")
if st.button("Build FAISS index now"):
    key = _get_openai_key()
    if not key.startswith("sk-"):
        st.warning("No valid OPENAI_API_KEY found; the app will continue using TF-IDF.")
    else:
        with st.spinner("Building FAISS index…"):
            try:
                out_path = generate_index(api_key=key)
                st.success(f"FAISS index built at {out_path}")
            except Exception as e:
                st.error(f"Failed to build: {e}")

st.markdown("---")

# ---------------------------
# Probe the local KB
# ---------------------------
st.subheader("Probe the local KB")

probe_q = st.text_input("Test a query against the local medical KB", "latest CKD treatments", key="kb_probe_q")
if st.button("Retrieve top-3", key="kb_probe_btn"):
    try:
        backend = getattr(rag, "backend", getattr(rag, "backend_label", "unknown"))
        is_faiss = (backend == "openai")
        label = "distance (lower=better)" if is_faiss else "similarity (higher=better)"
        st.caption(f"Backend in use: {backend} — showing {label}")

        pairs = rag.retrieve(probe_q, k=3)
        if not pairs:
            st.info("No results from KB.")
            st.write("**Diagnostics:**")
            st.json(rag.status())
            st.markdown(
                "- Ensure the KB directory contains **text-bearing files** (.txt/.md/.pdf/.docx/.json).\n"
                "- For .pdf/.docx, optional deps: `PyPDF2`, `python-docx` (or `docx2txt`).\n"
                "- Try a broader query (e.g., “chronic kidney disease treatments”).\n"
                "- After uploading a ZIP or changing the path, click **Apply & Rebuild**."
            )
        else:
            for i, (text, score) in enumerate(pairs, 1):
                st.write(f"**{i}.** {label.split()[0]}={float(score):.4f}")
                st.write(text[:1000] + ("…" if len(text) > 1000 else ""))
    except Exception as e:
        st.error(f"Probe failed: {e}")

st.markdown("---")

# ---------------------------
# Q&A Eval (LLM-as-judge) — supports multiple JSONL
# ---------------------------
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

def _local_eval_table(examples, predictions, dataset_name=""):
    rows = []
    for ex, pr in zip(examples, predictions):
        q, gold, pred = ex["query"], ex["answer"], pr["result"]
        score = _f1_local(pred or "", gold or "")
        rows.append({
            "dataset": dataset_name,
            "query": q, "prediction": pred, "gold": gold,
            "score": round(score, 3), "why": "Local token F1 overlap (no LLM key)."
        })
    return rows

def _parse_many(files):
    all_examples, all_predictions = [], []
    for f in files:
        content = f.read().decode("utf-8", errors="ignore")
        ex, pr = _parse_jsonl(content)
        all_examples.append((f.name, ex))
        all_predictions.append((f.name, pr))
        f.seek(0)
    return all_examples, all_predictions

uploaded_files = st.file_uploader(
    "Upload one or more JSONL files (fields: query, answer, result)",
    type=["jsonl"], accept_multiple_files=True,
)

has_key = _get_openai_key().startswith("sk-")
use_llm = st.toggle("Use LLM judge (QAEvalChain)", value=has_key)
model_name = st.text_input("LLM model", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

if uploaded_files and st.button("Run Evaluation"):
    try:
        all_ex, all_pr = _parse_many(uploaded_files)
    except Exception as e:
        st.error(f"Parse error: {e}")
        st.stop()

    rows = []
    with st.spinner("Scoring…"):
        if use_llm and has_key:
            try:
                llm = ChatOpenAI(model=model_name, temperature=0)
                evaluator = load_evaluator("qa", llm=llm)
                # flatten with dataset tag
                examples = []
                predictions = []
                for (name_e, ex), (name_p, pr) in zip(all_ex, all_pr):
                    # names should align in same order
                    for x in ex:
                        x["__dataset"] = name_e
                    for y in pr:
                        y["__dataset"] = name_p
                    examples.extend(ex)
                    predictions.extend(pr)
                results = evaluator.evaluate(examples, predictions)
                for ex, pr, r in zip(examples, predictions, results):
                    rows.append({
                        "dataset": ex.get("__dataset", ""),
                        "query": ex["query"], "prediction": pr["result"], "gold": ex["answer"],
                        "judgment": r.get("score") or r.get("label") or "",
                        "why": (r.get("text") or "").strip(),
                    })
            except Exception as e:
                st.error(f"LLM eval failed: {e}")
                # fallback local per dataset
                for (name, ex), (_, pr) in zip(all_ex, all_pr):
                    rows.extend(_local_eval_table(ex, pr, dataset_name=name))
        else:
            for (name, ex), (_, pr) in zip(all_ex, all_pr):
                rows.extend(_local_eval_table(ex, pr, dataset_name=name))

    import pandas as pd
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

st.markdown("---")

# ---------------------------
# Metrics & Traces
# ---------------------------
st.subheader("Metrics & Traces")
summary = get_metrics_summary()
st.json(summary)

st.markdown("### Recent traces")
rows = iter_traces(limit=200)
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=420)
else:
    st.info("No traces yet. Interact with the app to generate activity.")

# ---- Diagnostics ----
st.markdown("---")
st.subheader("Diagnostics")
st.write({
    "OPENAI key detected": _get_openai_key().startswith("sk-"),
    "Model": os.environ.get("OPENAI_MODEL"),
    "FAISS index exists": os.path.exists("vector_store/faiss_index.bin"),
    "Offline KB dir": rag.kb_dir,
    "Offline KB docs": rag.status().get("num_docs", 0),
    "OFFLINE_PATIENT_DIR": os.environ.get("OFFLINE_PATIENT_DIR", pmemory.seed_dir),
    "Loaded patients": len(pmemory.patients),
})
