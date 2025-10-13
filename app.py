import os, json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
from langchain_core.messages import HumanMessage
from agents.graph_agent import build_graph

load_dotenv()

for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "SERPAPI_API_KEY", "TAVILY_API_KEY"):
    if k in st.secrets and st.secrets[k]:
        os.environ[k] = str(st.secrets[k]).strip()

if os.getenv("OPENAI_API_KEY"):
    try:
        if not os.path.exists("vector_store/faiss_index.bin"):
            from generate_faiss_index import generate_index
            with st.spinner("Building FAISS index for local KB…"):
                generate_index()  # uses data/medical_kb/
            st.success("FAISS index built.")
    except Exception as e:
        st.warning(f"Could not build FAISS index: {e}")
else:
    st.info("No OPENAI_API_KEY detected — RAG will use a TF-IDF fallback (still works).")

st.set_page_config(page_title="Grothko Agentic Healthcare Assistant", layout="wide")
st.title("👨‍⚕️ Grothko Agentic Healthcare Assistant")
st.caption("Not medical advice. High-level info & logistics only.")

graph = build_graph(model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

with st.sidebar:
    st.header("Context")
    patient_id = st.text_input("Patient ID", "patient_001")
    st.info("The agent routes: booking / records / search / RAG. Offline Mini-KB ensures results even if web is blocked.")
    st.markdown("---")
    st.caption("Set OPENAI_API_KEY in .env for LLM polishing.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about logistics or high-level info…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                state = {"messages": [HumanMessage(content=prompt)], "intent": None, "result": None, "patient_id": patient_id}
                result_state = graph.invoke(state)
                answer = result_state["messages"][-1].content
            except Exception as e:
                answer = f"There was an error: {e}"
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ============================
# Eval tab: QAEvalChain in UI
# ============================
def _parse_jsonl(s: str):
    """Returns (examples, predictions) for LangChain 'qa' evaluator."""
    examples, predictions = []
    examples, predictions = [], []
    for i, line in enumerate(s.splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        # expected keys:
        #   query   -> the question
        #   answer  -> ground truth
        #   result  -> your model's answer
        q = row.get("query", "")
        gold = row.get("answer", "")
        pred = row.get("result", "")
        if not q:
            raise ValueError(f"Line {i+1}: missing 'query'")
        examples.append({"query": q, "answer": gold})
        predictions.append({"query": q, "result": pred})
    return examples, predictions

def _f1_local(pred: str, gold: str) -> float:
    """Very simple token F1 for offline fallback."""
    import re
    tok = lambda s: [w for w in re.findall(r"\w+", s.lower()) if w]
    p, g = set(tok(pred)), set(tok(gold))
    if not p and not g: 
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec  = tp / len(g) if g else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

def _local_eval_table(examples, predictions):
    """Local, non-LLM baseline (F1)."""
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

# Render the tab
eval_tab, = st.tabs(["Eval"])
with eval_tab:
    st.subheader("Q&A Evaluation (LLM-as-judge)")
    st.caption("Upload a JSONL with fields: query, answer (gold), result (your model’s answer).")

    uploaded = st.file_uploader("Upload JSONL", type=["jsonl"])

    # Default to using LLM when key is present
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    use_llm = st.toggle("Use LLM judge (QAEvalChain)", value=has_key,
                        help="If off (or no key), a local F1 overlap is used instead.")
    model_name = st.text_input("LLM model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    if uploaded is not None:
        content = uploaded.read().decode("utf-8", errors="ignore")
        try:
            examples, predictions = _parse_jsonl(content)
        except Exception as e:
            st.error(f"Parse error: {e}")
            st.stop()

        st.write(f"Loaded **{len(examples)}** items.")
        if st.button("Run Evaluation"):
            rows = []
            with st.spinner("Evaluating…"):
                if use_llm and has_key:
                    try:
                        llm = ChatOpenAI(model=model_name, temperature=0)
                        evaluator = load_evaluator("qa", llm=llm)
                        results = evaluator.evaluate(examples, predictions)
                        for ex, pr, r in zip(examples, predictions, results):
                            q, gold, pred = ex["query"], ex["answer"], pr["result"]
                            verdict = r.get("score") or r.get("label") or ""
                            why = (r.get("text") or "").strip()
                            rows.append({
                                "query": q,
                                "prediction": pred,
                                "gold": gold,
                                "judgment": verdict,
                                "why": why
                            })
                    except Exception as e:
                        st.error(f"LLM eval failed: {e}")
                        st.info("Falling back to local F1 overlap.")
                        rows = _local_eval_table(examples, predictions)
                else:
                    rows = _local_eval_table(examples, predictions)

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=400)

            # Quick summary
            if "judgment" in df.columns:
                correct = df["judgment"].astype(str).str.contains("CORRECT", case=False, na=False).sum()
                st.write(f"**Judged CORRECT:** {correct}/{len(df)}")
            elif "score" in df.columns:
                st.write(f"**Avg F1:** {df['score'].mean():.3f}")

            # Download
            st.download_button(
                "Download results (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="qa_eval_results.csv",
                mime="text/csv",
            )

st.sidebar.markdown("---")
st.sidebar.header("LLMOps (placeholder)")
st.sidebar.write("Add traces, token usage, and evals here.")
