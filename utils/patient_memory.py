# utils/patient_memory.py
import os, sqlite3, json, datetime
from typing import List, Tuple, Optional

# Optional LLM summary (OpenAI) if key present; fallback to heuristic
def _summarize_with_llm(text: str) -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        # simple fallback: first 5 sentences
        import re
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return " ".join(sents[:5])
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"), temperature=0)
        prompt = [
            SystemMessage(content="You are a clinical admin assistant. Summarize key admin-relevant facts only (who/what/when/where), no medical advice."),
            HumanMessage(content=f"Conversation notes:\n{text}\n\nSummarize in 5-8 bullet points suitable to prime future admin tasks.")
        ]
        out = llm.invoke(prompt)
        return out.content
    except Exception:
        return text[:1200]

class PatientMemory:
    """SQLite-backed long-term memory for patient context."""
    def __init__(self, db_path: str = "data/memory.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          patient_id TEXT, role TEXT, content TEXT, ts TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
          patient_id TEXT PRIMARY KEY, summary TEXT, ts TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS facts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          patient_id TEXT, text TEXT, meta TEXT, ts TEXT
        )""")
        self.conn.commit()

    # --------- Writes ----------
    def add_message(self, patient_id: str, role: str, content: str):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        self.conn.execute(
            "INSERT INTO messages(patient_id, role, content, ts) VALUES(?,?,?,?)",
            (patient_id, role, content, ts)
        )
        self.conn.commit()

    def record_event(self, patient_id: str, text: str, meta: Optional[dict] = None):
        """For tool outputs (booking confirmations, etc.)."""
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        self.conn.execute(
            "INSERT INTO facts(patient_id, text, meta, ts) VALUES(?,?,?,?)",
            (patient_id, text, json.dumps(meta or {}), ts)
        )
        # also store as a 'tool' message to keep chronology
        self.conn.execute(
            "INSERT INTO messages(patient_id, role, content, ts) VALUES(?,?,?,?)",
            (patient_id, "tool", text, ts)
        )
        self.conn.commit()

    def upsert_summary(self, patient_id: str, summary: str):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        self.conn.execute(
            "INSERT INTO summaries(patient_id, summary, ts) VALUES(?,?,?) "
            "ON CONFLICT(patient_id) DO UPDATE SET summary=excluded.summary, ts=excluded.ts",
            (patient_id, summary, ts)
        )
        self.conn.commit()

    # --------- Reads ----------
    def get_window(self, patient_id: str, k: int = 8) -> List[Tuple[str, str, str]]:
        """Return last k messages as (role, content, ts)."""
        rows = self.conn.execute(
            "SELECT role, content, ts FROM messages WHERE patient_id=? ORDER BY id DESC LIMIT ?",
            (patient_id, k)
        ).fetchall()
        return list(reversed(rows))

    def get_summary(self, patient_id: str) -> str:
        row = self.conn.execute(
            "SELECT summary FROM summaries WHERE patient_id=?",
            (patient_id,)
        ).fetchone()
        return row[0] if row else ""

    def dump_messages(self, patient_id: str, limit: int = 200) -> str:
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE patient_id=? ORDER BY id ASC LIMIT ?",
            (patient_id, limit)
        ).fetchall()
        return "\n".join(f"[{r}] {c}" for r, c in rows)

    # --------- Semantic recall (TF-IDF; no key required) ----------
    def search(self, patient_id: str, query: str, k: int = 3) -> list[str]:
        rows = self.conn.execute(
            "SELECT text FROM facts WHERE patient_id=? ORDER BY id DESC LIMIT 500",
            (patient_id,)
        ).fetchall()
        corpus = [r[0] for r in rows]
        if not corpus:
            return []
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import linear_kernel
        except Exception:
            # fallback: contains-based naive ranking
            q = query.lower()
            hits = [t for t in corpus if q in t.lower()]
            return hits[:k]
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(corpus)
        qv = vec.transform([query])
        sims = linear_kernel(qv, X).ravel()
        idxs = sims.argsort()[::-1][:k]
        return [corpus[i] for i in idxs if sims[i] > 0.0]

    # --------- Maintenance ----------
    def maybe_autosummarize(self, patient_id: str, every_n: int = 12):
        """Periodically condense long chat history into summary to keep prompts lean."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE patient_id=?", (patient_id,)
        ).fetchone()[0]
        if count == 0 or count % every_n != 0:
            return
        text = self.dump_messages(patient_id, limit=400)
        summary = _summarize_with_llm(text)
        self.upsert_summary(patient_id, summary)
