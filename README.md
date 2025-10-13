# Grothko-Agentic-Healthcare-Assistant (Streamlit + LangChain + RAG)

> ⚠️ **Disclaimer:** This app is for demonstrations and admin-style automation only.  
> It does **not** provide medical advice, diagnosis, or treatment recommendations.

This project is a clean, production-ready starter implementing an **agentic orchestrator** with modular tools:
- **Booking tool** for appointment confirmations (mock).
- **Record tools** to read/write a JSON "EHR".
- **Search tool** for web lookups (SerpAPI if available, else DuckDuckGo fallback).
- **RAG pipeline** over a local medical knowledge base (FAISS).
- **Main agent** (LangChain) with memory and tools.
- **Streamlit UI** with chat, sidebar patient context, and a safe system prompt.
