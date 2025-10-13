# Grothko-Agentic-Healthcare-Assistant

A practical starter to explore agentic patterns (planner → tool using agents) for healthcare admin workflows:
- **BookingAgent**: mock doctor lookup + appointment booking (SQLite)
- **HistoryAgent**: CRUD + summarization of patient histories
- **InfoSearchAgent**: pulls trusted web info (DuckDuckGo fallback), extracts/cleans snippets, and summarizes
- **MemoryStore**: lightweight TF IDF vector memory for patient context & prior answers (no API keys needed)
- **Eval hooks**: quick plausibility checks and rubric scoring


⚠️ Disclaimer: This assistant is not medical advice and is intended only for demos and admin-style automation.
