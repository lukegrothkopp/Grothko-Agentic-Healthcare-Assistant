# utils/secret_env.py
from __future__ import annotations
import os

try:
    import streamlit as st  # available at runtime; safe if missing (e.g., unit tests)
except Exception:  # pragma: no cover
    st = None

DEFAULT_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "SERPAPI_API_KEY",
    "ADMIN_TOKEN",
    "TAVILY_API_KEY",
)

def export_secrets_to_env(keys=DEFAULT_KEYS) -> None:
    """
    Copies Streamlit secrets into os.environ when present.
    No-ops locally if st.secrets is not available.
    """
    if st is None:
        return
    for k in keys:
        try:
            v = st.secrets.get(k)
        except Exception:
            v = None
        if v and str(v).strip():
            os.environ[k] = str(v).strip()
