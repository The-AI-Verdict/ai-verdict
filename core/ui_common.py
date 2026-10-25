"""Shared helpers for every Streamlit page: env loading, DB bootstrap, and
small UI utilities that would otherwise be duplicated across pages/*.py.
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from core import db

_bootstrapped = False


def bootstrap() -> None:
    global _bootstrapped
    if _bootstrapped:
        return
    load_dotenv()
    db.init_db()
    _bootstrapped = True


def api_key_configured() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


def mask_key(key: str) -> str:
    if not key:
        return "(not set)"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"


def experiment_selector(label: str = "Experiment", key: str = "selected_experiment_id"):
    with db.get_connection() as conn:
        experiments = db.list_experiments(conn)
    if not experiments:
        st.info("No experiments yet. Create one on the **Create Experiment** page.")
        return None

    options = {f"#{e['id']} - {e['name']} ({e['status']})": e["id"] for e in experiments}
    current_id = st.session_state.get(key)
    default_label = next((lbl for lbl, eid in options.items() if eid == current_id), list(options.keys())[0])
    chosen_label = st.selectbox(label, list(options.keys()), index=list(options.keys()).index(default_label))
    chosen_id = options[chosen_label]
    st.session_state[key] = chosen_id

    with db.get_connection() as conn:
        return db.get_experiment(conn, chosen_id)


def require_api_key() -> bool:
    if not api_key_configured():
        st.warning(
            "No OPENROUTER_API_KEY configured. Add it to a `.env` file in the project root "
            "(see `.env.example`), then restart Streamlit. You can still create and configure "
            "experiments without a key, but running them requires one."
        )
        return False
    return True
