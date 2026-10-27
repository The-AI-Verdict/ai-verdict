from __future__ import annotations

import os

import streamlit as st

from core import db, openrouter_client, ui_common

st.set_page_config(page_title="Settings - The AI Verdict", layout="wide")
ui_common.bootstrap()

st.title("Settings and API configuration")

st.warning(
    "API keys are never shown or editable in this UI - not even masked - because this app is designed to "
    "be screen-recorded for video production. Set them in a `.env` file in the project root (outside the "
    "browser) and restart Streamlit to pick up changes."
)

st.subheader("OpenRouter API key")
key_set = ui_common.api_key_configured()
st.write("Status:", ":green[configured]" if key_set else ":red[not configured]")
st.caption("Set `OPENROUTER_API_KEY` in `.env` (see `.env.example`). Get a key at https://openrouter.ai/keys")

if st.button("Test connection", disabled=not key_set):
    try:
        models = openrouter_client.list_models()
        st.success(f"Connected. OpenRouter currently lists {len(models)} models.")
    except openrouter_client.OpenRouterError as exc:
        st.error(f"Connection failed: {exc}")

st.divider()
st.subheader("Other configuration")
st.write("Database file:", f"`{db.get_db_path()}`")
st.write("Default evaluator model (`AI_VERDICT_EVALUATOR_MODEL`):",
         f"`{os.environ.get('AI_VERDICT_EVALUATOR_MODEL', 'openai/gpt-4.1-mini (default)')}`")
st.write("OpenRouter site URL header (`OPENROUTER_SITE_URL`):", os.environ.get("OPENROUTER_SITE_URL", "(not set)"))
st.write("OpenRouter app name header (`OPENROUTER_APP_NAME`):", os.environ.get("OPENROUTER_APP_NAME", "(not set)"))

st.divider()
st.subheader("Model catalog cache")
with db.get_connection() as conn:
    cached = db.get_cached_models(conn)
last_fetched = max((c["fetched_at"] for c in cached), default=None)
st.write(f"{len(cached)} models cached." + (f" Last fetched: {last_fetched}" if last_fetched else ""))
if st.button("Refresh model catalog", disabled=not key_set):
    try:
        models = openrouter_client.list_models()
        with db.get_connection() as conn:
            db.cache_models(conn, models)
        st.success(f"Cached {len(models)} models.")
        st.rerun()
    except openrouter_client.OpenRouterError as exc:
        st.error(f"Refresh failed: {exc}")

st.divider()
st.subheader("Experimental integrity notes")
st.markdown(
    "- The same prompt text and generation settings are sent to every selected model for a given run; any "
    "setting a model doesn't support is recorded per-run (see the `unsupported_params` field in each run), "
    "not silently dropped.\n"
    "- Every run's exact request parameters, the provider's echoed `model` string, token usage, cost (when "
    "reported), latency, and any error are stored - raw responses are never edited or discarded.\n"
    "- OpenRouter can route a model id to different upstream provider deployments over time; this app "
    "cannot force full reproducibility of a specific provider snapshot, only record what was actually "
    "returned for each call.\n"
    "- Failed calls are retried with exponential backoff only when no response was already received, to "
    "avoid duplicate charges; retries are counted and stored per run."
)
