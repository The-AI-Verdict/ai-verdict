from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from core import db, openrouter_client, runner, ui_common

st.set_page_config(page_title="Run Experiment - The AI Verdict", layout="wide")
ui_common.bootstrap()

st.title("Run Experiment")

experiment = ui_common.experiment_selector()
if experiment is None:
    st.stop()

exp_id = experiment["id"]
running_key = f"running_{exp_id}"

with db.get_connection() as conn:
    prompt_versions = db.list_prompt_versions(conn, exp_id, latest_only=True)
    models = db.list_experiment_models(conn, exp_id)
    runs = db.list_runs(conn, exp_id)

st.subheader("Experiment summary")
c1, c2, c3 = st.columns(3)
c1.metric("Prompt variants", len(prompt_versions))
c2.metric("Models", len(models))
c3.metric("Repetitions per combination", experiment["repetitions"])

estimated = runner.estimate_call_count(max(len(prompt_versions), 1), max(len(models), 1), experiment["repetitions"])
st.caption(f"Planned tested-model calls: {estimated}. Evaluator calls happen automatically after each "
           f"successful run (approximately 1 more per run, plus any repair retries).")

with st.expander("Prompt variants (locked once runs start)"):
    for pv in prompt_versions:
        st.markdown(f"**{pv['variant_name']}** (v{pv['version_number']}) - locked: {bool(pv['is_locked'])}")
        st.code(pv["prompt_text"])

st.divider()
st.subheader("Models")
if models:
    for m in models:
        mc1, mc2 = st.columns([5, 1])
        mc1.write(f"`{m['model_id']}`" + (f" - {m['display_name']}" if m.get("display_name") else ""))
        if not runs and mc2.button("Remove", key=f"rm_{m['model_id']}"):
            with db.get_connection() as conn:
                db.remove_experiment_model(conn, exp_id, m["model_id"])
            st.rerun()
else:
    st.info("No models selected yet. Add at least one below before planning runs.")

with st.expander("Add a model", expanded=not models):
    key_available_for_discovery = ui_common.api_key_configured()
    if st.button("Discover models from OpenRouter", disabled=not key_available_for_discovery, key="discover_run_page"):
        try:
            discovered = openrouter_client.list_models()
            with db.get_connection() as conn:
                db.cache_models(conn, discovered)
            st.success(f"Cached {len(discovered)} models.")
        except openrouter_client.OpenRouterError as exc:
            st.error(f"Could not list models: {exc}")

    with db.get_connection() as conn:
        cached_models = db.get_cached_models(conn)
    existing_ids = {m["model_id"] for m in models}
    catalog_choices = {f"{m['model_id']}  -  {m['name'] or ''}".strip(): m
                        for m in cached_models if m["model_id"] not in existing_ids}
    picked = st.multiselect("Select from discovered models", list(catalog_choices.keys()), key="pick_catalog_run_page")
    manual_id = st.text_input("Or add a model ID manually (e.g. `openai/gpt-4o-mini`)", key="manual_model_run_page")

    if st.button("Add selected model(s)", key="add_models_run_page"):
        with db.get_connection() as conn:
            for label in picked:
                m = catalog_choices[label]
                db.add_experiment_model(conn, exp_id, m["model_id"], display_name=m.get("name"))
            if manual_id.strip():
                db.add_experiment_model(conn, exp_id, manual_id.strip())
        st.rerun()

st.divider()

if not runs:
    st.info("No runs planned yet.")
    if st.button("Plan runs", type="primary", disabled=not models):
        try:
            with db.get_connection() as conn:
                created = runner.plan_runs(conn, exp_id)
            st.success(f"Planned {len(created)} runs.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    st.stop()

df = pd.DataFrame(runs)
total = len(df)
success = int((df["status"] == "success").sum())
failed = int((df["status"] == "failed").sum())
pending = int((df["status"].isin(["pending", "running"])).sum())

st.subheader("Progress")
st.progress((success + failed) / total if total else 0.0)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total runs", total)
m2.metric("Success", success)
m3.metric("Failed", failed)
m4.metric("Pending", pending)

st.markdown("**By model**")
by_model = df.groupby("model_id")["status"].value_counts().unstack(fill_value=0)
st.dataframe(by_model, use_container_width=True)

if pending > 0:
    extra = runner.estimate_call_count(
        max(len({r["prompt_version_id"] for r in runs}), 1), max(len(models), 1), experiment["repetitions"]
    ) - total
    if extra > 0:
        st.caption(f"Note: {extra} additional combination(s) exist in the current config but haven't been planned yet. "
                   f"Use 'Plan runs' again after adding models/variants/repetitions.")

st.divider()
st.subheader("Execution")

key_available = ui_common.api_key_configured()
if not key_available:
    st.warning("OPENROUTER_API_KEY is not set - cannot execute runs. Configure it on the Settings page.")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Start / Resume", type="primary", disabled=not key_available or pending == 0):
        st.session_state[running_key] = True
with col2:
    if st.button("Pause", disabled=not st.session_state.get(running_key, False)):
        st.session_state[running_key] = False
with col3:
    if st.button("Refresh"):
        st.rerun()

if pending == 0:
    st.success("All planned runs have completed (success or failed). See Results Dashboard for analysis.")

if st.session_state.get(running_key, False):
    status_placeholder = st.empty()
    try:
        api_key = openrouter_client.get_api_key()
    except openrouter_client.OpenRouterError as exc:
        st.error(str(exc))
        st.session_state[running_key] = False
    else:
        outcome = runner.execute_one_pending_run(
            None, exp_id, api_key, randomize_order=bool(experiment["randomize_order"]),
        )
        if outcome is None:
            status_placeholder.success("No more pending runs.")
            st.session_state[running_key] = False
        else:
            if outcome.status == "success":
                status_placeholder.info(f"Completed run #{outcome.run_id} on `{outcome.model_id}`.")
            else:
                status_placeholder.error(f"Run #{outcome.run_id} on `{outcome.model_id}` failed: {outcome.error}")
            st.rerun()

st.divider()
with st.expander("Failed runs (errors kept separate from completed responses)"):
    failed_rows = df[df["status"] == "failed"]
    if failed_rows.empty:
        st.caption("No failures.")
    else:
        st.dataframe(failed_rows[["id", "model_id", "run_number", "error_message", "http_status", "retry_count"]],
                     use_container_width=True)
