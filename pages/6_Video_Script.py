from __future__ import annotations

import streamlit as st

from core import db, export, sentencing_export, ui_common

st.set_page_config(page_title="Video Script - The AI Verdict", layout="wide")
ui_common.bootstrap()

st.title("Video Script")
st.caption(
    "A full, read-on-camera narration script generated from this experiment's actually recorded data - "
    "not just an outline. Every number and quote in it is sourced and traceable; nothing is invented for "
    "effect. Edit it freely once generated - this is a starting draft, not a final cut."
)

experiment = ui_common.experiment_selector()
if experiment is None:
    st.stop()
exp_id = experiment["id"]
safe_name = "".join(c if c.isalnum() else "_" for c in experiment["name"])[:40]

domain = experiment.get("evaluation_domain") or "gender_healthcare"

with db.get_connection() as conn:
    rows_check = db.list_runs(conn, exp_id)
countable_known = any(r["status"] == "success" for r in rows_check)

if not countable_known:
    st.info("No successful runs yet for this experiment - go to Run Experiment first, then come back here.")
    st.stop()

companion_id = None
if domain == "sentencing_bias":
    with db.get_connection() as conn:
        other_sentencing_experiments = [
            e for e in db.list_experiments(conn)
            if e["id"] != exp_id and (e.get("evaluation_domain") or "gender_healthcare") == "sentencing_bias"
        ]
    if other_sentencing_experiments:
        companion_choice = st.selectbox(
            "Also mention a companion experiment's result (optional)",
            ["(none)"] + [f"#{e['id']} - {e['name']}" for e in other_sentencing_experiments],
            help="If the companion found a genuine null result, one sentence summarising it is folded in - "
                 "never fabricated if the companion actually found something or has no data yet.",
        )
        if companion_choice != "(none)":
            companion_id = int(companion_choice.split(" - ")[0].lstrip("#"))

if st.button("Generate / regenerate script", type="primary"):
    st.session_state[f"script_{exp_id}"] = None  # force regeneration below

cache_key = f"script_{exp_id}"
if cache_key not in st.session_state or st.session_state.get(cache_key) is None:
    with db.get_connection() as conn:
        if domain == "sentencing_bias":
            st.session_state[cache_key] = sentencing_export.generate_full_script(
                conn, exp_id, companion_experiment_id=companion_id,
            )
        else:
            st.session_state[cache_key] = export.generate_full_script(conn, exp_id)

script_text = st.session_state[cache_key]

st.download_button(
    "Download script.md", script_text, file_name=f"{safe_name}_script.md", mime="text/markdown",
)

st.divider()

view_mode = st.radio("View as", ["Formatted", "Plain text (for editing/copying)"], horizontal=True)
if view_mode == "Formatted":
    st.markdown(script_text)
else:
    st.text_area("Script", script_text, height=700, label_visibility="collapsed")

st.divider()
if domain == "sentencing_bias":
    st.caption(
        "How this is structured: if one model accounts for nearly every custodial recommendation, the "
        "script builds a three-act structure around that model specifically - what it did, a confound "
        "check if a dialect_formal_control variant exists, and why the finding doesn't generalise to "
        "'AI sentencing bias' as a whole. If no single model dominates, it falls back to a single, simpler "
        "results readout instead. It never invents a one-model narrative the data doesn't actually support."
    )
else:
    st.caption(
        "How this is structured: if this experiment has a prompt variant where models stayed almost entirely "
        "neutral and another where they were forced to commit to a pronoun, the script builds a two-act "
        "'spontaneous vs. forced' narrative around that contrast - with a per-model breakdown of which gender "
        "each model defaulted to. If there's no such contrast in the data, it falls back to a single, simpler "
        "results-and-caveats structure instead. It never invents a dramatic structure the numbers don't "
        "actually support."
    )
