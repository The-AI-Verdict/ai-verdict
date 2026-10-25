from __future__ import annotations

import uuid

import streamlit as st

from core import db, openrouter_client, runner, ui_common

st.set_page_config(page_title="Create Experiment - The AI Verdict", layout="wide")
ui_common.bootstrap()

st.title("Create Experiment")

if "draft_variants" not in st.session_state:
    st.session_state["draft_variants"] = [
        {"uid": uuid.uuid4().hex, "variant_name": "baseline", "prompt_text": "", "role_reversed": False, "notes": ""}
    ]
if "draft_manual_models" not in st.session_state:
    st.session_state["draft_manual_models"] = []

st.subheader("1. Basics")
name = st.text_input("Experiment name", value=st.session_state.get("draft_name", ""))
hypothesis = st.text_area("Hypothesis being tested", value=st.session_state.get("draft_hypothesis", ""), height=80)
description = st.text_area("Description (optional)", value=st.session_state.get("draft_description", ""), height=60)

st.divider()
st.subheader("2. Prompt variants")
st.caption(
    "Add one or more prompt variants (e.g. a baseline scenario, a role-reversed version, an alternative "
    "profession). Each becomes an immutable, versioned prompt once the first run against it starts."
)

to_remove = None
for i, variant in enumerate(st.session_state["draft_variants"]):
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            variant["variant_name"] = st.text_input(
                "Variant name", value=variant["variant_name"], key=f"vname_{variant['uid']}"
            )
        with c2:
            variant["role_reversed"] = st.checkbox(
                "Role-reversed variant", value=variant["role_reversed"], key=f"vrev_{variant['uid']}"
            )
        variant["prompt_text"] = st.text_area(
            "Prompt text", value=variant["prompt_text"], height=140, key=f"vtext_{variant['uid']}"
        )
        variant["notes"] = st.text_input("Notes (optional)", value=variant["notes"], key=f"vnotes_{variant['uid']}")
        if len(st.session_state["draft_variants"]) > 1 and st.button("Remove variant", key=f"vremove_{variant['uid']}"):
            to_remove = i

if to_remove is not None:
    st.session_state["draft_variants"].pop(to_remove)
    st.rerun()

if st.button("Add another prompt variant"):
    st.session_state["draft_variants"].append(
        {"uid": uuid.uuid4().hex, "variant_name": "", "prompt_text": "", "role_reversed": False, "notes": ""}
    )
    st.rerun()

st.divider()
st.subheader("3. Models")

key_available = ui_common.api_key_configured()
col1, col2 = st.columns(2)
with col1:
    if st.button("Discover models from OpenRouter", disabled=not key_available):
        try:
            models = openrouter_client.list_models()
            with db.get_connection() as conn:
                db.cache_models(conn, models)
            st.success(f"Discovered and cached {len(models)} models.")
        except openrouter_client.OpenRouterError as exc:
            st.error(f"Could not list models: {exc}")
with col2:
    if not key_available:
        st.caption("Set OPENROUTER_API_KEY to enable live model discovery.")

with db.get_connection() as conn:
    cached_models = db.get_cached_models(conn)

catalog_options = {f"{m['model_id']}  -  {m['name'] or ''}".strip(): m["model_id"] for m in cached_models}
selected_catalog = st.multiselect(
    "Select from discovered models", options=list(catalog_options.keys()),
    default=st.session_state.get("draft_catalog_selection", []),
)
st.session_state["draft_catalog_selection"] = selected_catalog

manual_models_text = st.text_area(
    "Add model IDs manually (one per line, e.g. `openai/gpt-4o-mini`)",
    value="\n".join(st.session_state["draft_manual_models"]), height=80,
)
st.session_state["draft_manual_models"] = [m.strip() for m in manual_models_text.splitlines() if m.strip()]

all_model_ids = sorted(set([catalog_options[k] for k in selected_catalog] + st.session_state["draft_manual_models"]))
if all_model_ids:
    st.write("Selected models:", ", ".join(f"`{m}`" for m in all_model_ids))

st.divider()
st.subheader("4. Repetitions and run settings")
repetitions = st.number_input("Repetitions per prompt variant per model", min_value=1, max_value=200, value=5)
randomize_order = st.checkbox("Randomise run execution order", value=True,
                               help="Plans runs in a fixed order but executes them shuffled, to avoid "
                                    "confounding model/prompt effects with time-of-run effects.")
reveal_hypothesis = st.checkbox(
    "Reveal hypothesis to tested models (NOT recommended - will prime responses)", value=False,
)
if reveal_hypothesis:
    st.error("Revealing the hypothesis to tested models will bias their responses and invalidate the experiment.")

st.divider()
st.subheader("5. Generation parameters")
st.caption("Leave a parameter unchecked to use each provider's default (recorded as 'not set').")

params: dict[str, float | int | None] = {}


def optional_number(label: str, key: str, value_type, default):
    enabled = st.checkbox(f"Set {label}", key=f"enable_{key}")
    if not enabled:
        return None
    if value_type is int:
        return st.number_input(label, value=default, step=1, key=f"val_{key}")
    return st.number_input(label, value=default, step=0.1, key=f"val_{key}")


c1, c2, c3 = st.columns(3)
with c1:
    params["temperature"] = optional_number("temperature", "temperature", float, 0.7)
    params["max_tokens"] = optional_number("max_tokens", "max_tokens", int, 300)
with c2:
    params["top_p"] = optional_number("top_p", "top_p", float, 1.0)
    params["seed"] = optional_number("seed", "seed", int, 42)
with c3:
    params["frequency_penalty"] = optional_number("frequency_penalty", "frequency_penalty", float, 0.0)
    params["presence_penalty"] = optional_number("presence_penalty", "presence_penalty", float, 0.0)

st.divider()
st.subheader("6. Evaluator")
evaluator_model = st.text_input(
    "Evaluator model override (optional)",
    help="Leave blank to use AI_VERDICT_EVALUATOR_MODEL from .env. Prefer a different model from whatever "
         "is under test.",
)

st.divider()
num_variants = len([v for v in st.session_state["draft_variants"] if v["prompt_text"].strip()])
estimated_calls = runner.estimate_call_count(max(num_variants, 1), max(len(all_model_ids), 1), int(repetitions))
st.subheader("Estimated API calls")
st.metric("Tested-model calls (before any retries)", estimated_calls)
st.caption("Evaluator calls are separate: approximately one additional call per successfully completed run.")

if st.button("Save experiment", type="primary"):
    errors = []
    if not name.strip():
        errors.append("Experiment name is required.")
    if num_variants == 0:
        errors.append("At least one prompt variant with text is required.")
    if not all_model_ids:
        errors.append("At least one model must be selected.")
    if errors:
        for e in errors:
            st.error(e)
    else:
        with db.get_connection() as conn:
            experiment_id = db.create_experiment(
                conn, name=name.strip(), hypothesis=hypothesis.strip(), description=description.strip(),
                repetitions=int(repetitions), randomize_order=randomize_order,
                temperature=params["temperature"], max_tokens=params["max_tokens"], top_p=params["top_p"],
                frequency_penalty=params["frequency_penalty"], presence_penalty=params["presence_penalty"],
                seed=params["seed"], reveal_hypothesis_to_model=reveal_hypothesis,
                evaluator_model_id=evaluator_model.strip() or None,
            )
            for variant in st.session_state["draft_variants"]:
                if not variant["prompt_text"].strip():
                    continue
                db.create_prompt_version(
                    conn, experiment_id, variant["variant_name"].strip() or "variant",
                    variant["prompt_text"], notes=variant["notes"], is_role_reversed=variant["role_reversed"],
                )
            for model_id in all_model_ids:
                cached = next((m for m in cached_models if m["model_id"] == model_id), None)
                db.add_experiment_model(
                    conn, experiment_id, model_id,
                    display_name=(cached["name"] if cached else None),
                )
        st.session_state["selected_experiment_id"] = experiment_id
        for k in ("draft_variants", "draft_manual_models", "draft_catalog_selection"):
            st.session_state.pop(k, None)
        st.success(f"Experiment #{experiment_id} saved.")
        st.page_link("pages/2_Run_Experiment.py", label="Go to Run Experiment ->")
