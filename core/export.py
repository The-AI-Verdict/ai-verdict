"""CSV / Markdown report / JSON archive generation.

Hard rule enforced throughout this module: every number, quote, and example
written into a report comes directly from the experiment's own dataframe or
database rows. Nothing here invents a result, a quote, or a sample size.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from io import StringIO
from typing import Any

import pandas as pd

from core import analysis, db

REPORT_COLUMNS = [
    "run_id", "model_id", "prompt_variant", "run_number", "run_status",
    "doctor_final_classification", "doctor_final_assignment_type",
    "nurse_final_classification", "nurse_final_assignment_type",
    "has_human_review", "evaluation_is_valid", "latency_ms", "cost_usd", "evaluation_cost_usd", "total_tokens",
]


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Minimal markdown table renderer so we don't need the optional
    `tabulate` dependency just for report generation."""
    if df.empty:
        return "_(no data)_"
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def dataframe_to_csv(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    cols = [c for c in (columns or REPORT_COLUMNS) if c in df.columns]
    buf = StringIO()
    df[cols].to_csv(buf, index=False)
    return buf.getvalue()


def build_json_archive(conn: sqlite3.Connection, experiment_id: int) -> dict[str, Any]:
    experiment = db.get_experiment(conn, experiment_id)
    prompt_versions = db.list_prompt_versions(conn, experiment_id)
    models = db.list_experiment_models(conn, experiment_id)
    runs = db.list_runs(conn, experiment_id)
    evaluations = db.list_evaluations_for_experiment(conn, experiment_id)
    reviews = []
    for run in runs:
        reviews.extend(db.list_human_review_history(conn, run["id"]))

    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "experiment": experiment,
        "prompt_versions": prompt_versions,
        "models": models,
        "runs": runs,
        "evaluations": evaluations,
        "human_reviews": reviews,
    }


def _fmt_pct(result: analysis.ProportionResult) -> str:
    lo, hi = result.ci
    exploratory = " (exploratory - small sample)" if result.is_exploratory else ""
    return f"{result.numerator}/{result.denominator} = {result.proportion*100:.1f}% [95% CI {lo*100:.1f}%-{hi*100:.1f}%]{exploratory}"


def generate_markdown_report(conn: sqlite3.Connection, experiment_id: int) -> str:
    experiment = db.get_experiment(conn, experiment_id)
    prompt_versions = db.list_prompt_versions(conn, experiment_id, latest_only=True)
    models = db.list_experiment_models(conn, experiment_id)
    rows = analysis.build_experiment_rows(conn, experiment_id)
    df = analysis.rows_to_dataframe(rows)
    counts = analysis.run_counts(df)
    missing = analysis.missing_and_invalid_counts(df)

    lines: list[str] = []
    lines.append(f"# Experiment Report: {experiment['name']}")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Research question / hypothesis")
    lines.append("")
    lines.append(experiment.get("hypothesis") or "_No hypothesis recorded._")
    lines.append("")
    if experiment.get("description"):
        lines.append(experiment["description"])
        lines.append("")

    lines.append("## Prompt versions tested")
    lines.append("")
    for pv in prompt_versions:
        lines.append(f"### Variant: {pv['variant_name']} (v{pv['version_number']})")
        lines.append("")
        lines.append("```")
        lines.append(pv["prompt_text"])
        lines.append("```")
        lines.append("")

    lines.append("## Models tested")
    lines.append("")
    for m in models:
        lines.append(f"- `{m['model_id']}`" + (f" ({m['provider']})" if m.get("provider") else ""))
    lines.append("")
    drift = analysis.reproducibility_drift_rows(df)
    lines.append(
        "Note: OpenRouter routes requests to upstream providers and does not always guarantee a fixed, "
        "reproducible model snapshot behind a given model id. The exact `model` string echoed back by the "
        "API for every individual run is stored and exported in the raw data / JSON archive. "
        + (
            f"**{len(drift)} run(s) in this experiment actually got a different model string back than "
            f"what was requested** (e.g. a bare id routed to a dated snapshot) - see the Results Dashboard "
            f"or the raw data for exactly which ones."
            if not drift.empty else
            "In this experiment, every successful run's requested model id matched what was echoed back."
        )
    )
    lines.append("")

    lines.append("## Experimental settings")
    lines.append("")
    settings_cols = ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "seed"]
    for col in settings_cols:
        if experiment.get(col) is not None:
            lines.append(f"- {col}: {experiment[col]}")
    lines.append(f"- repetitions per model/prompt: {experiment['repetitions']}")
    lines.append(f"- run order randomised: {'yes' if experiment['randomize_order'] else 'no'}")
    lines.append("")

    lines.append("## Run outcomes")
    lines.append("")
    lines.append(f"- Total planned runs: {counts.total}")
    lines.append(f"- Successful: {counts.success}")
    lines.append(f"- Failed (API errors / exhausted retries): {counts.failed}")
    lines.append(f"- Still pending: {counts.pending}")
    lines.append(f"- Successful responses with an unresolved invalid evaluation (excluded from stats below, need human review): {missing['invalid_unreviewed']}")
    lines.append(f"- Responses marked 'evidence insufficient' by a human reviewer (excluded from stats below): {missing['evidence_insufficient']}")
    lines.append(f"- Responses not yet human-reviewed: {missing['unreviewed_total']}")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(
        "All proportions below use the number of **valid, classified responses** as the denominator "
        "(successful API call + either a valid automated evaluation or a human-reviewed override). "
        "Failed calls and unresolved invalid evaluations are excluded and counted separately above."
    )
    lines.append("")
    for role in ("doctor", "nurse"):
        lines.append(f"### {role.capitalize()}")
        lines.append("")
        summary = analysis.role_summary(df, role)
        for key in ("explicit", "implied", "any_assignment", "unspecified", "ambiguous"):
            lines.append(f"- {summary[key].label}: {_fmt_pct(summary[key])}")
        lines.append("")

    if len(prompt_versions) > 1:
        lines.append("### By prompt variant (across all models)")
        lines.append("")
        for role in ("doctor", "nurse"):
            by_variant = analysis.summarize_by(df, role, ["prompt_variant"])
            if by_variant.empty:
                continue
            lines.append(f"**{role.capitalize()}**")
            lines.append("")
            lines.append(_df_to_markdown_table(by_variant[["prompt_variant", "explicit_count", "explicit_n", "explicit_pct",
                                      "implied_count", "implied_pct", "unspecified_count", "unspecified_pct",
                                      "is_exploratory"]]))
            lines.append("")

        lines.append("### By model, within each prompt variant")
        lines.append("")
        lines.append(
            "A single 'by model' figure averaged across all variants would blend together very different "
            "conditions (e.g. a variant that forces a pronoun choice vs. one that doesn't) into a number "
            "that represents neither - so each variant is broken down separately below."
        )
        lines.append("")
        for pv in prompt_versions:
            variant_df = df[df["prompt_variant"] == pv["variant_name"]]
            if variant_df.empty:
                continue
            lines.append(f"#### Variant: {pv['variant_name']}")
            lines.append("")
            for role in ("doctor", "nurse"):
                by_model = analysis.summarize_by(variant_df, role, ["model_id"])
                if by_model.empty:
                    continue
                lines.append(f"**{role.capitalize()}**")
                lines.append("")
                lines.append(_df_to_markdown_table(by_model[["model_id", "explicit_count", "explicit_n", "explicit_pct",
                                        "implied_count", "implied_pct", "unspecified_count", "unspecified_pct",
                                        "is_exploratory"]]))
                lines.append("")
    else:
        lines.append("### By model")
        lines.append("")
        for role in ("doctor", "nurse"):
            by_model = analysis.summarize_by(df, role, ["model_id"])
            if by_model.empty:
                continue
            lines.append(f"**{role.capitalize()}**")
            lines.append("")
            lines.append(_df_to_markdown_table(by_model[["model_id", "explicit_count", "explicit_n", "explicit_pct",
                                    "implied_count", "implied_pct", "unspecified_count", "unspecified_pct",
                                    "is_exploratory"]]))
            lines.append("")

    disagreements = analysis.disagreement_rows(df)
    lines.append("## Evaluator vs. human reviewer disagreements")
    lines.append("")
    lines.append(f"{len(disagreements)} run(s) where a human reviewer's final classification differed from the automated evaluator.")
    lines.append("")

    lines.append("## Evaluation criteria")
    lines.append("")
    lines.append(
        "An independent model classified each response's references to \"the doctor\" and \"the nurse\" as "
        "explicit_male / explicit_female / explicit_other / unspecified / ambiguous, and separately as an "
        "explicit, implied, or no gender assignment - using only textual evidence (pronouns, gendered titles) "
        "and never inferring gender from the profession, a name, or a stereotype. Every automated "
        "classification required an exact supporting quote from the response; outputs that failed schema or "
        "quote validation were flagged for human review rather than counted automatically. See `core/evaluator.py` "
        "for the exact instructions given to the evaluator model."
    )
    lines.append("")

    lines.append("## Limitations and possible confounders")
    lines.append("")
    lines.append(
        "- A single prompt, even run many times, cannot establish a general stereotype for a model; this "
        "experiment should be read alongside its other prompt variants (role-reversed / alternative "
        "professions) before drawing any broader conclusion.\n"
        "- Repeated samples from the same model at the same prompt are not fully independent observations - "
        "they share the same model weights, system behaviour, and (for deterministic settings) may be highly "
        "correlated. Treat repetition counts as reducing noise, not as independent trials in the classical "
        "statistical sense.\n"
        "- Confidence intervals (Wilson score) describe sampling uncertainty in the observed proportion only; "
        "they say nothing about whether this prompt generalises to other phrasings, domains, or professions.\n"
        "- OpenRouter can route a model id to different underlying provider deployments over time; the exact "
        "`model` string returned per call is recorded, but exact reproducibility of a specific provider's "
        "serving stack is not guaranteed by OpenRouter.\n"
        "- Findings below are exploratory for any group with fewer than "
        f"{analysis.SMALL_SAMPLE_THRESHOLD} valid responses and are labelled as such."
    )
    lines.append("")

    lines.append("## Findings summary (observed data only)")
    lines.append("")
    lines.append(_observed_findings_summary(df))
    lines.append("")

    return "\n".join(lines)


def _observed_findings_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No runs have completed yet."
    bullets = []
    for role in ("doctor", "nurse"):
        s = analysis.role_summary(df, role)
        n = s["explicit"].denominator
        if n == 0:
            bullets.append(f"- {role.capitalize()}: no valid classified responses yet.")
            continue
        exploratory = " (exploratory, small sample)" if n < analysis.SMALL_SAMPLE_THRESHOLD else ""
        bullets.append(
            f"- {role.capitalize()} (n={n}){exploratory}: explicit gender assigned in "
            f"{s['explicit'].proportion*100:.1f}% of responses, implied in {s['implied'].proportion*100:.1f}%, "
            f"unspecified/gender-neutral in {s['unspecified'].proportion*100:.1f}%."
        )
    return "\n".join(bullets)

