"""CSV / Markdown report / JSON archive / YouTube brief generation.

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
    "has_human_review", "evaluation_is_valid", "latency_ms", "cost_usd", "total_tokens",
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
    lines.append(
        "Note: OpenRouter routes requests to upstream providers and does not always guarantee a fixed, "
        "reproducible model snapshot behind a given model id. The exact `model` string echoed back by the "
        "API for every individual run is stored and exported in the raw data / JSON archive."
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

    if len(prompt_versions) > 1:
        lines.append("### By prompt variant")
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


def generate_video_brief(conn: sqlite3.Connection, experiment_id: int) -> str:
    experiment = db.get_experiment(conn, experiment_id)
    rows = analysis.build_experiment_rows(conn, experiment_id)
    df = analysis.rows_to_dataframe(rows)

    lines: list[str] = []
    lines.append(f"# YouTube Production Brief: {experiment['name']}")
    lines.append("")
    lines.append(
        "Everything below is derived from the experiment's actually recorded runs. Sections marked "
        "**[interpretation]** are suggested framing for you to confirm on camera - they are not themselves "
        "experimental findings."
    )
    lines.append("")

    lines.append("## Three possible titles **[interpretation]**")
    lines.append("")
    lines.append(f"1. \"{experiment['name']}\"")
    lines.append("2. \"I Tested AI Models For Hidden Gender Bias - Here's What They Said\"")
    lines.append("3. \"Does ChatGPT Assume Your Nurse Is a Woman? I Ran The Experiment\"")
    lines.append("")

    lines.append("## Suggested opening hook **[interpretation]**")
    lines.append("")
    lines.append(
        "Read the exact prompt on screen, then ask the audience to guess: will the AI assume a gender for "
        "the doctor or the nurse, even though none was given? Reveal that you ran this prompt dozens of times "
        "across multiple models to find out - with receipts."
    )
    lines.append("")

    lines.append("## Narration outline **[interpretation]**")
    lines.append("")
    lines.append(
        "1. State the hypothesis being tested (from the experiment record) without revealing it to the models.\n"
        "2. Show the exact prompt text.\n"
        "3. Explain the method: same prompt, N repetitions, multiple models, independent evaluator model scoring "
        "each response for explicit vs implied vs no gender assignment.\n"
        "4. Reveal the per-model results (see table below).\n"
        "5. Show 2-3 real response excerpts on screen (see 'Real outputs to show' below).\n"
        "6. State the limitations plainly - one prompt, small-sample caveats, non-independence of repeated samples.\n"
        "7. Balanced conclusion."
    )
    lines.append("")

    lines.append("## Suggested screen-recording sequence **[interpretation]**")
    lines.append("")
    lines.append(
        "- Record the Create Experiment screen showing the prompt and model selection.\n"
        "- Record the Run Experiment screen with live progress as calls complete.\n"
        "- Record the Human Review screen showing a real response alongside its automated classification and quote.\n"
        "- Record the Results Dashboard with the final per-model chart."
    )
    lines.append("")

    lines.append("## Real outputs to show on screen (actual recorded data)")
    lines.append("")
    lines.append(_pick_example_quotes(df))
    lines.append("")

    lines.append("## Results-reveal sequence (actual recorded numbers)")
    lines.append("")
    for role in ("doctor", "nurse"):
        by_model = analysis.summarize_by(df, role, ["model_id"])
        if by_model.empty:
            lines.append(f"No completed, classified data yet for {role}.")
            continue
        lines.append(f"**{role.capitalize()}** - explicit assignment rate by model:")
        lines.append("")
        for _, r in by_model.iterrows():
            flag = " (exploratory)" if r["is_exploratory"] else ""
            lines.append(f"- {r['model_id']}: {r['explicit_count']}/{r['explicit_n']} = {r['explicit_pct']}%{flag}")
        lines.append("")

    lines.append("## Balanced conclusion **[interpretation - confirm against final data before recording]**")
    lines.append("")
    lines.append(_observed_findings_summary(df))
    lines.append("")
    lines.append(
        "Remind viewers: this is one prompt tested a limited number of times. It does not prove a model is "
        "'biased' in general - it shows what that model did, in these specific recorded runs, on this specific "
        "prompt. Encourage checking the other prompt variants in this experiment (role-reversed, alternative "
        "professions) before generalising."
    )
    lines.append("")

    lines.append("## Methodology description (for the video description box)")
    lines.append("")
    lines.append(
        f"We ran the prompt \"{experiment['name']}\" against multiple language models via the OpenRouter API, "
        f"{experiment['repetitions']} time(s) per model per prompt variant. Responses were never shown the "
        "hypothesis being tested. Each response was then classified by an independent model using a structured "
        "rubric that only counts explicit textual evidence (pronouns, gendered titles) - never profession, name, "
        "or stereotype - as a gender assignment, with every automated classification backed by an exact quote. "
        "A subset of responses was manually reviewed by a human to check the automated classifier's work. Full "
        "methodology, raw responses, and code: [link]."
    )
    lines.append("")

    return "\n".join(lines)


def _pick_example_quotes(df: pd.DataFrame, max_examples: int = 4) -> str:
    if df.empty:
        return "No completed runs yet - nothing to show."
    candidates = df[df["run_status"] == "success"]
    with_assignment = candidates[
        (candidates["doctor_final_assignment_type"].isin(["explicit", "implied"]))
        | (candidates["nurse_final_assignment_type"].isin(["explicit", "implied"]))
    ]
    chosen = with_assignment.head(max_examples)
    if chosen.empty:
        return (
            "No response in the currently recorded data had an explicit or implied gender assignment for "
            "either role - i.e. every classified response so far was gender-neutral. That is itself a real, "
            "reportable finding; do not fabricate an example to the contrary."
        )
    out = []
    for _, r in chosen.iterrows():
        excerpt = (r["raw_response"] or "")[:400]
        out.append(
            f"- **{r['model_id']}** (run #{r['run_number']}, variant `{r['prompt_variant']}`): "
            f"doctor={r['doctor_final_classification']}/{r['doctor_final_assignment_type']}, "
            f"nurse={r['nurse_final_classification']}/{r['nurse_final_assignment_type']}\n\n"
            f"  > {excerpt}{'...' if len(r['raw_response'] or '') > 400 else ''}"
        )
    return "\n\n".join(out)
