"""CSV / Markdown report / JSON archive / YouTube brief generation.

Hard rule enforced throughout this module: every number, quote, and example
written into a report comes directly from the experiment's own dataframe or
database rows. Nothing here invents a result, a quote, or a sample size.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
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


def generate_video_brief(conn: sqlite3.Connection, experiment_id: int) -> str:
    experiment = db.get_experiment(conn, experiment_id)
    prompt_versions = db.list_prompt_versions(conn, experiment_id, latest_only=True)
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
    if len(prompt_versions) > 1:
        lines.append(
            "Broken down per prompt variant, not averaged together - a figure blended across variants with "
            "very different conditions (e.g. one that forces a pronoun choice and one that doesn't) would "
            "misrepresent all of them."
        )
        lines.append("")
        for pv in prompt_versions:
            variant_df = df[df["prompt_variant"] == pv["variant_name"]]
            if variant_df.empty:
                continue
            lines.append(f"### Variant: {pv['variant_name']}")
            lines.append("")
            for role in ("doctor", "nurse"):
                by_model = analysis.summarize_by(variant_df, role, ["model_id"])
                if by_model.empty:
                    continue
                lines.append(f"**{role.capitalize()}** - explicit assignment rate by model:")
                lines.append("")
                for _, r in by_model.iterrows():
                    flag = " (exploratory)" if r["is_exploratory"] else ""
                    lines.append(f"- {r['model_id']}: {r['explicit_count']}/{r['explicit_n']} = {r['explicit_pct']}%{flag}")
                lines.append("")
    else:
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


# ---------------------------------------------------------------------------
# Full narration script generator
#
# Separate from generate_video_brief (which is a production OUTLINE: titles,
# beat list, numbers). This generates an actual word-for-word narration
# script. It still obeys the same hard rule as the rest of this module: every
# number and quote is pulled live from the experiment's own data, nothing is
# invented. The "narrative" parts (the exact phrasing, jokes, pacing) are
# synthesised, but every factual claim embedded in that narration is sourced
# and the source is traceable (n, %, run id) right next to the claim.
# ---------------------------------------------------------------------------

GENDERED_CLASSIFICATIONS = {"explicit_male", "explicit_female", "explicit_other"}
CONTRAST_THRESHOLD = 0.20  # percentage-point gap that counts as a real "twist" between two variants
PAIRING_MAJORITY = 0.6  # a model's dominant classification must hold at least this share of its reps


def _variant_headline_rate(
    df: pd.DataFrame, variant_name: str, field_specs: list[tuple[str, set]],
) -> tuple[float, int]:
    """Combined headline rate for one variant, across one or more
    (field_col, positive_values) pairs - e.g. the gender domain combines
    doctor_final_assignment_type and nurse_final_assignment_type into one
    rate; a single-field domain (sentencing) passes just one pair. This is
    the domain-agnostic engine behind every narrative generator's variant
    ranking - a new domain needs no new ranking code, just its own field
    names and positive-value set."""
    vdf = df[(df["prompt_variant"] == variant_name) & (df["is_countable"])]
    if vdf.empty:
        return 0.0, 0
    total_num = 0
    total_den = 0
    for field_col, positive_values in field_specs:
        r = analysis.categorical_rate(vdf, field_col, positive_values, countable_col="is_countable")
        total_num += r.numerator
        total_den += r.denominator
    rate = total_num / total_den if total_den else 0.0
    return rate, total_den


def _variant_any_assignment_rate(df: pd.DataFrame, variant_name: str) -> tuple[float, int]:
    """Combined (doctor + nurse) any-assignment rate for one variant, used
    only to rank variants by how "revealing" they were - the underlying
    doctor/nurse breakdown is still reported separately everywhere else.
    The gender domain's own field_specs, passed into the generic engine."""
    return _variant_headline_rate(df, variant_name, [
        ("doctor_final_assignment_type", {"explicit", "implied"}),
        ("nurse_final_assignment_type", {"explicit", "implied"}),
    ])


def _pick_quote(df: pd.DataFrame, variant_name: str, role: str, assignment_types: list[str],
                 exclude_models: set[str] | None = None) -> dict[str, Any] | None:
    vdf = df[
        (df["prompt_variant"] == variant_name) & (df["is_countable"])
        & (df[f"{role}_final_assignment_type"].isin(assignment_types))
    ]
    if exclude_models:
        vdf = vdf[~vdf["model_id"].isin(exclude_models)]
    if vdf.empty:
        return None
    row = vdf.iloc[0]
    return {
        "model_id": row["model_id"], "run_number": row["run_number"],
        "classification": row[f"{role}_final_classification"], "quote": row[f"{role}_auto_quote"],
        "response": row["raw_response"],
    }


def _blockquote(text: str) -> str:
    """Markdown blockquote syntax requires '> ' on every line, not just the
    first - a quoted model response with internal newlines (markdown
    headers, bullet points, blank lines between paragraphs) would otherwise
    fall out of the blockquote after the first line break and render as
    regular body text. Shared by every domain's narrative generator that
    quotes a raw model response verbatim."""
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _truncate_at_word(text: str, limit: int) -> str:
    if not text or len(text) <= limit:
        return text or ""
    cut = text[:limit]
    last_space = cut.rfind(" ")
    if last_space > limit * 0.6:  # don't chop off most of the text looking for a boundary
        cut = cut[:last_space]
    return cut.rstrip() + "..."


def _speakable(variant_name: str) -> str:
    """Variant slugs like 'forced_pronoun' are fine in [ON SCREEN] tags but
    nobody says an underscore out loud - this is for text meant to be read
    as narration."""
    return variant_name.replace("_", " ")


def _model_pairings(df: pd.DataFrame, variant_name: str) -> list[dict[str, Any]]:
    """Per model (within one variant): which gender classification it most
    often used for the doctor vs. the nurse, and how consistent that was.
    Used to find models worth naming on camera - not to compute any of the
    headline statistics, which always come from analysis.role_summary."""
    vdf = df[(df["prompt_variant"] == variant_name) & (df["is_countable"])]
    out = []
    for model_id in sorted(vdf["model_id"].unique()):
        sub = vdf[vdf["model_id"] == model_id]
        n = len(sub)
        if n == 0:
            continue
        doc_counts = sub["doctor_final_classification"].value_counts()
        nur_counts = sub["nurse_final_classification"].value_counts()
        doc_mode, doc_mode_n = doc_counts.index[0], int(doc_counts.iloc[0])
        nur_mode, nur_mode_n = nur_counts.index[0], int(nur_counts.iloc[0])
        out.append({
            "model_id": model_id, "n": n,
            "doctor_mode": doc_mode, "doctor_mode_pct": doc_mode_n / n,
            "nurse_mode": nur_mode, "nurse_mode_pct": nur_mode_n / n,
        })
    return out


def _classify_pairing(p: dict[str, Any]) -> str:
    doc_consistent = p["doctor_mode_pct"] >= PAIRING_MAJORITY and p["doctor_mode"] in GENDERED_CLASSIFICATIONS
    nur_consistent = p["nurse_mode_pct"] >= PAIRING_MAJORITY and p["nurse_mode"] in GENDERED_CLASSIFICATIONS
    if not (doc_consistent and nur_consistent):
        return "mixed"
    if p["doctor_mode"] == p["nurse_mode"]:
        return "same_gender_both"
    if p["doctor_mode"] == "explicit_male" and p["nurse_mode"] == "explicit_female":
        return "matches_stereotype"
    if p["doctor_mode"] == "explicit_female" and p["nurse_mode"] == "explicit_male":
        return "inverts_stereotype"
    return "mixed"


@dataclass
class VariantRanking:
    variant_ranking: list[tuple[dict, float, int]]  # (prompt_version, combined_rate, n), ascending by rate
    has_contrast: bool
    low_pv: dict | None
    low_rate: float
    low_n: int
    high_pv: dict | None
    high_rate: float
    high_n: int


def _rank_variants_by_headline_rate(
    df: pd.DataFrame, prompt_versions: list[dict], field_specs: list[tuple[str, set]],
) -> VariantRanking:
    """Domain-agnostic engine behind every narrative generator's variant
    ranking: decides whether this experiment has a 'before vs after' contrast
    worth building a two-part structure around, and which variant is the
    thematically central one to lead with. A new domain's ranking is one call
    to this with its own field_specs - no new ranking/contrast-detection code
    needed. Tie-break always prefers 'baseline' among equal rates."""
    variant_ranking = []
    for pv in prompt_versions:
        rate, n = _variant_headline_rate(df, pv["variant_name"], field_specs)
        if n > 0:
            variant_ranking.append((pv, rate, n))
    variant_ranking.sort(key=lambda t: (t[1], 0 if t[0]["variant_name"] == "baseline" else 1))

    has_contrast = (
        len(variant_ranking) >= 2
        and (variant_ranking[-1][1] - variant_ranking[0][1]) >= CONTRAST_THRESHOLD
    )
    low_pv, low_rate, low_n = variant_ranking[0] if variant_ranking else (None, 0.0, 0)
    high_pv, high_rate, high_n = variant_ranking[-1] if variant_ranking else (None, 0.0, 0)
    return VariantRanking(variant_ranking, has_contrast, low_pv, low_rate, low_n, high_pv, high_rate, high_n)


def _rank_variants_by_assignment_rate(df: pd.DataFrame, prompt_versions: list[dict]) -> VariantRanking:
    """The gender domain's own field_specs, passed into the generic ranking
    engine - kept under its original name since generate_full_script and
    generate_blog_post already call it by this name."""
    return _rank_variants_by_headline_rate(df, prompt_versions, [
        ("doctor_final_assignment_type", {"explicit", "implied"}),
        ("nurse_final_assignment_type", {"explicit", "implied"}),
    ])


def _already_narrated_variant_names(ranking: VariantRanking) -> set[str]:
    """Which variant(s) the narrative body (Act 1/Act 2, or Round 1/Round 2)
    actually discusses, so a closing 'I also ran N other versions to check'
    caveat doesn't double-count a variant as unmentioned robustness-checking
    when it was literally the subject of an earlier section. In the
    has_contrast case both low_pv and high_pv get their own section; outside
    that, only high_pv does."""
    names: set[str] = set()
    if ranking.has_contrast:
        if ranking.low_pv:
            names.add(ranking.low_pv["variant_name"])
        if ranking.high_pv:
            names.add(ranking.high_pv["variant_name"])
    elif ranking.high_pv:
        names.add(ranking.high_pv["variant_name"])
    return names


def _fmt_pairing_line(p: dict[str, Any]) -> str:
    doc = p["doctor_mode"].replace("explicit_", "")
    nur = p["nurse_mode"].replace("explicit_", "")
    return (f"`{p['model_id']}` - doctor: {doc} ({p['doctor_mode_pct']*100:.0f}% of {p['n']}), "
            f"nurse: {nur} ({p['nurse_mode_pct']*100:.0f}% of {p['n']})")


def generate_full_script(conn: sqlite3.Connection, experiment_id: int) -> str:
    """A complete, read-on-camera narration script built from this
    experiment's actual recorded data. Structure adapts to what the data
    actually shows (a single-variant experiment gets a single-act script; a
    multi-variant experiment with a sharp contrast between two variants gets
    a two-act 'before/after' structure) - it never forces a narrative the
    numbers don't support."""
    experiment = db.get_experiment(conn, experiment_id)
    prompt_versions = db.list_prompt_versions(conn, experiment_id, latest_only=True)
    rows = analysis.build_experiment_rows(conn, experiment_id)
    df = analysis.rows_to_dataframe(rows)
    counts = analysis.run_counts(df)
    missing = analysis.missing_and_invalid_counts(df)
    n_models = df["model_id"].nunique() if not df.empty else 0

    L: list[str] = []
    L.append(f"# Video Script: {experiment['name']}")
    L.append("")
    L.append(
        "This is a full narration script, generated from this experiment's actually recorded data - not a "
        "summary or an outline. Read the plain text as spoken narration. Lines in **[brackets]** are "
        "production notes (on-screen text, cuts, b-roll) - don't read those aloud. Every number and quote "
        "below is sourced from this experiment and traceable in the raw data / JSON archive; nothing is "
        "invented for effect."
    )
    L.append("")

    countable_total = int(df["is_countable"].sum()) if not df.empty else 0
    if countable_total == 0:
        L.append(
            "No classified results are available yet for this experiment - run it and review/resolve any "
            "invalid evaluations first, then regenerate this script."
        )
        return "\n".join(L)

    # Rank variants by how "revealing" they were, to decide the script structure.
    ranking = _rank_variants_by_assignment_rate(df, prompt_versions)
    has_contrast = ranking.has_contrast
    low_pv, low_rate, low_n = ranking.low_pv, ranking.low_rate, ranking.low_n
    high_pv, high_rate, high_n = ranking.high_pv, ranking.high_rate, ranking.high_n

    # ---------------- COLD OPEN ----------------
    L.append("## [COLD OPEN - 0:00]")
    L.append("")
    if has_contrast:
        L.append(f'**[ON SCREEN: the exact prompt text from "{low_pv["variant_name"]}"]**')
        L.append("")
        L.append(
            f'"{n_models} AI models. The exact same underlying scenario. Nobody told them what I was '
            f'actually testing. Under one version of the prompt, the combined rate of assigning a gender '
            f'to anyone was just {low_rate*100:.0f}%."'
        )
        L.append("")
        L.append("**[BEAT - let that number sit on screen]**")
        L.append("")
        L.append(
            f'"Under a different version of the exact same scenario - you\'ll see exactly what changed - '
            f'that number jumps to {high_rate*100:.0f}%."'
        )
        L.append("")
        L.append('"So which is the real answer? Stick around, because it\'s more interesting than either number alone."')
    else:
        overall_doc = analysis.role_summary(df[df["is_countable"]], "doctor")["any_assignment"]
        overall_nur = analysis.role_summary(df[df["is_countable"]], "nurse")["any_assignment"]
        L.append('**[ON SCREEN: the exact prompt text]**')
        L.append("")
        L.append(
            f'"I ran this exact prompt against {n_models} AI models, {counts.success} times total. No model '
            f'was told this was a test of anything. For the doctor, {overall_doc.numerator}/{overall_doc.denominator} '
            f'responses ({overall_doc.proportion*100:.0f}%) assigned a gender. For the nurse, '
            f'{overall_nur.numerator}/{overall_nur.denominator} ({overall_nur.proportion*100:.0f}%). '
            f'Here\'s exactly what that looked like."'
        )
    L.append("")

    # ---------------- INTRO ----------------
    L.append("## [INTRO - METHODOLOGY]")
    L.append("")
    L.append(f'"Here\'s the hypothesis I went in with:"')
    L.append("")
    L.append(f"> {experiment['hypothesis'] or '(no hypothesis recorded)'}")
    L.append("")
    L.append(
        f'"To test it fairly, I used the same prompt, the same settings, and multiple repetitions per '
        f'model - {experiment["repetitions"]} per model per variant - run in randomised order so I '
        f'couldn\'t accidentally cherry-pick a good run. I also used a completely separate AI model as an '
        f'independent judge, which never saw the hypothesis and was only shown the response text - '
        f'scoring it on whether it used an actual gendered pronoun or title, never guessing from the '
        f'profession, a name, or a stereotype. Every single classification had to be backed by an exact '
        f'quote pulled from the response - no quote, no count."'
    )
    L.append("")
    intro_stats = f'"{counts.total} total calls, {counts.success} completed successfully'
    if counts.failed:
        intro_stats += f", {counts.failed} failed and excluded"
    evidence_insufficient_n = missing["evidence_insufficient"]
    if evidence_insufficient_n:
        intro_stats += f", {evidence_insufficient_n} excluded for insufficient evidence (I'll show you why later)"
    intro_stats += '."'
    L.append(intro_stats)
    L.append("")

    # ---------------- ACT 1 ----------------
    def _act(pv, rate, n, act_label, framing_open, framing_close):
        vdf = df[(df["prompt_variant"] == pv["variant_name"]) & (df["is_countable"])]
        doc = analysis.role_summary(vdf, "doctor")
        nur = analysis.role_summary(vdf, "nurse")
        L.append(act_label)
        L.append("")
        L.append(f'**[ON SCREEN: variant "{pv["variant_name"]}" - full prompt text below]**')
        L.append("")
        L.append(f'> {pv["prompt_text"]}')
        L.append("")
        L.append(f'"{framing_open}"')
        L.append("")
        L.append(f'"For the **doctor**: {doc["explicit"].numerator}/{doc["explicit"].denominator} explicit '
                  f'({doc["explicit"].proportion*100:.1f}%), {doc["implied"].numerator} implied, '
                  f'{doc["unspecified"].numerator} stayed neutral."')
        L.append("")
        L.append(f'"For the **nurse**: {nur["explicit"].numerator}/{nur["explicit"].denominator} explicit '
                  f'({nur["explicit"].proportion*100:.1f}%), {nur["implied"].numerator} implied, '
                  f'{nur["unspecified"].numerator} stayed neutral."')
        L.append("")
        quote = _pick_quote(df, pv["variant_name"], "doctor", ["explicit", "implied"]) \
            or _pick_quote(df, pv["variant_name"], "nurse", ["explicit", "implied"])
        if quote:
            L.append(f'**[ON SCREEN: real response from `{quote["model_id"]}`, run #{quote["run_number"]}]**')
            L.append("")
            L.append(f'> {_truncate_at_word(quote["response"], 350)}')
            L.append("")
            if quote.get("quote"):
                quote_text = quote["quote"].strip().rstrip(".")
                L.append(f'"Notice the exact phrase here: \'{quote_text}\'. That\'s the evidence - not a guess."')
            L.append("")
        else:
            L.append(
                '"I don\'t have a gendered example to show you from this variant - every single response '
                'that counted stayed neutral. That absence is itself the finding here, so I\'m not going '
                'to manufacture a dramatic clip that doesn\'t exist."'
            )
            L.append("")
        L.append(f'"{framing_close}"')
        L.append("")

    if has_contrast:
        _act(
            low_pv, low_rate, low_n, f"## [ACT 1 - THE \"{_speakable(low_pv['variant_name']).upper()}\" VERSION]",
            "Here's what happened under this version of the prompt - full text on screen, so you can "
            "judge for yourself exactly what it does and doesn't ask for.",
            f"Combined across doctor and nurse, that's a {low_rate*100:.0f}% gender-assignment rate under "
            f"this version.",
        )
        _act(
            high_pv, high_rate, high_n, f"## [ACT 2 - THE \"{_speakable(high_pv['variant_name']).upper()}\" VERSION]",
            "Now here's the exact same underlying scenario, but a different version of the prompt. Look "
            "closely at what's actually different on screen - I'm not going to characterise the change "
            "for you, because the prompt speaks for itself.",
            f"The combined rate jumps to {high_rate*100:.0f}%. Whatever this version of the prompt changes, "
            f"it clearly matters - that's not noise, that's a {abs(high_rate-low_rate)*100:.0f} point swing.",
        )
    else:
        pv = high_pv
        if pv:
            _act(pv, high_rate, high_n, "## [THE RESULTS]",
                 "Here's what actually happened across every model I tested.",
                 "That's the topline. Now let's look at whether every model behaved the same way - "
                 "because averages hide a lot.")

    # ---------------- ACT 3: PER-MODEL ----------------
    focus_pv = high_pv
    if focus_pv:
        pairings = [p for p in _model_pairings(df, focus_pv["variant_name"]) if p["n"] > 0]
        by_class = {"matches_stereotype": [], "inverts_stereotype": [], "same_gender_both": [], "mixed": []}
        for p in pairings:
            by_class[_classify_pairing(p)].append(p)

        L.append("## [ACT 3 - NOT EVERY MODEL AGREES]")
        L.append("")
        L.append(
            f'"Here\'s where it gets interesting. I broke this down model by model, within the '
            f'{_speakable(focus_pv["variant_name"])} variant. These are small samples per model - '
            f'{experiment["repetitions"]} each - so treat this as a lead to investigate further, not a '
            f'final verdict on any single model."'
        )
        L.append("")
        if by_class["matches_stereotype"]:
            L.append(f'"{len(by_class["matches_stereotype"])} model(s) went the stereotypical way every time - '
                      f'male doctor, female nurse:"')
            L.append("")
            for p in by_class["matches_stereotype"][:5]:
                L.append(f"- {_fmt_pairing_line(p)}")
            L.append("")
        if by_class["inverts_stereotype"]:
            L.append(f'"But {len(by_class["inverts_stereotype"])} model(s) did the exact opposite - every '
                      f'single time:"')
            L.append("")
            for p in by_class["inverts_stereotype"][:5]:
                L.append(f"- {_fmt_pairing_line(p)}")
            L.append("")
            L.append(
                '"That\'s not noise. That\'s a model with a consistent internal default that runs the '
                'other way. Which already tells you this isn\'t one universal AI stereotype - it\'s a '
                'per-model one."'
            )
            L.append("")
        if by_class["same_gender_both"]:
            L.append(f'"{len(by_class["same_gender_both"])} model(s) assigned the SAME gender to both the '
                      f'doctor and the nurse, consistently:"')
            L.append("")
            for p in by_class["same_gender_both"][:5]:
                L.append(f"- {_fmt_pairing_line(p)}")
            L.append("")
        if by_class["mixed"]:
            L.append(f'"And {len(by_class["mixed"])} model(s) didn\'t show a consistent pattern at all across '
                      f'its {experiment["repetitions"]} runs - it varied."')
            L.append("")
        if not (by_class["matches_stereotype"] or by_class["inverts_stereotype"]):
            L.append(
                '"No model showed a clean, consistent male-doctor/female-nurse pattern in this sample - '
                'worth being upfront about that rather than forcing a cleaner story than the data supports."'
            )
            L.append("")

    # ---------------- LIMITATIONS ----------------
    L.append("## [WHAT THIS DOES AND DOESN'T PROVE]")
    L.append("")
    narrated = _already_narrated_variant_names(ranking)
    other_variants = [pv["variant_name"] for pv in prompt_versions if pv["variant_name"] not in narrated]
    if other_variants:
        variant_list = ", ".join(_speakable(v) for v in other_variants)
        version_word = "version" if len(other_variants) == 1 else "versions"
        L.append(
            f'"A few honest caveats before I wrap up. This is one scenario. I also ran {len(other_variants)} '
            f'other {version_word} of it - {variant_list} - to check whether this holds up outside one '
            f'specific framing, and I\'ll put those results in the full report, linked below."'
        )
    else:
        L.append(
            '"A few honest caveats before I wrap up. This is one scenario, run one way. A single prompt - '
            'however many times you repeat it - can\'t establish a general pattern on its own; it needs '
            'variants that change the framing to see whether the result holds up."'
        )
    L.append("")
    L.append(
        '"The per-model breakdown you just saw is based on a handful of runs per model - that\'s '
        'exploratory, not statistically conclusive. If a result surprised you, that\'s exactly the kind of '
        'thing worth re-running with a bigger sample before treating it as settled."'
    )
    L.append("")
    L.append(
        '"And repeated runs from the same model aren\'t fully independent samples - they share the same '
        'underlying weights. More repetitions reduce noise, they don\'t turn this into a controlled trial."'
    )
    L.append("")
    if missing["evidence_insufficient"] > 0:
        L.append(
            f'"One more thing, because I want to show my work: {missing["evidence_insufficient"]} response(s) '
            f'in this dataset got excluded entirely - not because the result was inconvenient, but because '
            f'the model\'s answer was cut off before it finished, usually a reasoning model that burned its '
            f'whole budget thinking instead of answering. That\'s a token-limit bug, not a finding, so I '
            f'pulled it out rather than let it quietly skew the numbers."'
        )
        L.append("")

    # ---------------- CLOSING ----------------
    L.append("## [CLOSING - CALL TO ACTION]")
    L.append("")
    L.append(
        '"So: these models mostly don\'t volunteer a gender. But the moment you take away their ability '
        'to dodge the question, most of them default to something - and which default depends on which '
        'model you\'re talking to."'
    )
    L.append("")
    L.append(
        '"Full dataset, every raw response, the exact evaluator prompt I used to score this fairly - all '
        'linked below, so you can check my work instead of just taking my word for it. If you want me to '
        'run this on a different profession pair, or test a specific model, tell me in the comments."'
    )
    L.append("")

    # ---------------- PRODUCTION CHECKLIST ----------------
    L.append("## [PRODUCTION CHECKLIST]")
    L.append("")
    L.append("- [ ] Record Create Experiment screen (prompt + model list)")
    L.append("- [ ] Record Run Experiment screen with live progress")
    L.append("- [ ] Record Human Review screen showing a real response + automated classification side by side")
    L.append("- [ ] Record Results Dashboard with the per-model chart for the focus variant")
    L.append("- [ ] On-screen callouts for every quoted number above (use the exact n/% shown, not rounded further)")
    L.append("- [ ] Link in description: full report, JSON archive, and (if public) the repo")
    L.append("")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Medium-style blog post generator
#
# Different job from the video script: this is prose meant to be read, not
# heard, and it has to not read as machine-generated. That rules out the
# tells AI writing is known for - hedge-every-sentence, "moreover"/"delve
# into"/"navigate the landscape"/"it's important to note", rule-of-three
# lists, uniform paragraph lengths, a perfectly symmetric intro-body-
# conclusion skeleton. The fix isn't a disclaimer; it's actually varying
# sentence length, stating opinions plainly, and leaning on specific real
# numbers and quotes instead of generic summary language - specificity is
# the thing generic AI copy is missing, so the rule here is simple: never
# say something vague when a real number or a real quote is sitting right
# there in the data.
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def _prose_list(items: list[str], max_items: int = 6) -> str:
    """'a', 'a and b', 'a, b and c' - proper list grammar instead of a bare
    comma-join, which reads like a machine-generated enumeration. Beyond
    max_items this names the first few and states a count for the rest,
    since naming ten things in a row stops being readable prose."""
    if not items:
        return ""
    if len(items) > max_items:
        shown = items[: max_items - 1]
        return f"{', '.join(shown)}, and {len(items) - len(shown)} others"
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def generate_blog_post(conn: sqlite3.Connection, experiment_id: int) -> str:
    """A full Medium-ready draft (headline options, dek, body copy) built
    from this experiment's real data. Works for any experiment - the
    narrative shape adapts to what the numbers actually show instead of
    forcing a fixed template. This is a draft to personalise, not a
    finished article: the one thing it genuinely can't write for you is
    your own voice, so read it out loud before publishing and cut anything
    that doesn't sound like you."""
    experiment = db.get_experiment(conn, experiment_id)
    prompt_versions = db.list_prompt_versions(conn, experiment_id, latest_only=True)
    rows = analysis.build_experiment_rows(conn, experiment_id)
    df = analysis.rows_to_dataframe(rows)
    counts = analysis.run_counts(df)
    missing = analysis.missing_and_invalid_counts(df)
    n_models = df["model_id"].nunique() if not df.empty else 0

    countable_total = int(df["is_countable"].sum()) if not df.empty else 0
    if countable_total == 0:
        return (
            f"# {experiment['name']}\n\n"
            "No classified results are available yet for this experiment. Run it and resolve any "
            "invalid evaluations first, then come back and generate the post."
        )

    ranking = _rank_variants_by_assignment_rate(df, prompt_versions)
    P: list[str] = []

    # ---------------- HEADLINE OPTIONS ----------------
    P.append(f"# {experiment['name']}")
    P.append("")
    P.append(
        "*(Pick one of these as your actual Medium title, or rewrite it - these are starting points, not "
        "a final answer. Delete this note before publishing.)*"
    )
    P.append("")
    if ranking.has_contrast:
        P.append(f"- \"{experiment['name']}\"")
        P.append(
            f"- \"I Asked {n_models} AI Models the Same Question Two Ways. The Numbers Didn't Match.\""
        )
        P.append(f"- \"What Happens When You Take Away an AI's Favourite Way to Dodge a Question\"")
    else:
        P.append(f"- \"{experiment['name']}\"")
        P.append(f"- \"I Tested {n_models} AI Models on the Same Prompt. Here's the Actual Data.\"")
    P.append("")
    P.append(f"*{experiment['hypothesis'] or 'An experiment in finding out, instead of assuming.'}*")
    P.append("")
    P.append("---")
    P.append("")

    # ---------------- OPENING ----------------
    if ranking.has_contrast:
        low_pct, high_pct = ranking.low_rate * 100, ranking.high_rate * 100
        P.append(
            f"Here's a number first, because it's more interesting than anything I could say about it: "
            f"{low_pct:.0f}%. That's how often {n_models} different AI models assumed anything at all "
            f"about two people in a short story I gave them, when I didn't give them a reason to."
        )
        P.append("")
        P.append(
            f"Then I changed one thing about the prompt - not what it asked, just how much room it left "
            f"to avoid answering - and that number went to {high_pct:.0f}%."
        )
        P.append("")
        P.append(
            "Same models. Same underlying scenario. A swing that big, from one small change, isn't "
            "noise. It's the actual finding, and it's more useful than either number would have been on "
            "its own."
        )
    else:
        overall_doc = analysis.role_summary(df[df["is_countable"]], "doctor")["any_assignment"]
        P.append(
            f"I ran the same prompt against {n_models} AI models, {counts.success} times total, and "
            f"nobody was told what I was actually looking for. {overall_doc.numerator} of "
            f"{overall_doc.denominator} responses assigned a gender to someone the prompt never "
            f"described. That's the headline number. Here's how I got it, and what it does and doesn't "
            f"tell you."
        )
    P.append("")

    # ---------------- WHY ----------------
    P.append(
        "I didn't tell the models what I was testing. That part matters more than it sounds like it "
        "should - ask a model directly whether it's biased and you'll get a very correct, very useless "
        "answer about fairness and context, the kind of answer that's been trained into it precisely "
        "because someone asked that question before. I wanted to know what it actually does when "
        "nobody's watching for the answer, not what it says when it is."
    )
    P.append("")
    P.append(
        "So instead of asking, I built something that runs the same prompt against a model many times, "
        "swaps in a second model as a blind grader that never sees what I'm testing for, and only ever "
        "counts a gender as \"assigned\" if the grader can point to an exact phrase in the response that "
        "says so. No phrase, no count. That's the whole trick, and it's less a trick than just... doing "
        "the obvious thing instead of asking an AI to grade its own homework, or grading it myself with "
        "whatever bias I walked in with."
    )
    P.append("")

    # ---------------- SECTION: LOW VARIANT (or sole variant) ----------------
    def _section(pv: dict, rate: float, n: int, heading: str, lead: str, tail: str) -> None:
        vdf = df[(df["prompt_variant"] == pv["variant_name"]) & (df["is_countable"])]
        doc = analysis.role_summary(vdf, "doctor")
        nur = analysis.role_summary(vdf, "nurse")
        P.append(f"## {heading}")
        P.append("")
        P.append(lead)
        P.append("")
        combined_rate = (doc["explicit"].proportion + nur["explicit"].proportion) / 2
        if combined_rate < 0.2:
            closing_clause = "The rest just didn't - no pronoun, no gendered title, nothing to classify."
        elif combined_rate > 0.8:
            closing_clause = "The handful that didn't were the exception here, not the rule."
        else:
            closing_clause = "Roughly a coin flip, in other words - it could go either way."
        P.append(
            f"{doc['explicit'].numerator} of {doc['explicit'].denominator} responses ("
            f"{doc['explicit'].proportion*100:.0f}%) gave the first person an explicit gender. "
            f"{nur['explicit'].numerator} of {nur['explicit'].denominator} ("
            f"{nur['explicit'].proportion*100:.0f}%) did it for the second. {closing_clause}"
        )
        P.append("")
        quote = _pick_quote(df, pv["variant_name"], "doctor", ["explicit", "implied"]) \
            or _pick_quote(df, pv["variant_name"], "nurse", ["explicit", "implied"])
        if quote:
            quote_excerpt = _truncate_at_word(quote["response"], 280)
            P.append(f"One actual response, from `{quote['model_id']}`, unedited:")
            P.append("")
            P.append(_blockquote(quote_excerpt))
            P.append("")
        P.append(tail)
        P.append("")

    if ranking.has_contrast:
        _section(
            ranking.low_pv, ranking.low_rate, ranking.low_n,
            "Round one: give it an easy out",
            "First version of the prompt. Nothing in it requires a pronoun at all - a model can just say "
            "\"the doctor\" and \"the nurse\" as many times as it wants and never commit to anything.",
            "Which, mostly, is what happened. I'll be honest, my first reaction was relief - clean "
            "numbers, no villain, nice simple finding. Then I got suspicious of my own relief. Staying "
            "quiet isn't the same thing as not having an opinion. It might just mean I gave it an easy "
            "way to avoid showing one.",
        )
        _section(
            ranking.high_pv, ranking.high_rate, ranking.high_n,
            "Round two: close the exit",
            "So I ran a second version. Same people, same scene - I just required a pronoun after the "
            "first mention of each person, instead of letting the model keep repeating \"the doctor.\" I "
            "didn't say why. I didn't mention gender. I just removed the easy way out.",
            "And that's the number from the top of this post. Whatever was sitting underneath the polite "
            "silence in round one, it didn't go away. It just needed a reason to show up.",
        )
    else:
        pv = ranking.high_pv
        if pv:
            _section(
                pv, ranking.high_rate, ranking.high_n,
                "What actually happened",
                "Here's the breakdown, not the summary.",
                "That's the real distribution - not an average, not a vibe. The table below has it broken "
                "out by model if you want to check a specific one.",
            )

    # ---------------- SECTION: PER-MODEL PERSONALITIES ----------------
    focus_pv = ranking.high_pv
    if focus_pv:
        pairings = [p for p in _model_pairings(df, focus_pv["variant_name"]) if p["n"] > 0]
        by_class: dict[str, list[dict]] = {"matches_stereotype": [], "inverts_stereotype": [],
                                            "same_gender_both": [], "mixed": []}
        for p in pairings:
            by_class[_classify_pairing(p)].append(p)

        if by_class["matches_stereotype"] or by_class["inverts_stereotype"]:
            P.append("## Not every model defaults the same way")
            P.append("")
            P.append(
                "This is the part I didn't expect going in, and it's the part worth actually sitting "
                "with. I was ready to write a post about whether AI models default to a stereotype. I "
                "wasn't ready for them to split."
            )
            P.append("")
            P.append(
                "Quick definitions, since I'm about to use them a lot: \"stereotypical pairing\" means "
                "the first role came back male and the second came back female - the direction you'd "
                "guess if you'd never seen the data. \"Opposite pairing\" means exactly that, flipped. "
                "Both are a model picking *something* once you take the easy out away - the question is "
                "just which way it leans, and whether it's consistent about it."
            )
            P.append("")
            if by_class["matches_stereotype"]:
                names = _prose_list([f"`{p['model_id']}`" for p in by_class["matches_stereotype"]])
                P.append(
                    f"{names} landed on the stereotypical pairing every single time I asked. "
                    f"Not mostly - every time, across {experiment['repetitions']} separate attempts each."
                )
                P.append("")
            if by_class["inverts_stereotype"]:
                names = _prose_list([f"`{p['model_id']}`" for p in by_class["inverts_stereotype"]])
                P.append(
                    f"{names} did the exact opposite, just as consistently. Not a mix, not a "
                    f"coin flip - the other pairing, every time."
                )
                P.append("")
                P.append(
                    "That's the detail that actually changes what this experiment means. If every model "
                    "had landed the same way, you could write a tidy sentence about \"AI\" and move on. "
                    "They didn't. Whatever's driving this lives in each model separately, not in some "
                    "shared training-data ghost that all of them inherited the same way."
                )
                P.append("")
            P.append(
                f"Small sample, to be clear - {experiment['repetitions']} runs per model here. I wouldn't "
                f"bet money on any single model's exact number. I'd absolutely bet on the fact that they "
                f"don't all land in the same place, because that pattern showed up consistently within "
                f"each model, not just once."
            )
            P.append("")

    # ---------------- SECTION: THE MESS (data quality, honestly) ----------------
    mess_notes = []
    if missing["evidence_insufficient"] > 0:
        n_ei = missing["evidence_insufficient"]
        resp_word = "response" if n_ei == 1 else "responses"
        was_were = "was" if n_ei == 1 else "were"
        mess_notes.append(
            f"{n_ei} {resp_word} {was_were} thrown out entirely partway through this - not because the "
            f"result was inconvenient, but because a couple of the models spent their entire response "
            f"budget on invisible reasoning tokens and never actually wrote an answer. One of them handed "
            f"back a response that was, in full, a single asterisk. That's not a finding, that's a "
            f"token-limit bug, so it's not in the numbers above."
        )
    if missing["failed_runs"] > 0:
        n_failed = missing["failed_runs"]
        call_word = "call" if n_failed == 1 else "calls"
        mess_notes.append(
            f"{n_failed} API {call_word} just failed outright and {'is' if n_failed == 1 else 'are'} "
            f"excluded, not quietly folded into \"neutral.\""
        )
    if mess_notes:
        P.append("## The part that doesn't make it into the chart")
        P.append("")
        P.append(
            "Every tidy results post hides a less tidy process, so here's mine: " + " ".join(mess_notes)
        )
        P.append("")
        P.append(
            "I'm mentioning it because the alternative - quietly excluding the mess and only showing you "
            "the clean chart - is exactly how you end up trusting a number you shouldn't."
        )
        P.append("")

    # ---------------- CAVEATS ----------------
    P.append("## What this isn't")
    P.append("")
    narrated = _already_narrated_variant_names(ranking)
    other_variant_names = [pv["variant_name"] for pv in prompt_versions if pv["variant_name"] not in narrated]
    caveat_variant_note = (
        f" I ran {len(other_variant_names)} other "
        f"{'version' if len(other_variant_names) == 1 else 'versions'} of this scenario to check - see "
        f"the full data for those." if other_variant_names else ""
    )
    P.append(
        f"This is one scenario, tested {experiment['repetitions']} times per model.{caveat_variant_note} "
        f"It's not a universal verdict on any model, and repeated runs from the same model aren't fully "
        f"independent evidence - they share the same weights, so they're more like asking the same "
        f"person the same question several times than polling several different people. Treat the "
        f"repetitions as noise reduction, not as proof."
    )
    P.append("")
    P.append(
        "What it is: a real, repeatable test, with every raw response saved, that anyone can run again "
        "and check against mine."
    )
    P.append("")

    # ---------------- CLOSING ----------------
    P.append("## So, is the AI biased?")
    P.append("")
    if ranking.has_contrast:
        P.append(
            "Wrong question, honestly. The more useful one is: biased compared to what baseline? These "
            "models mostly won't say anything about gender if you don't make them. That's worth knowing. "
            "It's also not the whole story, because the moment you take away the option to stay quiet, "
            "most of them land somewhere - and it's not always the somewhere you'd guess."
        )
    else:
        P.append(
            "I'll let the numbers above answer that one rather than doing it for you in a tidy closing "
            "line - that's kind of the point of running the experiment instead of just asking the "
            "question out loud."
        )
    P.append("")
    P.append(
        "If you want to poke at this yourself - the raw responses, the evaluator prompt, the full "
        "per-model breakdown - it's all linked below. Rerun it, change the scenario, prove me wrong. "
        "That's a better use of a comment section than an argument about what AI models \"are like\" in "
        "the abstract."
    )
    P.append("")
    P.append("*[Link to full data / report / repo here. Your name, your usual sign-off.]*")

    return "\n".join(P)
