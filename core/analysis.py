"""Analysis layer: turns raw runs/evaluations/human-reviews into the results
table and summary statistics shown on the dashboard.

Design choices that matter for correctness (see README "Analysis methodology"):
- The denominator for every proportion is the number of VALID classified
  responses for that group - i.e. the API call succeeded AND there is either
  a valid automated evaluation or a human review that resolved it. Failed
  API calls and unresolved-invalid evaluations are excluded from the
  denominator and reported separately as missing/invalid/unreviewed counts.
- Explicit and implied assignments are always reported as separate figures.
  Any "combined" figure is an explicit sum of the two, clearly labelled.
- No single cross-model "bias score" is computed anywhere in this module.
- Confidence intervals use the Wilson score interval (valid for small n,
  unlike the naive normal approximation), computed with no scipy dependency.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from core import db

Role = Literal["doctor", "nurse"]

SMALL_SAMPLE_THRESHOLD = 20

# Shared by wilson_ci (proportions) and custodial_sentence_length_stats (a
# normal-approximation CI on a continuous measure) - no scipy dependency
# anywhere in this project, so this is the full set of z-scores available.
Z_TABLE = {0.90: 1.6448536, 0.95: 1.9599640, 0.99: 2.5758293}


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval. Returns (0.0, 0.0) when n == 0."""
    if n <= 0:
        return (0.0, 0.0)
    z = Z_TABLE.get(confidence, 1.9599640)
    phat = successes / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z ** 2 / (4 * n ** 2)))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


def build_experiment_rows(conn: sqlite3.Connection, experiment_id: int) -> list[dict[str, Any]]:
    """One row per run, joined with its latest evaluation and latest human
    review. This is the single source of truth the dashboard and exports
    both read from."""
    runs = db.list_runs(conn, experiment_id)
    prompt_versions = {pv["id"]: pv for pv in db.list_prompt_versions(conn, experiment_id)}
    evaluations = db.get_latest_evaluations_by_run(conn, experiment_id)
    reviews = db.get_latest_human_reviews_by_run(conn, experiment_id)

    rows = []
    for run in runs:
        pv = prompt_versions.get(run["prompt_version_id"], {})
        ev = evaluations.get(run["id"])
        hr = reviews.get(run["id"])
        rows.append(_flatten_row(run, pv, ev, hr))
    return rows


def _flatten_row(run: dict, pv: dict, ev: dict | None, hr: dict | None) -> dict[str, Any]:
    has_human_review = hr is not None
    auto_valid = bool(ev and ev.get("is_valid"))

    def resolved(role: Role, field: str) -> Any:
        if has_human_review:
            key = f"final_{role}_{field}"
            val = hr.get(key)
            if val is not None:
                return val
        if ev:
            # evaluations columns are named doctor_gender_classification /
            # doctor_assignment_type - "classification" maps to the former.
            ev_field = "gender_classification" if field == "classification" else field
            return ev.get(f"{role}_{ev_field}")
        return None

    evidence_insufficient = bool(hr.get("evidence_insufficient")) if hr else False
    review_is_resolving = has_human_review and not evidence_insufficient \
        and hr.get("final_doctor_classification") is not None \
        and hr.get("final_nurse_classification") is not None
    # A human marking evidence_insufficient overrides even a schema-valid
    # automated classification - the UI's classification dropdowns always
    # have *some* value selected, but that's a form default, not a genuine
    # judgement the reviewer is standing behind.
    is_countable = run["status"] == "success" and not evidence_insufficient and (auto_valid or review_is_resolving)

    row = {
        "run_id": run["id"],
        "experiment_id": run["experiment_id"],
        "model_id": run["model_id"],
        "response_model": run.get("response_model"),
        "prompt_variant": pv.get("variant_name"),
        "prompt_version_id": run["prompt_version_id"],
        "prompt_version_number": pv.get("version_number"),
        "is_role_reversed": bool(pv.get("is_role_reversed")),
        "run_number": run["run_number"],
        "run_status": run["status"],
        "error_message": run.get("error_message"),
        "latency_ms": run.get("latency_ms"),
        "cost_usd": run.get("cost_usd"),
        "evaluation_cost_usd": ev.get("cost_usd") if ev else None,
        "total_tokens": run.get("total_tokens"),
        "raw_response": run.get("raw_response"),
        "has_evaluation": ev is not None,
        "evaluation_is_valid": auto_valid,
        "evaluation_status": ev.get("status") if ev else "missing",
        "has_human_review": has_human_review,
        "evidence_insufficient": evidence_insufficient,
        "reviewer_note": hr.get("reviewer_note") if hr else None,
        "is_countable": is_countable,
        "doctor_auto_classification": ev.get("doctor_gender_classification") if ev else None,
        "doctor_auto_assignment_type": ev.get("doctor_assignment_type") if ev else None,
        "doctor_auto_quote": ev.get("doctor_quote") if ev else None,
        "nurse_auto_classification": ev.get("nurse_gender_classification") if ev else None,
        "nurse_auto_assignment_type": ev.get("nurse_assignment_type") if ev else None,
        "nurse_auto_quote": ev.get("nurse_quote") if ev else None,
        "doctor_final_classification": resolved("doctor", "classification"),
        "doctor_final_assignment_type": resolved("doctor", "assignment_type"),
        "nurse_final_classification": resolved("nurse", "classification"),
        "nurse_final_assignment_type": resolved("nurse", "assignment_type"),
        "doctor_disagreement": bool(
            has_human_review and auto_valid and not evidence_insufficient and hr.get("final_doctor_classification")
            and hr.get("final_doctor_classification") != ev.get("doctor_gender_classification")
        ),
        "nurse_disagreement": bool(
            has_human_review and auto_valid and not evidence_insufficient and hr.get("final_nurse_classification")
            and hr.get("final_nurse_classification") != ev.get("nurse_gender_classification")
        ),
    }
    return row


def rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def filter_rows(
    df: pd.DataFrame,
    model_id: str | None = None,
    prompt_variant: str | None = None,
    review_status: str | None = None,
) -> pd.DataFrame:
    out = df
    if model_id:
        out = out[out["model_id"] == model_id]
    if prompt_variant:
        out = out[out["prompt_variant"] == prompt_variant]
    if review_status == "reviewed":
        out = out[out["has_human_review"]]
    elif review_status == "unreviewed":
        out = out[~out["has_human_review"]]
    return out


@dataclass
class RunCounts:
    total: int
    success: int
    failed: int
    pending: int

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 0.0


def run_counts(df: pd.DataFrame) -> RunCounts:
    if df.empty:
        return RunCounts(0, 0, 0, 0)
    counts = df["run_status"].value_counts()
    return RunCounts(
        total=len(df),
        success=int(counts.get("success", 0)),
        failed=int(counts.get("failed", 0)),
        pending=int(counts.get("pending", 0) + counts.get("running", 0)),
    )


@dataclass
class CostSummary:
    total_usd: float  # tested-model cost + evaluator cost combined - the true total spend
    runs_with_known_cost: int
    runs_missing_cost: int  # successful runs where the provider didn't report a cost
    tested_model_cost_usd: float
    evaluation_cost_usd: float
    evaluations_with_known_cost: int
    evaluations_missing_cost: int  # an evaluation exists but its cost wasn't recorded


def cost_summary(df: pd.DataFrame) -> CostSummary:
    """Sums whatever cost_usd OpenRouter actually reported, for BOTH the
    tested model's calls (cost_usd) and the independent evaluator's calls
    (evaluation_cost_usd) - total_usd is the true combined spend. The
    evaluator's cost went completely untracked in this project for a long
    time (every "cost so far" figure silently excluded it, undercounting
    real spend by however much evaluation traffic cost); this is the fix.

    Cost reporting isn't guaranteed for every provider/model, and rows
    created before evaluator cost tracking existed have no evaluation cost
    recorded at all, so this also surfaces how many rows have no cost
    figure - the total is a floor, not a guaranteed-complete spend figure,
    and callers should say so."""
    if df.empty:
        return CostSummary(0.0, 0, 0, 0.0, 0.0, 0, 0)
    success = df[df["run_status"] == "success"]
    known_run_cost = success["cost_usd"].dropna()

    evaluated = success[success["has_evaluation"]]
    known_eval_cost = evaluated["evaluation_cost_usd"].dropna()

    tested_cost = float(known_run_cost.sum())
    eval_cost = float(known_eval_cost.sum())
    return CostSummary(
        total_usd=tested_cost + eval_cost,
        runs_with_known_cost=int(known_run_cost.shape[0]),
        runs_missing_cost=int(success["cost_usd"].isna().sum()),
        tested_model_cost_usd=tested_cost,
        evaluation_cost_usd=eval_cost,
        evaluations_with_known_cost=int(known_eval_cost.shape[0]),
        evaluations_missing_cost=int(evaluated["evaluation_cost_usd"].isna().sum()),
    )


@dataclass
class ProportionResult:
    label: str
    numerator: int
    denominator: int

    @property
    def proportion(self) -> float:
        return self.numerator / self.denominator if self.denominator else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_ci(self.numerator, self.denominator)

    @property
    def is_exploratory(self) -> bool:
        return self.denominator < SMALL_SAMPLE_THRESHOLD


# ---------------------------------------------------------------------------
# Generic statistical primitives - domain-agnostic engine
#
# Every per-domain "headline rate" or "headline continuous measure" in this
# file (role_summary's explicit/implied/any_assignment, the sentencing
# domain's custodial_rate/custodial_sentence_length_stats) is a thin wrapper
# around ONE of these three functions, parameterised with that domain's own
# column names and "which value(s) count as positive" set. A third domain
# needs no new statistics code - just its own column names and positive-value
# set passed into these same primitives.
# ---------------------------------------------------------------------------

def categorical_rate(
    df: pd.DataFrame, field_col: str, positive_values: set, countable_col: str = "is_countable",
    label: str = "",
) -> ProportionResult:
    """Share of countable rows where field_col's value is in positive_values.
    The shared engine behind every "headline rate" in this module."""
    countable = df[df[countable_col]] if not df.empty and countable_col in df.columns else df
    n = len(countable)
    numerator = int(countable[field_col].isin(positive_values).sum()) if n else 0
    return ProportionResult(label or f"{field_col} in {sorted(positive_values, key=str)}", numerator, n)


def value_distribution(df: pd.DataFrame, field_col: str, countable_col: str = "is_countable") -> pd.DataFrame:
    """Count of countable rows per distinct value of field_col - the generic
    engine behind sentence_type_distribution (and usable directly by any new
    domain's full categorical breakdown without writing new code)."""
    countable = df[df[countable_col]] if not df.empty and countable_col in df.columns else df
    if countable.empty:
        return pd.DataFrame(columns=[field_col, "count"])
    counts = countable[field_col].value_counts().reset_index()
    counts.columns = [field_col, "count"]
    return counts


@dataclass
class ContinuousStats:
    n: int
    mean: float | None
    median: float | None
    std_dev: float | None
    ci: tuple[float, float]  # (0.0, 0.0) when n == 0; (mean, mean) when n == 1

    @property
    def is_exploratory(self) -> bool:
        return self.n < SMALL_SAMPLE_THRESHOLD


def continuous_stats(
    df: pd.DataFrame, numeric_col: str, countable_col: str = "is_countable",
    mask_col: str | None = None, confidence: float = 0.95,
) -> ContinuousStats:
    """Mean/median/normal-approximation CI of numeric_col among countable rows
    (optionally further restricted to rows where mask_col is True - e.g. only
    custodial-type rows for a sentence-length average). The shared engine
    behind custodial_sentence_length_stats.

    Uses a normal approximation CI on the mean. A t-distribution would be
    more correct for small n, but this project takes no scipy dependency
    anywhere; callers must treat ContinuousStats.is_exploratory the same way
    ProportionResult.is_exploratory is already treated elsewhere - a flag
    that gates how confidently a figure can be reported, not a reason to
    hide it."""
    countable = df[df[countable_col]] if not df.empty and countable_col in df.columns else df
    if mask_col and not countable.empty:
        countable = countable[countable[mask_col]]
    if countable.empty or numeric_col not in countable.columns:
        return ContinuousStats(0, None, None, None, (0.0, 0.0))
    values = countable[numeric_col].dropna()
    n = len(values)
    if n == 0:
        return ContinuousStats(0, None, None, None, (0.0, 0.0))
    mean = float(values.mean())
    median = float(values.median())
    if n < 2:
        return ContinuousStats(n, mean, median, None, (mean, mean))
    std_dev = float(values.std(ddof=1))
    z = Z_TABLE.get(confidence, 1.9599640)
    margin = z * std_dev / math.sqrt(n)
    return ContinuousStats(n, mean, median, std_dev, (mean - margin, mean + margin))


def grouped_categorical_rate(
    df: pd.DataFrame, field_col: str, positive_values: set, group_cols: list[str],
    countable_col: str = "is_countable",
) -> pd.DataFrame:
    """categorical_rate, broken out by arbitrary grouping columns - usable
    directly by a new domain's "by model" / "by variant" breakdown without
    writing a new per-domain summarize_by function."""
    if df.empty:
        return pd.DataFrame()
    records = []
    for key, group in df.groupby(group_cols):
        keys = key if isinstance(key, tuple) else (key,)
        result = categorical_rate(group, field_col, positive_values, countable_col)
        lo, hi = result.ci
        record = dict(zip(group_cols, keys))
        record.update({
            "n": result.denominator, "count": result.numerator, "pct": round(result.proportion * 100, 1),
            "ci": f"[{lo*100:.1f}%, {hi*100:.1f}%]", "is_exploratory": result.is_exploratory,
        })
        records.append(record)
    return pd.DataFrame(records)


def grouped_continuous_stats(
    df: pd.DataFrame, numeric_col: str, group_cols: list[str], countable_col: str = "is_countable",
    mask_col: str | None = None,
) -> pd.DataFrame:
    """continuous_stats, broken out by arbitrary grouping columns."""
    if df.empty:
        return pd.DataFrame()
    records = []
    for key, group in df.groupby(group_cols):
        keys = key if isinstance(key, tuple) else (key,)
        stats = continuous_stats(group, numeric_col, countable_col, mask_col)
        record = dict(zip(group_cols, keys))
        record.update({
            "n": stats.n, "mean": round(stats.mean, 1) if stats.mean is not None else None,
            "median": stats.median,
            "ci_low": round(stats.ci[0], 1) if stats.n >= 2 else None,
            "ci_high": round(stats.ci[1], 1) if stats.n >= 2 else None,
            "is_exploratory": stats.is_exploratory,
        })
        records.append(record)
    return pd.DataFrame(records)


def role_summary(df: pd.DataFrame, role: Role) -> dict[str, ProportionResult]:
    """Core per-role breakdown used throughout the dashboard. denominator is
    always the count of countable (valid) rows in the given slice of df.
    Each figure is categorical_rate with this domain's own column names and
    positive-value set - no bespoke counting logic lives here anymore."""
    classification_col = f"{role}_final_classification"
    assignment_col = f"{role}_final_assignment_type"

    explicit = categorical_rate(df, assignment_col, {"explicit"})
    implied = categorical_rate(df, assignment_col, {"implied"})
    any_assignment = categorical_rate(df, assignment_col, {"explicit", "implied"})
    unspecified = categorical_rate(df, classification_col, {"unspecified"})
    ambiguous = categorical_rate(df, classification_col, {"ambiguous"})
    none_assigned = categorical_rate(df, assignment_col, {"none"})

    return {
        "explicit": ProportionResult(f"{role}: explicit gender assignment", explicit.numerator, explicit.denominator),
        "implied": ProportionResult(f"{role}: implied gender assignment", implied.numerator, implied.denominator),
        "any_assignment": ProportionResult(
            f"{role}: any assignment (explicit + implied)", any_assignment.numerator, any_assignment.denominator,
        ),
        "unspecified": ProportionResult(
            f"{role}: gender-neutral / unspecified", unspecified.numerator, unspecified.denominator,
        ),
        "ambiguous": ProportionResult(
            f"{role}: ambiguous / insufficient evidence", ambiguous.numerator, ambiguous.denominator,
        ),
        "none_assigned": ProportionResult(
            f"{role}: no assignment detected", none_assigned.numerator, none_assigned.denominator,
        ),
    }


def missing_and_invalid_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {"failed_runs": 0, "invalid_unreviewed": 0, "unreviewed_total": 0, "evidence_insufficient": 0}
    failed_runs = int((df["run_status"] == "failed").sum())
    invalid_unreviewed = int(((~df["evaluation_is_valid"]) & (~df["has_human_review"]) & (df["run_status"] == "success")).sum())
    unreviewed_total = int(((~df["has_human_review"]) & (df["run_status"] == "success")).sum())
    evidence_insufficient = int((df["evidence_insufficient"] & (df["run_status"] == "success")).sum())
    return {
        "failed_runs": failed_runs,
        "evidence_insufficient": evidence_insufficient,
        "invalid_unreviewed": invalid_unreviewed,
        "unreviewed_total": unreviewed_total,
    }


def summarize_by(df: pd.DataFrame, role: Role, group_cols: list[str]) -> pd.DataFrame:
    """Breakdown by arbitrary grouping columns (e.g. ['model_id'] or
    ['model_id', 'prompt_variant'])."""
    if df.empty:
        return pd.DataFrame()
    records = []
    for key, group in df.groupby(group_cols):
        keys = key if isinstance(key, tuple) else (key,)
        summary = role_summary(group, role)
        record = dict(zip(group_cols, keys))
        for metric_name, result in summary.items():
            record[f"{metric_name}_n"] = result.denominator
            record[f"{metric_name}_count"] = result.numerator
            record[f"{metric_name}_pct"] = round(result.proportion * 100, 1)
            lo, hi = result.ci
            record[f"{metric_name}_ci"] = f"[{lo*100:.1f}%, {hi*100:.1f}%]"
        record["is_exploratory"] = group.pipe(lambda g: len(g[g["is_countable"]])) < SMALL_SAMPLE_THRESHOLD
        records.append(record)
    return pd.DataFrame(records)


def reproducibility_drift_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Successful runs where the model string OpenRouter actually served
    differs from the model id that was requested - e.g. the request asked
    for 'openai/gpt-4o-mini' and the response was echoed back as
    'openai/gpt-4o-mini-2024-07-18'. This isn't necessarily a problem (many
    providers route a bare id to a specific dated snapshot as a matter of
    course), but it IS exactly the kind of provider behaviour the project
    brief asks to be recorded rather than silently assumed away - an
    unpinned model id is not a guarantee of a fixed, reproducible model
    version."""
    if df.empty:
        return df
    success = df[df["run_status"] == "success"]
    has_response_model = success["response_model"].notna()
    drifted = success["model_id"] != success["response_model"]
    return success[has_response_model & drifted]


def disagreement_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["doctor_disagreement"] | df["nurse_disagreement"]]


# ---------------------------------------------------------------------------
# Sentencing-bias domain (evaluation_domain == "sentencing_bias")
#
# Shares the run/evaluation/human-review infrastructure above, but the
# measure under study is a sentence recommendation, not a gender
# classification, so it gets its own row-flattening and summary functions
# rather than overloading role_summary's doctor/nurse shape.
# ---------------------------------------------------------------------------

def build_sentencing_rows(conn: sqlite3.Connection, experiment_id: int) -> list[dict[str, Any]]:
    """One row per run, joined with its latest evaluation and latest human
    review - the sentencing-domain equivalent of build_experiment_rows."""
    runs = db.list_runs(conn, experiment_id)
    prompt_versions = {pv["id"]: pv for pv in db.list_prompt_versions(conn, experiment_id)}
    evaluations = db.get_latest_evaluations_by_run(conn, experiment_id)
    reviews = db.get_latest_human_reviews_by_run(conn, experiment_id)

    rows = []
    for run in runs:
        pv = prompt_versions.get(run["prompt_version_id"], {})
        ev = evaluations.get(run["id"])
        hr = reviews.get(run["id"])
        rows.append(_flatten_sentencing_row(run, pv, ev, hr))
    return rows


def _flatten_sentencing_row(run: dict, pv: dict, ev: dict | None, hr: dict | None) -> dict[str, Any]:
    has_human_review = hr is not None
    auto_valid = bool(ev and ev.get("is_valid"))
    evidence_insufficient = bool(hr.get("evidence_insufficient")) if hr else False
    # A human review only "resolves" a row if it actually recorded a final
    # sentence_type - the UI's dropdown always has some value selected, but
    # that alone isn't evidence a reviewer looked at this specific row.
    review_is_resolving = has_human_review and not evidence_insufficient and hr.get("final_sentence_type") is not None

    if review_is_resolving:
        final_sentence_type = hr.get("final_sentence_type")
        final_custodial_months = hr.get("final_sentence_custodial_months")
    elif ev:
        final_sentence_type = ev.get("sentence_type")
        final_custodial_months = ev.get("sentence_custodial_months")
    else:
        final_sentence_type = None
        final_custodial_months = None

    is_countable = run["status"] == "success" and not evidence_insufficient and (auto_valid or review_is_resolving)

    return {
        "run_id": run["id"],
        "experiment_id": run["experiment_id"],
        "model_id": run["model_id"],
        "response_model": run.get("response_model"),
        "prompt_variant": pv.get("variant_name"),
        "prompt_version_id": run["prompt_version_id"],
        "prompt_version_number": pv.get("version_number"),
        "run_number": run["run_number"],
        "run_status": run["status"],
        "error_message": run.get("error_message"),
        "latency_ms": run.get("latency_ms"),
        "cost_usd": run.get("cost_usd"),
        "evaluation_cost_usd": ev.get("cost_usd") if ev else None,
        "total_tokens": run.get("total_tokens"),
        "raw_response": run.get("raw_response"),
        "has_evaluation": ev is not None,
        "evaluation_is_valid": auto_valid,
        "evaluation_status": ev.get("status") if ev else "missing",
        "has_human_review": has_human_review,
        "evidence_insufficient": evidence_insufficient,
        "reviewer_note": hr.get("reviewer_note") if hr else None,
        "is_countable": is_countable,
        "sentence_auto_type": ev.get("sentence_type") if ev else None,
        "sentence_auto_custodial_months": ev.get("sentence_custodial_months") if ev else None,
        "sentence_auto_quote": ev.get("sentence_quote") if ev else None,
        "sentence_final_type": final_sentence_type,
        "sentence_final_custodial_months": final_custodial_months,
        "is_custodial": final_sentence_type in ("custodial", "suspended_custodial"),
        "no_recommendation": final_sentence_type == "no_recommendation",
        "sentence_disagreement": bool(
            has_human_review and auto_valid and not evidence_insufficient and hr.get("final_sentence_type")
            and hr.get("final_sentence_type") != ev.get("sentence_type")
        ),
    }


def custodial_rate(df: pd.DataFrame) -> ProportionResult:
    """Share of countable rows that recommended a custodial or
    suspended-custodial sentence (the two sentence_types with an actual
    prison-term length) - the sentencing-domain analogue of role_summary's
    any_assignment figure. categorical_rate with this domain's own boolean
    column, same as every other domain's headline rate."""
    result = categorical_rate(df, "is_custodial", {True})
    return ProportionResult("custodial or suspended-custodial sentence", result.numerator, result.denominator)


@dataclass
class SentenceLengthStats:
    n: int
    mean: float | None
    median: float | None
    std_dev: float | None
    ci: tuple[float, float]  # (0.0, 0.0) when n == 0; (mean, mean) when n == 1

    @property
    def is_exploratory(self) -> bool:
        return self.n < SMALL_SAMPLE_THRESHOLD


def custodial_sentence_length_stats(df: pd.DataFrame, confidence: float = 0.95) -> SentenceLengthStats:
    """Mean/median custodial-or-suspended sentence length in months, among
    countable custodial-type rows only - a fine/community-order/discharge/
    no_recommendation row has no length to average into this figure.
    continuous_stats with this domain's own column names, same as every
    other domain's headline continuous measure would be."""
    stats = continuous_stats(df, "sentence_final_custodial_months", mask_col="is_custodial", confidence=confidence)
    return SentenceLengthStats(stats.n, stats.mean, stats.median, stats.std_dev, stats.ci)


def summarize_sentencing_by(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Breakdown by arbitrary grouping columns (e.g. ['prompt_variant'] or
    ['model_id', 'prompt_variant']) - custodial rate plus sentence-length
    stats per group."""
    if df.empty:
        return pd.DataFrame()
    records = []
    for key, group in df.groupby(group_cols):
        keys = key if isinstance(key, tuple) else (key,)
        countable = group[group["is_countable"]]
        n = len(countable)
        rate = custodial_rate(group)
        length_stats = custodial_sentence_length_stats(group)
        lo, hi = rate.ci
        record = dict(zip(group_cols, keys))
        record.update({
            "n": n,
            # Named with the "custodial_" prefix (custodial_n/_count/_pct) so
            # this dataframe can be passed straight to
            # charts.proportion_by_group_chart(df, group_col, "custodial",
            # title), which expects exactly {prefix}_n/_count/_pct columns.
            "custodial_n": n,
            "custodial_count": rate.numerator,
            "custodial_pct": round(rate.proportion * 100, 1),
            "custodial_rate_ci": f"[{lo*100:.1f}%, {hi*100:.1f}%]",
            "custodial_length_n": length_stats.n,
            "mean_custodial_months": round(length_stats.mean, 1) if length_stats.mean is not None else None,
            "median_custodial_months": length_stats.median,
            "custodial_months_ci_low": round(length_stats.ci[0], 1) if length_stats.n >= 2 else None,
            "custodial_months_ci_high": round(length_stats.ci[1], 1) if length_stats.n >= 2 else None,
            "custodial_months_ci": (
                f"[{length_stats.ci[0]:.1f}, {length_stats.ci[1]:.1f}]" if length_stats.n >= 2 else "n/a"
            ),
            "no_recommendation_count": int(countable["no_recommendation"].sum()) if n else 0,
            "is_exploratory": n < SMALL_SAMPLE_THRESHOLD,
        })
        records.append(record)
    return pd.DataFrame(records)


def sentence_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count of countable rows per final sentence_type - the sentencing
    equivalent of a single role_summary breakdown, but across all six
    categories at once rather than a binary custodial/non-custodial split.
    value_distribution with this domain's own column name."""
    return value_distribution(df, "sentence_final_type").rename(columns={"sentence_final_type": "sentence_type"})


def sentencing_disagreement_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["sentence_disagreement"]]
