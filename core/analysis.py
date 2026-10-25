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


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval. Returns (0.0, 0.0) when n == 0."""
    if n <= 0:
        return (0.0, 0.0)
    z_table = {0.90: 1.6448536, 0.95: 1.9599640, 0.99: 2.5758293}
    z = z_table.get(confidence, 1.9599640)
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
    evaluations = {}
    for ev in db.list_evaluations_for_experiment(conn, experiment_id):
        # list is ordered attempt_number DESC per run already; keep first seen (= latest)
        evaluations.setdefault(ev["run_id"], ev)
    reviews = {r["run_id"]: r for r in db.list_human_reviews_for_experiment(conn, experiment_id)}

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

    review_is_resolving = has_human_review and hr.get("final_doctor_classification") is not None \
        and hr.get("final_nurse_classification") is not None
    is_countable = run["status"] == "success" and (auto_valid or review_is_resolving)

    row = {
        "run_id": run["id"],
        "experiment_id": run["experiment_id"],
        "model_id": run["model_id"],
        "prompt_variant": pv.get("variant_name"),
        "prompt_version_id": run["prompt_version_id"],
        "prompt_version_number": pv.get("version_number"),
        "is_role_reversed": bool(pv.get("is_role_reversed")),
        "run_number": run["run_number"],
        "run_status": run["status"],
        "error_message": run.get("error_message"),
        "latency_ms": run.get("latency_ms"),
        "cost_usd": run.get("cost_usd"),
        "total_tokens": run.get("total_tokens"),
        "raw_response": run.get("raw_response"),
        "has_evaluation": ev is not None,
        "evaluation_is_valid": auto_valid,
        "evaluation_status": ev.get("status") if ev else "missing",
        "has_human_review": has_human_review,
        "evidence_insufficient": bool(hr.get("evidence_insufficient")) if hr else False,
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
            has_human_review and auto_valid and hr.get("final_doctor_classification")
            and hr.get("final_doctor_classification") != ev.get("doctor_gender_classification")
        ),
        "nurse_disagreement": bool(
            has_human_review and auto_valid and hr.get("final_nurse_classification")
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


def role_summary(df: pd.DataFrame, role: Role) -> dict[str, ProportionResult]:
    """Core per-role breakdown used throughout the dashboard. denominator is
    always the count of countable (valid) rows in the given slice of df."""
    countable = df[df["is_countable"]] if not df.empty else df
    n = len(countable)
    classification_col = f"{role}_final_classification"
    assignment_col = f"{role}_final_assignment_type"

    explicit = int((countable[assignment_col] == "explicit").sum()) if n else 0
    implied = int((countable[assignment_col] == "implied").sum()) if n else 0
    none_assigned = int((countable[assignment_col] == "none").sum()) if n else 0
    unspecified = int((countable[classification_col] == "unspecified").sum()) if n else 0
    ambiguous = int((countable[classification_col] == "ambiguous").sum()) if n else 0

    return {
        "explicit": ProportionResult(f"{role}: explicit gender assignment", explicit, n),
        "implied": ProportionResult(f"{role}: implied gender assignment", implied, n),
        "any_assignment": ProportionResult(f"{role}: any assignment (explicit + implied)", explicit + implied, n),
        "unspecified": ProportionResult(f"{role}: gender-neutral / unspecified", unspecified, n),
        "ambiguous": ProportionResult(f"{role}: ambiguous / insufficient evidence", ambiguous, n),
        "none_assigned": ProportionResult(f"{role}: no assignment detected", none_assigned, n),
    }


def missing_and_invalid_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {"failed_runs": 0, "invalid_unreviewed": 0, "unreviewed_total": 0}
    failed_runs = int((df["run_status"] == "failed").sum())
    invalid_unreviewed = int(((~df["evaluation_is_valid"]) & (~df["has_human_review"]) & (df["run_status"] == "success")).sum())
    unreviewed_total = int(((~df["has_human_review"]) & (df["run_status"] == "success")).sum())
    return {
        "failed_runs": failed_runs,
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


def disagreement_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["doctor_disagreement"] | df["nurse_disagreement"]]
