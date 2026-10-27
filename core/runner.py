"""Experiment execution: turns an experiment's config into a concrete list of
runs, executes them against OpenRouter, and immediately evaluates each
successful response. Designed to be safely pausable and resumable - every
run is a durable DB row, and nothing is re-sent once it has a recorded
success or failure.
"""
from __future__ import annotations

import json
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable

from core import db, evaluator, openrouter_client, sentencing_evaluator

GENERATION_PARAM_NAMES = [
    "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "seed",
]


def estimate_call_count(num_prompt_variants: int, num_models: int, repetitions: int) -> int:
    """Number of tested-model API calls. Evaluator calls are separate (one
    per successfully completed run, plus any repair retries)."""
    return num_prompt_variants * num_models * repetitions


def compute_unsupported_params(experiment: dict[str, Any], cached_model: dict[str, Any] | None) -> list[str]:
    requested = [p for p in GENERATION_PARAM_NAMES if experiment.get(p) is not None]
    if not cached_model:
        return []
    supported = set(json.loads(cached_model.get("supported_parameters_json") or "[]"))
    if not supported:
        return []
    return [p for p in requested if p not in supported]


def plan_runs(conn: sqlite3.Connection, experiment_id: int) -> list[int]:
    """Creates `runs` rows for every (prompt_version x model x repetition)
    combination that doesn't already have one, and locks every prompt
    version used so it becomes immutable. Idempotent: calling it again after
    some runs already exist will not create duplicates for combinations that
    already have the full set of repetitions planned."""
    experiment = db.get_experiment(conn, experiment_id)
    if experiment is None:
        raise ValueError(f"experiment {experiment_id} not found")

    prompt_versions = db.list_prompt_versions(conn, experiment_id, latest_only=True)
    models = db.list_experiment_models(conn, experiment_id)
    if not prompt_versions:
        raise ValueError("experiment has no prompt versions")
    if not models:
        raise ValueError("experiment has no selected models")

    cached_models = {m["model_id"]: m for m in db.get_cached_models(conn)}
    existing = db.list_runs(conn, experiment_id)
    existing_keys = {(r["prompt_version_id"], r["model_id"], r["run_number"]) for r in existing}
    next_sequence = (max((r["sequence_number"] for r in existing), default=0)) + 1

    created_ids: list[int] = []
    for pv in prompt_versions:
        db.lock_prompt_version(conn, pv["id"])
        for model in models:
            unsupported = compute_unsupported_params(experiment, cached_models.get(model["model_id"]))
            for rep in range(1, experiment["repetitions"] + 1):
                key = (pv["id"], model["model_id"], rep)
                if key in existing_keys:
                    continue
                params = {p: experiment.get(p) for p in GENERATION_PARAM_NAMES if experiment.get(p) is not None}
                run_id = db.create_run(
                    conn, experiment_id, pv["id"], model["model_id"], rep, next_sequence,
                    request_params=params, unsupported_params=unsupported,
                )
                created_ids.append(run_id)
                existing_keys.add(key)
                next_sequence += 1

    db.update_experiment_status(conn, experiment_id, "ready")
    return created_ids


@dataclass
class RunOutcome:
    run_id: int
    status: str
    model_id: str
    error: str | None = None


def _reset_stale_running(conn: sqlite3.Connection, experiment_id: int) -> None:
    """A 'running' row with no completed_at means a previous process was
    interrupted mid-call (crash, kill). No response was persisted for it, so
    it's safe to simply re-queue it - we never retry a call that actually
    produced a stored response."""
    conn.execute(
        "UPDATE runs SET status = 'pending' WHERE experiment_id = ? AND status = 'running'",
        (experiment_id,),
    )


def _runs_needing_evaluation(conn: sqlite3.Connection, experiment_id: int) -> list[dict[str, Any]]:
    """Successful runs with no evaluation yet, or whose most recent evaluation
    attempt never finished (status still 'pending'). Both indicate the process
    was killed somewhere between mark_run_success and evaluate_response
    returning. These are swept and retried rather than left stuck forever,
    mirroring how _reset_stale_running recovers a crashed run attempt. Only
    called when run_evaluation=True, so a deliberate per-call opt-out of
    evaluation (used in tests) never triggers this sweep."""
    latest_by_run = db.get_latest_evaluations_by_run(conn, experiment_id)
    needing = []
    for run in db.list_runs(conn, experiment_id):
        if run["status"] != "success":
            continue
        latest = latest_by_run.get(run["id"])
        if latest is None or latest["status"] == "pending":
            needing.append(run)
    return needing


def count_errored_evaluations(conn: sqlite3.Connection, experiment_id: int) -> int:
    """Successful runs whose most recent evaluation ended in status='error' -
    the evaluator API call itself failed (bad/deprecated model id, rate
    limit exhausted, etc.), as opposed to 'invalid_needs_review' (the
    evaluator responded but its output failed validation, which is correctly
    routed to a human rather than auto-retried)."""
    latest_by_run = db.get_latest_evaluations_by_run(conn, experiment_id)
    count = 0
    for run in db.list_runs(conn, experiment_id):
        if run["status"] != "success":
            continue
        latest = latest_by_run.get(run["id"])
        if latest is not None and latest["status"] == "error":
            count += 1
    return count


def retry_failed_runs(conn: sqlite3.Connection, experiment_id: int) -> list[int]:
    """Creates a fresh pending run for every currently-failed run in this
    experiment that doesn't already have a retry (same prompt version,
    model, and request params), so it gets picked up by the normal
    pending-run execution flow. The original failed row is never touched -
    the failure stays in the audit trail exactly as it happened; this only
    adds a new attempt with a new run_number, linked back via
    retried_from_run_id, the same way a human would if they re-ran a single
    cell by hand (this is exactly what had to be done manually, via a
    one-off script, for 20 truncated Gemini runs before this function
    existed). Calling this again before re-executing is a safe no-op - a
    failed run that already has a pending/running/successful retry is
    skipped, so repeated clicks don't pile up duplicate retries."""
    all_runs = db.list_runs(conn, experiment_id)
    already_retried_ids = {r["retried_from_run_id"] for r in all_runs if r["retried_from_run_id"] is not None}
    failed_runs = [r for r in all_runs if r["status"] == "failed" and r["id"] not in already_retried_ids]
    if not failed_runs:
        return []

    next_seq = max((r["sequence_number"] for r in all_runs), default=0) + 1
    max_run_number: dict[tuple[int, str], int] = {}
    for r in all_runs:
        key = (r["prompt_version_id"], r["model_id"])
        max_run_number[key] = max(max_run_number.get(key, 0), r["run_number"])

    created = []
    for r in failed_runs:
        key = (r["prompt_version_id"], r["model_id"])
        max_run_number[key] += 1
        new_id = db.create_run(
            conn, experiment_id, r["prompt_version_id"], r["model_id"],
            run_number=max_run_number[key], sequence_number=next_seq,
            request_params=json.loads(r["request_params_json"]),
            retried_from_run_id=r["id"],
            unsupported_params=json.loads(r["unsupported_params_json"] or "[]"),
        )
        created.append(new_id)
        next_seq += 1
    return created


def retry_errored_evaluations(
    db_path: str | None, experiment_id: int, api_key: str, delay_seconds: float = 0.5,
) -> int:
    """Re-attempts evaluation for every successful run whose most recent
    evaluation ended in status='error' - e.g. after fixing a deprecated
    evaluator model id. Each retry creates a new evaluation attempt; the
    failed one is preserved, not overwritten. Returns the number retried.

    A small delay is inserted between calls: free-tier / low-spend
    OpenRouter keys enforce a per-account cap on simultaneously in-flight
    credit reservations, independent of total balance. Firing many
    evaluation calls back-to-back with no pacing can trip "would exceed your
    available credits given your current in-flight requests" even when the
    account has ample balance - this showed up in practice retrying 150
    evaluations at once."""
    with db.get_connection(db_path) as conn:
        latest_by_run = db.get_latest_evaluations_by_run(conn, experiment_id)
        to_retry = []
        for run in db.list_runs(conn, experiment_id):
            if run["status"] != "success":
                continue
            latest = latest_by_run.get(run["id"])
            if latest is not None and latest["status"] == "error":
                to_retry.append(run)
        prompt_versions = {pv["id"]: pv for pv in db.list_prompt_versions(conn, experiment_id)}

    for i, run in enumerate(to_retry):
        if i > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)
        prompt_version = prompt_versions[run["prompt_version_id"]]
        _evaluate_run(db_path, run["id"], prompt_version["prompt_text"], run["raw_response"], api_key)

    return len(to_retry)


def _order_pending_runs(conn: sqlite3.Connection, experiment_id: int, randomize: bool) -> list[dict[str, Any]]:
    pending = db.get_pending_runs(conn, experiment_id)
    already_ordered = [r for r in pending if r["execution_order"] is not None]
    not_ordered = [r for r in pending if r["execution_order"] is None]

    if randomize:
        random.shuffle(not_ordered)
    next_order = (max((r["execution_order"] for r in already_ordered), default=0)) + 1
    for r in not_ordered:
        db.set_run_execution_order(conn, r["id"], next_order)
        r["execution_order"] = next_order
        next_order += 1

    return sorted(already_ordered + not_ordered, key=lambda r: r["execution_order"])


def execute_pending_runs(
    db_path: str | None,
    experiment_id: int,
    api_key: str,
    randomize_order: bool = False,
    should_stop: Callable[[], bool] | None = None,
    on_run_complete: Callable[[RunOutcome], None] | None = None,
    run_evaluation: bool = True,
) -> list[RunOutcome]:
    """Executes all pending runs for an experiment, one at a time. Safe to
    call repeatedly (pause = stop calling it; resume = call it again)."""
    with db.get_connection(db_path) as conn:
        _reset_stale_running(conn, experiment_id)
        db.update_experiment_status(conn, experiment_id, "running")
        ordered = _order_pending_runs(conn, experiment_id, randomize_order)
        prompt_cache: dict[int, dict[str, Any]] = {}

    outcomes: list[RunOutcome] = []

    for run in ordered:
        if should_stop and should_stop():
            with db.get_connection(db_path) as conn:
                db.update_experiment_status(conn, experiment_id, "paused")
            break

        with db.get_connection(db_path) as conn:
            db.mark_run_running(conn, run["id"])
            pv_id = run["prompt_version_id"]
            if pv_id not in prompt_cache:
                prompt_cache[pv_id] = db.get_prompt_version(conn, pv_id)
            prompt_version = prompt_cache[pv_id]

        outcome = _execute_single_run(db_path, run, prompt_version, api_key, run_evaluation)
        outcomes.append(outcome)
        if on_run_complete:
            on_run_complete(outcome)

    with db.get_connection(db_path) as conn:
        remaining = db.get_pending_runs(conn, experiment_id)
        if not remaining:
            db.update_experiment_status(conn, experiment_id, "completed")

    return outcomes


def _execute_single_run(
    db_path: str | None,
    run: dict[str, Any],
    prompt_version: dict[str, Any],
    api_key: str,
    run_evaluation: bool,
) -> RunOutcome:
    params = json.loads(run["request_params_json"])
    messages = [{"role": "user", "content": prompt_version["prompt_text"]}]

    try:
        result = openrouter_client.chat_completion(
            model=run["model_id"], messages=messages, api_key=api_key, **params,
        )
    except openrouter_client.OpenRouterError as exc:
        with db.get_connection(db_path) as conn:
            db.mark_run_failed(conn, run["id"], exc.message, exc.status_code, latency_ms=0, retry_count=0)
        return RunOutcome(run["id"], "failed", run["model_id"], error=exc.message)

    with db.get_connection(db_path) as conn:
        db.mark_run_success(
            conn, run["id"], result.content, result.response_id, result.response_model,
            result.finish_reason, result.prompt_tokens, result.completion_tokens, result.total_tokens,
            result.cost_usd, result.latency_ms, result.retry_count,
        )

    if run_evaluation and result.content.strip():
        _evaluate_run(db_path, run["id"], prompt_version["prompt_text"], result.content, api_key)

    return RunOutcome(run["id"], "success", run["model_id"])


def execute_one_pending_run(
    db_path: str | None,
    experiment_id: int,
    api_key: str,
    randomize_order: bool = False,
    run_evaluation: bool = True,
) -> RunOutcome | None:
    """Executes exactly one pending run and returns. Intended for UI loops
    that want to show live progress and support a responsive pause button
    between steps (see pages/2_Run_Experiment.py) rather than blocking for
    the whole experiment inside one call.

    Before picking a new pending run, this also checks for any already-
    successful run whose evaluation never finished (a prior crash) and
    recovers exactly one of those first - see _runs_needing_evaluation."""
    stuck_eval_run: dict[str, Any] | None = None
    with db.get_connection(db_path) as conn:
        _reset_stale_running(conn, experiment_id)
        db.update_experiment_status(conn, experiment_id, "running")

        if run_evaluation:
            stuck = _runs_needing_evaluation(conn, experiment_id)
            stuck_eval_run = stuck[0] if stuck else None

        if stuck_eval_run is not None:
            run = stuck_eval_run
            prompt_version = db.get_prompt_version(conn, run["prompt_version_id"])
        else:
            ordered = _order_pending_runs(conn, experiment_id, randomize_order)
            if not ordered:
                db.update_experiment_status(conn, experiment_id, "completed")
                return None
            run = ordered[0]
            db.mark_run_running(conn, run["id"])
            prompt_version = db.get_prompt_version(conn, run["prompt_version_id"])

    if stuck_eval_run is not None:
        _evaluate_run(db_path, run["id"], prompt_version["prompt_text"], run["raw_response"], api_key)
        outcome = RunOutcome(run["id"], "success", run["model_id"])
    else:
        outcome = _execute_single_run(db_path, run, prompt_version, api_key, run_evaluation)

    with db.get_connection(db_path) as conn:
        nothing_pending = not db.get_pending_runs(conn, experiment_id)
        nothing_stuck = (not run_evaluation) or (not _runs_needing_evaluation(conn, experiment_id))
        if nothing_pending and nothing_stuck:
            db.update_experiment_status(conn, experiment_id, "completed")

    return outcome


def _evaluate_run(db_path: str | None, run_id: int, original_prompt: str, response_text: str, api_key: str) -> None:
    """Dispatches to the gender-healthcare or sentencing-bias evaluator based
    on the owning experiment's evaluation_domain. Both domains share the same
    generic run/retry/pause-resume machinery above this function - only the
    evaluator call and the result-persisting write differ."""
    with db.get_connection(db_path) as conn:
        experiment_row = conn.execute(
            "SELECT e.evaluator_model_id, e.evaluation_domain FROM runs r "
            "JOIN experiments e ON e.id = r.experiment_id WHERE r.id = ?",
            (run_id,),
        ).fetchone()
        domain = (experiment_row["evaluation_domain"] if experiment_row else None) or "gender_healthcare"
        evaluator_model = (experiment_row["evaluator_model_id"] if experiment_row else None) or evaluator.default_evaluator_model()
        prompt_version_tag = (
            sentencing_evaluator.SENTENCING_EVALUATOR_PROMPT_VERSION if domain == "sentencing_bias"
            else evaluator.EVALUATOR_PROMPT_VERSION
        )
        evaluation_id = db.create_evaluation(conn, run_id, evaluator_model, prompt_version_tag)

    try:
        if domain == "sentencing_bias":
            outcome = sentencing_evaluator.evaluate_sentencing_response(
                original_prompt, response_text, evaluator_model, api_key=api_key,
            )
        else:
            outcome = evaluator.evaluate_response(original_prompt, response_text, evaluator_model, api_key=api_key)
    except openrouter_client.OpenRouterError as exc:
        with db.get_connection(db_path) as conn:
            db.mark_evaluation_error(conn, evaluation_id, str(exc))
        return

    with db.get_connection(db_path) as conn:
        parsed_dict = outcome.parsed.model_dump() if outcome.parsed else None
        if domain == "sentencing_bias":
            db.update_sentencing_evaluation_result(
                conn, evaluation_id, outcome.raw_text, outcome.is_valid, outcome.error, parsed_dict,
                cost_usd=outcome.cost_usd,
            )
        else:
            db.update_evaluation_result(
                conn, evaluation_id, outcome.raw_text, outcome.is_valid, outcome.error, parsed_dict,
                cost_usd=outcome.cost_usd,
            )
