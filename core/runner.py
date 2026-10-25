"""Experiment execution: turns an experiment's config into a concrete list of
runs, executes them against OpenRouter, and immediately evaluates each
successful response. Designed to be safely pausable and resumable - every
run is a durable DB row, and nothing is re-sent once it has a recorded
success or failure.
"""
from __future__ import annotations

import random
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable

from core import db, evaluator, openrouter_client

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
    import json
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
    needing = []
    for run in db.list_runs(conn, experiment_id):
        if run["status"] != "success":
            continue
        latest = db.get_latest_evaluation_for_run(conn, run["id"])
        if latest is None or latest["status"] == "pending":
            needing.append(run)
    return needing


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
    import json
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
    with db.get_connection(db_path) as conn:
        experiment_row = conn.execute(
            "SELECT e.evaluator_model_id FROM runs r JOIN experiments e ON e.id = r.experiment_id WHERE r.id = ?",
            (run_id,),
        ).fetchone()
        evaluator_model = (experiment_row["evaluator_model_id"] if experiment_row else None) or evaluator.default_evaluator_model()
        evaluation_id = db.create_evaluation(conn, run_id, evaluator_model, evaluator.EVALUATOR_PROMPT_VERSION)

    try:
        outcome = evaluator.evaluate_response(original_prompt, response_text, evaluator_model, api_key=api_key)
    except openrouter_client.OpenRouterError as exc:
        with db.get_connection(db_path) as conn:
            db.mark_evaluation_error(conn, evaluation_id, str(exc))
        return

    with db.get_connection(db_path) as conn:
        parsed_dict = outcome.parsed.model_dump() if outcome.parsed else None
        db.update_evaluation_result(conn, evaluation_id, outcome.raw_text, outcome.is_valid, outcome.error, parsed_dict)
