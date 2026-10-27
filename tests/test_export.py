from __future__ import annotations

import csv
import json
from io import StringIO

from core import db, export
from tests.fixtures import sample_responses as fx


def _seed_full_run(db_path, response_text: str, parsed_eval: dict | None, is_valid: bool = True):
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(conn, name="Gender test", hypothesis="h", repetitions=1)
        pv_id = db.create_prompt_version(conn, exp_id, "baseline", "A doctor and a nurse...")
        db.add_experiment_model(conn, exp_id, "openai/gpt-4o-mini", provider="openai")
        run_id = db.create_run(conn, exp_id, pv_id, "openai/gpt-4o-mini", run_number=1, sequence_number=1,
                                request_params={"temperature": 0.7})
        db.mark_run_success(conn, run_id, response_text, "gen-1", "openai/gpt-4o-mini-2024", "stop",
                             50, 20, 70, 0.001, 500, 0)
        eval_id = db.create_evaluation(conn, run_id, "anthropic/claude-3.5-haiku", "v1")
        db.update_evaluation_result(conn, eval_id, raw_evaluator_response=json.dumps(parsed_eval or {}),
                                     is_valid=is_valid, validation_errors=None if is_valid else "bad",
                                     parsed=parsed_eval)
    return exp_id


def test_dataframe_to_csv_has_header_and_rows(db_path):
    parsed = {
        "doctor": {"gender_classification": "explicit_male", "assignment_type": "explicit", "quote": "his", "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }
    exp_id = _seed_full_run(db_path, fx.EXPLICIT_MALE_DOCTOR_FEMALE_NURSE_RESPONSE, parsed)
    from core import analysis
    with db.get_connection(db_path) as conn:
        rows = analysis.build_experiment_rows(conn, exp_id)
    df = analysis.rows_to_dataframe(rows)

    csv_text = export.dataframe_to_csv(df)
    reader = csv.DictReader(StringIO(csv_text))
    out_rows = list(reader)
    assert len(out_rows) == 1
    assert out_rows[0]["model_id"] == "openai/gpt-4o-mini"
    assert out_rows[0]["doctor_final_classification"] == "explicit_male"


def test_markdown_report_contains_required_sections(db_path):
    parsed = {
        "doctor": {"gender_classification": "explicit_male", "assignment_type": "explicit", "quote": "his", "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }
    exp_id = _seed_full_run(db_path, fx.EXPLICIT_MALE_DOCTOR_FEMALE_NURSE_RESPONSE, parsed)
    with db.get_connection(db_path) as conn:
        report = export.generate_markdown_report(conn, exp_id)

    for heading in ("Research question", "Prompt versions tested", "Models tested",
                    "Experimental settings", "Run outcomes", "## Results", "Evaluation criteria",
                    "Limitations and possible confounders", "Findings summary"):
        assert heading in report, f"missing section: {heading}"
    assert "1/1" in report or "100.0%" in report


def test_markdown_report_labels_small_samples_exploratory(db_path):
    parsed = {
        "doctor": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }
    exp_id = _seed_full_run(db_path, fx.NEUTRAL_RESPONSE, parsed)
    with db.get_connection(db_path) as conn:
        report = export.generate_markdown_report(conn, exp_id)
    assert "exploratory" in report.lower()


def test_json_archive_round_trips_core_entities(db_path):
    parsed = {
        "doctor": {"gender_classification": "explicit_male", "assignment_type": "explicit", "quote": "his", "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }
    exp_id = _seed_full_run(db_path, fx.EXPLICIT_MALE_DOCTOR_FEMALE_NURSE_RESPONSE, parsed)
    with db.get_connection(db_path) as conn:
        archive = export.build_json_archive(conn, exp_id)

    assert archive["experiment"]["id"] == exp_id
    assert len(archive["runs"]) == 1
    assert archive["runs"][0]["raw_response"] == fx.EXPLICIT_MALE_DOCTOR_FEMALE_NURSE_RESPONSE
    assert len(archive["evaluations"]) == 1
    # must be JSON-serialisable end to end
    json.dumps(archive, default=str)


def _neutral_eval():
    return {
        "doctor": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }


def _explicit_eval():
    return {
        "doctor": {"gender_classification": "explicit_male", "assignment_type": "explicit", "quote": "he", "reasoning": "r"},
        "nurse": {"gender_classification": "explicit_female", "assignment_type": "explicit", "quote": "she", "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }


def test_by_model_breakdown_does_not_average_away_a_variant_specific_signal(db_path):
    """Regression test: a 'by model' table that pools every prompt variant
    together would blend a variant where every response is gender-neutral
    with one where every response is explicitly gendered into a flat,
    misleading ~50% for every model - hiding the actual finding. The report
    must break 'by model' down per variant instead."""
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(conn, name="Multi-variant test", repetitions=1)
        neutral_pv = db.create_prompt_version(conn, exp_id, "neutral_variant", "Describe neutrally.")
        forced_pv = db.create_prompt_version(conn, exp_id, "forced_variant", "Describe with pronouns.")
        db.add_experiment_model(conn, exp_id, "model-a")

        neutral_run = db.create_run(conn, exp_id, neutral_pv, "model-a", run_number=1, sequence_number=1,
                                     request_params={})
        db.mark_run_success(conn, neutral_run, "neutral text", "g1", "model-a", "stop", 10, 5, 15, 0.0, 10, 0)
        ev1 = db.create_evaluation(conn, neutral_run, "evaluator/x", "v1")
        db.update_evaluation_result(conn, ev1, "{}", True, None, _neutral_eval())

        forced_run = db.create_run(conn, exp_id, forced_pv, "model-a", run_number=1, sequence_number=2,
                                    request_params={})
        db.mark_run_success(conn, forced_run, "he and she text", "g2", "model-a", "stop", 10, 5, 15, 0.0, 10, 0)
        ev2 = db.create_evaluation(conn, forced_run, "evaluator/x", "v1")
        db.update_evaluation_result(conn, ev2, "{}", True, None, _explicit_eval())

    with db.get_connection(db_path) as conn:
        report = export.generate_markdown_report(conn, exp_id)

    assert "neutral_variant" in report and "forced_variant" in report
    # the forced variant's 100% explicit rate for model-a must appear
    # undiluted - not averaged with the neutral variant down to ~50%
    assert "100.0%" in report or "100.0" in report
