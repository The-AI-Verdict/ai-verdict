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


def test_video_brief_never_fabricates_example_when_none_exists(db_path):
    parsed = {
        "doctor": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }
    exp_id = _seed_full_run(db_path, fx.NEUTRAL_RESPONSE, parsed)
    with db.get_connection(db_path) as conn:
        brief = export.generate_video_brief(conn, exp_id)
    assert "do not fabricate an example" in brief.lower()


def test_video_brief_shows_real_quote_when_one_exists(db_path):
    parsed = {
        "doctor": {"gender_classification": "explicit_male", "assignment_type": "explicit", "quote": "his long shift", "reasoning": "r"},
        "nurse": {"gender_classification": "explicit_female", "assignment_type": "explicit", "quote": "her hands steady", "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }
    exp_id = _seed_full_run(db_path, fx.EXPLICIT_MALE_DOCTOR_FEMALE_NURSE_RESPONSE, parsed)
    with db.get_connection(db_path) as conn:
        brief = export.generate_video_brief(conn, exp_id)
    assert fx.EXPLICIT_MALE_DOCTOR_FEMALE_NURSE_RESPONSE[:50] in brief


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
