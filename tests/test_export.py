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
        brief = export.generate_video_brief(conn, exp_id)

    for doc in (report, brief):
        assert "neutral_variant" in doc and "forced_variant" in doc
        # the forced variant's 100% explicit rate for model-a must appear
        # undiluted - not averaged with the neutral variant down to ~50%
        assert "100.0%" in doc or "100.0" in doc


def test_generate_full_script_single_variant_no_crash(db_path):
    parsed = {
        "doctor": {"gender_classification": "explicit_male", "assignment_type": "explicit", "quote": "he", "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }
    exp_id = _seed_full_run(db_path, "he reviewed the chart. " * 3, parsed)
    with db.get_connection(db_path) as conn:
        script = export.generate_full_script(conn, exp_id)

    assert "[COLD OPEN" in script
    assert "[CLOSING" in script
    assert "## ##" not in script, "act headers must not be double-prefixed"
    assert '""' not in script, "narration lines must not have doubled quote marks"


def test_generate_full_script_never_fabricates_a_quote_when_none_exists(db_path):
    parsed = {
        "doctor": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }
    exp_id = _seed_full_run(db_path, fx.NEUTRAL_RESPONSE, parsed)
    with db.get_connection(db_path) as conn:
        script = export.generate_full_script(conn, exp_id)
    assert "manufacture a dramatic clip" in script


def test_generate_full_script_contrast_structure_and_tie_break_prefers_baseline(db_path):
    """Regression test: when multiple variants tie at the minimum assignment
    rate, the cold open / Act 1 must reference 'baseline' specifically, not
    an arbitrary other variant - a cold open built around a role-reversed or
    alternative-profession prompt while the channel's hypothesis is about
    doctors and nurses would be a confusing, wrong-sounding hook."""
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(conn, name="Contrast test", hypothesis="Models assume gender.", repetitions=5)
        alt_pv = db.create_prompt_version(conn, exp_id, "alternative_profession", "Pilot scenario.")
        base_pv = db.create_prompt_version(conn, exp_id, "baseline", "Doctor/nurse scenario.")
        forced_pv = db.create_prompt_version(conn, exp_id, "forced_pronoun", "Forced scenario.")
        db.add_experiment_model(conn, exp_id, "model-a")

        seq = 1
        for pv_id, variant_name in ((alt_pv, "alt"), (base_pv, "base")):
            for i in range(5):
                run_id = db.create_run(conn, exp_id, pv_id, "model-a", run_number=i + 1, sequence_number=seq,
                                        request_params={})
                seq += 1
                db.mark_run_success(conn, run_id, f"neutral text {i}", f"g{seq}", "model-a", "stop", 10, 5, 15, 0.0, 10, 0)
                ev = db.create_evaluation(conn, run_id, "evaluator/x", "v1")
                db.update_evaluation_result(conn, ev, "{}", True, None, _neutral_eval())
        for i in range(5):
            run_id = db.create_run(conn, exp_id, forced_pv, "model-a", run_number=i + 1, sequence_number=seq,
                                    request_params={})
            seq += 1
            db.mark_run_success(conn, run_id, f"he and she text {i}", f"g{seq}", "model-a", "stop", 10, 5, 15, 0.0, 10, 0)
            ev = db.create_evaluation(conn, run_id, "evaluator/x", "v1")
            db.update_evaluation_result(conn, ev, "{}", True, None, _explicit_eval())

    with db.get_connection(db_path) as conn:
        script = export.generate_full_script(conn, exp_id)

    cold_open_section = script.split("[INTRO")[0]
    assert 'the exact prompt text from "baseline"' in cold_open_section
    assert "alternative_profession" not in cold_open_section
    assert "[ACT 2" in script
    assert "[ACT 3" in script

    # Regression test: baseline (Act 1) and forced_pronoun (Act 2) are both
    # narrated in the body, so the closing caveat's "I also ran N other
    # versions" must only count alternative_profession - not re-count a
    # variant that was literally the subject of an earlier act.
    caveat_section = script.split("[WHAT THIS DOES AND DOESN'T PROVE]")[1]
    assert "1 other version" in caveat_section
    assert "2 other version" not in caveat_section
    assert "baseline" not in caveat_section.split("linked below")[0]


def test_already_narrated_variant_names_covers_both_acts_when_contrasting():
    low = {"variant_name": "baseline"}
    high = {"variant_name": "forced_pronoun"}
    contrast_ranking = export.VariantRanking([], True, low, 0.0, 10, high, 0.9, 10)
    assert export._already_narrated_variant_names(contrast_ranking) == {"baseline", "forced_pronoun"}

    no_contrast_ranking = export.VariantRanking([], False, low, 0.3, 10, high, 0.4, 10)
    assert export._already_narrated_variant_names(no_contrast_ranking) == {"forced_pronoun"}


def test_prose_list_grammar():
    assert export._prose_list([]) == ""
    assert export._prose_list(["a"]) == "a"
    assert export._prose_list(["a", "b"]) == "a and b"
    assert export._prose_list(["a", "b", "c"]) == "a, b and c"
    many = [f"m{i}" for i in range(9)]
    out = export._prose_list(many, max_items=6)
    assert out.endswith("and 4 others")
    assert "m0" in out and "m8" not in out


def testword_count_helper():
    assert export.word_count("one two three") == 3
    assert export.word_count("") == 0


def test_generate_blog_post_single_variant_no_crash(db_path):
    parsed = {
        "doctor": {"gender_classification": "explicit_male", "assignment_type": "explicit", "quote": "he", "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }
    exp_id = _seed_full_run(db_path, "he reviewed the chart. " * 3, parsed)
    with db.get_connection(db_path) as conn:
        post = export.generate_blog_post(conn, exp_id)

    assert post.startswith("#")
    assert "Pick one of these" in post
    assert export.word_count(post) > 50


def test_generate_blog_post_never_fabricates_a_quote_when_none_exists(db_path):
    exp_id = _seed_full_run(db_path, fx.NEUTRAL_RESPONSE, _neutral_eval())
    with db.get_connection(db_path) as conn:
        post = export.generate_blog_post(conn, exp_id)
    assert "One actual response" not in post, "must not claim a quote exists when nothing was gendered"


def test_generate_blog_post_contrast_structure_and_no_duplicate_boilerplate(db_path):
    """Regression test for a real bug: the shared section-builder used the
    exact same closing sentence ('The rest just... didn't...') in both the
    near-zero round and the near-universal round, which is the kind of
    verbatim repetition that makes generated prose read as templated rather
    than written."""
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(conn, name="Contrast test", hypothesis="Models assume gender.", repetitions=5)
        base_pv = db.create_prompt_version(conn, exp_id, "baseline", "Doctor/nurse scenario.")
        forced_pv = db.create_prompt_version(conn, exp_id, "forced_pronoun", "Forced scenario.")
        db.add_experiment_model(conn, exp_id, "model-a")
        db.add_experiment_model(conn, exp_id, "model-b")

        seq = 1
        for model_id in ("model-a", "model-b"):
            for i in range(5):
                run_id = db.create_run(conn, exp_id, base_pv, model_id, run_number=i + 1, sequence_number=seq,
                                        request_params={})
                seq += 1
                db.mark_run_success(conn, run_id, f"neutral {i}", f"g{seq}", model_id, "stop", 10, 5, 15, 0.0, 10, 0)
                ev = db.create_evaluation(conn, run_id, "evaluator/x", "v1")
                db.update_evaluation_result(conn, ev, "{}", True, None, _neutral_eval())
            for i in range(5):
                run_id = db.create_run(conn, exp_id, forced_pv, model_id, run_number=i + 1, sequence_number=seq,
                                        request_params={})
                seq += 1
                db.mark_run_success(conn, run_id, f"he and she {i}", f"g{seq}", model_id, "stop", 10, 5, 15, 0.0, 10, 0)
                ev = db.create_evaluation(conn, run_id, "evaluator/x", "v1")
                db.update_evaluation_result(conn, ev, "{}", True, None, _explicit_eval())

    with db.get_connection(db_path) as conn:
        post = export.generate_blog_post(conn, exp_id)

    assert "Round one" in post and "Round two" in post
    # the near-zero round and the near-universal round must get different
    # closing clauses, not the same templated sentence repeated verbatim
    assert post.count("The rest just didn't") <= 1
    assert "The handful that didn't were the exception" in post
    assert "model-a and model-b" in post or "`model-a`" in post  # Oxford-comma-correct list, no raw join artefact

    # Regression test: baseline and forced_pronoun are the ONLY two variants
    # and both get their own narrated section (Round one / Round two), so
    # there are zero genuinely-unmentioned variants left for the closing
    # caveat to reference - it must not claim "I ran N other versions".
    assert "other version" not in post


def test_generate_blog_post_mentions_data_quality_mess_only_when_present(db_path):
    parsed = {
        "doctor": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "nurse": {"gender_classification": "unspecified", "assignment_type": "none", "quote": None, "reasoning": "r"},
        "uncertain": False, "uncertain_reason": None,
    }
    exp_id = _seed_full_run(db_path, fx.NEUTRAL_RESPONSE, parsed)
    with db.get_connection(db_path) as conn:
        post_clean = export.generate_blog_post(conn, exp_id)
    assert "doesn't make it into the chart" not in post_clean

    # Add a second, independent successful run so marking the first one
    # evidence-insufficient still leaves at least one countable result -
    # otherwise the generator correctly (and separately, see the other
    # test) falls back to its "no classified results" early return instead
    # of exercising the mess section this test is actually checking.
    with db.get_connection(db_path) as conn:
        runs = db.list_runs(conn, exp_id)
        pv_id = runs[0]["prompt_version_id"]
        run2_id = db.create_run(conn, exp_id, pv_id, "openai/gpt-4o-mini", run_number=2, sequence_number=2,
                                 request_params={"temperature": 0.7})
        db.mark_run_success(conn, run2_id, fx.NEUTRAL_RESPONSE, "gen-2", "openai/gpt-4o-mini-2024", "stop",
                             50, 20, 70, 0.001, 500, 0)
        ev2 = db.create_evaluation(conn, run2_id, "anthropic/claude-3.5-haiku", "v1")
        db.update_evaluation_result(conn, ev2, raw_evaluator_response=json.dumps(parsed), is_valid=True,
                                     validation_errors=None, parsed=parsed)

        run1_id = runs[0]["id"]
        ev1 = db.get_latest_evaluation_for_run(conn, run1_id)
        db.create_human_review(conn, run1_id, ev1["id"], "Alex", "unspecified", "none", "unspecified", "none",
                                evidence_insufficient=True, reviewer_note="truncated")
        post_with_mess = export.generate_blog_post(conn, exp_id)
    assert "doesn't make it into the chart" in post_with_mess
    assert "1 response was thrown out" in post_with_mess, "singular count must use correct grammar, not '1 responses'"

