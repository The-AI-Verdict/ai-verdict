from __future__ import annotations

from core import db, sentencing_seed


def test_create_builtin_sentencing_experiment_seeds_five_variants(db_path):
    with db.get_connection(db_path) as conn:
        exp_id = sentencing_seed.create_builtin_sentencing_experiment(conn)
        experiment = db.get_experiment(conn, exp_id)
        variants = {pv["variant_name"]: pv for pv in db.list_prompt_versions(conn, exp_id)}

    assert experiment["evaluation_domain"] == "sentencing_bias"
    assert set(variants) == {"baseline", "name_only", "dialect_aae", "dialect_nigerian", "dialect_glaswegian"}


def test_seed_is_idempotent(db_path):
    with db.get_connection(db_path) as conn:
        first_id = sentencing_seed.create_builtin_sentencing_experiment(conn)
        second_id = sentencing_seed.create_builtin_sentencing_experiment(conn)
    assert first_id == second_id


def test_only_name_changes_between_baseline_and_name_only():
    """The core experimental-design guarantee: name_only must differ from
    baseline in exactly the defendant's name and nowhere else, so a measured
    effect can't be blamed on anything but the name."""
    baseline_words = sentencing_seed.BASELINE_PROMPT.replace("James Mitchell", "<NAME>")
    name_only_words = sentencing_seed.NAME_ONLY_PROMPT.replace("Jamal Washington", "<NAME>")
    assert baseline_words == name_only_words


def test_dialect_variants_keep_the_baseline_name_and_only_change_the_statement():
    for prompt in (
        sentencing_seed.DIALECT_AAE_PROMPT,
        sentencing_seed.DIALECT_NIGERIAN_PROMPT,
        sentencing_seed.DIALECT_GLASWEGIAN_PROMPT,
    ):
        assert "James Mitchell" in prompt
        assert "Jamal Washington" not in prompt


def test_case_facts_are_byte_identical_across_all_five_variants():
    """The offence, value, and record (everything before the defendant's own
    statement) must be the exact same text in every variant - only the name
    placeholder or the statement itself may differ."""
    prompts = [
        sentencing_seed.BASELINE_PROMPT,
        sentencing_seed.NAME_ONLY_PROMPT,
        sentencing_seed.DIALECT_AAE_PROMPT,
        sentencing_seed.DIALECT_NIGERIAN_PROMPT,
        sentencing_seed.DIALECT_GLASWEGIAN_PROMPT,
    ]
    case_bodies = [p.split("CASE SUMMARY:")[1].split("'s statement to the court")[0] for p in prompts]
    normalized = [body.replace("James Mitchell", "<NAME>").replace("Jamal Washington", "<NAME>")
                  for body in case_bodies]
    assert len(set(normalized)) == 1


def test_stolen_value_is_in_pounds_not_dollars():
    assert "£1,200" in sentencing_seed.BASELINE_PROMPT
    assert "$1,200" not in sentencing_seed.BASELINE_PROMPT
    assert "$" not in sentencing_seed.BASELINE_PROMPT


def test_create_builtin_name_nationality_experiment_seeds_four_variants(db_path):
    with db.get_connection(db_path) as conn:
        exp_id = sentencing_seed.create_builtin_name_nationality_experiment(conn)
        experiment = db.get_experiment(conn, exp_id)
        variants = {pv["variant_name"]: pv for pv in db.list_prompt_versions(conn, exp_id)}

    assert experiment["evaluation_domain"] == "sentencing_bias"
    assert set(variants) == {"baseline", "name_afghan", "name_chinese", "name_nigerian"}


def test_name_nationality_seed_is_idempotent(db_path):
    with db.get_connection(db_path) as conn:
        first_id = sentencing_seed.create_builtin_name_nationality_experiment(conn)
        second_id = sentencing_seed.create_builtin_name_nationality_experiment(conn)
    assert first_id == second_id


def test_name_nationality_experiment_is_a_distinct_experiment_from_the_dialect_one(db_path):
    with db.get_connection(db_path) as conn:
        dialect_id = sentencing_seed.create_builtin_sentencing_experiment(conn)
        name_id = sentencing_seed.create_builtin_name_nationality_experiment(conn)
    assert dialect_id != name_id


def test_name_nationality_variants_change_only_the_name_never_the_statement():
    """The core experimental-design guarantee for this experiment: every
    variant must differ from baseline in exactly the defendant's name and
    nowhere else - no dialect/phrasing confound mixed in this time."""
    baseline_words = sentencing_seed.ENGLISH_NAME_PROMPT.replace(sentencing_seed.NAME_ENGLISH, "<NAME>")
    for prompt, name in (
        (sentencing_seed.AFGHAN_NAME_PROMPT, sentencing_seed.NAME_AFGHAN),
        (sentencing_seed.CHINESE_NAME_PROMPT, sentencing_seed.NAME_CHINESE),
        (sentencing_seed.NIGERIAN_NAME_PROMPT, sentencing_seed.NAME_NIGERIAN),
    ):
        assert prompt.replace(name, "<NAME>") == baseline_words


def test_name_nationality_case_facts_are_byte_identical_across_all_four_variants():
    prompts = [
        sentencing_seed.ENGLISH_NAME_PROMPT, sentencing_seed.AFGHAN_NAME_PROMPT,
        sentencing_seed.CHINESE_NAME_PROMPT, sentencing_seed.NIGERIAN_NAME_PROMPT,
    ]
    case_bodies = [p.split("CASE SUMMARY:")[1].split("'s statement to the court")[0] for p in prompts]
    normalized = [
        body.replace(sentencing_seed.NAME_ENGLISH, "<NAME>").replace(sentencing_seed.NAME_AFGHAN, "<NAME>")
            .replace(sentencing_seed.NAME_CHINESE, "<NAME>").replace(sentencing_seed.NAME_NIGERIAN, "<NAME>")
        for body in case_bodies
    ]
    assert len(set(normalized)) == 1


def test_name_nationality_statements_are_byte_identical_across_all_four_variants():
    """Unlike the dialect experiment, this one must hold the STATEMENT
    constant too - only the name may differ, so a result can't be blamed on
    phrasing/formality instead of the name."""
    for prompt in (
        sentencing_seed.ENGLISH_NAME_PROMPT, sentencing_seed.AFGHAN_NAME_PROMPT,
        sentencing_seed.CHINESE_NAME_PROMPT, sentencing_seed.NIGERIAN_NAME_PROMPT,
    ):
        statement_part = prompt.split("statement to the court, read as submitted:")[1]
        assert sentencing_seed.STANDARD_STATEMENT in statement_part


def test_add_formal_control_variant_is_idempotent(db_path):
    with db.get_connection(db_path) as conn:
        exp_id = sentencing_seed.create_builtin_sentencing_experiment(conn)
        first_id = sentencing_seed.add_formal_control_variant(conn, exp_id)
        second_id = sentencing_seed.add_formal_control_variant(conn, exp_id)
    assert first_id == second_id


def test_formal_control_variant_keeps_baseline_name_and_case_facts():
    assert "James Mitchell" in sentencing_seed.DIALECT_FORMAL_CONTROL_PROMPT
    baseline_words = sentencing_seed.BASELINE_PROMPT.replace(
        sentencing_seed.STANDARD_STATEMENT, "<STATEMENT>")
    control_words = sentencing_seed.DIALECT_FORMAL_CONTROL_PROMPT.replace(
        sentencing_seed.FORMAL_CONTROL_STATEMENT, "<STATEMENT>")
    assert baseline_words == control_words


def test_formal_control_statement_has_no_contractions_like_nigerian_does():
    """The whole point of this control: match dialect_nigerian's formal,
    no-contractions register without using any Nigerian-specific
    grammatical marker - so it must share that one property."""
    for contraction in ["don't", "I've", "I'm", "didn't", "wasn't"]:
        assert contraction not in sentencing_seed.FORMAL_CONTROL_STATEMENT
        assert contraction not in sentencing_seed.NIGERIAN_STATEMENT


def test_formal_control_statement_avoids_the_nigerian_specific_marker():
    """Must NOT reuse the progressive-stative construction ('I was owing')
    that is the genuine Nigerian-English feature being isolated out."""
    assert "owing" not in sentencing_seed.FORMAL_CONTROL_STATEMENT


def test_no_real_infamous_or_celebrity_names_used():
    """Guards the specific incident this experiment's docstring describes:
    an earlier draft used a real terrorist's surname and a real celebrity's
    full name - both must never appear anywhere in the seeded prompts."""
    all_text = " ".join([
        sentencing_seed.ENGLISH_NAME_PROMPT, sentencing_seed.AFGHAN_NAME_PROMPT,
        sentencing_seed.CHINESE_NAME_PROMPT, sentencing_seed.NIGERIAN_NAME_PROMPT,
    ])
    assert "bin Laden" not in all_text
    assert "Chris Brown" not in all_text
