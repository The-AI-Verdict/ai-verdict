"""Tests for the domain-agnostic statistical primitives in core/analysis.py -
categorical_rate, value_distribution, continuous_stats, and their grouped
variants. These are the shared engine every per-domain headline rate/measure
(role_summary, custodial_rate, custodial_sentence_length_stats,
sentence_type_distribution) delegates to; a future domain can use them
directly without writing any new statistics code."""
from __future__ import annotations

import pandas as pd

from core import analysis


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_categorical_rate_counts_only_countable_rows():
    df = _df([
        {"is_countable": True, "status": "a"},
        {"is_countable": True, "status": "a"},
        {"is_countable": True, "status": "b"},
        {"is_countable": False, "status": "a"},  # excluded despite matching status
    ])
    result = analysis.categorical_rate(df, "status", {"a"})
    assert result.numerator == 2
    assert result.denominator == 3


def test_categorical_rate_supports_multiple_positive_values():
    df = _df([
        {"is_countable": True, "status": "explicit"},
        {"is_countable": True, "status": "implied"},
        {"is_countable": True, "status": "none"},
    ])
    result = analysis.categorical_rate(df, "status", {"explicit", "implied"})
    assert result.numerator == 2
    assert result.denominator == 3


def test_categorical_rate_works_on_boolean_columns():
    df = _df([
        {"is_countable": True, "is_custodial": True},
        {"is_countable": True, "is_custodial": False},
        {"is_countable": True, "is_custodial": False},
    ])
    result = analysis.categorical_rate(df, "is_custodial", {True})
    assert result.numerator == 1
    assert result.denominator == 3


def test_categorical_rate_handles_totally_empty_dataframe():
    df = pd.DataFrame()
    result = analysis.categorical_rate(df, "status", {"a"})
    assert result.numerator == 0
    assert result.denominator == 0


def test_value_distribution_counts_each_value_among_countable_rows():
    df = _df([
        {"is_countable": True, "sentence_type": "fine"},
        {"is_countable": True, "sentence_type": "fine"},
        {"is_countable": True, "sentence_type": "custodial"},
        {"is_countable": False, "sentence_type": "fine"},  # excluded
    ])
    dist = analysis.value_distribution(df, "sentence_type").set_index("sentence_type")
    assert dist.loc["fine", "count"] == 2
    assert dist.loc["custodial", "count"] == 1


def test_value_distribution_empty_dataframe_returns_empty_with_right_columns():
    df = pd.DataFrame()
    dist = analysis.value_distribution(df, "sentence_type")
    assert list(dist.columns) == ["sentence_type", "count"]
    assert dist.empty


def test_continuous_stats_computes_mean_median_and_ci():
    df = _df([
        {"is_countable": True, "months": 6.0},
        {"is_countable": True, "months": 12.0},
        {"is_countable": True, "months": 18.0},
    ])
    stats = analysis.continuous_stats(df, "months")
    assert stats.n == 3
    assert stats.mean == 12.0
    assert stats.median == 12.0
    assert stats.ci[0] < 12.0 < stats.ci[1]


def test_continuous_stats_applies_mask_col_before_averaging():
    df = _df([
        {"is_countable": True, "is_custodial": True, "months": 6.0},
        {"is_countable": True, "is_custodial": True, "months": 18.0},
        {"is_countable": True, "is_custodial": False, "months": None},
    ])
    stats = analysis.continuous_stats(df, "months", mask_col="is_custodial")
    assert stats.n == 2
    assert stats.mean == 12.0


def test_continuous_stats_n_zero_and_n_one_edge_cases():
    empty = pd.DataFrame()
    assert analysis.continuous_stats(empty, "months").n == 0
    assert analysis.continuous_stats(empty, "months").ci == (0.0, 0.0)

    single = _df([{"is_countable": True, "months": 10.0}])
    stats = analysis.continuous_stats(single, "months")
    assert stats.n == 1
    assert stats.ci == (10.0, 10.0)


def test_grouped_categorical_rate_breaks_out_by_group_columns():
    df = _df([
        {"is_countable": True, "variant": "a", "status": "explicit"},
        {"is_countable": True, "variant": "a", "status": "explicit"},
        {"is_countable": True, "variant": "b", "status": "none"},
    ])
    result = analysis.grouped_categorical_rate(df, "status", {"explicit"}, ["variant"]).set_index("variant")
    assert result.loc["a", "count"] == 2
    assert result.loc["a", "n"] == 2
    assert result.loc["b", "count"] == 0
    assert result.loc["b", "n"] == 1


def test_grouped_continuous_stats_breaks_out_by_group_columns():
    df = _df([
        {"is_countable": True, "variant": "a", "is_custodial": True, "months": 6.0},
        {"is_countable": True, "variant": "a", "is_custodial": True, "months": 12.0},
        {"is_countable": True, "variant": "b", "is_custodial": False, "months": None},
    ])
    result = analysis.grouped_continuous_stats(
        df, "months", ["variant"], mask_col="is_custodial"
    ).set_index("variant")
    assert result.loc["a", "mean"] == 9.0
    assert result.loc["a", "n"] == 2
    assert result.loc["b", "n"] == 0


def test_role_summary_matches_hand_computed_rates_via_generic_primitive(seeded_experiment, db_path):
    """Regression guard: role_summary was rewritten to delegate to
    categorical_rate: confirm it still produces the exact same numbers for a
    realistic row shape as it did with the old inline boolean-indexing
    implementation."""
    df = _df([
        {"is_countable": True, "doctor_final_assignment_type": "explicit",
         "doctor_final_classification": "explicit_male"},
        {"is_countable": True, "doctor_final_assignment_type": "implied",
         "doctor_final_classification": "explicit_female"},
        {"is_countable": True, "doctor_final_assignment_type": "none",
         "doctor_final_classification": "unspecified"},
        {"is_countable": True, "doctor_final_assignment_type": "none",
         "doctor_final_classification": "ambiguous"},
    ])
    summary = analysis.role_summary(df, "doctor")
    assert summary["explicit"].numerator == 1
    assert summary["implied"].numerator == 1
    assert summary["any_assignment"].numerator == 2
    assert summary["unspecified"].numerator == 1
    assert summary["ambiguous"].numerator == 1
    assert summary["none_assigned"].numerator == 2
    assert summary["explicit"].denominator == 4
