"""Regression tests for core/charts.py.

The main thing worth guarding here isn't the visual output (not meaningfully
testable without a human looking at it) but the real production incident
this module caused: every chart function leaked its Figure into matplotlib's
global pyplot registry, since plt.subplots() was never paired with
plt.close(). In a long-running Streamlit process that regenerates charts on
every rerun, this leaked one Figure's worth of objects per chart per rerun
for the life of the process, and eventually surfaced as a live
"MemoryError: bad allocation" on the Results Dashboard page.
"""
from __future__ import annotations

import io

import pandas as pd
from matplotlib import _pylab_helpers

from core import charts


def _open_figure_count() -> int:
    return len(_pylab_helpers.Gcf.figs)


def test_run_status_chart_does_not_leak_into_pyplot_global_state():
    before = _open_figure_count()
    for _ in range(20):
        charts.run_status_chart(10, 2, 3)
    assert _open_figure_count() == before


def test_proportion_by_group_chart_does_not_leak_into_pyplot_global_state():
    df = pd.DataFrame({
        "model_id": ["a", "b", "c"],
        "custodial_pct": [50.0, 20.0, 80.0],
        "custodial_count": [5, 2, 8],
        "custodial_n": [10, 10, 10],
    })
    before = _open_figure_count()
    for _ in range(20):
        charts.proportion_by_group_chart(df, "model_id", "custodial", "Test chart")
    assert _open_figure_count() == before


def test_mean_value_by_group_chart_does_not_leak_into_pyplot_global_state():
    df = pd.DataFrame({
        "model_id": ["a", "b"],
        "mean_custodial_months": [6.0, 12.0],
        "custodial_length_n": [5, 3],
        "custodial_months_ci_low": [4.0, 8.0],
        "custodial_months_ci_high": [8.0, 16.0],
    })
    before = _open_figure_count()
    for _ in range(20):
        charts.mean_value_by_group_chart(
            df, "model_id", "mean_custodial_months", "custodial_length_n",
            "custodial_months_ci_low", "custodial_months_ci_high", "Test chart", "Months",
        )
    assert _open_figure_count() == before


def test_closed_figure_is_still_fully_renderable():
    """plt.close(fig) only unregisters the figure from pyplot's global
    state - the returned Figure object must still render correctly via its
    own savefig, since that's exactly what st.pyplot(fig) does."""
    fig = charts.run_status_chart(10, 2, 3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    assert buf.tell() > 0


def test_empty_dataframe_returns_none_without_leaking():
    before = _open_figure_count()
    result = charts.proportion_by_group_chart(pd.DataFrame(), "model_id", "custodial", "Test chart")
    assert result is None
    assert _open_figure_count() == before
