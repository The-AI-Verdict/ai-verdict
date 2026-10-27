"""Matplotlib chart builders for the Results Dashboard.

Colors follow a fixed-order categorical palette (one hue per model, assigned
by position, never reassigned when a filter changes the series set) plus a
dedicated status palette for success/failed/pending, per the project's
dataviz conventions. Values are always direct-labelled on the chart, since
there are rarely more than a handful of bars per chart - but model ids are
long strings and an experiment can easily have 10+ of them, so
proportion_by_group_chart uses horizontal bars specifically to avoid x-axis
label collisions at that scale (see its docstring).

Every function here calls plt.close(fig) right before returning it. This is
NOT optional cleanup - pyplot's stateful plt.subplots() registers every
figure in a global registry (matplotlib._pylab_helpers.Gcf) that is never
garbage-collected until explicitly closed. In a long-running server process
like this one, where the Dashboard page regenerates several charts on every
single Streamlit rerun (every filter change, every navigation), skipping
plt.close() leaks one Figure's worth of Python objects per chart per rerun
for the life of the process - this is exactly what caused a real
"MemoryError: bad allocation" in production after enough reruns accumulated.
plt.close(fig) only unregisters the figure from pyplot's global state; the
returned Figure object itself is still fully valid for st.pyplot(fig) to
render, since Streamlit renders directly from the object, not from pyplot's
registry.
"""
from __future__ import annotations

import matplotlib
# Force the non-interactive Agg backend explicitly, rather than relying on
# matplotlib's auto-detection. Streamlit happens to force Agg itself before
# this module is ever imported in the live app, which is the only reason
# auto-detection hasn't bitten production - but running this module directly
# (a script, a test suite) without that prior Agg-forcing step can have
# matplotlib pick an interactive GUI backend instead (TkAgg on Windows if
# tkinter looks available), which then fails outright on any machine where
# Tk isn't properly installed. A chart-generation module for a web app
# should never depend on a GUI toolkit being present at all.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"success": "#0ca30c", "failed": "#d03b3b", "pending": "#fab219"}
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "font.family": "sans-serif",
})


def run_status_chart(success: int, failed: int, pending: int):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    labels = ["Success", "Failed", "Pending"]
    values = [success, failed, pending]
    colors = [STATUS["success"], STATUS["failed"], STATUS["pending"]]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(v),
                ha="center", va="bottom", color=INK, fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])
    ax.grid(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def proportion_by_group_chart(df: pd.DataFrame, group_col: str, metric_prefix: str, title: str):
    """Horizontal bars, one per category. Model ids and variant names are
    long strings and there can be many of them (10 models is common) - with
    vertical bars, horizontal tick labels collide into unreadable mush
    regardless of rotation; horizontal bars put each label on its own row
    instead, which scales to any label length or category count without
    collision.

    df must have {metric_prefix}_pct, {metric_prefix}_count, {metric_prefix}_n
    columns plus a Wilson CI implicitly already computed into
    {metric_prefix}_ci (string) - we re-derive numeric lo/hi for the error
    bars from count/n directly to avoid string parsing."""
    from core.analysis import wilson_ci

    if df.empty:
        return None
    groups = df[group_col].astype(str).tolist()
    pcts = df[f"{metric_prefix}_pct"].tolist()
    counts = df[f"{metric_prefix}_count"].tolist()
    ns = df[f"{metric_prefix}_n"].tolist()

    lowers, uppers = [], []
    for pct, c, n in zip(pcts, counts, ns):
        lo, hi = wilson_ci(int(c), int(n))
        lowers.append(max(0.0, pct - lo * 100))
        uppers.append(max(0.0, hi * 100 - pct))

    fig_height = max(2.0, 0.5 * len(groups) + 1.0)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    y_pos = range(len(groups))
    colors = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(groups))]
    bars = ax.barh(y_pos, pcts, color=colors, height=0.6, xerr=[lowers, uppers], capsize=3,
                    error_kw={"ecolor": SECONDARY_INK, "elinewidth": 1})
    for bar, pct, n, upper in zip(bars, pcts, ns, uppers):
        label_x = min(bar.get_width() + upper + 2, 92)
        ax.text(label_x, bar.get_y() + bar.get_height() / 2, f"{pct:.0f}% (n={n})",
                ha="left", va="center", color=INK, fontsize=9)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(groups, fontsize=9)
    ax.invert_yaxis()  # first row of df reads at the top, matching the table below it
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("%")
    ax.set_xlim(0, 100)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    plt.close(fig)
    return fig


def mean_value_by_group_chart(
    df: pd.DataFrame, group_col: str, mean_col: str, n_col: str,
    ci_low_col: str, ci_high_col: str, title: str, x_label: str,
):
    """Horizontal bars for a continuous mean (e.g. mean custodial sentence
    length in months) by group, with a normal-approximation CI as an error
    bar - the sentencing domain's equivalent of proportion_by_group_chart,
    for a measure that isn't a 0-100% proportion. Rows with no CI (n<2,
    ci_low/ci_high are None/NaN) are drawn without an error bar rather than
    being dropped, so a single-observation group still shows its bar."""
    plotted = df[df[mean_col].notna()]
    if plotted.empty:
        return None
    groups = plotted[group_col].astype(str).tolist()
    means = plotted[mean_col].tolist()
    ns = plotted[n_col].tolist()
    los = plotted[ci_low_col].tolist()
    his = plotted[ci_high_col].tolist()

    lowers, uppers = [], []
    for mean, lo, hi in zip(means, los, his):
        if pd.isna(lo) or pd.isna(hi):
            lowers.append(0.0)
            uppers.append(0.0)
        else:
            lowers.append(max(0.0, mean - lo))
            uppers.append(max(0.0, hi - mean))

    fig_height = max(2.0, 0.5 * len(groups) + 1.0)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    y_pos = range(len(groups))
    colors = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(groups))]
    bars = ax.barh(y_pos, means, color=colors, height=0.6, xerr=[lowers, uppers], capsize=3,
                    error_kw={"ecolor": SECONDARY_INK, "elinewidth": 1})
    max_val = max(m + u for m, u in zip(means, uppers)) if means else 1.0
    for bar, mean, n, upper in zip(bars, means, ns, uppers):
        label_x = bar.get_width() + upper + max_val * 0.02
        ax.text(label_x, bar.get_y() + bar.get_height() / 2, f"{mean:.1f} (n={n})",
                ha="left", va="center", color=INK, fontsize=9)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(groups, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(x_label)
    ax.set_xlim(0, max_val * 1.25 if max_val else 1.0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    plt.close(fig)
    return fig
