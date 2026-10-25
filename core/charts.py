"""Matplotlib chart builders for the Results Dashboard.

Colors follow a fixed-order categorical palette (one hue per model, assigned
by position, never reassigned when a filter changes the series set) plus a
dedicated status palette for success/failed/pending, per the project's
dataviz conventions. Values are always direct-labelled on the chart since
there are only ever a handful of bars (models / prompt variants).
"""
from __future__ import annotations

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
    return fig


def proportion_by_group_chart(df: pd.DataFrame, group_col: str, metric_prefix: str, title: str):
    """df must have {metric_prefix}_pct, {metric_prefix}_count, {metric_prefix}_n columns
    plus a Wilson CI implicitly already computed into {metric_prefix}_ci (string) -
    we re-derive numeric lo/hi for the error bars from count/n directly to avoid
    string parsing."""
    from core.analysis import wilson_ci

    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(df)), 3.2))
    groups = df[group_col].astype(str).tolist()
    pcts = df[f"{metric_prefix}_pct"].tolist()
    counts = df[f"{metric_prefix}_count"].tolist()
    ns = df[f"{metric_prefix}_n"].tolist()

    lowers, uppers = [], []
    for pct, c, n in zip(pcts, counts, ns):
        lo, hi = wilson_ci(int(c), int(n))
        lowers.append(max(0.0, pct - lo * 100))
        uppers.append(max(0.0, hi * 100 - pct))

    colors = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(groups))]
    bars = ax.bar(groups, pcts, color=colors, width=0.55, yerr=[lowers, uppers], capsize=4,
                   error_kw={"ecolor": SECONDARY_INK, "elinewidth": 1})
    for bar, pct, n, upper in zip(bars, pcts, ns, uppers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + upper + 1,
                f"{pct:.0f}%\n(n={n})", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
