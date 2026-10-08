"""Matplotlib figures shared by the web app and compare_chart.py."""
import textwrap

import matplotlib.pyplot as plt
import numpy as np

INK = "#13241F"
GREEN = "#0B6B4F"
AMBER = "#C98A1B"
RED = "#B3261E"
GRID = "#D5DEDA"
MUTED = "#5E716B"


def _style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def comparison_figure(df, facecolor="none"):
    """df columns: Model, RMSE, MAE, R2. Error metrics left (lower is better), R2 right."""
    labels = [textwrap.fill(m, 12) for m in df["Model"]]
    x = np.arange(len(df))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor(facecolor)

    w = 0.36
    b1 = ax1.bar(x - w / 2, df["RMSE"], w, color=GREEN, label="RMSE")
    b2 = ax1.bar(x + w / 2, df["MAE"], w, color=AMBER, label="MAE")
    for bars in (b1, b2):
        ax1.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, color=INK)
    ax1.set_xticks(x, labels, fontsize=9, color=INK)
    ax1.set_title("Prediction error (lower is better)", loc="left", fontsize=11, color=INK)
    ax1.legend(frameon=False, fontsize=9, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    _style(ax1)

    best = int(np.argmax(df["R2"].to_numpy()))
    colors = [GREEN if i == best else "#9DB5AD" for i in range(len(df))]
    bars = ax2.bar(x, df["R2"], 0.55, color=colors)
    ax2.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, color=INK)
    ax2.set_xticks(x, labels, fontsize=9, color=INK)
    ax2.set_ylim(0, max(0.5, float(df["R2"].max()) * 1.25))
    ax2.set_title("R² score (higher is better)", loc="left", fontsize=11, color=INK)
    _style(ax2)

    fig.tight_layout()
    return fig


def drivers_figure(df, facecolor="none"):
    """df columns: Factor, Impact (signed). Positive pushes towards a pass."""
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, max(2.2, 0.42 * len(df) + 0.8)))
    fig.patch.set_facecolor(facecolor)
    colors = [GREEN if v >= 0 else RED for v in df["Impact"]]
    ax.barh(df["Factor"], df["Impact"], color=colors, height=0.6)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("Pushes towards fail   |   Pushes towards pass", fontsize=9, color=MUTED)
    ax.tick_params(axis="y", labelsize=9, labelcolor=INK, length=0)
    ax.tick_params(axis="x", labelsize=8, labelcolor=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
