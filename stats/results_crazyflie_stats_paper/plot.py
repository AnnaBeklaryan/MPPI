#!/usr/bin/env python3
"""
SVG plots for comparing MPPI, RAMPPI, DRMPPI, and DRAMPPI.

Generated figures:
  - combined_reference_boxplots.svg
  - combined_reference_violins.svg
  - combined_reference_histograms.svg

Usage:
  python plot.py \
      --run-metrics run_metrics.csv \
      --summary summary_table_numeric.csv \
      --outdir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import warnings

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Color-blind-friendly, highly separated colors
COLORS: Dict[str, str] = {
    "MPPI": "#0072B2",    # blue
    "RAMPPI": "#E69F00",  # orange
    "DRMPPI": "#009E73",  # green
    "DRAMPPI": "#D55E00", # vermillion/red-orange
}

ORDER: List[str] = ["MPPI", "RAMPPI", "DRMPPI", "DRAMPPI"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create publication-ready controller comparison plots.")
    parser.add_argument("--run-metrics", type=Path, default=Path("run_metrics.csv"), help="Path to run_metrics.csv")
    parser.add_argument("--summary", type=Path, default=Path("summary_table_numeric.csv"), help="Path to summary_table_numeric.csv")
    parser.add_argument("--outdir", type=Path, default=Path("figures"), help="Output directory for figures")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument("--pdf", action="store_true", help="Also save figures as PDF")
    return parser.parse_args()


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })


def load_data(run_metrics_path: Path, summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_df = pd.read_csv(run_metrics_path)
    summary_df = pd.read_csv(summary_path)

    required_run = {
        "Algorithm", "Run", "RunMinDistance", "SafetyViolation", "Collision", "TotalQSigmaInvStageCost"
    }
    required_summary = {
        "Algorithm", "Collision_Prob", "TotalQSigmaInvStageCost_Mean", "TotalQSigmaInvStageCost_Std",
        "RunMinDistance_Mean"
    }

    missing_run = required_run - set(run_df.columns)
    missing_summary = required_summary - set(summary_df.columns)

    if missing_run:
        raise ValueError(f"Missing columns in run_metrics.csv: {sorted(missing_run)}")
    if missing_summary:
        raise ValueError(f"Missing columns in summary_table_numeric.csv: {sorted(missing_summary)}")

    run_df = run_df.copy()
    summary_df = summary_df.copy()

    run_df["Algorithm"] = pd.Categorical(run_df["Algorithm"], categories=ORDER, ordered=True)
    summary_df["Algorithm"] = pd.Categorical(summary_df["Algorithm"], categories=ORDER, ordered=True)

    run_df = run_df.sort_values(["Algorithm", "Run"])
    summary_df = summary_df.sort_values("Algorithm")

    return run_df, summary_df


def save_current_figure(outdir: Path, name: str, dpi: int, save_pdf: bool) -> None:
    png_path = outdir / f"{name}.png"
    plt.savefig(png_path, dpi=dpi)
    if save_pdf:
        pdf_path = outdir / f"{name}.pdf"
        plt.savefig(pdf_path)
    plt.close()


def _paper_box_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, axis="y", color="#e5e5e5", alpha=0.9, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")


def _silverman_bandwidth(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    std = float(np.std(x, ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return np.nan
    return 1.06 * std * (x.size ** (-1.0 / 5.0))


def _gaussian_kde_pdf(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.full_like(grid, np.nan, dtype=float)
    h = _silverman_bandwidth(x)
    if not np.isfinite(h) or h <= 0.0:
        return np.full_like(grid, np.nan, dtype=float)
    z = (grid[:, None] - x[None, :]) / h
    kernel = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return np.mean(kernel, axis=1) / h


def plot_reference_style_boxplot(
    run_df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    outdir: Path,
    name: str,
    dpi: int,
    save_pdf: bool,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    _paper_box_axes(ax)

    data = []
    labels = []
    colors = []
    for algo in ORDER:
        vals = run_df.loc[run_df["Algorithm"] == algo, column].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if log_y:
            vals = vals[vals > 0.0]
        if vals.size == 0:
            continue
        data.append(vals)
        labels.append(algo)
        colors.append(COLORS[algo])

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="#444444", linewidth=1.2),
        capprops=dict(color="#444444", linewidth=1.2),
        boxprops=dict(edgecolor="#666666", linewidth=1.3),
        flierprops=dict(
            marker="o",
            markersize=2.6,
            alpha=0.55,
            markerfacecolor="#555555",
            markeredgecolor="#555555",
        ),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)

    y_min = np.inf
    y_max = -np.inf
    for idx, vals in enumerate(data, start=1):
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        ax.scatter(idx, mean, marker="D", s=26, color="black", zorder=4)
        y_min = min(y_min, float(np.min(vals)))
        y_max = max(y_max, float(np.max(vals)))
        if log_y:
            y_text = min(mean * 1.35, y_max * 1.08)
        else:
            y_text = float(np.max(vals)) + 0.03 * max(y_max - y_min, 1e-9)
        ax.text(
            idx,
            y_text,
            f"{mean:.4f}\n±{std:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    save_current_figure(outdir, name, dpi, save_pdf)


def _draw_reference_boxplot_on_axis(
    ax: plt.Axes,
    run_df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    labels: List[str],
    log_y: bool = False,
) -> None:
    _paper_box_axes(ax)

    data = []
    colors = []
    used_labels = []
    for algo in labels:
        vals = run_df.loc[run_df["Algorithm"] == algo, column].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if log_y:
            vals = vals[vals > 0.0]
        if vals.size == 0:
            continue
        data.append(vals)
        colors.append(COLORS[algo])
        used_labels.append(algo)

    bp = ax.boxplot(
        data,
        tick_labels=used_labels,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="#444444", linewidth=1.2),
        capprops=dict(color="#444444", linewidth=1.2),
        boxprops=dict(edgecolor="#666666", linewidth=1.3),
        flierprops=dict(
            marker="o",
            markersize=2.6,
            alpha=0.55,
            markerfacecolor="#555555",
            markeredgecolor="#555555",
        ),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)

    y_min = np.inf
    y_max = -np.inf
    stats = []
    for idx, vals in enumerate(data, start=1):
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        stats.append((idx, vals, mean, std))
        y_min = min(y_min, float(np.min(vals)))
        y_max = max(y_max, float(np.max(vals)))

    if log_y:
        bottom = max(y_min * 0.90, 1e-6)
        top = y_max * 2.35
        ax.set_ylim(bottom, top)
    else:
        span = max(y_max - y_min, 1e-9)
        bottom = y_min - 0.08 * span
        top = y_max + 0.22 * span
        ax.set_ylim(bottom, top)

    for idx, vals, mean, std in stats:
        ax.scatter(idx, mean, marker="D", s=26, color="black", zorder=4)
        if log_y:
            y_text = min(float(np.max(vals)) * 1.10, ax.get_ylim()[1] / 1.35)
        else:
            span = max(y_max - y_min, 1e-9)
            y_text = float(np.max(vals)) + 0.04 * span
        ax.text(
            idx,
            y_text,
            f"{mean:.4f}\n±{std:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )

    if log_y:
        ax.set_yscale("log")
    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)


def plot_combined_reference_boxplots(run_df: pd.DataFrame, outdir: Path) -> None:
    labels = ["MPPI", "RAMPPI", "DRMPPI", "DRAMPPI"]
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.2))

    _draw_reference_boxplot_on_axis(
        ax=axes[0],
        run_df=run_df,
        column="RunMinDistance",
        title="Run-min distance",
        ylabel="Distance",
        labels=labels,
        log_y=False,
    )
    _draw_reference_boxplot_on_axis(
        ax=axes[1],
        run_df=run_df,
        column="TotalQSigmaInvStageCost",
        title="Total cost",
        ylabel="Cost",
        labels=labels,
        log_y=False,
    )
    _draw_reference_boxplot_on_axis(
        ax=axes[2],
        run_df=run_df,
        column="RunMaxComputeTimeMs",
        title="Run-max compute time",
        ylabel="Time (ms; log scale)",
        labels=labels,
        log_y=True,
    )

    fig.subplots_adjust(left=0.04, right=0.995, top=0.86, bottom=0.13, wspace=0.24)
    out_path = outdir / "combined_reference_boxplots.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _draw_reference_violin_on_axis(
    ax: plt.Axes,
    run_df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    labels: List[str],
    log_y: bool = False,
) -> None:
    _paper_box_axes(ax)

    data = []
    positions = []
    colors = []
    used_labels = []
    for idx, algo in enumerate(labels, start=1):
        vals = run_df.loc[run_df["Algorithm"] == algo, column].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if log_y:
            vals = vals[vals > 0.0]
        if vals.size == 0:
            continue
        data.append(vals)
        positions.append(idx)
        colors.append(COLORS[algo])
        used_labels.append(algo)

    vp = ax.violinplot(
        data,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#555555")
        body.set_alpha(0.28)
        body.set_linewidth(1.1)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in vp:
            vp[key].set_color("#333333")
            vp[key].set_linewidth(1.1)

    y_min = np.inf
    y_max = -np.inf
    stats = []
    for idx, vals in zip(positions, data):
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        stats.append((idx, vals, mean, std))
        y_min = min(y_min, float(np.min(vals)))
        y_max = max(y_max, float(np.max(vals)))

    if log_y:
        bottom = max(y_min * 0.90, 1e-6)
        top = y_max * 2.35
        ax.set_yscale("log")
        ax.set_ylim(bottom, top)
    else:
        span = max(y_max - y_min, 1e-9)
        bottom = y_min - 0.08 * span
        top = y_max + 0.22 * span
        ax.set_ylim(bottom, top)

    for idx, vals, mean, std in stats:
        ax.scatter(idx, mean, marker="D", s=26, color="black", zorder=4)
        if log_y:
            y_text = min(float(np.max(vals)) * 1.10, ax.get_ylim()[1] / 1.35)
        else:
            span = max(y_max - y_min, 1e-9)
            y_text = float(np.max(vals)) + 0.04 * span
        ax.text(
            idx,
            y_text,
            f"{mean:.4f}\n±{std:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(used_labels)
    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)


def plot_combined_reference_violins(run_df: pd.DataFrame, outdir: Path) -> None:
    labels = ["MPPI", "RAMPPI", "DRMPPI", "DRAMPPI"]
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.2))

    _draw_reference_violin_on_axis(
        ax=axes[0],
        run_df=run_df,
        column="RunMinDistance",
        title="Run-min distance",
        ylabel="Distance",
        labels=labels,
        log_y=False,
    )
    _draw_reference_violin_on_axis(
        ax=axes[1],
        run_df=run_df,
        column="TotalQSigmaInvStageCost",
        title="Total cost",
        ylabel="Cost",
        labels=labels,
        log_y=False,
    )
    _draw_reference_violin_on_axis(
        ax=axes[2],
        run_df=run_df,
        column="RunMaxComputeTimeMs",
        title="Run-max compute time",
        ylabel="Time (ms; log scale)",
        labels=labels,
        log_y=True,
    )

    fig.subplots_adjust(left=0.04, right=0.995, top=0.86, bottom=0.13, wspace=0.24)
    out_path = outdir / "combined_reference_violins.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _draw_reference_histogram_on_axis(
    ax: plt.Axes,
    run_df: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    labels: List[str],
    log_x: bool = False,
) -> None:
    _paper_box_axes(ax)

    all_vals = []
    grouped = []
    for algo in labels:
        vals = run_df.loc[run_df["Algorithm"] == algo, column].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if log_x:
            vals = vals[vals > 0.0]
        grouped.append((algo, vals))
        if vals.size > 0:
            all_vals.append(vals)

    if not all_vals:
        ax.set_title(title, pad=10)
        ax.set_xlabel(xlabel)
        return

    stacked = np.concatenate(all_vals)
    if log_x:
        log_stacked = np.log10(stacked)
        xmin = float(np.min(log_stacked))
        xmax = float(np.max(log_stacked))
        if xmax <= xmin:
            xmin, xmax = xmin - 0.15, xmax + 0.15
        else:
            pad = max(0.12 * (xmax - xmin), 0.08)
            xmin, xmax = xmin - pad, xmax + pad
        bins = np.logspace(xmin, xmax, 16)
        kde_grid_log = np.linspace(xmin, xmax, 400)
        kde_grid = 10.0 ** kde_grid_log
    else:
        xmin = float(np.min(stacked))
        xmax = float(np.max(stacked))
        if xmax <= xmin:
            pad = 1e-6 if xmin == 0.0 else 0.08 * abs(xmin)
            xmin, xmax = xmin - pad, xmax + pad
        else:
            widths = []
            for vals in all_vals:
                h = _silverman_bandwidth(vals)
                if np.isfinite(h) and h > 0.0:
                    widths.append(h)
            bw_pad = max(widths) * 2.5 if widths else 0.0
            range_pad = 0.08 * (xmax - xmin)
            pad = max(range_pad, bw_pad)
            xmin, xmax = xmin - pad, xmax + pad
        bins = np.linspace(xmin, xmax, 16)
        kde_grid = np.linspace(xmin, xmax, 400)

    for algo, vals in grouped:
        if vals.size == 0:
            continue
        sns.histplot(
            vals,
            bins=bins,
            stat="count",
            kde=False,
            element="bars",
            fill=True,
            alpha=0.20,
            linewidth=1.2,
            edgecolor="#555555",
            color=COLORS[algo],
            label=algo,
            ax=ax,
        )
        if log_x:
            log_vals = np.log10(vals)
            pdf_log = _gaussian_kde_pdf(log_vals, kde_grid_log)
            if np.any(np.isfinite(pdf_log)):
                bin_width_log = float(np.diff(kde_grid_log[:2])[0])
                ax.plot(
                    kde_grid,
                    pdf_log * len(vals) * bin_width_log,
                    color=COLORS[algo],
                    linewidth=2.0,
                )
        else:
            pdf = _gaussian_kde_pdf(vals, kde_grid)
            if np.any(np.isfinite(pdf)):
                bin_width = float(np.diff(bins[:2])[0])
                ax.plot(
                    kde_grid,
                    pdf * len(vals) * bin_width,
                    color=COLORS[algo],
                    linewidth=2.0,
                )

    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(bins[0], bins[-1])
    else:
        ax.set_xlim(bins[0], bins[-1])
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(frameon=True, fontsize=9)


def plot_combined_reference_histograms(run_df: pd.DataFrame, outdir: Path) -> None:
    labels = ["MPPI", "RAMPPI", "DRMPPI", "DRAMPPI"]
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.2))

    _draw_reference_histogram_on_axis(
        ax=axes[0],
        run_df=run_df,
        column="RunMinDistance",
        title="Run-min distance",
        xlabel="Distance",
        labels=labels,
        log_x=False,
    )
    _draw_reference_histogram_on_axis(
        ax=axes[1],
        run_df=run_df,
        column="TotalQSigmaInvStageCost",
        title="Total cost",
        xlabel="Cost",
        labels=labels,
        log_x=False,
    )
    _draw_reference_histogram_on_axis(
        ax=axes[2],
        run_df=run_df,
        column="RunMaxComputeTimeMs",
        title="Run-max compute time",
        xlabel="Time (ms; log scale)",
        labels=labels,
        log_x=True,
    )

    fig.subplots_adjust(left=0.04, right=0.995, top=0.86, bottom=0.13, wspace=0.24)
    out_path = outdir / "combined_reference_histograms.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


# ------------------------------
# Figure 2: boxplot of minimum distance
# ------------------------------
def plot_boxplot_min_distance(run_df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    plt.figure()
    ax = sns.boxplot(
        data=run_df,
        x="Algorithm",
        y="RunMinDistance",
        order=ORDER,
        palette=COLORS,
        width=0.55,
        linewidth=1.3,
        showfliers=False,
    )
    sns.stripplot(
        data=run_df,
        x="Algorithm",
        y="RunMinDistance",
        order=ORDER,
        color="black",
        alpha=0.50,
        size=3.5,
        jitter=0.15,
    )
    ax.set_title("Figure 2: Minimum Distance Distribution")
    ax.set_xlabel("Controller")
    ax.set_ylabel("Minimum Distance to Obstacle")
    ax.grid(True, axis="y", alpha=0.3)
    save_current_figure(outdir, "figure2_boxplot_min_distance", dpi, save_pdf)


# ------------------------------
# Figure 3: collision probability bar chart
# ------------------------------
def plot_collision_probability(summary_df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    plt.figure()
    x = np.arange(len(summary_df))
    vals = summary_df["Collision_Prob"].to_numpy()
    labels = summary_df["Algorithm"].astype(str).tolist()
    colors = [COLORS[a] for a in labels]

    bars = plt.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8)
    plt.xticks(x, labels)
    plt.ylabel("Collision Probability")
    plt.xlabel("Controller")
    plt.title("Figure 3: Collision Probability")
    plt.ylim(0, max(0.05, vals.max() * 1.25 + 1e-9))
    plt.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save_current_figure(outdir, "figure3_collision_probability_bar", dpi, save_pdf)


# ------------------------------
# Figure 4: cost comparison bar chart
# ------------------------------
def plot_cost_comparison(summary_df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    plt.figure()
    x = np.arange(len(summary_df))
    means = summary_df["TotalQSigmaInvStageCost_Mean"].to_numpy()
    stds = summary_df["TotalQSigmaInvStageCost_Std"].to_numpy()
    labels = summary_df["Algorithm"].astype(str).tolist()
    colors = [COLORS[a] for a in labels]

    bars = plt.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    plt.xticks(x, labels)
    plt.ylabel(r"Total Cost")
    plt.xlabel("Controller")
    plt.title("Figure 4: Cost Comparison")
    plt.grid(True, axis="y", alpha=0.3)

    ymin = max(0.0, means.min() - max(stds.max() * 4, 0.02 * means.mean()))
    ymax = means.max() + max(stds.max() * 6, 0.02 * means.mean())
    plt.ylim(ymin, ymax)

    for bar, mean, std in zip(bars, means, stds):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.002 * means.mean(),
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save_current_figure(outdir, "figure4_cost_comparison_bar", dpi, save_pdf)


# ------------------------------
# Figure 5: cost vs safety scatter plot
# ------------------------------
def plot_cost_vs_safety(run_df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    plt.figure()
    ax = plt.gca()

    for algo in ORDER:
        subset = run_df[run_df["Algorithm"] == algo]
        if subset.empty:
            continue
        ax.scatter(
            subset["RunMinDistance"],
            subset["TotalQSigmaInvStageCost"],
            s=55,
            alpha=0.85,
            label=algo,
            color=COLORS[algo],
            edgecolor="black",
            linewidth=0.4,
        )

        # Mark the mean point for each controller
        ax.scatter(
            subset["RunMinDistance"].mean(),
            subset["TotalQSigmaInvStageCost"].mean(),
            s=140,
            marker="X",
            color=COLORS[algo],
            edgecolor="black",
            linewidth=0.9,
        )

    ax.set_title("Figure 5: Cost vs Safety Trade-off")
    ax.set_xlabel("Minimum Distance to Obstacle")
    ax.set_ylabel("Total Cost")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Controller", frameon=True)
    save_current_figure(outdir, "figure5_cost_vs_safety_scatter", dpi, save_pdf)


# ------------------------------
# Figure 6: histogram + KDE of minimum distance
# ------------------------------
def plot_hist_kde_min_distance(run_df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    plt.figure(figsize=(9, 5.5))
    ax = plt.gca()
    ax.set_facecolor("#ececec")
    ax.grid(True, color="#c9c9c9", linewidth=1.0, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#333333")

    values = run_df["RunMinDistance"].dropna().to_numpy(dtype=float)
    if values.size > 0:
        vmin, vmax = float(np.min(values)), float(np.max(values))
        if vmax <= vmin:
            pad = 1e-6 if vmin == 0.0 else 0.05 * abs(vmin)
            vmin, vmax = vmin - pad, vmax + pad
        else:
            pad = 0.05 * (vmax - vmin)
            vmin, vmax = vmin - pad, vmax + pad
        bins = np.linspace(vmin, vmax, 22)
    else:
        bins = 12

    for algo in ORDER:
        subset = run_df[run_df["Algorithm"] == algo]["RunMinDistance"].dropna()
        if subset.empty:
            continue
        sns.histplot(
            subset,
            bins=bins,
            stat="count",
            kde=True,
            element="bars",
            fill=True,
            alpha=0.22,
            linewidth=1.6,
            line_kws={"linewidth": 2.3},
            edgecolor="#555555",
            color=COLORS[algo],
            label=algo,
            ax=ax,
        )

    ax.set_title("Figure 6: Distribution of Minimum Separation Distance")
    ax.set_xlabel("Minimum separation distance (m)")
    ax.set_ylabel("Count")
    ax.legend(title="Controller", frameon=True)
    save_current_figure(outdir, "figure6_hist_kde_min_distance", dpi, save_pdf)


# ------------------------------
# Figure 7: violin plot of minimum distance
# ------------------------------
def plot_violin_min_distance(run_df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    plt.figure()
    ax = sns.violinplot(
        data=run_df,
        x="Algorithm",
        y="RunMinDistance",
        order=ORDER,
        palette=COLORS,
        inner="quartile",
        linewidth=1.2,
        cut=0,
    )
    ax.set_title("Figure 7: Violin Plot of Minimum Distance")
    ax.set_xlabel("Controller")
    ax.set_ylabel("Minimum Distance to Obstacle")
    ax.grid(True, axis="y", alpha=0.3)
    save_current_figure(outdir, "figure7_violin_min_distance", dpi, save_pdf)


# ------------------------------
# Figure 8: compact box plot for paper layout
# ------------------------------
def plot_compact_box_min_distance(run_df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    plt.figure(figsize=(7.2, 4.8))
    ax = sns.boxplot(
        data=run_df,
        x="Algorithm",
        y="RunMinDistance",
        order=ORDER,
        palette=COLORS,
        width=0.48,
        linewidth=1.2,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.65},
    )
    sns.swarmplot(
        data=run_df,
        x="Algorithm",
        y="RunMinDistance",
        order=ORDER,
        color="black",
        size=2.7,
        alpha=0.65,
    )
    ax.set_title("Figure 8: Box Plot with Individual Runs")
    ax.set_xlabel("Controller")
    ax.set_ylabel("Minimum Distance to Obstacle")
    ax.grid(True, axis="y", alpha=0.25)
    save_current_figure(outdir, "figure8_boxplot_with_runs", dpi, save_pdf)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_style()
    run_df, _summary_df = load_data(args.run_metrics, args.summary)
    plot_combined_reference_boxplots(run_df, args.outdir)
    plot_combined_reference_violins(run_df, args.outdir)
    plot_combined_reference_histograms(run_df, args.outdir)

    print(f"Saved figures to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()  
