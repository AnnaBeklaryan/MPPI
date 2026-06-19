#!/usr/bin/env python3
"""
Create Crazyflie collision-probability-vs-epsilon plots from the CSV files in
this results directory.

Examples:
  python3 results_dr_cf_diff_eps_stats/plot_dr_cf_epsilon_summary.py
  python3 results_dr_cf_diff_eps_stats/plot_dr_cf_epsilon_summary.py --x-min 0.02 --x-max 0.08
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, MultipleLocator
import numpy as np
import pandas as pd


DEFAULT_X_MAX = None


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot Crazyflie DR-MPPI collision probability against epsilon."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=here / "dr_cf_epsilon_summary.csv",
        help="Path to dr_cf_epsilon_summary.csv.",
    )
    parser.add_argument(
        "--run-csv",
        type=Path,
        default=here / "dr_cf_epsilon_run_metrics.csv",
        help="Path to dr_cf_epsilon_run_metrics.csv.",
    )
    parser.add_argument(
        "--moving-csv",
        type=Path,
        default=here / "dr_cf_moving_obstacle_collision_probability.csv",
        help="Path to dr_cf_moving_obstacle_collision_probability.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "dr_cf_epsilon_collision_probability.svg",
        help="Output plot path for the linear-x plot.",
    )
    parser.add_argument(
        "--log-out",
        type=Path,
        default=None,
        help="Optional output path for the log-x plot. Defaults to '<out stem>_logx<suffix>'.",
    )
    parser.add_argument(
        "--moving-out",
        type=Path,
        default=here / "dr_cf_moving_obstacle_collision_probability.svg",
        help="Output plot path for per-moving-obstacle collision probabilities.",
    )
    parser.add_argument("--title", type=str, default="", help="Optional plot title.")
    parser.add_argument("--x-min", type=float, default=0.0, help="Lower x-axis limit.")
    parser.add_argument(
        "--x-max",
        type=float,
        default=DEFAULT_X_MAX,
        help="Upper x-axis limit. Defaults to the maximum epsilon in the summary CSV.",
    )
    return parser.parse_args()


def load_summary(csv_path: Path) -> pd.DataFrame:
    summary_df = pd.read_csv(csv_path)
    required_columns = {"epsilon", "collision_prob"}
    missing_columns = required_columns - set(summary_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing_columns)}")
    return summary_df.sort_values("epsilon").reset_index(drop=True)


def load_run_metrics(csv_path: Path) -> pd.DataFrame:
    run_df = pd.read_csv(csv_path)
    required_columns = {"epsilon", "Run", "Seed", "Collision"}
    missing_columns = required_columns - set(run_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing_columns)}")
    return run_df.sort_values(["epsilon", "Run"]).reset_index(drop=True)


def load_moving_obstacle_summary(csv_path: Path) -> pd.DataFrame:
    moving_df = pd.read_csv(csv_path)
    required_columns = {"epsilon", "moving_obstacle_index", "collision_prob"}
    missing_columns = required_columns - set(moving_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing_columns)}")
    return moving_df.sort_values(["epsilon", "moving_obstacle_index"]).reset_index(drop=True)


def _apply_common_style(ax: plt.Axes) -> None:
    ax.set_xlabel("Wasserstein radius, $\\varepsilon$")
    ax.set_ylabel("Collision probability")
    ax.set_ylim(-0.03, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, which="major", color="#b0b0b0", alpha=0.35, linewidth=0.9)
    ax.grid(True, which="minor", color="#b0b0b0", alpha=0.35, linewidth=0.9)
    ax.axhline(0.05, color="#0072B2", linestyle="--", linewidth=1.5)


def _linear_x_step(x_span: float) -> float:
    if x_span <= 0.05:
        return 0.01
    if x_span <= 0.1:
        return 0.02
    if x_span <= 0.5:
        return 0.05
    return 0.1


def apply_linear_style(ax: plt.Axes, x_min: float, x_max: float) -> None:
    _apply_common_style(ax)
    x_step = _linear_x_step(x_max - x_min)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.xaxis.set_minor_locator(MultipleLocator(x_step / 2.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def epsilon_subset(summary_df: pd.DataFrame, x_min: float, x_max: float) -> pd.DataFrame:
    eps = summary_df["epsilon"].to_numpy(dtype=float)
    mask = np.isfinite(eps) & (eps >= x_min) & (eps <= x_max)
    return summary_df.loc[mask].copy()


def positive_log_subset(summary_df: pd.DataFrame, x_min: float, x_max: float) -> pd.DataFrame:
    eps = summary_df["epsilon"].to_numpy(dtype=float)
    mask = np.isfinite(eps) & (eps > 0.0) & (eps >= x_min) & (eps <= x_max)
    return summary_df.loc[mask].copy()


def apply_log_style(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    _apply_common_style(ax)
    eps = summary_df["epsilon"].to_numpy(dtype=float)
    ax.set_xscale("log")
    ax.set_xlim(float(np.min(eps)), float(np.max(eps)))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def plot_overall_collision(
    plot_df: pd.DataFrame,
    out_path: Path,
    x_min: float,
    x_max: float,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        plot_df["epsilon"].to_numpy(dtype=float),
        plot_df["collision_prob"].to_numpy(dtype=float),
        color="#0072B2",
        linewidth=2.0,
    )
    if title:
        ax.set_title(title)
    apply_linear_style(ax, x_min, x_max)
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"[OK] wrote: {out_path}")


def plot_log_collision(
    summary_df: pd.DataFrame,
    out_path: Path,
    x_min: float,
    x_max: float,
    title: str,
) -> None:
    log_df = positive_log_subset(summary_df, x_min, x_max)
    if log_df.empty:
        print(
            "[WARN] skipped log-x plot: there are no positive epsilon values "
            f"within {x_min:.3f} <= x <= {x_max:.3f}."
        )
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        log_df["epsilon"].to_numpy(dtype=float),
        log_df["collision_prob"].to_numpy(dtype=float),
        color="#0072B2",
        linewidth=2.0,
    )
    if title:
        ax.set_title(title)
    apply_log_style(ax, log_df)
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"[OK] wrote: {out_path}")
    if np.any(summary_df["epsilon"].to_numpy(dtype=float) <= 0.0):
        print("[note] omitted epsilon=0.0 from the log-x plot because logarithmic axes cannot display zero.")


def plot_moving_obstacle_collision(
    moving_df: pd.DataFrame,
    out_path: Path,
    x_min: float,
    x_max: float,
) -> None:
    plot_df = epsilon_subset(moving_df, x_min, x_max)
    if plot_df.empty:
        print(
            "[WARN] skipped moving-obstacle plot: no epsilon values found "
            f"within {x_min:.3f} <= x <= {x_max:.3f}."
        )
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for obs_idx, group in plot_df.groupby("moving_obstacle_index", sort=True):
        group = group.sort_values("epsilon")
        ax.plot(
            group["epsilon"].to_numpy(dtype=float),
            group["collision_prob"].to_numpy(dtype=float),
            linewidth=2.0,
            label=f"Moving obstacle {int(obs_idx)}",
        )
    _apply_common_style(ax)
    ax.set_xlim(x_min, x_max)
    x_step = _linear_x_step(x_max - x_min)
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.xaxis.set_minor_locator(MultipleLocator(x_step / 2.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend(frameon=True, fontsize=10)
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"[OK] wrote: {out_path}")


def main() -> None:
    args = parse_args()
    summary_df = load_summary(args.csv)
    run_df = load_run_metrics(args.run_csv)
    moving_df = load_moving_obstacle_summary(args.moving_csv)

    x_min = float(args.x_min)
    x_max = (
        float(args.x_max)
        if args.x_max is not None
        else float(np.nanmax(summary_df["epsilon"].to_numpy(dtype=float)))
    )
    if not (np.isfinite(x_min) and np.isfinite(x_max) and x_min < x_max):
        raise ValueError(f"Expected --x-min < --x-max with finite values, got {x_min} and {x_max}.")

    plot_df = epsilon_subset(summary_df, x_min, x_max)
    if plot_df.empty:
        raise ValueError(f"No epsilon values found in [{x_min}, {x_max}] from {args.csv}.")

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    print(f"[data] summary_rows={len(summary_df)} run_rows={len(run_df)} moving_rows={len(moving_df)}")
    plot_overall_collision(plot_df, args.out, x_min, x_max, args.title)

    log_out = args.log_out
    if log_out is None:
        log_out = args.out.with_name(f"{args.out.stem}_logx{args.out.suffix}")
    plot_log_collision(summary_df, log_out, x_min, x_max, args.title)
    plot_moving_obstacle_collision(moving_df, args.moving_out, x_min, x_max)


if __name__ == "__main__":
    main()
