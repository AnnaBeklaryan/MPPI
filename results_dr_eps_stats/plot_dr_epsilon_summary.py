#!/usr/bin/env python3
"""
Create a collision-probability-vs-epsilon plot from dr_epsilon_summary.csv.

Examples:
  python3 results_dr_eps_stats/plot_dr_epsilon_summary.py
  python3 results_dr_eps_stats/plot_dr_epsilon_summary.py \
      --csv results_dr_eps_stats/dr_epsilon_summary.csv \
      --out results_dr_eps_stats/dr_epsilon_collision_probability.svg
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


DEFAULT_X_MAX = 0.05


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot collision probability against epsilon from a DR-MPPI summary CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=here / "dr_epsilon_summary.csv",
        help="Path to dr_epsilon_summary.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "dr_epsilon_collision_probability.svg",
        help="Output plot path for the linear-x plot.",
    )
    parser.add_argument(
        "--log-out",
        type=Path,
        default=None,
        help="Optional output path for the log-x plot. Defaults to '<out stem>_logx<suffix>'.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional plot title. Leave empty to match the reference style.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=DEFAULT_X_MAX,
        help="Upper x-axis limit for the exported plots.",
    )
    return parser.parse_args()


def load_summary(csv_path: Path) -> pd.DataFrame:
    summary_df = pd.read_csv(csv_path)
    required_columns = {"epsilon", "collision_prob"}
    missing_columns = required_columns - set(summary_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {csv_path}: {sorted(missing_columns)}"
        )
    return summary_df.sort_values("epsilon").reset_index(drop=True)


def _apply_common_style(ax: plt.Axes) -> None:
    ax.set_xlabel("Wasserstein radius, $\\varepsilon$")
    ax.set_ylabel("Collision probability")
    ax.set_ylim(-0.02, 0.6)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, which="major", color="#b0b0b0", alpha=0.35, linewidth=0.9)
    ax.grid(True, which="minor", color="#b0b0b0", alpha=0.35, linewidth=0.9)


def _linear_x_step(x_max: float) -> float:
    if x_max <= 0.05:
        return 0.01
    if x_max <= 0.1:
        return 0.02
    if x_max <= 0.5:
        return 0.05
    return 0.1


def apply_linear_style(ax: plt.Axes, x_max: float) -> None:
    _apply_common_style(ax)
    x_step = _linear_x_step(x_max)
    ax.set_xlim(0.0, x_max)
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.xaxis.set_minor_locator(MultipleLocator(x_step / 2.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def positive_log_subset(summary_df: pd.DataFrame, x_max: float) -> pd.DataFrame:
    eps = summary_df["epsilon"].to_numpy(dtype=float)
    mask = np.isfinite(eps) & (eps > 0.0) & (eps <= x_max)
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


def main() -> None:
    args = parse_args()
    summary_df = load_summary(args.csv)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        summary_df["epsilon"].to_numpy(dtype=float),
        summary_df["collision_prob"].to_numpy(dtype=float),
        color="#0072B2",
        linewidth=2.0,
    )
    if args.title:
        ax.set_title(args.title)
    apply_linear_style(ax, float(args.x_max))
    save_figure(fig, args.out)
    plt.close(fig)
    print(f"[OK] wrote: {args.out}")

    log_out = args.log_out
    if log_out is None:
        log_out = args.out.with_name(f"{args.out.stem}_logx{args.out.suffix}")

    log_df = positive_log_subset(summary_df, float(args.x_max))
    if log_df.empty:
        print(
            "[WARN] skipped log-x plot: there are no positive epsilon values "
            f"within x <= {float(args.x_max):.3f}."
        )
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        log_df["epsilon"].to_numpy(dtype=float),
        log_df["collision_prob"].to_numpy(dtype=float),
        color="#0072B2",
        linewidth=2.0,
    )
    if args.title:
        ax.set_title(args.title)
    apply_log_style(ax, log_df)
    save_figure(fig, log_out)
    plt.close(fig)
    print(f"[OK] wrote: {log_out}")
    if np.any(summary_df["epsilon"].to_numpy(dtype=float) <= 0.0):
        print("[note] omitted epsilon=0.0 from the log-x plot because logarithmic axes cannot display zero.")


if __name__ == "__main__":
    main()
