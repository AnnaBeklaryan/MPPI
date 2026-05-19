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
import math
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
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import pandas as pd


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
        help="Output SVG path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional plot title. Leave empty to match the reference style.",
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


def apply_style(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    epsilon_values = summary_df["epsilon"].to_numpy(dtype=float)
    max_epsilon = float(epsilon_values.max())
    x_pad = max(0.0025, 0.02 * max_epsilon)

    ax.set_xlabel("Wasserstein radius, $\\varepsilon$")
    ax.set_ylabel("Collision probability")
    ax.set_ylim(-0.02, 0.6)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # Match the reference DR plot: keep the epsilon axis linear so dense values
    # near zero do not collapse into overlapping symlog tick labels.
    if max_epsilon <= 0.1:
        x_step = 0.02
    elif max_epsilon <= 0.5:
        x_step = 0.05
    elif max_epsilon <= 1.0:
        x_step = 0.1
    elif max_epsilon <= 5.0:
        x_step = 0.5
    else:
        rough_step = max_epsilon / 5.0
        magnitude = 10 ** math.floor(math.log10(rough_step))
        normalized = rough_step / magnitude
        if normalized <= 1.0:
            step_multiplier = 1.0
        elif normalized <= 2.0:
            step_multiplier = 2.0
        elif normalized <= 5.0:
            step_multiplier = 5.0
        else:
            step_multiplier = 10.0
        x_step = step_multiplier * magnitude

    ax.set_xlim(-x_pad, max_epsilon + x_pad)
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # ax.grid(True, color="#b0b0b0", alpha=0.35, linewidth=1.0)


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
    apply_style(ax, summary_df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote: {args.out}")


if __name__ == "__main__":
    main()
