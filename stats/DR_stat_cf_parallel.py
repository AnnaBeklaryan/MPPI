#!/usr/bin/env python3
"""
Parallel epsilon sweep for DR_mppi_crazyflie.py.

Example:
python3 stats/DR_stat_cf_parallel.py --runs 100 --eps-min 0.0 --eps-max 0.5 --eps-step 0.005 --workers 15 --obs-update-steps 15 --steps 400 --use_gpu
python3 stats/DR_stat_cf_parallel.py --runs 100 --eps-min 0.505 --eps-max 1.0 --eps-step 0.005 --workers 15 --obs-update-steps 15 --steps 400 --append
python3 stats/DR_stat_cf_parallel.py --runs 100 --eps-min 0.505 --eps-max 1 --eps-step 0.005 --workers 15 --obs-update-steps 15 --steps 400 --append true
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import io
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import DR_mppi_crazyflie as cf_dr


DEFAULT_EPSILONS = np.arange(0.0, 0.1001, 0.005, dtype=np.float64)
RUN_COLUMNS = [
    "epsilon",
    "Run",
    "Seed",
    "RunMinDistance",
    "SafetyViolation",
    "Collision",
    "RunMaxComputeTimeMs",
]
SUMMARY_COLUMNS = [
    "epsilon",
    "collision_prob",
    "collision_free_prob",
    "safety_violation_prob",
    "run_min_distance_min",
    "run_min_distance_mean",
    "run_min_distance_std",
    "run_max_compute_time_ms_mean",
    "run_max_compute_time_ms_std",
]


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value_l = str(value).strip().lower()
    if value_l in ("1", "true", "yes", "y", "on"):
        return True
    if value_l in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run DR-MPPI Crazyflie epsilon statistics in parallel.")
    ap.add_argument("--runs", type=int, default=50, help="Number of repeated simulations per epsilon.")
    ap.add_argument("--outdir", type=str, default="results_dr_cf_eps_stats")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument(
        "--append",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Append new epsilon results to existing CSV files instead of rewriting them. Accepts true/false.",
    )
    ap.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit list of dr_eps_cvar values.",
    )
    ap.add_argument("--eps-min", type=float, default=0.00, help="Minimum epsilon for a sweep.")
    ap.add_argument("--eps-max", type=float, default=0.00, help="Maximum epsilon for a sweep.")
    ap.add_argument("--eps-step", type=float, default=0.2, help="Epsilon step when --eps-max is used.")
    ap.add_argument(
        "--obs-update-steps",
        type=int,
        default=15,
        help="Read fresh moving-obstacle observations every N control steps and hold between reads.",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=400,
        help="Number of simulation control steps to run for each epsilon/run pair.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel epsilon worker processes. Use 0 for auto.",
    )
    ap.add_argument(
        "--torch-threads-per-worker",
        type=int,
        default=0,
        help="CPU torch threads per worker. Use 0 for default, or auto-cap to 1 for multi-worker CPU.",
    )
    ap.add_argument(
        "--log-runs-every",
        type=int,
        default=1,
        help="Print progress every N runs per epsilon. Use 0 to disable.",
    )
    ap.add_argument("--quiet-sim", action="store_true", default=True, help="Suppress per-step simulation logs.")
    ap.add_argument("--no-quiet-sim", dest="quiet_sim", action="store_false", help="Show per-step simulation logs.")
    return ap.parse_args()


def build_epsilon_sweep(args: argparse.Namespace) -> np.ndarray:
    if args.epsilons is not None:
        eps = np.asarray(args.epsilons, dtype=np.float64)
    elif args.eps_max is not None:
        eps_min = float(args.eps_min)
        eps_max = float(args.eps_max)
        eps_step = float(args.eps_step)
        if not (np.isfinite(eps_min) and eps_min >= 0.0):
            raise ValueError(f"--eps-min must be >= 0, got {args.eps_min}")
        if not (np.isfinite(eps_max) and eps_max >= 0.0):
            raise ValueError(f"--eps-max must be >= 0, got {args.eps_max}")
        if not (np.isfinite(eps_step) and eps_step > 0.0):
            raise ValueError(f"--eps-step must be > 0, got {args.eps_step}")
        if eps_max < eps_min:
            raise ValueError(f"--eps-max must be >= --eps-min, got {args.eps_max} < {args.eps_min}")
        count = int(np.floor((eps_max - eps_min) / eps_step + 1e-12)) + 1
        eps = eps_min + eps_step * np.arange(count, dtype=np.float64)
        if eps.size == 0 or abs(float(eps[-1]) - eps_max) > 1e-12:
            eps = np.append(eps, eps_max)
    else:
        eps = DEFAULT_EPSILONS.copy()

    if eps.size == 0 or np.any(~np.isfinite(eps)) or np.any(eps < 0.0):
        raise ValueError("Epsilon values must be finite and non-negative.")
    return np.sort(np.unique(np.round(eps, decimals=12)))


def _choose_workers(args: argparse.Namespace, num_eps: int) -> int:
    requested = int(args.workers)
    if requested < 0:
        raise ValueError(f"--workers must be >= 0, got {args.workers}")
    if requested == 0:
        if bool(args.use_gpu):
            return 1
        requested = max(1, (os.cpu_count() or 1) - 1)
    if bool(args.use_gpu) and requested > 1:
        print(
            f"[parallel] --use_gpu with workers={requested}; launching multiple CUDA worker processes. "
            "Monitor VRAM with nvidia-smi.",
            flush=True,
        )
    return max(1, min(requested, num_eps))


def _choose_torch_threads(args: argparse.Namespace, workers: int) -> int:
    requested = int(args.torch_threads_per_worker)
    if requested < 0:
        raise ValueError(f"--torch-threads-per-worker must be >= 0, got {requested}")
    if requested > 0:
        return requested
    if bool(args.use_gpu) or workers <= 1:
        return 0
    return 1


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _summarize_one_epsilon(eps: float, run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    min_d = np.asarray([row["RunMinDistance"] for row in run_rows], dtype=float)
    min_d_f = min_d[np.isfinite(min_d)]
    safety = np.asarray([row["SafetyViolation"] for row in run_rows], dtype=float)
    collision = np.asarray([row["Collision"] for row in run_rows], dtype=float)
    tmax = np.asarray([row["RunMaxComputeTimeMs"] for row in run_rows], dtype=float)
    collision_prob = float(np.mean(collision))
    return dict(
        epsilon=float(eps),
        collision_prob=collision_prob,
        collision_free_prob=1.0 - collision_prob,
        safety_violation_prob=float(np.mean(safety)),
        run_min_distance_min=float(np.min(min_d_f)) if min_d_f.size else np.nan,
        run_min_distance_mean=float(np.mean(min_d_f)) if min_d_f.size else np.nan,
        run_min_distance_std=float(np.std(min_d_f, ddof=1)) if min_d_f.size > 1 else 0.0,
        run_max_compute_time_ms_mean=float(np.mean(tmax)) if tmax.size else np.nan,
        run_max_compute_time_ms_std=float(np.std(tmax, ddof=1)) if tmax.size > 1 else 0.0,
    )


def _run_one_simulation(eps: float, run_seed: int, args_dict: dict[str, Any]) -> dict[str, Any]:
    np.random.seed(int(run_seed) % (2**32 - 1))
    torch.manual_seed(int(run_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(run_seed))

    call = lambda: cf_dr.simulate(
        save_dir=None,
        obs_update_steps=int(args_dict["obs_update_steps"]),
        use_gpu=bool(args_dict["use_gpu"]),
        dr_eps_cvar=float(eps),
        sim_steps=int(args_dict["steps"]),
    )
    if bool(args_dict["quiet_sim"]):
        with contextlib.redirect_stdout(io.StringIO()):
            return call()
    return call()


def _run_one_epsilon_worker(eps: float, run_seeds: tuple[int, ...], args_dict: dict[str, Any]) -> dict[str, Any]:
    if int(args_dict["torch_threads"]) > 0 and not bool(args_dict["use_gpu"]):
        torch.set_num_threads(int(args_dict["torch_threads"]))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    job_t0 = time.perf_counter()
    pid = os.getpid()
    total_runs = len(run_seeds)
    log_runs_every = int(args_dict["log_runs_every"])
    print(
        f"[worker pid={pid}] eps={eps:.6f} started runs={total_runs} "
        f"obs_update_steps={int(args_dict['obs_update_steps'])}",
        flush=True,
    )

    run_rows: list[dict[str, Any]] = []
    for run_idx, run_seed in enumerate(run_seeds):
        run_data = _run_one_simulation(eps=float(eps), run_seed=int(run_seed), args_dict=args_dict)
        min_d = np.asarray(run_data.get("min_obstacle_dist", []), dtype=float)
        min_d = min_d[np.isfinite(min_d)]
        collision_flags = np.asarray(run_data.get("collision_flags", []), dtype=bool)
        safety_flags = np.asarray(run_data.get("safety_flags", []), dtype=bool)
        solve_ms = np.asarray(run_data.get("solve_ms", []), dtype=float)
        row = dict(
            epsilon=float(eps),
            Run=int(run_idx),
            Seed=int(run_seed),
            RunMinDistance=float(np.min(min_d)) if min_d.size else np.nan,
            SafetyViolation=1.0 if np.any(safety_flags) else 0.0,
            Collision=1.0 if np.any(collision_flags) else 0.0,
            RunMaxComputeTimeMs=float(np.nanmax(solve_ms)) if solve_ms.size else np.nan,
        )
        run_rows.append(row)

        runs_done = run_idx + 1
        if log_runs_every > 0 and (runs_done == 1 or runs_done == total_runs or runs_done % log_runs_every == 0):
            elapsed = time.perf_counter() - job_t0
            eta = (elapsed / float(runs_done)) * float(total_runs - runs_done)
            print(
                f"[worker pid={pid}] eps={eps:.6f} run={runs_done}/{total_runs} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                flush=True,
            )

    return dict(
        epsilon=float(eps),
        duration_s=float(time.perf_counter() - job_t0),
        run_rows=run_rows,
        summary_row=_summarize_one_epsilon(eps=float(eps), run_rows=run_rows),
    )


def _log_one_result(result: dict[str, Any], done_count: int, total_count: int) -> None:
    summary = result["summary_row"]
    elapsed_total = float(result["elapsed_total_s"])
    remaining = max(0, total_count - done_count)
    eta_total = (elapsed_total / float(done_count)) * float(remaining) if done_count > 0 else 0.0
    print(
        f"[done {done_count:3d}/{total_count}] eps={summary['epsilon']:.6f} "
        f"job_time={_format_duration(result['duration_s'])} "
        f"total_elapsed={_format_duration(elapsed_total)} "
        f"overall_eta={_format_duration(eta_total)} "
        f"collision_prob={summary['collision_prob']:.4f} "
        f"collision_free_prob={summary['collision_free_prob']:.4f}",
        flush=True,
    )


def save_collision_plot(summary_df: pd.DataFrame, outdir: Path, save_pdf: bool = False) -> list[Path]:
    saved: list[Path] = []
    eps = summary_df["epsilon"].to_numpy(dtype=float)
    cp = summary_df["collision_prob"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(eps, cp, linewidth=2.0, color="#0072B2", marker="o")
    ax.set_xlabel("Wasserstein radius, $\\varepsilon$")
    ax.set_ylabel("Collision probability")
    ax.set_ylim(-0.02, 1.0)
    ax.grid(True, color="#b0b0b0", alpha=0.35)
    png = outdir / "dr_cf_epsilon_collision_probability.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    saved.append(png)
    if save_pdf:
        pdf = outdir / "dr_cf_epsilon_collision_probability.pdf"
        fig.savefig(pdf, bbox_inches="tight")
        saved.append(pdf)
    plt.close(fig)
    return saved


def _init_csv_outputs(run_csv: Path, summary_csv: Path) -> None:
    pd.DataFrame(columns=RUN_COLUMNS).to_csv(run_csv, index=False)
    pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(summary_csv, index=False)


def _read_csv_if_exists(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[columns].copy()


def _append_result_to_csv(result: dict[str, Any], run_csv: Path, summary_csv: Path) -> None:
    pd.DataFrame(result["run_rows"], columns=RUN_COLUMNS).to_csv(
        run_csv,
        mode="a",
        header=False,
        index=False,
    )
    pd.DataFrame([result["summary_row"]], columns=SUMMARY_COLUMNS).to_csv(
        summary_csv,
        mode="a",
        header=False,
        index=False,
    )


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_csv = outdir / "dr_cf_epsilon_run_metrics.csv"
    summary_csv = outdir / "dr_cf_epsilon_summary.csv"
    if bool(args.append):
        if not run_csv.exists() or not summary_csv.exists():
            _init_csv_outputs(run_csv, summary_csv)
        print(f"[csv] append=1 keeping existing rows in {outdir}", flush=True)
    else:
        _init_csv_outputs(run_csv, summary_csv)
        print(f"[csv] append=0 rewriting CSV outputs in {outdir}", flush=True)

    eps_sweep = build_epsilon_sweep(args)
    workers = _choose_workers(args, num_eps=len(eps_sweep))
    torch_threads = _choose_torch_threads(args, workers)
    run_seeds = tuple(
        int(seed)
        for seed in np.random.default_rng(int(args.seed)).integers(
            0, 2**31 - 1, size=int(args.runs), dtype=np.int64
        )
    )
    args_dict = dict(
        runs=int(args.runs),
        use_gpu=bool(args.use_gpu),
        obs_update_steps=max(1, int(args.obs_update_steps)),
        steps=max(1, int(args.steps)),
        log_runs_every=max(0, int(args.log_runs_every)),
        quiet_sim=bool(args.quiet_sim),
        torch_threads=int(torch_threads),
    )

    print(f"[epsilon sweep] values={eps_sweep.tolist()}", flush=True)
    print(
        f"[parallel] workers={workers} start_method=spawn "
        f"torch_threads_per_worker={torch_threads if torch_threads > 0 else 'default'}",
        flush=True,
    )
    print(
        f"[parallel] submitted_eps={len(eps_sweep)} runs_per_eps={int(args.runs)} "
        f"obs_update_steps={int(args.obs_update_steps)} use_gpu={int(bool(args.use_gpu))}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    sweep_t0 = time.perf_counter()
    if workers == 1:
        for idx, eps in enumerate(eps_sweep, start=1):
            result = _run_one_epsilon_worker(float(eps), run_seeds, args_dict)
            result["elapsed_total_s"] = float(time.perf_counter() - sweep_t0)
            results.append(result)
            _append_result_to_csv(result, run_csv, summary_csv)
            _log_one_result(result, done_count=idx, total_count=len(eps_sweep))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            future_to_eps = {
                executor.submit(_run_one_epsilon_worker, float(eps), run_seeds, args_dict): float(eps)
                for eps in eps_sweep
            }
            done_count = 0
            for future in as_completed(future_to_eps):
                eps = future_to_eps[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Parallel Crazyflie epsilon job failed for dr_eps_cvar={eps:.6f}") from exc
                result["elapsed_total_s"] = float(time.perf_counter() - sweep_t0)
                results.append(result)
                done_count += 1
                _append_result_to_csv(result, run_csv, summary_csv)
                _log_one_result(result, done_count=done_count, total_count=len(eps_sweep))

    results.sort(key=lambda item: float(item["epsilon"]))
    if bool(args.append):
        run_df = _read_csv_if_exists(run_csv, RUN_COLUMNS)
        summary_df = _read_csv_if_exists(summary_csv, SUMMARY_COLUMNS)
    else:
        run_rows = [row for result in results for row in result["run_rows"]]
        summary_rows = [result["summary_row"] for result in results]
        run_df = pd.DataFrame(run_rows, columns=RUN_COLUMNS)
        summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    run_df = run_df.sort_values(["epsilon", "Run"]).reset_index(drop=True)
    summary_df = summary_df.sort_values("epsilon").reset_index(drop=True)

    run_df.to_csv(run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    saved_plot_paths = save_collision_plot(summary_df, outdir=outdir)

    print(f"[OK] wrote: {run_csv}", flush=True)
    print(f"[OK] wrote: {summary_csv}", flush=True)
    for path in saved_plot_paths:
        print(f"[OK] wrote: {path}", flush=True)


if __name__ == "__main__":
    main()
