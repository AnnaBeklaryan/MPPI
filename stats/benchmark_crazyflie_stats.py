#!/usr/bin/env python3
"""
Benchmark statistics for the current Crazyflie MPPI scripts.

This driver intentionally calls each standalone simulator's simulate(...) so the
benchmark stays aligned with the live scenario/controller code:

  - mppi_crazyflie.py
  - RA_mppi_crazyflie.py
  - DR_mppi_crazyflie.py
  - DRA_mppi_crazyflie.py

Example:
python3 stats/benchmark_crazyflie_stats.py --runs 100 --steps 520 \
  --obs-update-steps 15 --outdir stats/results_crazyflie_stats_paper \
  --use_gpu  --dr-eps-cvar 0.0015
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    method: str
    path: Path


def _default_algorithm_specs() -> list[AlgorithmSpec]:
    return [
        AlgorithmSpec("MPPI", "mppi", ROOT_DIR / "mppi_crazyflie.py"),
        AlgorithmSpec("RAMPPI", "ramppi", ROOT_DIR / "RA_mppi_crazyflie.py"),
        AlgorithmSpec("DRMPPI", "drmppi", ROOT_DIR / "DR_mppi_crazyflie.py"),
        AlgorithmSpec("DRAMPPI", "dramppi", ROOT_DIR / "DRA_mppi_crazyflie.py"),
    ]


def _load_module(module_name: str, path: Path):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing simulator file: {path}")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "simulate"):
        raise AttributeError(f"{path} does not expose simulate(...)")
    return mod


def _set_run_seed(seed: int):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fmt_mean_std(vals: np.ndarray, digits: int = 4) -> str:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "nan +/- (nan)"
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return f"{mean:.{digits}f} +/- ({std:.{digits}f})"


def _fmt_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0.0:
        return "unknown"
    seconds_i = int(round(float(seconds)))
    h, rem = divmod(seconds_i, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def _as_float_array(data: dict, key: str) -> np.ndarray:
    return np.asarray(data.get(key, []), dtype=np.float64)


def _scalar(data: dict, key: str, default=np.nan):
    if key not in data:
        return default
    arr = np.asarray(data[key])
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def _control_effort(data: dict) -> float:
    u = _as_float_array(data, "U_applied")
    if u.size == 0:
        return np.nan
    return float(np.sum(u * u))


def _total_full_cost(data: dict) -> float:
    if "total_full_cost" in data:
        return float(_scalar(data, "total_full_cost", np.nan))
    return _control_effort(data)


def _total_tracking_control_cost(data: dict) -> float:
    if "total_tracking_cost" in data:
        return float(_scalar(data, "total_tracking_cost", np.nan))
    return np.nan


def _summarize_run(spec: AlgorithmSpec, run_idx: int, run_seed: int, data: dict, wall_s: float) -> dict:
    min_dist = _as_float_array(data, "min_obstacle_dist")
    min_dist = min_dist[np.isfinite(min_dist)]
    solve_ms = _as_float_array(data, "solve_ms")
    solve_ms = solve_ms[np.isfinite(solve_ms)]
    safety_flags = _as_float_array(data, "safety_flags")
    collision_flags = _as_float_array(data, "collision_flags")

    run_min_dist = float(np.min(min_dist)) if min_dist.size > 0 else np.nan
    safety = bool(np.any(safety_flags > 0.5)) if safety_flags.size > 0 else False
    collision = bool(np.any(collision_flags > 0.5)) if collision_flags.size > 0 else False
    solve_max = float(np.max(solve_ms)) if solve_ms.size > 0 else np.nan
    solve_mean = float(np.mean(solve_ms)) if solve_ms.size > 0 else np.nan
    first_collision = int(_scalar(data, "first_collision_step", -1))

    return {
        "Algorithm": spec.name,
        "Method": str(_scalar(data, "method", spec.method)),
        "SourceFile": str(spec.path.relative_to(ROOT_DIR)),
        "Run": int(run_idx),
        "Seed": int(run_seed),
        "SimSteps": int(_scalar(data, "sim_steps", len(min_dist))),
        "Dt": float(_scalar(data, "dt", np.nan)),
        "ObsUpdateSteps": int(_scalar(data, "obs_update_steps", -1)),
        "RunMinDistance": run_min_dist,
        "SafetyViolation": float(safety),
        "Collision": float(collision),
        "FirstCollisionStep": first_collision,
        "CollisionSteps": int(np.count_nonzero(collision_flags > 0.5)) if collision_flags.size > 0 else 0,
        "RunMaxComputeTimeMs": solve_max,
        "RunMeanComputeTimeMs": solve_mean,
        "WallTimeS": float(wall_s),
        "TotalControlEffort": _control_effort(data),
        "TotalFullCost": _total_full_cost(data),
        "TerminalFullCost": float(_scalar(data, "terminal_full_cost", np.nan)),
        "TotalTrackingControlCost": _total_tracking_control_cost(data),
        "TerminalTrackingCost": float(_scalar(data, "terminal_tracking_cost", np.nan)),
        # Legacy names kept so older plotting/analysis notebooks still open.
        "TotalQRStageCost": _total_tracking_control_cost(data),
        "TotalQSigmaInvStageCost": _total_tracking_control_cost(data),
        "MovingCollisionRadius": float(_scalar(data, "moving_collision_radius", np.nan)),
        "MovingSafeRadius": float(_scalar(data, "moving_safe_radius", np.nan)),
        "DroneRadius": float(_scalar(data, "drone_radius", np.nan)),
        "DrEpsCvar": float(_scalar(data, "dr_eps_cvar", np.nan)),
        "CvarAlpha": float(_scalar(data, "cvar_alpha", np.nan)),
        "CvarN": float(_scalar(data, "cvar_N", np.nan)),
        "RiskCostA": float(_scalar(data, "risk_cost_A", np.nan)),
        "RiskCostCu": float(_scalar(data, "" \
        "_cost_Cu", np.nan)),
        "SigmaCp": float(_scalar(data, "sigma_cp", np.nan)),
        "Nmc": float(_scalar(data, "Nmc", np.nan)),
        "OmegaSoft": float(_scalar(data, "omega_soft", np.nan)),
        "OmegaHard": float(_scalar(data, "omega_hard", np.nan)),
    }


def _run_simulator(mod, spec: AlgorithmSpec, args, run_seed: int):
    _set_run_seed(run_seed)
    kwargs = dict(
        save_dir=None,
        obs_update_steps=int(args.obs_update_steps),
        use_gpu=bool(args.use_gpu),
        sim_steps=int(args.steps),
    )
    if spec.method == "drmppi" and args.dr_eps_cvar is not None:
        kwargs["dr_eps_cvar"] = float(args.dr_eps_cvar)
    if args.verbose:
        return mod.simulate(**kwargs)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return mod.simulate(**kwargs)


def _write_summary_tables(rows: list[dict], outdir: Path, algorithm_names: list[str]):
    raw_df = pd.DataFrame(rows)
    raw_path = outdir / "run_metrics.csv"
    raw_df.to_csv(raw_path, index=False)

    row_names = [
        "1) Min distance across runs (min of run minima)",
        "2) Run-min distance (mean +/- std)",
        "3) Safety violation probability using simulator flags",
        "4) Collision probability using simulator flags",
        "5) Total tracking/control cost over collision-free runs (mean +/- std)",
        "6) Max compute time across runs (ms)",
        "7) Run-max compute time (mean +/- std) (ms)",
        "8) Mean compute time per step (mean +/- std) (ms)",
        "9) Wall time per episode (mean +/- std) (s)",
    ]
    paper = pd.DataFrame({"Metric": row_names})

    compact_rows = []
    for alg in algorithm_names:
        sub = raw_df[raw_df["Algorithm"] == alg]
        run_min = sub["RunMinDistance"].to_numpy(dtype=np.float64)
        run_min_f = run_min[np.isfinite(run_min)]
        safety = sub["SafetyViolation"].to_numpy(dtype=np.float64)
        collision = sub["Collision"].to_numpy(dtype=np.float64)
        success_sub = sub[sub["Collision"] < 0.5]
        tracking_cost = success_sub["TotalTrackingControlCost"].to_numpy(dtype=np.float64)
        full_cost = sub["TotalFullCost"].to_numpy(dtype=np.float64)
        max_ms = sub["RunMaxComputeTimeMs"].to_numpy(dtype=np.float64)
        mean_ms = sub["RunMeanComputeTimeMs"].to_numpy(dtype=np.float64)
        wall_s = sub["WallTimeS"].to_numpy(dtype=np.float64)

        paper[alg] = [
            f"{(float(np.min(run_min_f)) if run_min_f.size > 0 else np.nan):.4f}",
            _fmt_mean_std(run_min_f),
            f"{float(np.mean(safety)):.4f}" if safety.size > 0 else "nan",
            f"{float(np.mean(collision)):.4f}" if collision.size > 0 else "nan",
            _fmt_mean_std(tracking_cost),
            f"{(float(np.nanmax(max_ms)) if max_ms.size > 0 else np.nan):.4f}",
            _fmt_mean_std(max_ms),
            _fmt_mean_std(mean_ms),
            _fmt_mean_std(wall_s),
        ]

        compact_rows.append(
            {
                "Algorithm": alg,
                "Runs": int(len(sub)),
                "CollisionFreeRuns": int(len(success_sub)),
                "RunMinDistance_Min": float(np.min(run_min_f)) if run_min_f.size > 0 else np.nan,
                "RunMinDistance_Mean": float(np.mean(run_min_f)) if run_min_f.size > 0 else np.nan,
                "RunMinDistance_Std": float(np.std(run_min_f, ddof=1)) if run_min_f.size > 1 else 0.0,
                "SafetyViolation_Prob": float(np.mean(safety)) if safety.size > 0 else np.nan,
                "Collision_Prob": float(np.mean(collision)) if collision.size > 0 else np.nan,
                "TotalFullCost_Mean": float(np.nanmean(full_cost)) if full_cost.size > 0 else np.nan,
                "TotalFullCost_Std": float(np.nanstd(full_cost, ddof=1)) if full_cost.size > 1 else 0.0,
                "TotalTrackingControlCost_Mean": float(np.nanmean(tracking_cost)) if tracking_cost.size > 0 else np.nan,
                "TotalTrackingControlCost_Std": float(np.nanstd(tracking_cost, ddof=1)) if tracking_cost.size > 1 else 0.0,
                "TotalQRStageCost_Mean": float(np.nanmean(tracking_cost)) if tracking_cost.size > 0 else np.nan,
                "TotalQRStageCost_Std": float(np.nanstd(tracking_cost, ddof=1)) if tracking_cost.size > 1 else 0.0,
                "TotalQSigmaInvStageCost_Mean": float(np.nanmean(tracking_cost)) if tracking_cost.size > 0 else np.nan,
                "TotalQSigmaInvStageCost_Std": float(np.nanstd(tracking_cost, ddof=1)) if tracking_cost.size > 1 else 0.0,
                "RunMaxComputeTimeMs_Max": float(np.nanmax(max_ms)) if max_ms.size > 0 else np.nan,
                "RunMaxComputeTimeMs_Mean": float(np.nanmean(max_ms)) if max_ms.size > 0 else np.nan,
                "RunMaxComputeTimeMs_Std": float(np.nanstd(max_ms, ddof=1)) if max_ms.size > 1 else 0.0,
                "RunMeanComputeTimeMs_Mean": float(np.nanmean(mean_ms)) if mean_ms.size > 0 else np.nan,
                "WallTimeS_Mean": float(np.nanmean(wall_s)) if wall_s.size > 0 else np.nan,
            }
        )

    paper_path = outdir / "summary_table_paper.csv"
    numeric_path = outdir / "summary_table_numeric.csv"
    paper.to_csv(paper_path, index=False)
    pd.DataFrame(compact_rows).to_csv(numeric_path, index=False)
    return paper, paper_path, numeric_path, raw_path


def parse_args():
    ap = argparse.ArgumentParser(description="Run benchmark stats using the current Crazyflie simulator files.")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--obs-update-steps", type=int, default=15)
    ap.add_argument("--outdir", type=str, default="stats/results_crazyflie_stats_paper")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument(
        "--methods",
        type=str,
        default="mppi,ramppi,drmppi,dramppi",
        help="Comma-separated subset: mppi,ramppi,drmppi,dramppi",
    )
    ap.add_argument("--use_gpu", "--cuda", dest="use_gpu", action="store_true", default=True)
    ap.add_argument("--cpu", dest="use_gpu", action="store_false")
    ap.add_argument(
        "--dr-eps-cvar",
        type=float,
        default=None,
        help="Optional DRMPPI epsilon override. Omit to use DR_mppi_crazyflie.py's current simulate(...) default.",
    )
    ap.add_argument("--verbose", action="store_true", help="Do not suppress per-step simulator logs.")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    wanted = {m.strip().lower() for m in args.methods.split(",") if m.strip()}
    specs = [spec for spec in _default_algorithm_specs() if spec.method in wanted]
    if not specs:
        raise ValueError("No valid methods selected.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"[setup] runs={args.runs} steps={args.steps} obs_update_steps={args.obs_update_steps} "
        f"use_gpu={int(bool(args.use_gpu))} cuda_available={int(torch.cuda.is_available())}",
        flush=True,
    )
    print("[setup] simulator files:", flush=True)

    modules = {}
    for idx, spec in enumerate(specs):
        print(f"  {spec.name}: {spec.path.relative_to(ROOT_DIR)}", flush=True)
        modules[spec.method] = _load_module(f"cf_bench_{spec.method}_{idx}", spec.path)

    master = np.random.default_rng(int(args.seed))
    run_seeds = master.integers(0, 2**31 - 1, size=(args.runs, len(specs)), dtype=np.int64)

    rows = []
    benchmark_t0 = time.perf_counter()
    total = args.runs * len(specs)
    completed = 0
    for run_idx in range(args.runs):
        print(f"[run {run_idx + 1:3d}/{args.runs}] started", flush=True)
        for alg_idx, spec in enumerate(specs):
            run_seed = int(run_seeds[run_idx, alg_idx])
            episode_t0 = time.perf_counter()
            print(f"  [controller] {spec.name} seed={run_seed} ...", flush=True)
            data = _run_simulator(modules[spec.method], spec, args, run_seed)
            wall_s = time.perf_counter() - episode_t0
            row = _summarize_run(spec, run_idx, run_seed, data, wall_s)
            rows.append(row)

            completed += 1
            elapsed_s = time.perf_counter() - benchmark_t0
            avg_episode_s = elapsed_s / max(completed, 1)
            remaining_s = avg_episode_s * max(total - completed, 0)
            print(
                f"  [done] {spec.name}: min_dist={row['RunMinDistance']:.4f}, "
                f"safety={int(row['SafetyViolation'])}, collision={int(row['Collision'])}, "
                f"max_solve_ms={row['RunMaxComputeTimeMs']:.2f}",
                flush=True,
            )
            print(
                f"  [eta] completed={completed}/{total}, elapsed={_fmt_duration(elapsed_s)}, "
                f"remaining~{_fmt_duration(remaining_s)}",
                flush=True,
            )
        print(f"[run {run_idx + 1:3d}/{args.runs}] done", flush=True)

    paper, paper_path, numeric_path, raw_path = _write_summary_tables(
        rows=rows,
        outdir=outdir,
        algorithm_names=[spec.name for spec in specs],
    )

    print("\n=== Benchmark Table ===", flush=True)
    print(paper.to_string(index=False), flush=True)
    print(f"\n[OK] wrote: {paper_path}", flush=True)
    print(f"[OK] wrote: {numeric_path}", flush=True)
    print(f"[OK] wrote: {raw_path}", flush=True)


if __name__ == "__main__":
    main()
