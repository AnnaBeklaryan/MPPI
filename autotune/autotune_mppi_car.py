#!/usr/bin/env python3
"""
Autotune plain MPPI parameters for the car setup in MPPI_final/mppi.py.

This runs the same closed-loop car scenario used by the main simulation script,
but headless, and performs a simple random search over tracking/controller
parameters. The best configuration is written to JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


MPPI_DIR = Path(__file__).resolve().parents[1]
if str(MPPI_DIR) not in sys.path:
    sys.path.insert(0, str(MPPI_DIR))

from mppi import (  # noqa: E402
    MPPI,
    MovingObstacleCSV,
    angle_wrap,
    diffdrive_dynamics,
    dyn_diffdrive,
    running_cost_lane_obs,
    terminal_cost_track,
)


def sample_cfg(rng: random.Random):
    return {
        "T": rng.choice([15, 20, 25, 30, 35]),
        "M": rng.choice([512, 768, 1024, 1200, 1536]),
        "I": rng.choice([1, 2, 3]),
        "lam": rng.uniform(0.01, 0.30),
        "sigma_v": rng.uniform(0.2, 1.8),
        "sigma_w_deg": rng.uniform(3.0, 35.0),
        "u_max_v": rng.uniform(7.0, 12.0),
        "u_max_w_deg": rng.uniform(80.0, 180.0),
        "dyn_w_max_deg": rng.uniform(90.0, 180.0),
        "Qx": rng.uniform(0.05, 3.0),
        "Qy": rng.uniform(1.0, 12.0),
        "Qpsi": rng.uniform(0.05, 3.0),
        "Qf_x_scale": rng.uniform(1.5, 5.0),
        "Qf_y_scale": rng.uniform(1.5, 5.0),
        "Qf_psi_scale": rng.uniform(1.5, 5.0),
        "Rv": rng.uniform(0.1, 2.5),
        "Rw": rng.uniform(0.1, 2.5),
        "L_ref": rng.uniform(8.0, 20.0),
        "v_des": rng.uniform(4.0, 10.0),
        "u_blend_v": rng.uniform(0.0, 0.7),
        "obs_w": rng.uniform(1e3, 2e4),
        "extra_margin": rng.uniform(0.0, 0.15),
    }


def evaluate_cfg(cfg, obs_csv: MovingObstacleCSV, pos_scale: float, max_steps: int, device: str):
    dt = float(np.round(np.min(np.diff(obs_csv.times)), 6))
    if not (np.isfinite(dt) and dt > 0):
        return 1e12, {"error": "invalid_dt"}

    T = int(cfg["T"])
    M = int(cfg["M"])
    lam = float(cfg["lam"])
    sigma = np.array([float(cfg["sigma_v"]), math.radians(float(cfg["sigma_w_deg"]))], dtype=np.float32)
    u_min = np.array([0.0, -math.radians(float(cfg["u_max_w_deg"]))], dtype=np.float32)
    u_max = np.array([float(cfg["u_max_v"]), math.radians(float(cfg["u_max_w_deg"]))], dtype=np.float32)
    dyn_w_max = math.radians(float(cfg["dyn_w_max_deg"]))

    Q = np.array([float(cfg["Qx"]), float(cfg["Qy"]), float(cfg["Qpsi"])], dtype=np.float32)
    Qf = np.array([
        float(cfg["Qf_x_scale"]) * float(cfg["Qx"]),
        float(cfg["Qf_y_scale"]) * float(cfg["Qy"]),
        float(cfg["Qf_psi_scale"]) * float(cfg["Qpsi"]),
    ], dtype=np.float32)
    R = np.array([float(cfg["Rv"]), float(cfg["Rw"])], dtype=np.float32)

    mppi = MPPI(
        dt=dt,
        T=T,
        M=M,
        lam=lam,
        noise_sigma=sigma,
        u_min=u_min,
        u_max=u_max,
        dynamics=dyn_diffdrive,
        running_cost=running_cost_lane_obs,
        terminal_cost=terminal_cost_track,
        device=device,
        dtype=torch.float32,
        I=int(cfg["I"]),
        dyn_kwargs=dict(w_max=dyn_w_max),
        cost_kwargs=dict(ref=None, Q=None, R=None, Qf=None, O_mean=None, radii=None, obs_w=float(cfg["obs_w"])),
        verbose=False,
    )

    Q_t = torch.as_tensor(Q, device=mppi.device, dtype=mppi.dtype)
    R_t = torch.as_tensor(R, device=mppi.device, dtype=mppi.dtype)
    Qf_t = torch.as_tensor(Qf, device=mppi.device, dtype=mppi.dtype)

    lane_psi = 0.0
    road_center = 1.0
    lane_w = 0.70
    lane_y = road_center + 0.5 * lane_w
    x_mppi = np.array([0.0, lane_y, 0.0], dtype=np.float32)

    L_ref = float(cfg["L_ref"])
    v_des = float(cfg["v_des"])
    blend = float(cfg["u_blend_v"])

    ego_length = 4.5 * pos_scale
    ego_width = 1.8 * pos_scale
    ego_radius = 0.5 * np.sqrt(ego_length**2 + ego_width**2)
    obs_radius = ego_radius
    extra_margin = float(cfg["extra_margin"])
    max_obs_draw = 20

    steps = min(int(max_steps), len(obs_csv.times))
    if steps < 2:
        return 1e12, {"error": "too_few_steps"}

    min_solve = float("inf")
    max_solve = 0.0
    sum_solve = 0.0
    lane_err_sq = 0.0
    psi_err_sq = 0.0
    du_sq = 0.0
    coll_viol_sq = 0.0
    last_u = np.zeros((2,), dtype=np.float32)

    for k in range(steps):
        t_now = float(obs_csv.times[k])
        obs_now = obs_csv.obstacles_now(t_now)

        dx = obs_now["x"].to_numpy(float) - float(x_mppi[0])
        dy = obs_now["y"].to_numpy(float) - float(x_mppi[1])
        mask = (dx > -4.0) & (dx < 28.0) & (np.abs(dy) < 4.8)
        obs_now = obs_now[mask].copy()

        if len(obs_now) > 0:
            dist2 = (obs_now["x"] - float(x_mppi[0])) ** 2 + (obs_now["y"] - float(x_mppi[1])) ** 2
            obs_now = obs_now.iloc[np.argsort(dist2.to_numpy())]
            obs_now = obs_now.iloc[:max_obs_draw].copy()

        ids, O_mean, _ = obs_csv.build_prediction_for_mppi(obs_df=obs_now, dt=dt, T=T, max_obs=max_obs_draw)
        if len(ids) > 0:
            radii_for_mppi = np.full((len(ids),), obs_radius + ego_radius + extra_margin, dtype=np.float32)
        else:
            radii_for_mppi = np.array([], dtype=np.float32)

        ref = np.array([float(x_mppi[0]) + L_ref, lane_y, lane_psi], dtype=np.float32)
        mppi.cost_kwargs["ref"] = torch.as_tensor(ref, device=mppi.device, dtype=mppi.dtype)
        mppi.cost_kwargs["Q"] = Q_t
        mppi.cost_kwargs["R"] = R_t
        mppi.cost_kwargs["Qf"] = Qf_t

        if O_mean is not None and O_mean.shape[1] > 0:
            mppi.cost_kwargs["O_mean"] = torch.as_tensor(O_mean, device=mppi.device, dtype=mppi.dtype)
            mppi.cost_kwargs["radii"] = torch.as_tensor(radii_for_mppi, device=mppi.device, dtype=mppi.dtype)
        else:
            mppi.cost_kwargs["O_mean"] = None
            mppi.cost_kwargs["radii"] = None

        if mppi.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        U, _ = mppi.plan(x_mppi, return_samples=False)
        if mppi.device.type == "cuda":
            torch.cuda.synchronize()
        solve_t = time.perf_counter() - t0

        min_solve = min(min_solve, solve_t)
        max_solve = max(max_solve, solve_t)
        sum_solve += solve_t

        u0 = U[0].copy()
        u0[0] = np.clip(blend * u0[0] + (1.0 - blend) * v_des, 0.0, float(u_max[0]))
        x_next = diffdrive_dynamics(x_mppi, u0, dt, v_max=float(u_max[0]), w_max=dyn_w_max).astype(np.float32)

        lane_err = float(x_next[1] - lane_y)
        psi_err = float(angle_wrap(x_next[2] - lane_psi))
        lane_err_sq += lane_err * lane_err
        psi_err_sq += psi_err * psi_err

        du = u0 - last_u
        du_sq += float(du[0] * du[0] + du[1] * du[1])
        last_u = u0.copy()

        if len(obs_now) > 0:
            obs_xy = obs_now[["x", "y"]].to_numpy(float)
            d = np.linalg.norm(obs_xy - x_next[None, :2], axis=1)
            safe = obs_radius + ego_radius + extra_margin
            viol = np.maximum(0.0, safe - d)
            coll_viol_sq += float(np.sum(viol * viol))

        x_mppi = x_next

        mppi.U[:-1] = mppi.U[1:]
        mppi.U[-1] = np.array([v_des, 0.0], dtype=np.float32)
        mppi.U_cpu = mppi.U

        if not np.all(np.isfinite(x_mppi)):
            return 1e12, {"error": "non_finite_state"}

    lane_rmse = math.sqrt(lane_err_sq / steps)
    psi_rmse = math.sqrt(psi_err_sq / steps)
    du_rms = math.sqrt(du_sq / max(1, steps - 1))
    coll_viol_mean = coll_viol_sq / steps
    mean_solve_ms = 1000.0 * (sum_solve / steps)

    score = (
        2.0 * lane_rmse
        + 1.0 * psi_rmse
        + 0.25 * du_rms
        + 8.0 * coll_viol_mean
        + 0.001 * mean_solve_ms
    )

    metrics = {
        "lane_rmse": lane_rmse,
        "psi_rmse": psi_rmse,
        "du_rms": du_rms,
        "collision_violation_mean": coll_viol_mean,
        "mean_solve_ms": mean_solve_ms,
        "min_solve_ms": 1000.0 * min_solve,
        "max_solve_ms": 1000.0 * max_solve,
        "steps": steps,
    }
    return float(score), metrics


def main():
    ap = argparse.ArgumentParser(description="Autotune plain MPPI car parameters")
    ap.add_argument("--trials", type=int, default=60, help="Random candidates to evaluate")
    ap.add_argument("--max-steps", type=int, default=420, help="Max simulation steps per candidate")
    ap.add_argument("--seed", type=int, default=11, help="Random seed")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--csv-path", type=str, default="", help="Optional override CSV path")
    ap.add_argument("--save-json", type=str, default="mppi_car_autotune_best.json")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    csv_path = args.csv_path if args.csv_path else os.path.join(str(MPPI_DIR), "Data/features_dir2_5_10s.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    pos_scale = 0.10
    vel_scale = 0.10
    acc_scale = 0.10
    y_offset = 23.0

    obs_csv = MovingObstacleCSV(
        csv_path=csv_path,
        x_offset=0.0,
        y_offset=y_offset,
        pos_scale=pos_scale,
        vel_scale=vel_scale,
        acc_scale=acc_scale,
    )

    best_score = float("inf")
    best_cfg = None
    best_metrics = None
    tuned_keys = sorted(sample_cfg(random.Random(args.seed)).keys())

    print("Tuning keys:", ", ".join(tuned_keys))
    print("Device:", device)

    for trial in range(1, int(args.trials) + 1):
        cfg = sample_cfg(random)
        score, metrics = evaluate_cfg(cfg, obs_csv, pos_scale=pos_scale, max_steps=int(args.max_steps), device=device)
        print(
            f"[trial {trial:03d}/{args.trials}] score={score:.4f} "
            f"lane_rmse={metrics.get('lane_rmse', float('nan')):.3f} "
            f"psi_rmse={metrics.get('psi_rmse', float('nan')):.3f} "
            f"coll={metrics.get('collision_violation_mean', float('nan')):.5f} "
            f"solve_ms={metrics.get('mean_solve_ms', float('nan')):.2f}"
        )
        if score < best_score:
            best_score = score
            best_cfg = cfg
            best_metrics = metrics
            print(f"  -> new best score={best_score:.4f}")

    out = {
        "best_score": best_score,
        "best_cfg": best_cfg,
        "best_metrics": best_metrics,
        "tuned_keys": tuned_keys,
        "notes": "Plain MPPI car tuning for MPPI_final/mppi.py using the CSV moving-obstacle scenario.",
    }

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nBest configuration:")
    print(json.dumps(out, indent=2))
    print(f"\nSaved to: {args.save_json}")


if __name__ == "__main__":
    main()
