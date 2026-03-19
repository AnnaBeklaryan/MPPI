#!/usr/bin/env python3
"""
Tracking-focused autotune for MPPI_final/DR_mppi_crazyflie.py.

This tuner is intentionally constrained for your use case:
- fixed planner timing/sample settings:
    dt=0.02
    horizon_steps=35
    rollouts=1000
    iterations=1
- DR risk parameters are NOT tuned and remain at DRParams defaults
- objective prioritizes path tracking quality, then final convergence, then smoothness

Example:
python3 MPPI_final/autotune/autotune_dr_mppi_crazyflie.py \
  --trials 300 \
  --max-steps 600 \
  --device auto \
  --seed 7 \
  --save-json MPPI_final/crazyflie_dr_tracking_autotune_fixed_best.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


MPPI_DIR = Path(__file__).resolve().parents[1]
if str(MPPI_DIR) not in sys.path:
    sys.path.insert(0, str(MPPI_DIR))

from DR_mppi_crazyflie import DRParams, TorchDRMPPIQuadOuter, build_min_snap_3d  # noqa: E402


FIXED_DT = 0.02
FIXED_HORIZON_STEPS = 35
FIXED_ROLLOUTS = 1000
FIXED_ITERATIONS = 1


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_follow_from_vel(v: np.ndarray, fallback: float) -> float:
    vx, vy = float(v[0]), float(v[1])
    if vx * vx + vy * vy < 1e-6:
        return fallback
    return math.atan2(vy, vx)


def make_world():
    ref_waypoints = np.array([
        [2.5, 2.0, 0.0],
        [0.0, 3.5, 2.0],
        [-3.0, 1.5, 4.5],
        [-2.0, -2.5, 3.0],
        [2.0, -3.0, 1.0],
        [3.0, 0.0, 0.5],
        [2.5, 2.0, 0.0],
    ], dtype=float)

    moving_waypoints_a = np.array([
        [2.5, 2.0, 0.0],
        [3.0, 0.0, 0.5],
        [2.0, -3.0, 1.0],
        [-2.0, -2.5, 3.0],
        [-3.0, 1.5, 4.5],
        [0.0, 3.5, 2.0],
        [2.5, 2.0, 0.0],
    ], dtype=float)

    moving_waypoints_b = np.array([
        [2.5, 2.0, 0.0],
        [-2.0, -2.5, 3.0],
        [0.0, 3.5, 2.0],
        [3.0, 0.0, 0.5],
        [2.5, 2.0, 0.0],
    ], dtype=float)

    cylinders = [
        {"cx": 0.5, "cy": 1.0, "r": 0.6, "zmin": 0.0, "zmax": 4.5},
        {"cx": -1.8, "cy": -0.5, "r": 0.7, "zmin": 0.0, "zmax": 5.0},
        {"cx": 1.5, "cy": -1.8, "r": 0.5, "zmin": 0.0, "zmax": 3.0},
        {"cx": -1.0, "cy": 4.0, "r": 0.7, "zmin": 0.0, "zmax": 4.5},
    ]

    traj = build_min_snap_3d(ref_waypoints, avg_speed=1.8)
    moving_a = build_min_snap_3d(moving_waypoints_a + np.array([0.40, -0.30, 0.00], dtype=float), avg_speed=1.8)
    moving_b = build_min_snap_3d(moving_waypoints_b + np.array([-0.25, 0.20, 0.10], dtype=float), avg_speed=1.8)
    return traj, [moving_a, moving_b], cylinders


def build_ref_and_obs(traj, moving_traj, t: float, dt: float, horizon: int, last_yaw_ref: float, lead_time: float):
    ref_seq = np.zeros((horizon + 1, 4), dtype=float)
    ref_seq[0, 0:3], v0 = traj.eval(min(t, traj.total_time))
    psi0_raw = wrap_pi(yaw_follow_from_vel(v0, last_yaw_ref))
    ref_seq[0, 3] = last_yaw_ref + wrap_pi(psi0_raw - last_yaw_ref)

    for k in range(1, horizon + 1):
        tk = min(traj.total_time, t + k * dt)
        pk, vk = traj.eval(tk)
        psi_raw = wrap_pi(yaw_follow_from_vel(vk, ref_seq[k - 1, 3]))
        prev = ref_seq[k - 1, 3]
        ref_seq[k, 0:3] = pk
        ref_seq[k, 3] = prev + wrap_pi(psi_raw - prev)

    obs_seq = np.zeros((horizon + 1, 3), dtype=float)
    for k in range(horizon + 1):
        tk = min(moving_traj.total_time, t + lead_time + k * dt)
        op, _ = moving_traj.eval(tk)
        obs_seq[k] = op

    return ref_seq, obs_seq, float(ref_seq[min(1, horizon), 3])


def sample_candidate(rng: random.Random):
    return {
        "lam": rng.uniform(0.35, 2.20),
        "ang_max_deg": rng.uniform(24.0, 38.0),
        "yawrate_max_deg": rng.uniform(120.0, 240.0),
        "tau_phi": rng.uniform(0.08, 0.22),
        "tau_theta": rng.uniform(0.08, 0.22),
        "phi_rate_max_deg": rng.uniform(150.0, 320.0),
        "theta_rate_max_deg": rng.uniform(150.0, 320.0),
        "sigma_T_ratio": rng.uniform(0.03, 0.18),
        "sigma_phi_deg": rng.uniform(0.8, 8.0),
        "sigma_theta_deg": rng.uniform(0.8, 8.0),
        "sigma_yawrate_deg": rng.uniform(2.0, 20.0),
        "Rd_T": rng.uniform(0.0, 1.0),
        "Rd_roll": rng.uniform(0.0, 6.0),
        "Rd_pitch": rng.uniform(0.0, 6.0),
        "Rd_yaw": rng.uniform(0.0, 1.5),
        "Qx": rng.uniform(30.0, 220.0),
        "Qy": rng.uniform(30.0, 220.0),
        "Qz": rng.uniform(25.0, 260.0),
        "Qpsi": 0.0,
        "Qf_x_scale": rng.uniform(1.5, 6.0),
        "Qf_y_scale": rng.uniform(1.5, 6.0),
        "Qf_z_scale": rng.uniform(1.5, 6.0),
        "Qf_psi_scale": 0.0,
        "lead_time": rng.uniform(0.25, 1.60),
    }


def build_controller(cfg: dict, cylinders, device: str):
    hover = 0.028 * 9.81
    params = DRParams(
        dt=FIXED_DT,
        horizon_steps=FIXED_HORIZON_STEPS,
        rollouts=FIXED_ROLLOUTS,
        iterations=FIXED_ITERATIONS,
        lam=float(cfg["lam"]),
        ang_max=math.radians(float(cfg["ang_max_deg"])),
        yawrate_max=math.radians(float(cfg["yawrate_max_deg"])),
        tau_phi=float(cfg["tau_phi"]),
        tau_theta=float(cfg["tau_theta"]),
        phi_rate_max=math.radians(float(cfg["phi_rate_max_deg"])),
        theta_rate_max=math.radians(float(cfg["theta_rate_max_deg"])),
        sigma=np.array([
            float(cfg["sigma_T_ratio"]) * hover,
            math.radians(float(cfg["sigma_phi_deg"])),
            math.radians(float(cfg["sigma_theta_deg"])),
            math.radians(float(cfg["sigma_yawrate_deg"])),
        ], dtype=np.float32),
        Rd_u=(
            float(cfg["Rd_T"]),
            float(cfg["Rd_roll"]),
            float(cfg["Rd_pitch"]),
            float(cfg["Rd_yaw"]),
        ),
    )
    ctrl = TorchDRMPPIQuadOuter(mass=0.028, g=9.81, params=params, cylinders=cylinders, device=device)
    return ctrl, params


def evaluate_scenario(ctrl, params, traj, moving_traj, cfg: dict, max_steps: int):
    dt = float(params.dt)
    Q = np.array([
        float(cfg["Qx"]),
        float(cfg["Qy"]),
        float(cfg["Qz"]),
        float(cfg["Qpsi"]),
    ], dtype=np.float32)
    Qf = np.array([
        float(cfg["Qf_x_scale"]) * float(cfg["Qx"]),
        float(cfg["Qf_y_scale"]) * float(cfg["Qy"]),
        float(cfg["Qf_z_scale"]) * float(cfg["Qz"]),
        float(cfg["Qf_psi_scale"]) * float(cfg["Qpsi"]),
    ], dtype=np.float32)

    p0, _ = traj.eval(0.0)
    x = np.zeros(9, dtype=float)
    x[0:3] = p0
    last_yaw_ref = float(x[6])
    lead_time = float(cfg["lead_time"])

    pos_err_sq = 0.0
    yaw_err_sq = 0.0
    final_pos_err = 0.0
    max_pos_err = 0.0
    du_sq = 0.0
    solve_ms_sum = 0.0
    u_prev = np.zeros(4, dtype=float)

    sim_T = max(traj.total_time, moving_traj.total_time)
    steps = min(int(sim_T / dt) + 1, int(max_steps))
    if steps < 2:
        return 1e12, {"error": "too_few_steps"}

    for i in range(steps):
        t = i * dt
        ref_seq, obs_seq, last_yaw_ref = build_ref_and_obs(
            traj=traj,
            moving_traj=moving_traj,
            t=t,
            dt=dt,
            horizon=ctrl.T,
            last_yaw_ref=last_yaw_ref,
            lead_time=lead_time,
        )

        if ctrl.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        u = ctrl.plan(x, ref_seq, Q, Qf, obs_seq)
        if ctrl.device.type == "cuda":
            torch.cuda.synchronize()
        solve_ms_sum += (time.perf_counter() - t0) * 1000.0

        Tcmd, phi_cmd, theta_cmd, yawrate = map(float, u)
        Tcmd = clamp(Tcmd, float(ctrl.u_min[0]), float(ctrl.u_max[0]))
        phi_cmd = clamp(phi_cmd, -float(params.ang_max), float(params.ang_max))
        theta_cmd = clamp(theta_cmd, -float(params.ang_max), float(params.ang_max))
        yawrate = clamp(yawrate, -float(params.yawrate_max), float(params.yawrate_max))
        u_cur = np.array([Tcmd, phi_cmd, theta_cmd, yawrate], dtype=float)

        phi = float(x[7])
        theta = float(x[8])
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        cth = math.cos(theta)
        sth = math.sin(theta)
        cpsi = math.cos(x[6])
        spsi = math.sin(x[6])
        zb = np.array([
            cpsi * sth * cphi + spsi * sphi,
            spsi * sth * cphi - cpsi * sphi,
            cth * cphi,
        ], dtype=float)

        a = (Tcmd / ctrl.m) * zb - np.array([0.0, 0.0, ctrl.g], dtype=float)
        phi_dot = (phi_cmd - phi) / max(1e-6, float(params.tau_phi))
        theta_dot = (theta_cmd - theta) / max(1e-6, float(params.tau_theta))
        phi_dot = clamp(phi_dot, -float(params.phi_rate_max), float(params.phi_rate_max))
        theta_dot = clamp(theta_dot, -float(params.theta_rate_max), float(params.theta_rate_max))

        x[3:6] = x[3:6] + dt * a
        x[0:3] = x[0:3] + dt * x[3:6]
        x[6] = wrap_pi(x[6] + dt * yawrate)
        x[7] = clamp(x[7] + dt * phi_dot, -float(params.ang_max), float(params.ang_max))
        x[8] = clamp(x[8] + dt * theta_dot, -float(params.ang_max), float(params.ang_max))

        e_pos = x[0:3] - ref_seq[0, 0:3]
        e_yaw = wrap_pi(x[6] - ref_seq[0, 3])
        pos_err = float(np.linalg.norm(e_pos))
        pos_err_sq += float(np.dot(e_pos, e_pos))
        yaw_err_sq += e_yaw * e_yaw
        max_pos_err = max(max_pos_err, pos_err)
        final_pos_err = pos_err

        if i > 0:
            du = u_cur - u_prev
            du_sq += float(np.dot(du[1:3], du[1:3]))
        u_prev = u_cur

        if not np.all(np.isfinite(x)):
            return 1e12, {"error": "non_finite_state"}

    rmse_pos = math.sqrt(pos_err_sq / steps)
    rmse_yaw = math.sqrt(yaw_err_sq / steps)
    du_rms = math.sqrt(du_sq / max(1, steps - 1))
    mean_solve_ms = solve_ms_sum / steps
    score = (
        2.8 * rmse_pos
        + 1.6 * final_pos_err
        + 0.35 * max_pos_err
        + 0.18 * du_rms
        + 0.04 * rmse_yaw
        + 0.0008 * mean_solve_ms
    )
    metrics = {
        "rmse_pos": rmse_pos,
        "rmse_yaw": rmse_yaw,
        "final_pos_error": final_pos_err,
        "max_pos_error": max_pos_err,
        "du_rms_roll_pitch": du_rms,
        "mean_solve_ms": mean_solve_ms,
        "steps": steps,
    }
    return float(score), metrics


def evaluate_candidate(cfg: dict, device: str, max_steps: int):
    traj, moving_trajs, cylinders = make_world()
    ctrl, params = build_controller(cfg, cylinders, device)

    scenario_scores = []
    scenario_metrics = []
    for moving_traj in moving_trajs:
        score, metrics = evaluate_scenario(ctrl, params, traj, moving_traj, cfg, max_steps)
        if not np.isfinite(score):
            return 1e12, metrics
        scenario_scores.append(float(score))
        scenario_metrics.append(metrics)

    avg_metrics = {
        "rmse_pos": float(np.mean([m["rmse_pos"] for m in scenario_metrics])),
        "rmse_yaw": float(np.mean([m["rmse_yaw"] for m in scenario_metrics])),
        "final_pos_error": float(np.mean([m["final_pos_error"] for m in scenario_metrics])),
        "max_pos_error": float(np.mean([m["max_pos_error"] for m in scenario_metrics])),
        "du_rms_roll_pitch": float(np.mean([m["du_rms_roll_pitch"] for m in scenario_metrics])),
        "mean_solve_ms": float(np.mean([m["mean_solve_ms"] for m in scenario_metrics])),
        "steps": int(np.mean([m["steps"] for m in scenario_metrics])),
        "scenario_scores": scenario_scores,
    }
    return float(np.mean(scenario_scores)), avg_metrics


def main():
    ap = argparse.ArgumentParser(description="Tracking-only autotune for DR_mppi_crazyflie.py")
    ap.add_argument("--trials", type=int, default=120, help="Random candidates to evaluate")
    ap.add_argument("--max-steps", type=int, default=600, help="Max simulated steps per scenario")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--save-json", type=str, default="crazyflie_dr_tracking_autotune_fixed_best.json")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    best_score = float("inf")
    best_cfg = None
    best_metrics = None
    tuned_keys = sorted(sample_candidate(random.Random(args.seed)).keys())

    print("Device:", device)
    print("Tuned keys:", ", ".join(tuned_keys))
    print(
        "Fixed planner params:",
        f"dt={FIXED_DT}, horizon_steps={FIXED_HORIZON_STEPS}, "
        f"rollouts={FIXED_ROLLOUTS}, iterations={FIXED_ITERATIONS}",
    )
    print("Risk params: kept at DRParams defaults in DR_mppi_crazyflie.py")

    for trial in range(1, int(args.trials) + 1):
        cfg = sample_candidate(random)
        score, metrics = evaluate_candidate(cfg, device=device, max_steps=int(args.max_steps))
        print(
            f"[trial {trial:03d}/{args.trials}] score={score:.4f} "
            f"rmse={metrics.get('rmse_pos', float('nan')):.3f} "
            f"final={metrics.get('final_pos_error', float('nan')):.3f} "
            f"max={metrics.get('max_pos_error', float('nan')):.3f} "
            f"du={metrics.get('du_rms_roll_pitch', float('nan')):.3f}"
        )
        if score < best_score:
            best_score = score
            best_cfg = dict(cfg)
            best_metrics = metrics
            print(f"  -> new best score={best_score:.4f}")

    out = {
        "profile": "tracking_only_fixed_planner",
        "best_score": best_score,
        "best_cfg": best_cfg,
        "best_metrics": best_metrics,
        "tuned_keys": tuned_keys,
        "fixed_planner_params": {
            "dt": FIXED_DT,
            "horizon_steps": FIXED_HORIZON_STEPS,
            "rollouts": FIXED_ROLLOUTS,
            "iterations": FIXED_ITERATIONS,
        },
        "risk_params_policy": "Kept at DRParams defaults; not part of the search space.",
        "notes": "Use best_cfg for tracking weights, smoothing, dynamics limits, sampling noise, and lead_time only.",
    }

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nBest configuration:")
    print(json.dumps(out, indent=2))
    print(f"\nSaved to: {args.save_json}")


if __name__ == "__main__":
    main()
