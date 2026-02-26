#!/usr/bin/env python3
"""
Autotune DR-MPPI parameters for MPPI/DR_mppi_crazyflie.py using headless rollouts.

This script does NOT plot. It runs repeated simulations, scores each candidate, and
saves the best configuration to JSON.
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


# Make MPPI/ importable regardless of current working directory
MPPI_DIR = Path(__file__).resolve().parents[1]
if str(MPPI_DIR) not in sys.path:
    sys.path.insert(0, str(MPPI_DIR))

from DR_mppi_crazyflie import DRParams, TorchDRMPPIQuadOuter, build_min_snap_3d, clamp, wrap_pi  # noqa: E402


def make_world():
    waypoints = np.array([
        [2.5, 2.0, 0.0],
        [0.0, 3.5, 2.0],
        [-3.0, 1.5, 4.5],
        [-2.0, -2.5, 3.0],
        [2.0, -3.0, 1.0],
        [3.0, 0.0, 0.5],
        [2.5, 2.0, 0.0],
    ], dtype=float)

    waypoints_1 = np.array([
        [2.5, 2.0, 0.0],
        [3.0, 0.0, 0.5],
        [2.0, -3.0, 1.0],
        [-2.0, -2.5, 3.0],
        [-3.0, 1.5, 4.5],
        [0.0, 3.5, 2.0],
        [2.5, 2.0, 0.0],
    ], dtype=float)

    cylinders = [
        {"cx": 0.5, "cy": 1.0, "r": 0.6, "zmin": 0.0, "zmax": 4.5},
        {"cx": -1.8, "cy": -0.5, "r": 0.7, "zmin": 0.0, "zmax": 5.0},
        {"cx": 1.5, "cy": -1.8, "r": 0.5, "zmin": 0.0, "zmax": 3.0},
        {"cx": -1.0, "cy": 4.0, "r": 0.7, "zmin": 0.0, "zmax": 4.5},
    ]

    traj = build_min_snap_3d(waypoints, avg_speed=1.8)
    moving_traj = build_min_snap_3d(waypoints_1 + np.array([0.40, -0.30, 0.00], dtype=float), avg_speed=1.8)
    return traj, moving_traj, cylinders


def yaw_follow_from_vel(v, fallback):
    vx, vy = float(v[0]), float(v[1])
    if vx * vx + vy * vy < 1e-6:
        return fallback
    return math.atan2(vy, vx)


def build_ref_and_obs(traj, moving_traj, t, dt, horizon, last_yaw_ref, lead_time):
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

    return ref_seq, obs_seq, float(ref_seq[1, 3])


def sample_candidate(rng: random.Random):
    rollouts_choices = [512, 768, 1096, 1536, 2048]
    iterations_choices = [1, 2, 3]
    horizon_choices = [40, 50, 60, 70, 80]
    cvar_N_choices = [32, 48, 64, 96]
    noise_mode_choices = ["static", "per_step"]
    return {
        # DRParams: timing / horizon / solver
        "dt": rng.uniform(0.015, 0.035),
        "horizon_steps": horizon_choices[rng.randrange(len(horizon_choices))],
        "rollouts": rollouts_choices[rng.randrange(len(rollouts_choices))],
        "iterations": iterations_choices[rng.randrange(len(iterations_choices))],
        "lam": rng.uniform(0.3, 2.5),
        # bounds / attitude
        "ang_max_deg": rng.uniform(20.0, 45.0),
        "yawrate_max_deg": rng.uniform(120.0, 260.0),
        "tau_phi": rng.uniform(0.08, 0.28),
        "tau_theta": rng.uniform(0.08, 0.28),
        "phi_rate_max_deg": rng.uniform(120.0, 360.0),
        "theta_rate_max_deg": rng.uniform(120.0, 360.0),
        # running soft costs
        "w_cyl": rng.uniform(100.0, 900.0),
        "cyl_safety_margin": rng.uniform(0.10, 0.60),
        "cyl_alpha": rng.uniform(4.0, 18.0),
        "w_moving": rng.uniform(200.0, 1800.0),
        "moving_r": rng.uniform(0.20, 0.60),
        "moving_safety_margin": rng.uniform(0.20, 1.00),
        "moving_alpha": rng.uniform(4.0, 20.0),
        # DR feasibility
        "cvar_alpha": rng.uniform(0.75, 0.95),
        "cvar_N": cvar_N_choices[rng.randrange(len(cvar_N_choices))],
        "obs_sigma_xy": (rng.uniform(0.10, 0.45), rng.uniform(0.10, 0.45)),
        "noise_mode": noise_mode_choices[rng.randrange(len(noise_mode_choices))],
        "dr_eps_cvar": rng.uniform(0.0, 0.35),
        "drone_radius": rng.uniform(0.15, 0.40),
        # control noise (sigma)
        "sigma_T_ratio": rng.uniform(0.02, 0.20),
        "sigma_phi_deg": rng.uniform(0.2, 8.0),
        "sigma_theta_deg": rng.uniform(0.2, 8.0),
        "sigma_yawrate_deg": rng.uniform(1.0, 45.0),
        # smoothing
        "Rd_T": rng.uniform(0.0, 1.0),
        "Rd_roll": rng.uniform(0.0, 15.0),
        "Rd_pitch": rng.uniform(0.0, 15.0),
        "Rd_yaw": rng.uniform(0.0, 2.0),
        # Q / Qf / lead time
        "Qx": rng.uniform(10.0, 120.0),
        "Qy": rng.uniform(10.0, 120.0),
        "Qz": rng.uniform(10.0, 220.0),
        "Qpsi": 0.0,
        "Qf_x_scale": rng.uniform(2.0, 6.0),
        "Qf_y_scale": rng.uniform(2.0, 6.0),
        "Qf_z_scale": rng.uniform(2.0, 6.0),
        "Qf_psi_scale": 0.0,
        "lead_time": rng.uniform(0.3, 1.8),
    }


def evaluate_candidate(cfg, device: str, max_steps: int):
    traj, moving_traj, cylinders = make_world()
    dt = float(cfg["dt"])
    hover = 0.028 * 9.81
    params = DRParams(
        dt=dt,
        horizon_steps=int(cfg["horizon_steps"]),
        rollouts=int(cfg["rollouts"]),
        iterations=int(cfg["iterations"]),
        lam=float(cfg["lam"]),
        ang_max=math.radians(float(cfg["ang_max_deg"])),
        yawrate_max=math.radians(float(cfg["yawrate_max_deg"])),
        tau_phi=float(cfg["tau_phi"]),
        tau_theta=float(cfg["tau_theta"]),
        phi_rate_max=math.radians(float(cfg["phi_rate_max_deg"])),
        theta_rate_max=math.radians(float(cfg["theta_rate_max_deg"])),
        w_cyl=float(cfg["w_cyl"]),
        cyl_safety_margin=float(cfg["cyl_safety_margin"]),
        cyl_alpha=float(cfg["cyl_alpha"]),
        w_moving=float(cfg["w_moving"]),
        moving_r=float(cfg["moving_r"]),
        moving_safety_margin=float(cfg["moving_safety_margin"]),
        moving_alpha=float(cfg["moving_alpha"]),
        cvar_alpha=float(cfg["cvar_alpha"]),
        cvar_N=int(cfg["cvar_N"]),
        obs_pos_sigma_xy=(float(cfg["obs_sigma_xy"][0]), float(cfg["obs_sigma_xy"][1])),
        noise_mode=str(cfg["noise_mode"]),
        dr_eps_cvar=float(cfg["dr_eps_cvar"]),
        drone_radius=float(cfg["drone_radius"]),
        sigma=np.array(
            [
                float(cfg["sigma_T_ratio"]) * hover,
                math.radians(float(cfg["sigma_phi_deg"])),
                math.radians(float(cfg["sigma_theta_deg"])),
                math.radians(float(cfg["sigma_yawrate_deg"])),
            ],
            dtype=np.float32,
        ),
        Rd_u=(float(cfg["Rd_T"]), float(cfg["Rd_roll"]), float(cfg["Rd_pitch"]), float(cfg["Rd_yaw"])),
    )

    ctrl = TorchDRMPPIQuadOuter(mass=0.028, g=9.81, params=params, cylinders=cylinders, device=device)

    Q = np.array([float(cfg["Qx"]), float(cfg["Qy"]), float(cfg["Qz"]), float(cfg["Qpsi"])], dtype=np.float32)
    Qf = np.array([
        float(cfg["Qf_x_scale"]) * float(cfg["Qx"]),
        float(cfg["Qf_y_scale"]) * float(cfg["Qy"]),
        float(cfg["Qf_z_scale"]) * float(cfg["Qz"]),
        float(cfg["Qf_psi_scale"]) * float(cfg["Qpsi"]),
    ], dtype=np.float32)

    p0, _ = traj.eval(0.0)
    x = np.zeros(9, dtype=float)
    x[0:3] = p0
    x[7] = 0.0
    x[8] = 0.0
    last_yaw_ref = float(x[6])
    lead_time = float(cfg["lead_time"])

    pos_err_sq = 0.0
    du_sq = 0.0
    solve_ms_sum = 0.0
    safety_viol = 0.0
    u_prev = np.zeros(4, dtype=float)

    sim_T = max(traj.total_time, moving_traj.total_time)
    steps = min(int(sim_T / dt) + 1, int(max_steps))

    for i in range(steps):
        t = i * dt
        ref_seq, obs_seq, last_yaw_ref = build_ref_and_obs(
            traj, moving_traj, t, dt, ctrl.T, last_yaw_ref, lead_time
        )

        if ctrl.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        u = ctrl.plan(x, ref_seq, Q, Qf, obs_seq)
        if ctrl.device.type == "cuda":
            torch.cuda.synchronize()
        solve_ms_sum += (time.perf_counter() - t0) * 1000.0

        Tcmd, phi_cmd, theta_cmd, yawrate = map(float, u)
        Tcmd = clamp(Tcmd, ctrl.u_min[0], ctrl.u_max[0])
        phi_cmd = clamp(phi_cmd, -params.ang_max, params.ang_max)
        theta_cmd = clamp(theta_cmd, -params.ang_max, params.ang_max)
        yawrate = clamp(yawrate, -params.yawrate_max, params.yawrate_max)
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
        phi_dot = (phi_cmd - phi) / max(1e-6, params.tau_phi)
        theta_dot = (theta_cmd - theta) / max(1e-6, params.tau_theta)
        phi_dot = clamp(phi_dot, -params.phi_rate_max, params.phi_rate_max)
        theta_dot = clamp(theta_dot, -params.theta_rate_max, params.theta_rate_max)
        x[3:6] = x[3:6] + dt * a
        x[0:3] = x[0:3] + dt * x[3:6]
        x[6] = wrap_pi(x[6] + dt * yawrate)
        x[7] = clamp(x[7] + dt * phi_dot, -params.ang_max, params.ang_max)
        x[8] = clamp(x[8] + dt * theta_dot, -params.ang_max, params.ang_max)

        e = x[0:3] - ref_seq[0, 0:3]
        pos_err_sq += float(np.dot(e, e))
        if i > 0:
            du = u_cur - u_prev
            du_sq += float(du[1] * du[1] + du[2] * du[2])
        u_prev = u_cur

        # moving obstacle safety violation proxy
        dmov = np.linalg.norm(x[0:3] - obs_seq[0])
        safe = params.moving_r + params.moving_safety_margin + params.drone_radius
        safety_viol += max(0.0, safe - dmov) ** 2

    if steps < 2:
        return 1e9, {}

    rmse = math.sqrt(pos_err_sq / steps)
    du_rms = math.sqrt(du_sq / max(1, steps - 1))
    mean_solve_ms = solve_ms_sum / steps

    # Lower is better
    score = (1.0 * rmse) + (0.25 * du_rms) + (2.0 * safety_viol / steps) + (0.002 * mean_solve_ms)
    metrics = {
        "rmse_pos": rmse,
        "du_rms_roll_pitch": du_rms,
        "mean_solve_ms": mean_solve_ms,
        "safety_violation_mean": safety_viol / steps,
        "steps": steps,
    }
    return float(score), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100, help="Number of random candidates.")
    ap.add_argument("--max-steps", type=int, default=400, help="Max sim steps per evaluation.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed.")
    ap.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--save-json", type=str, default="dr_mppi_autotune_best.json")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    best_score = float("inf")
    best_cfg = None
    best_metrics = None
    tuned_keys = sorted(sample_candidate(random.Random(args.seed)).keys())
    print("Tuning keys:", ", ".join(tuned_keys))

    for trial in range(1, int(args.trials) + 1):
        cfg = sample_candidate(random)
        score, metrics = evaluate_candidate(cfg, device=dev, max_steps=int(args.max_steps))
        print(
            f"[trial {trial:03d}/{args.trials}] score={score:.4f} "
            f"rmse={metrics.get('rmse_pos', float('nan')):.3f} "
            f"du={metrics.get('du_rms_roll_pitch', float('nan')):.3f} "
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
        "notes": "Use best_cfg to populate DRParams + Q/Qf + lead_time in DR_mppi_crazyflie.py",
    }
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nBest configuration:")
    print(json.dumps(out, indent=2))
    print(f"\nSaved to: {args.save_json}")


if __name__ == "__main__":
    main()
