#!/usr/bin/env python3
"""
Tracking-only autotune for Crazyflie MPPI controllers.

Supported modes:
- mppi : MPPI/mppi_crazyflie.py
- dr   : MPPI/DR_mppi_crazyflie.py
- ra   : MPPI/RA_mppi_crazyflie.py
- dra  : MPPI/DRA_mppi_crazyflie.py

Important:
- This script tunes tracking-related parameters only.
- Risk parameters are kept fixed at each controller dataclass defaults.

python3 MPPI/autotune/autotune_crazyflie_tracking.py \
  --mode dr \
  --trials 2000 \
  --max-steps 500 \
  --device auto \
  --seed 7 \
  --save-json crazyflie_dr_tracking_autotune_best.json

"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch


MPPI_DIR = Path(__file__).resolve().parents[1]
if str(MPPI_DIR) not in sys.path:
    sys.path.insert(0, str(MPPI_DIR))

from mppi_crazyflie import MPPIParams, TorchMPPIQuadOuter, build_min_snap_3d  # noqa: E402
from DR_mppi_crazyflie import DRParams, TorchDRMPPIQuadOuter  # noqa: E402
from RA_mppi_crazyflie import Params as RAParams, TorchRAQuad  # noqa: E402
from DRA_mppi_crazyflie import DRAParams, TorchDRAMPPIQuadOuter  # noqa: E402


FIXED_TRACKING_PARAMS = {
    "dr": {
        "horizon_steps": 30,
        "rollouts": 1096,
    }
}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_follow_from_vel(v: np.ndarray, fallback: float) -> float:
    vx, vy = float(v[0]), float(v[1])
    if vx * vx + vy * vy < 1e-6:
        return fallback
    return math.atan2(vy, vx)


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

    return ref_seq, obs_seq, float(ref_seq[1, 3])


def sample_candidate(rng: random.Random, mode: str):
    cfg = {
        "dt": rng.uniform(0.015, 0.030),
        "horizon_steps": rng.choice([35, 40, 50, 60, 70]),
        "rollouts": rng.choice([512, 768, 1096, 1536, 2048]),
        "iterations": rng.choice([1, 2, 3]),
        "lam": rng.uniform(0.4, 2.0),
        "ang_max_deg": rng.uniform(25.0, 45.0),
        "yawrate_max_deg": rng.uniform(120.0, 260.0),
        "tau_phi": rng.uniform(0.08, 0.24),
        "tau_theta": rng.uniform(0.08, 0.24),
        "phi_rate_max_deg": rng.uniform(140.0, 360.0),
        "theta_rate_max_deg": rng.uniform(140.0, 360.0),
        "Rd_T": rng.uniform(0.0, 1.0),
        "Rd_roll": rng.uniform(0.0, 6.0),
        "Rd_pitch": rng.uniform(0.0, 6.0),
        "Rd_yaw": rng.uniform(0.0, 2.0),
        "Qx": rng.uniform(20.0, 150.0),
        "Qy": rng.uniform(20.0, 150.0),
        "Qz": rng.uniform(10.0, 200.0),
        "Qpsi": 0.0,
        "Qf_x_scale": rng.uniform(1.5, 6.0),
        "Qf_y_scale": rng.uniform(1.5, 6.0),
        "Qf_z_scale": rng.uniform(1.5, 6.0),
        "Qf_psi_scale": 0.0,
        "lead_time": rng.uniform(0.3, 1.8),
    }
    if mode in ("mppi", "dr", "dra"):
        cfg.update({
            "sigma_T_ratio": rng.uniform(0.02, 0.20),
            "sigma_phi_deg": rng.uniform(0.4, 10.0),
            "sigma_theta_deg": rng.uniform(0.4, 10.0),
            "sigma_yawrate_deg": rng.uniform(1.0, 45.0),
        })
    cfg.update(FIXED_TRACKING_PARAMS.get(mode, {}))
    return cfg


def tuned_keys_for_mode(mode: str) -> list[str]:
    keys = sorted(sample_candidate(random.Random(0), mode).keys())
    fixed = set(FIXED_TRACKING_PARAMS.get(mode, {}).keys())
    return [k for k in keys if k not in fixed]


def build_controller(mode: str, cfg: dict, cylinders, device: str):
    hover = 0.028 * 9.81

    if mode == "mppi":
        p = MPPIParams()
        p.dt = float(cfg["dt"])
        p.horizon_steps = int(cfg["horizon_steps"])
        p.rollouts = int(cfg["rollouts"])
        p.iterations = int(cfg["iterations"])
        p.lam = float(cfg["lam"])
        p.ang_max = math.radians(float(cfg["ang_max_deg"]))
        p.yawrate_max = math.radians(float(cfg["yawrate_max_deg"]))
        p.tau_phi = float(cfg["tau_phi"])
        p.tau_theta = float(cfg["tau_theta"])
        p.phi_rate_max = math.radians(float(cfg["phi_rate_max_deg"]))
        p.theta_rate_max = math.radians(float(cfg["theta_rate_max_deg"]))
        p.Rd_u = (float(cfg["Rd_T"]), float(cfg["Rd_roll"]), float(cfg["Rd_pitch"]), float(cfg["Rd_yaw"]))
        p.sigma = np.array([
            float(cfg["sigma_T_ratio"]) * hover,
            math.radians(float(cfg["sigma_phi_deg"])),
            math.radians(float(cfg["sigma_theta_deg"])),
            math.radians(float(cfg["sigma_yawrate_deg"])),
        ], dtype=np.float32)
        ctrl = TorchMPPIQuadOuter(mass=0.028, g=9.81, params=p, cylinders=cylinders, device=device)
        return ctrl, p

    if mode == "dr":
        p = DRParams()
        p.dt = float(cfg["dt"])
        p.horizon_steps = int(cfg["horizon_steps"])
        p.rollouts = int(cfg["rollouts"])
        p.iterations = int(cfg["iterations"])
        p.lam = float(cfg["lam"])
        p.ang_max = math.radians(float(cfg["ang_max_deg"]))
        p.yawrate_max = math.radians(float(cfg["yawrate_max_deg"]))
        p.tau_phi = float(cfg["tau_phi"])
        p.tau_theta = float(cfg["tau_theta"])
        p.phi_rate_max = math.radians(float(cfg["phi_rate_max_deg"]))
        p.theta_rate_max = math.radians(float(cfg["theta_rate_max_deg"]))
        p.Rd_u = (float(cfg["Rd_T"]), float(cfg["Rd_roll"]), float(cfg["Rd_pitch"]), float(cfg["Rd_yaw"]))
        p.sigma = np.array([
            float(cfg["sigma_T_ratio"]) * hover,
            math.radians(float(cfg["sigma_phi_deg"])),
            math.radians(float(cfg["sigma_theta_deg"])),
            math.radians(float(cfg["sigma_yawrate_deg"])),
        ], dtype=np.float32)
        # Keep all DR-specific risk params untouched.
        ctrl = TorchDRMPPIQuadOuter(mass=0.028, g=9.81, params=p, cylinders=cylinders, device=device)
        return ctrl, p

    if mode == "ra":
        p = RAParams()
        p.dt = float(cfg["dt"])
        p.horizon_steps = int(cfg["horizon_steps"])
        p.rollouts = int(cfg["rollouts"])
        p.iterations = int(cfg["iterations"])
        p.lam = float(cfg["lam"])
        p.ang_max = math.radians(float(cfg["ang_max_deg"]))
        p.yawrate_max = math.radians(float(cfg["yawrate_max_deg"]))
        p.tau_phi = float(cfg["tau_phi"])
        p.tau_theta = float(cfg["tau_theta"])
        p.phi_rate_max = math.radians(float(cfg["phi_rate_max_deg"]))
        p.theta_rate_max = math.radians(float(cfg["theta_rate_max_deg"]))
        p.Rd_u = (float(cfg["Rd_T"]), float(cfg["Rd_roll"]), float(cfg["Rd_pitch"]), float(cfg["Rd_yaw"]))
        # Keep all RA-specific risk params untouched.
        ctrl = TorchRAQuad(mass=0.028, g=9.81, params=p, cylinders=cylinders, device=device)
        return ctrl, p

    if mode == "dra":
        p = DRAParams()
        p.dt = float(cfg["dt"])
        p.horizon_steps = int(cfg["horizon_steps"])
        p.rollouts = int(cfg["rollouts"])
        p.iterations = int(cfg["iterations"])
        p.lam = float(cfg["lam"])
        p.ang_max = math.radians(float(cfg["ang_max_deg"]))
        p.yawrate_max = math.radians(float(cfg["yawrate_max_deg"]))
        p.tau_phi = float(cfg["tau_phi"])
        p.tau_theta = float(cfg["tau_theta"])
        p.phi_rate_max = math.radians(float(cfg["phi_rate_max_deg"]))
        p.theta_rate_max = math.radians(float(cfg["theta_rate_max_deg"]))
        p.Rd_u = (float(cfg["Rd_T"]), float(cfg["Rd_roll"]), float(cfg["Rd_pitch"]), float(cfg["Rd_yaw"]))
        p.sigma = np.array([
            float(cfg["sigma_T_ratio"]) * hover,
            math.radians(float(cfg["sigma_phi_deg"])),
            math.radians(float(cfg["sigma_theta_deg"])),
            math.radians(float(cfg["sigma_yawrate_deg"])),
        ], dtype=np.float32)
        # Keep all DRA-specific risk params untouched.
        ctrl = TorchDRAMPPIQuadOuter(mass=0.028, g=9.81, params=p, cylinders=cylinders, device=device)
        return ctrl, p

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_candidate(
    mode: str,
    cfg: dict,
    device: str,
    max_steps: int,
):
    traj, moving_traj, cylinders = make_world()
    ctrl, p = build_controller(mode, cfg, cylinders, device)

    Q = np.array([float(cfg["Qx"]), float(cfg["Qy"]), float(cfg["Qz"]), float(cfg["Qpsi"])], dtype=np.float32)
    Qf = np.array([
        float(cfg["Qf_x_scale"]) * float(cfg["Qx"]),
        float(cfg["Qf_y_scale"]) * float(cfg["Qy"]),
        float(cfg["Qf_z_scale"]) * float(cfg["Qz"]),
        float(cfg["Qf_psi_scale"]) * float(cfg["Qpsi"]),
    ], dtype=np.float32)
    lead_time = float(cfg["lead_time"])
    dt = float(p.dt)

    p0, _ = traj.eval(0.0)
    x = np.zeros(9, dtype=float)
    x[0:3] = p0
    x[7] = 0.0
    x[8] = 0.0
    last_yaw_ref = float(x[6])

    pos_err_sq = 0.0
    yaw_err_sq = 0.0
    final_err = 0.0
    du_sq = 0.0
    u_prev = np.zeros(4, dtype=float)

    sim_T = max(traj.total_time, moving_traj.total_time)
    steps = min(int(sim_T / dt) + 1, int(max_steps))

    for i in range(steps):
        t = i * dt
        ref_seq, obs_seq, last_yaw_ref = build_ref_and_obs(
            traj, moving_traj, t, dt, ctrl.T, last_yaw_ref, lead_time
        )

        u = ctrl.plan(x, ref_seq, Q, Qf, obs_seq)

        Tcmd, phi_cmd, theta_cmd, yawrate = map(float, u)
        Tcmd = clamp(Tcmd, float(ctrl.u_min[0]), float(ctrl.u_max[0]))
        phi_cmd = clamp(phi_cmd, -float(p.ang_max), float(p.ang_max))
        theta_cmd = clamp(theta_cmd, -float(p.ang_max), float(p.ang_max))
        yawrate = clamp(yawrate, -float(p.yawrate_max), float(p.yawrate_max))
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
        phi_dot = (phi_cmd - phi) / max(1e-6, float(p.tau_phi))
        theta_dot = (theta_cmd - theta) / max(1e-6, float(p.tau_theta))
        phi_dot = clamp(phi_dot, -float(p.phi_rate_max), float(p.phi_rate_max))
        theta_dot = clamp(theta_dot, -float(p.theta_rate_max), float(p.theta_rate_max))

        x[3:6] = x[3:6] + dt * a
        x[0:3] = x[0:3] + dt * x[3:6]
        x[6] = wrap_pi(x[6] + dt * yawrate)
        x[7] = clamp(x[7] + dt * phi_dot, -float(p.ang_max), float(p.ang_max))
        x[8] = clamp(x[8] + dt * theta_dot, -float(p.ang_max), float(p.ang_max))

        e_pos = x[0:3] - ref_seq[0, 0:3]
        e_psi = wrap_pi(x[6] - ref_seq[0, 3])
        pos_err_sq += float(np.dot(e_pos, e_pos))
        yaw_err_sq += float(e_psi * e_psi)
        if i > 0:
            du = u_cur - u_prev
            du_sq += float(np.dot(du[1:4], du[1:4]))
        u_prev = u_cur

        if i == steps - 1:
            final_err = float(np.linalg.norm(e_pos))

    if steps < 2:
        return 1e9, {}

    rmse_pos = math.sqrt(pos_err_sq / steps)
    rmse_yaw = math.sqrt(yaw_err_sq / steps)
    du_rms = math.sqrt(du_sq / max(1, steps - 1))

    # Tracking-only objective (lower is better).
    score = (
        1.0 * rmse_pos
        + 0.15 * rmse_yaw
        + 0.20 * du_rms
        + 0.30 * final_err
    )
    metrics = {
        "rmse_pos": rmse_pos,
        "rmse_yaw": rmse_yaw,
        "final_pos_error": final_err,
        "du_rms": du_rms,
        "steps": steps,
    }
    return float(score), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="dr", choices=["mppi", "dr", "ra", "dra"])
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--save-json", type=str, default="")
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

    if args.save_json.strip():
        save_json = args.save_json
    else:
        save_json = f"crazyflie_{args.mode}_tracking_autotune_best.json"

    best_score = float("inf")
    best_cfg = None
    best_metrics = None
    tuned_keys = tuned_keys_for_mode(args.mode)
    print(f"Mode: {args.mode}")
    print("Tracking tuning keys:", ", ".join(tuned_keys))
    print("Risk parameters are fixed to controller dataclass defaults.")
    if args.mode in FIXED_TRACKING_PARAMS:
        print(f"Fixed tracking params: {FIXED_TRACKING_PARAMS[args.mode]}")

    for trial in range(1, int(args.trials) + 1):
        cfg = sample_candidate(random, args.mode)
        score, metrics = evaluate_candidate(
            mode=args.mode,
            cfg=cfg,
            device=dev,
            max_steps=int(args.max_steps),
        )
        print(
            f"[trial {trial:03d}/{args.trials}] score={score:.4f} "
            f"rmse_pos={metrics.get('rmse_pos', float('nan')):.3f} "
            f"rmse_yaw={metrics.get('rmse_yaw', float('nan')):.3f} "
            f"du={metrics.get('du_rms', float('nan')):.3f}"
        )
        if score < best_score:
            best_score = score
            best_cfg = cfg
            best_metrics = metrics
            print(f"  -> new best score={best_score:.4f}")

    out = {
        "mode": args.mode,
        "best_score": best_score,
        "best_cfg": best_cfg,
        "best_metrics": best_metrics,
        "tuned_keys": tuned_keys,
        "fixed_tracking_params": FIXED_TRACKING_PARAMS.get(args.mode, {}),
        "objective": "tracking_only",
        "notes": "Tracking params tuned without solve-time penalties; DR/RA/DRA risk params kept at dataclass defaults.",
    }
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nBest configuration:")
    print(json.dumps(out, indent=2))
    print(f"\nSaved to: {save_json}")


if __name__ == "__main__":
    main()
