#!/usr/bin/env python3
"""
Autotune tracking and DR-risk parameters for MPPI_final/DR_mppi_crazyflie.py.

This script intentionally tunes only:
- tracking/controller terms:
    lam, sigma, R_u, Rd_u, Q, Qf, lead_time
- DR risk terms:
    cvar_alpha, cvar_N, obs_pos_sigma_xy, noise_mode, dr_eps_cvar

Everything else is kept fixed:
- planner timing/sample settings
- attitude/dynamics limits and response
- obstacle soft-cost weights and margins
- drone geometry

Example:
python3 MPPI_final/autotune/autotune_dr_mppi_crazyflie_tracking_dr.py \
  --trials 300 \
  --max-steps 500 \
  --device auto \
  --seed 7 \
  --save-json MPPI_final/crazyflie_dr_tracking_dr_autotune_best.json
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

MASS = 0.028
GRAVITY = 9.81
HOVER = MASS * GRAVITY


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_follow_from_vel(v: np.ndarray, fallback: float) -> float:
    vx, vy = float(v[0]), float(v[1])
    if vx * vx + vy * vy < 1e-6:
        return fallback
    return math.atan2(vy, vx)


def set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sample_loguniform(rng: random.Random, low: float, high: float) -> float:
    return 10.0 ** rng.uniform(math.log10(low), math.log10(high))


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

    scenarios = [
        ("moving_a", [moving_a]),
        ("moving_b", [moving_b]),
        ("moving_pair", [moving_a, moving_b]),
    ]
    return traj, scenarios, cylinders


def build_ref_and_obs(
    traj,
    moving_trajs,
    t_now: float,
    dt: float,
    horizon: int,
    last_yaw_ref: float,
    lead_time: float,
):
    ref_seq = np.zeros((horizon + 1, 4), dtype=float)
    ref_seq[0, 0:3], v0 = traj.eval(min(t_now, traj.total_time))
    psi0_raw = wrap_pi(yaw_follow_from_vel(v0, last_yaw_ref))
    ref_seq[0, 3] = last_yaw_ref + wrap_pi(psi0_raw - last_yaw_ref)

    for k in range(1, horizon + 1):
        tk = min(traj.total_time, t_now + k * dt)
        pk, vk = traj.eval(tk)
        psi_raw = wrap_pi(yaw_follow_from_vel(vk, ref_seq[k - 1, 3]))
        prev = ref_seq[k - 1, 3]
        ref_seq[k, 0:3] = pk
        ref_seq[k, 3] = prev + wrap_pi(psi_raw - prev)

    n_obs = len(moving_trajs)
    if n_obs == 1:
        obs_seq = np.zeros((horizon + 1, 3), dtype=float)
        for k in range(horizon + 1):
            tk = min(moving_trajs[0].total_time, t_now + lead_time + k * dt)
            op, _ = moving_trajs[0].eval(tk)
            obs_seq[k] = op
    else:
        obs_seq = np.zeros((horizon + 1, n_obs, 3), dtype=float)
        for j, moving_traj in enumerate(moving_trajs):
            for k in range(horizon + 1):
                tk = min(moving_traj.total_time, t_now + lead_time + k * dt)
                op, _ = moving_traj.eval(tk)
                obs_seq[k, j] = op

    return ref_seq, obs_seq, float(ref_seq[min(1, horizon), 3])


def sample_candidate(rng: random.Random) -> dict:
    return {
        "lam": rng.uniform(0.35, 2.20),
        "sigma_T_ratio": rng.uniform(0.03, 0.18),
        "sigma_phi_deg": rng.uniform(0.8, 8.0),
        "sigma_theta_deg": rng.uniform(0.8, 8.0),
        "sigma_yawrate_deg": rng.uniform(2.0, 20.0),
        "R_T": sample_loguniform(rng, 200.0, 8000.0),
        "R_roll": sample_loguniform(rng, 20.0, 4000.0),
        "R_pitch": sample_loguniform(rng, 20.0, 4000.0),
        "R_yaw": sample_loguniform(rng, 5.0, 150.0),
        "Rd_T": rng.uniform(0.0, 1.0),
        "Rd_roll": rng.uniform(0.0, 6.0),
        "Rd_pitch": rng.uniform(0.0, 6.0),
        "Rd_yaw": rng.uniform(0.0, 1.5),
        "Qx": rng.uniform(25.0, 220.0),
        "Qy": rng.uniform(25.0, 220.0),
        "Qz": rng.uniform(25.0, 260.0),
        "Qf_x_scale": rng.uniform(1.5, 6.0),
        "Qf_y_scale": rng.uniform(1.5, 6.0),
        "Qf_z_scale": rng.uniform(1.5, 6.0),
        "lead_time": rng.uniform(0.25, 1.60),
        "cvar_alpha": rng.uniform(0.72, 0.97),
        "cvar_N": rng.choice([24, 32, 48, 64, 96]),
        "obs_sigma_x": rng.uniform(0.08, 0.35),
        "obs_sigma_y": rng.uniform(0.08, 0.35),
        "noise_mode": rng.choice(["static", "per_step"]),
        "dr_eps_cvar": rng.uniform(0.0, 0.08),
    }


def build_controller(cfg: dict, cylinders, device: str):
    p = DRParams()
    p.dt = FIXED_DT
    p.horizon_steps = FIXED_HORIZON_STEPS
    p.rollouts = FIXED_ROLLOUTS

    p.lam = float(cfg["lam"])
    p.sigma = np.array([
        float(cfg["sigma_T_ratio"]) * HOVER,
        math.radians(float(cfg["sigma_phi_deg"])),
        math.radians(float(cfg["sigma_theta_deg"])),
        math.radians(float(cfg["sigma_yawrate_deg"])),
    ], dtype=np.float32)
    p.R_u = (
        float(cfg["R_T"]),
        float(cfg["R_roll"]),
        float(cfg["R_pitch"]),
        float(cfg["R_yaw"]),
    )
    p.Rd_u = (
        float(cfg["Rd_T"]),
        float(cfg["Rd_roll"]),
        float(cfg["Rd_pitch"]),
        float(cfg["Rd_yaw"]),
    )

    p.cvar_alpha = float(cfg["cvar_alpha"])
    p.cvar_N = int(cfg["cvar_N"])
    p.obs_pos_sigma_xy = (float(cfg["obs_sigma_x"]), float(cfg["obs_sigma_y"]))
    p.noise_mode = str(cfg["noise_mode"])
    p.dr_eps_cvar = float(cfg["dr_eps_cvar"])

    ctrl = TorchDRMPPIQuadOuter(mass=MASS, g=GRAVITY, params=p, cylinders=cylinders, device=device)
    return ctrl, p


def reset_controller(ctrl) -> None:
    hover = float(ctrl.m * ctrl.g)
    ctrl.mppi.U_cpu[:] = 0.0
    ctrl.mppi.U_cpu[:, 0] = hover
    ctrl.mppi.U = ctrl.mppi.U_cpu


def apply_control_step(ctrl, params: DRParams, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt = float(params.dt)
    Tcmd, phi_cmd, theta_cmd, yawrate = map(float, u)
    Tcmd = clamp(Tcmd, float(ctrl.u_min[0]), float(ctrl.u_max[0]))
    phi_cmd = clamp(phi_cmd, -float(params.ang_max), float(params.ang_max))
    theta_cmd = clamp(theta_cmd, -float(params.ang_max), float(params.ang_max))
    yawrate = clamp(yawrate, -float(params.yawrate_max), float(params.yawrate_max))
    u_cur = np.array([Tcmd, phi_cmd, theta_cmd, yawrate], dtype=float)

    phi = float(x[7])
    theta = float(x[8])
    psi = float(x[6])
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)
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

    x_next = np.asarray(x, dtype=float).copy()
    x_next[3:6] = x_next[3:6] + dt * a
    x_next[0:3] = x_next[0:3] + dt * x_next[3:6]
    x_next[6] = wrap_pi(x_next[6] + dt * yawrate)
    x_next[7] = clamp(x_next[7] + dt * phi_dot, -float(params.ang_max), float(params.ang_max))
    x_next[8] = clamp(x_next[8] + dt * theta_dot, -float(params.ang_max), float(params.ang_max))
    return x_next, u_cur


def step_clearances(x: np.ndarray, obs_now: np.ndarray, params: DRParams, cylinders) -> tuple[float, float]:
    p = np.asarray(x[0:3], dtype=float)
    obs_now = np.asarray(obs_now, dtype=float)
    if obs_now.ndim == 1:
        obs_now = obs_now.reshape(1, 3)

    min_clearance = float("inf")
    min_safety_clearance = float("inf")

    for obs_pos in obs_now:
        d = float(np.linalg.norm(p - obs_pos))
        clear = d - (float(params.moving_r) + float(params.drone_radius))
        safe_clear = clear - float(params.moving_safety_margin)
        min_clearance = min(min_clearance, clear)
        min_safety_clearance = min(min_safety_clearance, safe_clear)

    pz = float(p[2])
    for cyl in cylinders:
        zmin = float(cyl.get("zmin", -1e9))
        zmax = float(cyl.get("zmax", 1e9))
        if not (zmin <= pz <= zmax):
            continue
        dxy = float(np.hypot(float(p[0]) - float(cyl["cx"]), float(p[1]) - float(cyl["cy"])))
        clear = dxy - (float(cyl["r"]) + float(params.drone_radius))
        safe_clear = clear - float(params.cyl_safety_margin)
        min_clearance = min(min_clearance, clear)
        min_safety_clearance = min(min_safety_clearance, safe_clear)

    return min_clearance, min_safety_clearance


def evaluate_scenario(ctrl, params: DRParams, traj, moving_trajs, cylinders, cfg: dict, max_steps: int):
    dt = float(params.dt)
    Q = np.array([
        float(cfg["Qx"]),
        float(cfg["Qy"]),
        float(cfg["Qz"]),
        0.0,
    ], dtype=np.float32)
    Qf = np.array([
        float(cfg["Qf_x_scale"]) * float(cfg["Qx"]),
        float(cfg["Qf_y_scale"]) * float(cfg["Qy"]),
        float(cfg["Qf_z_scale"]) * float(cfg["Qz"]),
        0.0,
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

    min_clearance = float("inf")
    min_safety_clearance = float("inf")
    safety_violation_steps = 0
    collision_steps = 0
    safety_shortfall_sum = 0.0
    collision_shortfall_sum = 0.0

    sim_T = max([float(traj.total_time)] + [float(moving_traj.total_time) for moving_traj in moving_trajs])
    steps = min(int(sim_T / dt) + 1, int(max_steps))
    if steps < 2:
        return 1e12, {"error": "too_few_steps"}

    reset_controller(ctrl)

    for i in range(steps):
        t_now = i * dt
        ref_seq, obs_seq, last_yaw_ref = build_ref_and_obs(
            traj=traj,
            moving_trajs=moving_trajs,
            t_now=t_now,
            dt=dt,
            horizon=ctrl.T,
            last_yaw_ref=last_yaw_ref,
            lead_time=lead_time,
        )
        obs_now = np.asarray(obs_seq[0], dtype=float)

        if ctrl.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        u = ctrl.plan(x, ref_seq, Q, Qf, obs_seq)
        if ctrl.device.type == "cuda":
            torch.cuda.synchronize()
        solve_ms_sum += (time.perf_counter() - t0) * 1000.0

        x, u_cur = apply_control_step(ctrl, params, x, u)

        e_pos = x[0:3] - ref_seq[0, 0:3]
        e_yaw = wrap_pi(float(x[6] - ref_seq[0, 3]))
        pos_err = float(np.linalg.norm(e_pos))
        pos_err_sq += float(np.dot(e_pos, e_pos))
        yaw_err_sq += e_yaw * e_yaw
        max_pos_err = max(max_pos_err, pos_err)
        final_pos_err = pos_err

        if i > 0:
            du = u_cur - u_prev
            du_sq += float(np.dot(du[1:3], du[1:3]))
        u_prev = u_cur

        step_clear, step_safe_clear = step_clearances(x, obs_now, params, cylinders)
        min_clearance = min(min_clearance, step_clear)
        min_safety_clearance = min(min_safety_clearance, step_safe_clear)
        if step_safe_clear < 0.0:
            safety_violation_steps += 1
            safety_shortfall_sum += -step_safe_clear
        if step_clear < 0.0:
            collision_steps += 1
            collision_shortfall_sum += -step_clear

        if not np.all(np.isfinite(x)):
            return 1e12, {"error": "non_finite_state"}

    rmse_pos = math.sqrt(pos_err_sq / steps)
    rmse_yaw = math.sqrt(yaw_err_sq / steps)
    du_rms = math.sqrt(du_sq / max(1, steps - 1))
    mean_solve_ms = solve_ms_sum / steps
    safety_violation_rate = float(safety_violation_steps) / float(steps)
    collision_rate = float(collision_steps) / float(steps)
    safety_shortfall_mean = float(safety_shortfall_sum) / float(steps)
    collision_shortfall_mean = float(collision_shortfall_sum) / float(steps)

    score = (
        2.8 * rmse_pos
        + 1.6 * final_pos_err
        + 0.35 * max_pos_err
        + 0.18 * du_rms
        + 0.04 * rmse_yaw
        + 0.0008 * mean_solve_ms
        + 30.0 * safety_violation_rate
        + 180.0 * collision_rate
        + 45.0 * safety_shortfall_mean
        + 220.0 * collision_shortfall_mean
    )
    if min_safety_clearance < 0.0:
        score += 40.0 * abs(min_safety_clearance)
    if min_clearance < 0.0:
        score += 140.0 * abs(min_clearance)

    metrics = {
        "rmse_pos": float(rmse_pos),
        "rmse_yaw": float(rmse_yaw),
        "final_pos_error": float(final_pos_err),
        "max_pos_error": float(max_pos_err),
        "du_rms_roll_pitch": float(du_rms),
        "mean_solve_ms": float(mean_solve_ms),
        "min_clearance": float(min_clearance),
        "min_safety_clearance": float(min_safety_clearance),
        "safety_violation_rate": float(safety_violation_rate),
        "collision_rate": float(collision_rate),
        "safety_shortfall_mean": float(safety_shortfall_mean),
        "collision_shortfall_mean": float(collision_shortfall_mean),
        "steps": int(steps),
    }
    return float(score), metrics


def evaluate_candidate(cfg: dict, device: str, max_steps: int, eval_seed: int):
    traj, scenarios, cylinders = make_world()

    scenario_scores = []
    scenario_metrics = []
    scenario_names = []

    for idx, (scenario_name, moving_trajs) in enumerate(scenarios):
        set_global_seed(eval_seed + idx)
        ctrl, params = build_controller(cfg, cylinders, device)
        score, metrics = evaluate_scenario(ctrl, params, traj, moving_trajs, cylinders, cfg, max_steps)
        if not np.isfinite(score):
            return 1e12, {"error": metrics.get("error", "invalid_score"), "scenario": scenario_name}
        scenario_scores.append(float(score))
        scenario_metrics.append(metrics)
        scenario_names.append(scenario_name)

    mean_score = float(np.mean(scenario_scores))
    worst_score = float(np.max(scenario_scores))
    total_score = 0.65 * mean_score + 0.35 * worst_score

    avg_metrics = {
        "rmse_pos": float(np.mean([m["rmse_pos"] for m in scenario_metrics])),
        "rmse_yaw": float(np.mean([m["rmse_yaw"] for m in scenario_metrics])),
        "final_pos_error": float(np.mean([m["final_pos_error"] for m in scenario_metrics])),
        "max_pos_error": float(np.mean([m["max_pos_error"] for m in scenario_metrics])),
        "du_rms_roll_pitch": float(np.mean([m["du_rms_roll_pitch"] for m in scenario_metrics])),
        "mean_solve_ms": float(np.mean([m["mean_solve_ms"] for m in scenario_metrics])),
        "safety_violation_rate": float(np.mean([m["safety_violation_rate"] for m in scenario_metrics])),
        "collision_rate": float(np.mean([m["collision_rate"] for m in scenario_metrics])),
        "safety_shortfall_mean": float(np.mean([m["safety_shortfall_mean"] for m in scenario_metrics])),
        "collision_shortfall_mean": float(np.mean([m["collision_shortfall_mean"] for m in scenario_metrics])),
        "min_clearance_across_scenarios": float(np.min([m["min_clearance"] for m in scenario_metrics])),
        "min_safety_clearance_across_scenarios": float(np.min([m["min_safety_clearance"] for m in scenario_metrics])),
        "steps": int(np.mean([m["steps"] for m in scenario_metrics])),
        "scenario_names": scenario_names,
        "scenario_scores": scenario_scores,
        "mean_score": float(mean_score),
        "worst_score": float(worst_score),
    }
    return float(total_score), avg_metrics


def best_cfg_payload(cfg: dict) -> dict:
    return {
        "dr_params": {
            "dt": FIXED_DT,
            "horizon_steps": FIXED_HORIZON_STEPS,
            "rollouts": FIXED_ROLLOUTS,
            "lam": float(cfg["lam"]),
            "sigma": [
                float(cfg["sigma_T_ratio"]) * HOVER,
                math.radians(float(cfg["sigma_phi_deg"])),
                math.radians(float(cfg["sigma_theta_deg"])),
                math.radians(float(cfg["sigma_yawrate_deg"])),
            ],
            "R_u": [
                float(cfg["R_T"]),
                float(cfg["R_roll"]),
                float(cfg["R_pitch"]),
                float(cfg["R_yaw"]),
            ],
            "Rd_u": [
                float(cfg["Rd_T"]),
                float(cfg["Rd_roll"]),
                float(cfg["Rd_pitch"]),
                float(cfg["Rd_yaw"]),
            ],
            "cvar_alpha": float(cfg["cvar_alpha"]),
            "cvar_N": int(cfg["cvar_N"]),
            "obs_pos_sigma_xy": [
                float(cfg["obs_sigma_x"]),
                float(cfg["obs_sigma_y"]),
            ],
            "noise_mode": str(cfg["noise_mode"]),
            "dr_eps_cvar": float(cfg["dr_eps_cvar"]),
        },
        "Q": [
            float(cfg["Qx"]),
            float(cfg["Qy"]),
            float(cfg["Qz"]),
            0.0,
        ],
        "Qf": [
            float(cfg["Qf_x_scale"]) * float(cfg["Qx"]),
            float(cfg["Qf_y_scale"]) * float(cfg["Qy"]),
            float(cfg["Qf_z_scale"]) * float(cfg["Qz"]),
            0.0,
        ],
        "lead_time": float(cfg["lead_time"]),
    }


def main():
    ap = argparse.ArgumentParser(description="Tune DR_mppi_crazyflie.py tracking and DR-risk params only")
    ap.add_argument("--trials", type=int, default=120, help="Random candidates to evaluate")
    ap.add_argument("--max-steps", type=int, default=500, help="Max simulated steps per scenario")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--save-json", type=str, default="crazyflie_dr_tracking_dr_autotune_best.json")
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
        f"dt={FIXED_DT}, horizon_steps={FIXED_HORIZON_STEPS}, rollouts={FIXED_ROLLOUTS}",
    )
    print("Fixed policy: obstacle soft-cost weights, dynamics limits, and geometry stay at DRParams defaults.")

    for trial in range(1, int(args.trials) + 1):
        cfg = sample_candidate(random)
        eval_seed = int(args.seed) + 1009 * trial
        score, metrics = evaluate_candidate(cfg, device=device, max_steps=int(args.max_steps), eval_seed=eval_seed)
        print(
            f"[trial {trial:03d}/{args.trials}] score={score:.4f} "
            f"rmse={metrics.get('rmse_pos', float('nan')):.3f} "
            f"final={metrics.get('final_pos_error', float('nan')):.3f} "
            f"safe={metrics.get('safety_violation_rate', float('nan')):.3f} "
            f"coll={metrics.get('collision_rate', float('nan')):.3f} "
            f"clear={metrics.get('min_clearance_across_scenarios', float('nan')):.3f}"
        )
        if score < best_score:
            best_score = float(score)
            best_cfg = dict(cfg)
            best_metrics = dict(metrics)
            print(f"  -> new best score={best_score:.4f}")

    out = {
        "profile": "tracking_plus_dr_risk_only",
        "best_score": float(best_score),
        "best_cfg": best_cfg,
        "best_metrics": best_metrics,
        "best_payload_for_dr_mppi_crazyflie": best_cfg_payload(best_cfg) if best_cfg is not None else None,
        "tuned_keys": tuned_keys,
        "fixed_planner_params": {
            "dt": FIXED_DT,
            "horizon_steps": FIXED_HORIZON_STEPS,
            "rollouts": FIXED_ROLLOUTS,
        },
        "fixed_policy": {
            "tracking_terms_tuned": ["lam", "sigma", "R_u", "Rd_u", "Q", "Qf", "lead_time"],
            "dr_risk_terms_tuned": ["cvar_alpha", "cvar_N", "obs_pos_sigma_xy", "noise_mode", "dr_eps_cvar"],
            "left_at_drparams_defaults": [
                "ang_max",
                "yawrate_max",
                "tau_phi",
                "tau_theta",
                "phi_rate_max",
                "theta_rate_max",
                "w_cyl",
                "cyl_safety_margin",
                "cyl_alpha",
                "w_moving",
                "moving_r",
                "moving_safety_margin",
                "moving_alpha",
                "drone_radius",
            ],
        },
        "notes": (
            "Objective mixes tracking quality with strong safety penalties so DR-risk terms stay meaningful. "
            "Scenario suite includes each moving obstacle separately plus a paired-obstacle case."
        ),
    }

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nBest configuration:")
    print(json.dumps(out, indent=2))
    print(f"\nSaved to: {args.save_json}")


if __name__ == "__main__":
    main()
