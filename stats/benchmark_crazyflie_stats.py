#!/usr/bin/env python3
"""
Paper-style benchmark statistics for Crazyflie controllers:
- MPPI
- RAMPPI
- DRMPPI
- DRAMPPI

Outputs:
- summary_table_paper.csv
- summary_table_numeric.csv
- run_metrics.csv

python benchmark_crazyflie_stats.py \
  --runs 30 \
  --steps 550 \
  --alpha 0.95 \
  --outdir results_crazyflie_stats_paper

"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import matplotlib.pyplot as plt

def _load_local_module(module_name: str, relative_path: str):
    here = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(here, relative_path)
    module_dir = os.path.dirname(module_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


plain_mod = _load_local_module("cf_plain_final_mod", os.path.join("..", "mppi_crazyflie.py"))
ra_mod = _load_local_module("cf_ra_final_mod", os.path.join("..", "RA_mppi_crazyflie.py"))
dr_mod = _load_local_module("cf_dr_final_mod", os.path.join("..", "DR_mppi_crazyflie.py"))


def _load_dra_module():
    return _load_local_module("cf_dra_final_mod", os.path.join("..", "DRA_mppi_crazyflie.py"))


def step_rng(run_seed: int, step_idx: int, stream_id: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(run_seed), int(step_idx), int(stream_id)])
    return np.random.default_rng(ss)


def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_follow_from_vel(v: np.ndarray, fallback: float) -> float:
    vx, vy = float(v[0]), float(v[1])
    if vx * vx + vy * vy < 1e-6:
        return fallback
    return float(np.arctan2(vy, vx))


def build_ref_and_obs_seq(traj, moving_traj, t_now: float, dt: float, T: int, lead_time: float, last_yaw_ref: float):
    ref_seq = np.zeros((T + 1, 4), dtype=float)

    p0, v0 = traj.eval(min(t_now, traj.total_time))
    psi0_raw = wrap_pi(yaw_follow_from_vel(v0, last_yaw_ref))
    ref_seq[0, 0:3] = p0
    ref_seq[0, 3] = last_yaw_ref + wrap_pi(psi0_raw - last_yaw_ref)

    for k in range(1, T + 1):
        tk = min(traj.total_time, t_now + k * dt)
        pk, vk = traj.eval(tk)
        psi_raw = wrap_pi(yaw_follow_from_vel(vk, ref_seq[k - 1, 3]))
        prev = ref_seq[k - 1, 3]
        ref_seq[k, 0:3] = pk
        ref_seq[k, 3] = prev + wrap_pi(psi_raw - prev)

    obs_seq = np.zeros((T + 1, 3), dtype=float)
    for k in range(T + 1):
        tk = min(moving_traj.total_time, t_now + lead_time + k * dt)
        op, _ = moving_traj.eval(tk)
        obs_seq[k] = op

    return ref_seq, obs_seq, float(ref_seq[1, 3])


def eval_stage_cost(x: np.ndarray, u: np.ndarray, ref: np.ndarray, Q: np.ndarray, R: np.ndarray) -> float:
    e = np.zeros((4,), dtype=float)
    e[0:3] = x[0:3] - ref[0:3]
    e[3] = wrap_pi(float(x[6] - ref[3]))
    return float(np.dot(e * e, Q) + np.dot(u * u, R))


def effective_control_weight(ctrl) -> np.ndarray:
    sigma = getattr(getattr(ctrl, "p", None), "sigma", None)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float).reshape(-1)
        if sigma.shape == (4,):
            return 1.0 / np.maximum(sigma ** 2, 1e-12)

    if hasattr(ctrl, "R_np"):
        return np.asarray(ctrl.R_np, dtype=float).reshape(4,)
    if hasattr(ctrl, "R"):
        return np.asarray(ctrl.R, dtype=float).reshape(4,)
    raise AttributeError("Controller does not expose sigma, R_np, or R for stage-cost evaluation.")


def _z_body_world(phi: float, theta: float, psi: float) -> np.ndarray:
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([cps * sth * cph + sps * sph, sps * sth * cph - cps * sph, cth * cph], dtype=float)


def apply_control_step(ctrl, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    Tcmd, phi_cmd, theta_cmd, yawrate = [float(v) for v in u]
    if hasattr(ctrl, "u_min") and hasattr(ctrl, "u_max"):
        Tcmd = np.clip(Tcmd, float(ctrl.u_min[0]), float(ctrl.u_max[0]))
        phi_cmd = np.clip(phi_cmd, float(ctrl.u_min[1]), float(ctrl.u_max[1]))
        theta_cmd = np.clip(theta_cmd, float(ctrl.u_min[2]), float(ctrl.u_max[2]))
        yawrate = np.clip(yawrate, float(ctrl.u_min[3]), float(ctrl.u_max[3]))
    else:
        Tcmd = np.clip(Tcmd, ctrl.p.T_min, ctrl.p.T_max)
        phi_cmd = np.clip(phi_cmd, -ctrl.p.ang_max, ctrl.p.ang_max)
        theta_cmd = np.clip(theta_cmd, -ctrl.p.ang_max, ctrl.p.ang_max)
        yawrate = np.clip(yawrate, -ctrl.p.yawrate_max, ctrl.p.yawrate_max)

    zb = _z_body_world(phi_cmd, theta_cmd, float(x[6]))
    a = (Tcmd / ctrl.m) * zb - np.array([0.0, 0.0, ctrl.g], dtype=float)

    x_next = x.copy()
    x_next[3:6] = x_next[3:6] + dt * a
    x_next[0:3] = x_next[0:3] + dt * x_next[3:6]
    x_next[6] = wrap_pi(float(x_next[6] + dt * yawrate))
    return x_next


def apply_control_step_dra(ctrl, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    Tcmd, phi_cmd, theta_cmd, yawrate = [float(v) for v in u]
    Tcmd = np.clip(Tcmd, float(ctrl.u_min[0]), float(ctrl.u_max[0]))
    phi_cmd = np.clip(phi_cmd, float(ctrl.u_min[1]), float(ctrl.u_max[1]))
    theta_cmd = np.clip(theta_cmd, float(ctrl.u_min[2]), float(ctrl.u_max[2]))
    yawrate = np.clip(yawrate, float(ctrl.u_min[3]), float(ctrl.u_max[3]))

    px, py, pz, vx, vy, vz, psi, phi, theta = [float(v) for v in x]

    phi_dot = (phi_cmd - phi) / max(1e-6, float(ctrl.p.tau_phi))
    theta_dot = (theta_cmd - theta) / max(1e-6, float(ctrl.p.tau_theta))
    phi_dot = np.clip(phi_dot, -float(ctrl.p.phi_rate_max), float(ctrl.p.phi_rate_max))
    theta_dot = np.clip(theta_dot, -float(ctrl.p.theta_rate_max), float(ctrl.p.theta_rate_max))
    phi_next = np.clip(phi + dt * phi_dot, -float(ctrl.p.ang_max), float(ctrl.p.ang_max))
    theta_next = np.clip(theta + dt * theta_dot, -float(ctrl.p.ang_max), float(ctrl.p.ang_max))

    zb = _z_body_world(phi, theta, psi)
    ax, ay, az = (Tcmd / float(ctrl.m)) * zb - np.array([0.0, 0.0, float(ctrl.g)], dtype=float)
    vx_next = vx + dt * ax
    vy_next = vy + dt * ay
    vz_next = vz + dt * az
    px_next = px + dt * vx_next
    py_next = py + dt * vy_next
    pz_next = pz + dt * vz_next
    psi_next = wrap_pi(psi + dt * yawrate)
    return np.array([px_next, py_next, pz_next, vx_next, vy_next, vz_next, psi_next, phi_next, theta_next], dtype=float)


def closest_distance_and_events(
    x: np.ndarray,
    obs_pos: np.ndarray,
    cylinders: list[dict],
    moving_r: float,
    drone_r: float,
    safety_margin: float,
) -> tuple[float, bool, bool]:
    p = x[0:3]

    d_move = float(np.linalg.norm(p - obs_pos))
    move_safety = d_move < (moving_r + drone_r + safety_margin)
    move_collision = d_move < (moving_r + drone_r)

    d_cyl_min = np.inf
    cyl_safety = False
    cyl_collision = False
    for c in cylinders:
        cx, cy, cr = float(c["cx"]), float(c["cy"]), float(c["r"])
        zmin = float(c.get("zmin", -1e9))
        zmax = float(c.get("zmax", 1e9))
        dxy = float(np.hypot(float(p[0]) - cx, float(p[1]) - cy))
        d_cyl_min = min(d_cyl_min, dxy)
        if zmin <= float(p[2]) <= zmax:
            if dxy < (cr + drone_r + safety_margin):
                cyl_safety = True
            if dxy < (cr + drone_r):
                cyl_collision = True

    d_min = min(d_move, d_cyl_min)
    return d_min, bool(move_safety or cyl_safety), bool(move_collision or cyl_collision)


def set_step_seed(run_seed: int, step_idx: int, stream_id: int, ctrl):
    seed = int(step_rng(run_seed, step_idx, stream_id).integers(0, 2**31 - 1, dtype=np.int64))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(ctrl, "use_gpu") and getattr(ctrl, "use_gpu", False):
        cp_mod = getattr(type(ctrl), "__module__", None)
        try:
            import cupy as cp  # optional
            cp.random.seed(seed)
        except Exception:
            pass

    if hasattr(ctrl, "mppi") and hasattr(ctrl.mppi, "_rng"):
        try:
            ctrl.mppi._rng.manual_seed(seed)
        except Exception:
            pass


def reset_controller(ctrl):
    hover = float(ctrl.m * ctrl.g)
    if hasattr(ctrl, "U"):
        ctrl.U[:] = 0.0
        ctrl.U[:, 0] = hover
        if hasattr(ctrl, "U_xp") and getattr(ctrl, "use_gpu", False):
            try:
                import cupy as cp
                ctrl.U_xp = cp.asarray(ctrl.U)
            except Exception:
                pass
    if hasattr(ctrl, "mppi") and hasattr(ctrl.mppi, "U_cpu"):
        ctrl.mppi.U_cpu[:] = 0.0
        ctrl.mppi.U_cpu[:, 0] = hover
        ctrl.mppi.U = ctrl.mppi.U_cpu


def _fmt_mean_std(vals: np.ndarray, digits: int = 4) -> str:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return "nan ± (nan)"
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return f"{mean:.{digits}f} ± ({std:.{digits}f})"


def _kde_pdf(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.full_like(grid, np.nan, dtype=np.float64)
    std = float(np.std(x, ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return np.full_like(grid, np.nan, dtype=np.float64)
    h = 1.06 * std * (n ** (-1.0 / 5.0))
    if not np.isfinite(h) or h <= 0.0:
        return np.full_like(grid, np.nan, dtype=np.float64)
    z = (grid[:, None] - x[None, :]) / h
    norm_const = 1.0 / np.sqrt(2.0 * np.pi)
    k = norm_const * np.exp(-0.5 * z * z)
    return np.mean(k, axis=1) / h


def _paper_axes(ax):
    ax.set_facecolor("#ececec")
    ax.grid(True, color="#9a9a9a", alpha=0.65, linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#222222")
    ax.tick_params(axis="both", labelsize=12, width=1.0, colors="#303030")


def _safe_plot_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _plot_histogram(per_alg: dict[str, np.ndarray], title: str, xlabel: str, out_path: str):
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    _paper_axes(ax)
    all_vals = []
    for vals in per_alg.values():
        x = np.asarray(vals, dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) > 0:
            all_vals.append(x)
    grid = None
    xlim = None
    if len(all_vals) > 0:
        stacked = np.concatenate(all_vals)
        xmin, xmax = float(np.min(stacked)), float(np.max(stacked))
        if np.isfinite(xmin) and np.isfinite(xmax):
            if xmax <= xmin:
                pad = 1.0 if xmin == 0.0 else 0.05 * abs(xmin)
                xmin, xmax = xmin - pad, xmax + pad
            else:
                pad = 0.05 * (xmax - xmin)
                xmin, xmax = xmin - pad, xmax + pad
            xlim = (xmin, xmax)
            grid = np.linspace(xmin, xmax, 300)

    for alg, vals in per_alg.items():
        x = np.asarray(vals, dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        bins = max(6, min(16, int(np.sqrt(len(x)) * 2)))
        counts, edges, _ = ax.hist(
            x,
            bins=bins,
            alpha=0.40,
            label=alg,
            color="#b56bb3",
            edgecolor="#3b2143",
            linewidth=1.6,
        )
        if grid is not None:
            pdf = _kde_pdf(x, grid)
            if np.any(np.isfinite(pdf)):
                bin_width = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
                ax.plot(grid, pdf * len(x) * bin_width, linewidth=2.0, color="#8f159d")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.legend(frameon=True, facecolor="white", edgecolor="#b0b0b0", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_violin(per_alg: dict[str, np.ndarray], title: str, ylabel: str, out_path: str):
    labels = list(per_alg.keys())
    data = []
    for alg in labels:
        x = np.asarray(per_alg[alg], dtype=np.float64)
        x = x[np.isfinite(x)]
        data.append(x)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    _paper_axes(ax)
    vp = ax.violinplot(data, showmeans=True, showmedians=True)
    for body in vp["bodies"]:
        body.set_facecolor("#b56bb3")
        body.set_edgecolor("#3b2143")
        body.set_alpha(0.45)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in vp:
            vp[key].set_color("#3b2143")
            vp[key].set_linewidth(1.4)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, pad=10)
    ax.grid(True, axis="y", color="#9a9a9a", alpha=0.65, linewidth=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_box(per_alg: dict[str, np.ndarray], title: str, ylabel: str, out_path: str):
    labels = list(per_alg.keys())
    data = []
    for alg in labels:
        x = np.asarray(per_alg[alg], dtype=np.float64)
        x = x[np.isfinite(x)]
        data.append(x)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    _paper_axes(ax)
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=True,
        patch_artist=True,
        medianprops=dict(color="#3b2143", linewidth=1.8),
        boxprops=dict(facecolor="#d5acd7", edgecolor="#3b2143", linewidth=1.4),
        whiskerprops=dict(color="#3b2143", linewidth=1.3),
        capprops=dict(color="#3b2143", linewidth=1.3),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="#8f159d", markeredgecolor="#3b2143", alpha=0.7),
    )
    _ = bp
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, pad=10)
    ax.grid(True, axis="y", color="#9a9a9a", alpha=0.65, linewidth=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_three_plots(metric_key: str, title_root: str, x_label: str, per_alg: dict[str, np.ndarray], outdir: str):
    safe_name = _safe_plot_name(metric_key)
    _plot_histogram(
        per_alg=per_alg,
        title=f"Distribution of {title_root}",
        xlabel=x_label,
        out_path=os.path.join(outdir, f"{safe_name}_histogram.png"),
    )
    _plot_violin(
        per_alg=per_alg,
        title=f"Violin Plot of {title_root}",
        ylabel=x_label,
        out_path=os.path.join(outdir, f"{safe_name}_violin.png"),
    )
    _plot_box(
        per_alg=per_alg,
        title=f"Box Plot of {title_root}",
        ylabel=x_label,
        out_path=os.path.join(outdir, f"{safe_name}_box.png"),
    )


def run_episode(
    ctrl,
    kind: str,
    traj,
    moving_traj,
    cylinders,
    Q,
    Qf,
    run_seed: int,
    steps: int,
    lead_time: float,
    drone_radius: float,
    safety_margin: float,
    R_eval: np.ndarray,
):
    dt = float(ctrl.p.dt)
    p0, _ = traj.eval(0.0)
    x = np.zeros((9,), dtype=float)
    x[0:3] = p0
    x[3:6] = 0.0
    x[6] = 0.0

    total_qr_cost = 0.0
    min_dists = []
    safety_violated = False
    collided = False
    step_compute_times = []

    last_yaw_ref = float(x[6])
    moving_r = float(getattr(ctrl.p, "moving_r", 0.35))

    for k in range(steps):
        t_now = k * dt
        ref_seq, obs_seq, last_yaw_ref = build_ref_and_obs_seq(
            traj=traj,
            moving_traj=moving_traj,
            t_now=t_now,
            dt=dt,
            T=int(ctrl.T),
            lead_time=lead_time,
            last_yaw_ref=last_yaw_ref,
        )

        set_step_seed(run_seed, k, stream_id=0, ctrl=ctrl)
        t0 = time.perf_counter()
        if kind == "dra":
            u = ctrl.plan(x, ref_seq, Q, Qf, obs_seq)
        else:
            u = ctrl.plan(x, ref_seq, Q, Qf, obs_seq)
        t1 = time.perf_counter()
        if k > 0:
            step_compute_times.append(t1 - t0)

        u = np.nan_to_num(np.asarray(u, dtype=float), nan=0.0)
        total_qr_cost += eval_stage_cost(x, u, ref_seq[1], Q, R_eval)
        x = apply_control_step_dra(ctrl, x, u, dt)

        dmin, hit_safety, hit_collision = closest_distance_and_events(
            x=x,
            obs_pos=obs_seq[0],
            cylinders=cylinders,
            moving_r=moving_r,
            drone_r=drone_radius,
            safety_margin=safety_margin,
        )
        min_dists.append(dmin)
        if hit_safety:
            safety_violated = True
        if hit_collision:
            collided = True

        if not np.all(np.isfinite(x)):
            safety_violated = True
            collided = True
            break

    max_compute_time = float(np.max(step_compute_times)) if len(step_compute_times) > 0 else np.nan
    return float(total_qr_cost), np.asarray(min_dists, dtype=np.float32), bool(safety_violated), bool(collided), max_compute_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="results_crazyflie_stats_paper")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--steps", type=int, default=550)
    ap.add_argument("--alpha", type=float, default=0.95, help="CVaR alpha for RA/DR")
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--dr_eps_cvar", type=float, default=0.02)
    args = ap.parse_args()

    horizon = 60
    rollouts = 1024
    iterations = 2
    lead_time = 1.0
    cvar_n = 64
    obs_noise_mode = "static"
    obs_sigma_xyz = (0.25, 0.25, 0.25)
    extra_margin = 0.0

    os.makedirs(args.outdir, exist_ok=True)
    dra_mod = _load_dra_module()

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

    traj = plain_mod.build_min_snap_3d(waypoints, avg_speed=1.8)
    moving_traj = plain_mod.build_min_snap_3d(waypoints_1 + np.array([0.40, -0.30, 0.00], dtype=float), avg_speed=1.8)

    dt = 0.02
    steps = min(int(args.steps), int(max(traj.total_time, moving_traj.total_time) / dt) + 1)

    Q = np.array([30.0, 30.0, 40.0, 0.0], dtype=float)
    Qf = np.array([120.0, 120.0, 160.0, 0.0], dtype=float)
    _ = Qf  # kept for parity with controller APIs

    mass = 0.028
    g = 9.81
    hover = mass * g
    common_sigma = np.array([0.15 * hover, np.deg2rad(6.0), np.deg2rad(6.0), np.deg2rad(30.0)], dtype=float)

    common_plain = dict(
        dt=dt,
        horizon_steps=int(horizon),
        rollouts=int(rollouts),
        iterations=int(iterations),
        lam=1.0,
        sigma=common_sigma.copy(),
        w_cyl=350.0,
        cyl_safety_margin=0.25,
        cyl_alpha=10.0,
    )

    alg_specs = [
        {"name": "MPPI", "kind": "plain"},
        {"name": "RAMPPI", "kind": "ra"},
        {"name": "DRMPPI", "kind": "dr"},
        {"name": "DRAMPPI", "kind": "dra"},
    ]

    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    print(
        f"[setup] requested_gpu={int(bool(args.use_gpu))}, "
        f"torch_cuda_available={int(bool(torch.cuda.is_available()))}, device={device}",
        flush=True,
    )
    controllers = []
    for spec in alg_specs:
        if spec["kind"] == "plain":
            p = plain_mod.MPPIParams(
                **common_plain,
                w_moving=1200.0,
                moving_r=0.35,
                moving_safety_margin=0.50,
                moving_alpha=12.0,
            )
            ctrl = plain_mod.TorchMPPIQuadOuter(mass=mass, g=g, params=p, cylinders=cylinders, device=device)
            drone_radius = 0.25
            R_eval = effective_control_weight(ctrl)

        elif spec["kind"] == "ra":
            p = ra_mod.Params(
                dt=dt,
                horizon_steps=int(horizon),
                rollouts=int(rollouts),
                iterations=int(iterations),
                lam=1.0,
                w_cyl=350.0,
                cyl_margin=0.25,
                cyl_alpha=10.0,
                w_moving_soft=1200.0,
                moving_r=0.35,
                moving_margin=0.50,
                moving_alpha=12.0,
                drone_radius=0.25,
                cvar_alpha=float(args.alpha),
                cvar_N=int(cvar_n),
                obs_pos_sigma_xy=tuple(float(v) for v in obs_sigma_xyz[:2]),
                obs_noise_mode=str(obs_noise_mode),
            )
            ctrl = ra_mod.TorchRAQuad(mass=mass, g=g, params=p, cylinders=cylinders, device=device)
            drone_radius = float(p.drone_radius)
            R_eval = effective_control_weight(ctrl)

        elif spec["kind"] == "dr":
            p = dr_mod.DRParams(
                dt=dt,
                horizon_steps=int(horizon),
                rollouts=int(rollouts),
                iterations=int(iterations),
                lam=1.0,
                sigma=common_sigma.copy(),
                w_cyl=350.0,
                cyl_safety_margin=0.25,
                cyl_alpha=10.0,
                w_moving=1200.0,
                moving_r=0.35,
                moving_safety_margin=0.50,
                moving_alpha=12.0,
                cvar_alpha=float(args.alpha),
                cvar_N=int(cvar_n),
                obs_pos_sigma_xy=tuple(float(v) for v in obs_sigma_xyz[:2]),
                noise_mode=str(obs_noise_mode),
                dr_eps_cvar=float(args.dr_eps_cvar),
                drone_radius=0.25,
            )
            ctrl = dr_mod.TorchDRMPPIQuadOuter(mass=mass, g=g, params=p, cylinders=cylinders, device=device)
            drone_radius = 0.25
            R_eval = effective_control_weight(ctrl)

        elif spec["kind"] == "dra":
            p = dra_mod.DRAParams(
                dt=dt,
                horizon_steps=int(horizon),
                rollouts=int(rollouts),
                iterations=int(iterations),
                lam=1.0,
                w_cyl=350.0,
                cyl_safety_margin=0.25,
                cyl_alpha=10.0,
                w_moving=1200.0,
                moving_r=0.35,
                moving_safety_margin=0.50,
                moving_alpha=12.0,
                sigma_cp=0.05,
                Nmc=1200,
                omega_soft=10.0,
                omega_hard=1000.0,
                sigma=common_sigma.copy(),
                obs_pos_sigma_xyz=tuple(float(v) for v in obs_sigma_xyz),
                mc_chunk=256,
                drone_radius=0.25,
            )
            ctrl = dra_mod.TorchDRAMPPIQuadOuter(mass=mass, g=g, params=p, cylinders=cylinders, device=device)
            drone_radius = float(p.drone_radius)
            R_eval = effective_control_weight(ctrl)

        else:
            raise RuntimeError("Unknown controller kind")

        controllers.append((spec, ctrl, float(drone_radius), R_eval))

    master = np.random.default_rng(args.seed)
    run_seeds = master.integers(0, 2**31 - 1, size=args.runs, dtype=np.int64)

    per_alg = {}
    for spec, _, _, _ in controllers:
        per_alg[spec["name"]] = dict(
            total_qr_cost=[],
            run_min_dist=[],
            safety_violation=[],
            collision=[],
            run_max_compute_time_s=[],
        )

    for r in range(args.runs):
        run_seed = int(run_seeds[r])
        print(f"[run {r + 1:3d}/{args.runs}] started (seed={run_seed})", flush=True)
        for spec, ctrl, drone_radius, r_eval in controllers:
            print(f"  [controller] {spec['name']} ...", flush=True)
            reset_controller(ctrl)
            total_qr_cost, dists, safety_violated, collided, max_compute_s = run_episode(
                ctrl=ctrl,
                kind=spec["kind"],
                traj=traj,
                moving_traj=moving_traj,
                cylinders=cylinders,
                Q=Q,
                Qf=Qf,
                run_seed=run_seed,
                steps=steps,
                lead_time=float(lead_time),
                drone_radius=drone_radius,
                safety_margin=float(extra_margin),
                R_eval=r_eval,
            )

            d = np.asarray(dists, dtype=np.float64)
            d = d[np.isfinite(d)]
            run_min_d = float(np.min(d)) if len(d) > 0 else np.nan

            rec = per_alg[spec["name"]]
            rec["total_qr_cost"].append(total_qr_cost)
            rec["run_min_dist"].append(run_min_d)
            rec["safety_violation"].append(1.0 if safety_violated else 0.0)
            rec["collision"].append(1.0 if collided else 0.0)
            rec["run_max_compute_time_s"].append(max_compute_s)
            print(
                f"  [done] {spec['name']}: cost={total_qr_cost:.4f}, "
                f"min_dist={run_min_d:.4f}, collided={int(collided)}",
                flush=True,
            )

        print(f"[run {r + 1:3d}/{args.runs}] done (seed={run_seed})", flush=True)

    raw_rows = []
    for alg_name, rec in per_alg.items():
        n = len(rec["total_qr_cost"])
        for i in range(n):
            raw_rows.append(
                dict(
                    Algorithm=alg_name,
                    Run=i,
                    RunMinDistance=rec["run_min_dist"][i],
                    SafetyViolation=rec["safety_violation"][i],
                    Collision=rec["collision"][i],
                    TotalQSigmaInvStageCost=rec["total_qr_cost"][i],
                    RunMaxComputeTimeMs=1000.0 * rec["run_max_compute_time_s"][i],
                )
            )
    raw_df = pd.DataFrame(raw_rows)
    out_raw = os.path.join(args.outdir, "run_metrics.csv")
    raw_df.to_csv(out_raw, index=False)

    row_names = [
        "1) Min distance across runs (min of run minima)",
        "2) Run-min distance (mean ± std)",
        "3) Safety violation probability using (r + r_s)",
        "4) Collision probability using (r)",
        "5) Total cost Q+Sigma^-2 only (mean ± std)",
        "6) Max compute time across runs (ms)",
        "7) Run-max compute time (mean ± std) (ms)",
    ]

    algorithm_names = [spec["name"] for spec, _, _, _ in controllers]
    table = pd.DataFrame({"Metric": row_names})
    for alg_name in algorithm_names:
        rec = per_alg[alg_name]
        run_min = np.asarray(rec["run_min_dist"], dtype=np.float64)
        run_min_f = run_min[np.isfinite(run_min)]
        safety = np.asarray(rec["safety_violation"], dtype=np.float64)
        coll = np.asarray(rec["collision"], dtype=np.float64)
        costs = np.asarray(rec["total_qr_cost"], dtype=np.float64)
        tmax_ms = 1000.0 * np.asarray(rec["run_max_compute_time_s"], dtype=np.float64)

        col = [
            f"{(float(np.min(run_min_f)) if len(run_min_f) > 0 else np.nan):.4f}",
            _fmt_mean_std(run_min_f, digits=4),
            f"{float(np.mean(safety)):.4f}",
            f"{float(np.mean(coll)):.4f}",
            _fmt_mean_std(costs, digits=4),
            f"{(float(np.max(tmax_ms)) if len(tmax_ms) > 0 else np.nan):.4f}",
            _fmt_mean_std(tmax_ms, digits=4),
        ]
        table[alg_name] = col

    out_table = os.path.join(args.outdir, "summary_table_paper.csv")
    table.to_csv(out_table, index=False)

    compact_rows = []
    for alg_name in algorithm_names:
        rec = per_alg[alg_name]
        run_min = np.asarray(rec["run_min_dist"], dtype=np.float64)
        run_min_f = run_min[np.isfinite(run_min)]
        safety = np.asarray(rec["safety_violation"], dtype=np.float64)
        coll = np.asarray(rec["collision"], dtype=np.float64)
        costs = np.asarray(rec["total_qr_cost"], dtype=np.float64)
        tmax_ms = 1000.0 * np.asarray(rec["run_max_compute_time_s"], dtype=np.float64)
        compact_rows.append(
            dict(
                Algorithm=alg_name,
                RunMinDistance_Min=float(np.min(run_min_f)) if len(run_min_f) > 0 else np.nan,
                RunMinDistance_Mean=float(np.mean(run_min_f)) if len(run_min_f) > 0 else np.nan,
                RunMinDistance_Std=float(np.std(run_min_f, ddof=1)) if len(run_min_f) > 1 else 0.0,
                SafetyViolation_Prob=float(np.mean(safety)),
                Collision_Prob=float(np.mean(coll)),
                TotalQSigmaInvStageCost_Mean=float(np.mean(costs)),
                TotalQSigmaInvStageCost_Std=float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
                RunMaxComputeTimeMs_Max=float(np.max(tmax_ms)) if len(tmax_ms) > 0 else np.nan,
                RunMaxComputeTimeMs_Mean=float(np.mean(tmax_ms)) if len(tmax_ms) > 0 else np.nan,
                RunMaxComputeTimeMs_Std=float(np.std(tmax_ms, ddof=1)) if len(tmax_ms) > 1 else 0.0,
            )
        )
    compact_df = pd.DataFrame(compact_rows)
    out_compact = os.path.join(args.outdir, "summary_table_numeric.csv")
    compact_df.to_csv(out_compact, index=False)

    print("\n=== Paper Table (rows=metrics, cols=algorithms) ===", flush=True)
    print(table.to_string(index=False), flush=True)
    print(f"\n[OK] wrote: {out_table}", flush=True)
    print(f"[OK] wrote: {out_compact}", flush=True)
    print(f"[OK] wrote: {out_raw}", flush=True)


if __name__ == "__main__":
    main()
