#!/usr/bin/env python3
"""
Benchmark statistics for MPPI variants with paper-ready outputs.

Algorithms (table columns):
- MPPI
- RAMPPI
- DRMPPI

Row metrics:
1) Global minimum of per-run minimum distance to obstacles
2) Mean ± std of per-run minimum distance
3) Safety violation probability using threshold (r + r_s)
4) Collision probability using threshold (r)
5) Mean ± std of total trajectory stage cost (Q and Sigma^-2 only; no terminal cost)
6) Global maximum of per-run maximum compute time
7) Mean ± std of per-run maximum compute time

Also saves histogram / violin / box plots for run-level distributions.

python benchmark_mppi_stats.py \
  --runs 30 \
  --steps 900 \
  --alpha 0.95 \
  --outdir results_mppi_stats_paper
"""

from __future__ import annotations

import argparse
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


# --------------------------
# Ensure imports resolve to /MPPI modules
# --------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MPPI_DIR = os.path.dirname(HERE)
if MPPI_DIR not in sys.path:
    sys.path.insert(0, MPPI_DIR)

from mppi_class import MPPI as PlainMPPIClass, RA_MPPI as RAMPPIClass, DR_MPPI as DRMPPIClass
import mppi as plain_mod
import RA_mppi as ra_mod
import DR_mppi as dr_mod
import DRA_mppi as dra_mod


def eval_stage_cost(x, u, ref, Q, R):
    e = x - ref
    e[2] = plain_mod.angle_wrap(float(e[2]))
    return float(np.dot(e * e, Q) + np.dot(u * u, R))


def min_dist_to_obstacles(ego_xy, obs_xy):
    if obs_xy.shape[0] == 0:
        return np.nan
    d = np.linalg.norm(obs_xy - ego_xy[None, :], axis=1)
    return float(np.min(d))


def all_obstacles_xy(obs_csv, t_query: float) -> np.ndarray:
    obs_df = obs_csv.obstacles_now(t_query)
    if len(obs_df) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return obs_df[["x", "y"]].to_numpy(dtype=np.float32)


def step_rng(run_seed: int, step_idx: int, stream_id: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(run_seed), int(step_idx), int(stream_id)])
    return np.random.default_rng(ss)


def make_obs_noise_cpu(run_seed: int, step_idx: int, noise_mode: str, T: int, K: int, N: int) -> np.ndarray:
    rng = step_rng(run_seed, step_idx, stream_id=1)
    if K <= 0:
        if noise_mode == "static":
            return np.zeros((0, N, 2), dtype=np.float32)
        return np.zeros((T, 0, N, 2), dtype=np.float32)
    if noise_mode == "static":
        return rng.standard_normal((K, N, 2), dtype=np.float32)
    return rng.standard_normal((T, K, N, 2), dtype=np.float32)


def set_step_seed(run_seed: int, step_idx: int, alg_stream: int):
    seed = int(step_rng(run_seed, step_idx, stream_id=alg_stream).integers(0, 2**31 - 1, dtype=np.int64))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fmt_mean_std(vals: np.ndarray, digits: int = 4) -> str:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return "nan ± (nan)"
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return f"{mean:.{digits}f} ± ({std:.{digits}f})"


def _kde_pdf(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Gaussian KDE using Silverman's rule-of-thumb bandwidth.
    Returns estimated PDF values on `grid`.
    """
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
        else:
            grid = None
    else:
        grid = None

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
    ax.boxplot(
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


def run_episode(ctrl, kind, obs_csv, times, params, run_seed):
    dt = params["dt"]
    steps = params["steps"]

    x = np.array([0.0, params["lane_y"], 0.0], dtype=np.float32)

    total_qr_cost = 0.0
    min_dists = []
    safety_violated = False
    collided = False
    step_compute_times = []

    safety_threshold = float(params["ego_radius"] + params["obs_radius"] + params["eval_extra_margin"])
    collision_threshold = float(params["ego_radius"] + params["obs_radius"])
    planning_R = float(params["ego_radius"] + params["obs_radius"] + params["plan_extra_margin"])

    if kind != "dra":
        Q_t = torch.as_tensor(params["Q"], device=ctrl.device, dtype=ctrl.dtype)
        R_t = torch.as_tensor(params["R"], device=ctrl.device, dtype=ctrl.dtype)
        Qf_t = torch.as_tensor(params["Qf"], device=ctrl.device, dtype=ctrl.dtype)

    for k in range(steps):
        t_now = float(times[k])
        obs_now = obs_csv.obstacles_now(t_now)

        dx = obs_now["x"].to_numpy(float) - float(x[0])
        dy = obs_now["y"].to_numpy(float) - float(x[1])
        mask = (dx > -params["x_behind"]) & (dx < params["x_ahead"] + 10.0) & (np.abs(dy) < (params["y_halfspan"] + 2.0))
        obs_now = obs_now[mask].copy()

        if len(obs_now) > 0:
            dist2 = (obs_now["x"] - float(x[0])) ** 2 + (obs_now["y"] - float(x[1])) ** 2
            obs_now = obs_now.iloc[np.argsort(dist2.to_numpy())]
        obs_now = obs_now.iloc[: params["max_obs_draw"]].copy()

        ids, O_mean, _ = obs_csv.build_prediction_for_mppi(
            obs_df=obs_now, dt=dt, T=params["T"], max_obs=params["max_obs_draw"]
        )
        K = len(ids)

        if K > 0:
            radii_for_mppi = np.full((K,), planning_R, dtype=np.float32)
        else:
            radii_for_mppi = np.array([], dtype=np.float32)

        ref = np.array([float(x[0]) + params["L_ref"], params["lane_y"], params["lane_psi"]], dtype=np.float32)

        if kind != "dra":
            ctrl.cost_kwargs["ref"] = torch.as_tensor(ref, device=ctrl.device, dtype=ctrl.dtype)
            ctrl.cost_kwargs["Q"] = Q_t
            ctrl.cost_kwargs["R"] = R_t
            ctrl.cost_kwargs["Qf"] = Qf_t
            if K > 0:
                ctrl.cost_kwargs["O_mean"] = torch.as_tensor(O_mean, device=ctrl.device, dtype=ctrl.dtype)
                ctrl.cost_kwargs["radii"] = torch.as_tensor(radii_for_mppi, device=ctrl.device, dtype=ctrl.dtype)
            else:
                ctrl.cost_kwargs["O_mean"] = None
                ctrl.cost_kwargs["radii"] = None

        set_step_seed(run_seed, k, alg_stream=0)

        t0 = time.perf_counter()
        if kind == "plain":
            U, _ = ctrl.plan(x, return_samples=False)
        elif kind in ("ra", "dr"):
            obs_noise_cpu = make_obs_noise_cpu(
                run_seed=run_seed,
                step_idx=k,
                noise_mode=ctrl.obs_noise_mode,
                T=params["T"],
                K=K,
                N=int(ctrl.cvar_N),
            )
            obs_noise_t = torch.as_tensor(obs_noise_cpu, device=ctrl.device, dtype=ctrl.dtype)
            U, _ = ctrl.plan(x, return_samples=False, obs_noise_std=obs_noise_t)
        elif kind == "dra":
            U = ctrl.plan(
                x0_cpu=x,
                ref_cpu=ref,
                O_mean_cpu=O_mean if K > 0 else None,
                radii_cpu=radii_for_mppi,
                return_samples=False,
            )
        else:
            raise ValueError(f"Unknown kind: {kind}")
        t1 = time.perf_counter()
        # Ignore first control-step timing (k=0) to avoid warm-start overhead bias.
        if k > 0:
            step_compute_times.append(t1 - t0)

        U = np.nan_to_num(U, nan=0.0).astype(np.float32)
        u0 = U[0].astype(np.float32).copy()
        u0[0] = np.clip(params["u_blend_v"] * u0[0] + (1.0 - params["u_blend_v"]) * params["v_des"], 0.0, float(params["u_max"][0]))

        # Q + Sigma^-2 stage cost only (terminal excluded by design).
        total_qr_cost += eval_stage_cost(x, u0, ref, params["Q"], params["R"])

        if kind == "plain":
            x = plain_mod.diffdrive_dynamics(x, u0, dt, v_max=float(params["u_max"][0])).astype(np.float32)
        elif kind in ("ra", "dr"):
            x = ra_mod.diffdrive_dynamics_cpu(
                x,
                u0,
                dt,
                v_max=float(params["u_max"][0]),
                w_max=float(params["dyn_w_max"]),
            ).astype(np.float32)
        elif kind == "dra":
            x = dra_mod.diffdrive_dynamics_cpu(
                x,
                u0,
                dt,
                v_max=float(params["u_max"][0]),
                w_max=float(params["dyn_w_max"]),
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        # Distance evaluated at the next timestamp.
        k_next = min(k + 1, steps - 1)
        t_next = float(times[k_next])
        obs_xy_eval = all_obstacles_xy(obs_csv, t_next)
        dmin = min_dist_to_obstacles(x[:2], obs_xy_eval)
        min_dists.append(dmin)

        if np.isfinite(dmin):
            if dmin < safety_threshold:
                safety_violated = True
            if dmin < collision_threshold:
                collided = True

        ctrl.U[:-1] = ctrl.U[1:]
        ctrl.U[-1] = np.array([params["v_des"], 0.0], dtype=np.float32)
        ctrl.U_cpu = ctrl.U

        if not np.all(np.isfinite(x)):
            collided = True
            safety_violated = True
            break

    max_compute_time = float(np.max(step_compute_times)) if len(step_compute_times) > 0 else np.nan
    return float(total_qr_cost), np.asarray(min_dists, dtype=np.float32), bool(safety_violated), bool(collided), max_compute_time


def main():
    ap = argparse.ArgumentParser()
    default_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "features_dir2_5_10s.csv")
    ap.add_argument("--csv", type=str, default=default_csv, help="Path to obstacle CSV")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--outdir", type=str, default="results_mppi_stats_paper")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--alpha", type=float, default=0.95, help="CVaR alpha used for both RA and DR MPPI")
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pos_scale = 0.10
    vel_scale = 0.10
    acc_scale = 0.10
    y_offset = 23.0

    obs_csv = plain_mod.MovingObstacleCSV(
        csv_path=args.csv,
        x_offset=0.0,
        y_offset=y_offset,
        pos_scale=pos_scale,
        vel_scale=vel_scale,
        acc_scale=acc_scale,
    )

    dt = float(np.round(np.min(np.diff(obs_csv.times)), 6))
    times = obs_csv.times
    steps = min(args.steps, len(times))

    T = 10
    M = 150
    lam = 2.0
    sigma = np.array([2.0, 1.04], dtype=np.float32)
    u_min = np.array([0.0, -np.deg2rad(180.0)], dtype=np.float32)
    u_max = np.array([10.0, np.deg2rad(180.0)], dtype=np.float32)

    Q = np.array([0.001, 0.001, 0.001], dtype=np.float32)
    Qf = np.array([0.5, 10.0, 1.0], dtype=np.float32)
    R = (1.0 / np.maximum(sigma ** 2, 1e-12)).astype(np.float32)

    lane_psi = 0.0
    L_ref = 14.0
    v_des = 7.0
    u_blend_v = 0.9

    ROAD_CENTER = 1.0
    LANE_W = 0.70
    lane_y = ROAD_CENTER + 0.5 * LANE_W

    ego_length = 4.5 * pos_scale
    ego_width = 1.8 * pos_scale
    ego_radius = 0.5 * np.sqrt(ego_length**2 + ego_width**2)
    obs_radius = ego_radius

    # r_s term in (r + r_s) safety threshold.
    plan_extra_margin = 0.03
    eval_extra_margin = 0.26

    params = dict(
        dt=dt,
        steps=steps,
        T=T,
        M=M,
        lane_y=lane_y,
        lane_psi=lane_psi,
        L_ref=L_ref,
        v_des=v_des,
        u_blend_v=u_blend_v,
        u_min=u_min,
        u_max=u_max,
        sigma=sigma,
        Q=Q,
        R=R,
        Qf=Qf,
        ego_radius=ego_radius,
        obs_radius=obs_radius,
        plan_extra_margin=plan_extra_margin,
        eval_extra_margin=eval_extra_margin,
        x_ahead=6.0,
        x_behind=4.0,
        y_halfspan=2.8,
        max_obs_draw=20,
        dyn_w_max=np.deg2rad(234.08634709537696),
    )

    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    print(
        f"[setup] csv={args.csv}, requested_gpu={int(bool(args.use_gpu))}, "
        f"torch_cuda_available={int(bool(torch.cuda.is_available()))}, device={device}",
        flush=True,
    )
    common_kwargs = dict(
        dt=dt,
        T=T,
        M=M,
        lam=lam,
        noise_sigma=sigma,
        u_min=u_min,
        u_max=u_max,
        I=1,
        device=device,
        dtype=torch.float32,
        verbose=False,
        dyn_kwargs=dict(w_max=float(params["dyn_w_max"])),
        cost_kwargs=dict(ref=None, Q=None, R=None, Qf=None, O_mean=None, radii=None, obs_w=0.0),
    )

    alg_specs = [
        {"name": "MPPI", "kind": "plain"},
        {"name": "RAMPPI", "kind": "ra"},
        {"name": "DRMPPI", "kind": "dr"},
        {"name": "DRAMPPI", "kind": "dra"},
    ]

    controllers = []
    for spec in alg_specs:
        if spec["kind"] == "plain":
            ctrl = PlainMPPIClass(
                dynamics=plain_mod.dyn_diffdrive,
                running_cost=plain_mod.running_cost_lane_obs,
                terminal_cost=plain_mod.terminal_cost_track,
                **common_kwargs,
            )
        elif spec["kind"] == "ra":
            ctrl = RAMPPIClass(
                dynamics=ra_mod.dyn_diffdrive,
                running_cost=ra_mod.running_cost_lane_obs,
                terminal_cost=ra_mod.terminal_cost_track,
                cvar_alpha=float(args.alpha),
                cvar_N=6,
                obs_pos_sigma=(0.06, 0.10),
                obs_noise_mode="per_step",
                **common_kwargs,
            )
        elif spec["kind"] == "dr":
            ctrl = DRMPPIClass(
                dynamics=dr_mod.dyn_diffdrive,
                running_cost=dr_mod.running_cost_lane_obs,
                terminal_cost=dr_mod.terminal_cost_track,
                cvar_alpha=float(args.alpha),
                cvar_N=6,
                obs_pos_sigma=(0.06, 0.10),
                obs_noise_mode="per_step",
                dr_eps_cvar=0.05,
                **common_kwargs,
            )
        elif spec["kind"] == "dra":
            ctrl = dra_mod.DRA_MPPI(
                dt=dt,
                T=T,
                M=M,
                lam=lam,
                sigma=sigma,
                Q=Q,
                R=R,
                Qf=Qf,
                u_min=u_min,
                u_max=u_max,
                I=1,
                sigma_cp=0.05,
                Nmc=400,
                omega_soft=10.0,
                omega_hard=1000.0,
                obs_pos_sigma=(0.06, 0.10),
                mc_chunk=50,
                seed=int(args.seed),
                device=device,
                dtype=torch.float32,
            )
        else:
            raise RuntimeError("Unknown controller spec")
        controllers.append((spec, ctrl))

    master = np.random.default_rng(args.seed)
    run_seeds = master.integers(0, 2**31 - 1, size=args.runs, dtype=np.int64)

    per_alg = {}
    for spec, _ in controllers:
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
        for spec, ctrl in controllers:
            print(f"  [controller] {spec['name']} ...", flush=True)
            ctrl.U[:] = 0.0
            ctrl.U[-1] = np.array([v_des, 0.0], dtype=np.float32)
            ctrl.U_cpu = ctrl.U

            total_qr_cost, dists, safety_violated, collided, max_compute_s = run_episode(
                ctrl=ctrl,
                kind=spec["kind"],
                obs_csv=obs_csv,
                times=times,
                params=params,
                run_seed=run_seed,
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

    # Save run-level raw metrics
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

    # Paper table: rows = metrics, columns = algorithms
    row_names = [
        "1) Min distance across runs (min of run minima)",
        "2) Run-min distance (mean ± std)",
        "3) Safety violation probability using (r + r_s)",
        "4) Collision probability using (r)",
        "5) Total cost Q+Sigma^-2 only (mean ± std)",
        "6) Max compute time across runs (ms)",
        "7) Run-max compute time (mean ± std) (ms)",
    ]

    algorithm_names = [spec["name"] for spec, _ in controllers]
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

    # Also keep a compact numeric summary for downstream processing.
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

    # Plots for requested run-level distributions.
    make_three_plots(
        metric_key="run_min_distance",
        title_root="Minimum Separation Distance Across Runs",
        x_label="Minimum separation distance (m)",
        per_alg={k: np.asarray(v["run_min_dist"], dtype=np.float64) for k, v in per_alg.items()},
        outdir=args.outdir,
    )
    make_three_plots(
        metric_key="total_q_sigma_inv_stage_cost",
        title_root="Total Tracking and Control Cost Across Runs",
        x_label=r"Total cost, $\sum_k (e_k^\top Q e_k + u_k^\top \Sigma^{-2} u_k)$",
        per_alg={k: np.asarray(v["total_qr_cost"], dtype=np.float64) for k, v in per_alg.items()},
        outdir=args.outdir,
    )
    make_three_plots(
        metric_key="run_max_compute_time_ms",
        title_root="Maximum Per-Run Solve Time",
        x_label="Maximum solve time per run (ms)",
        per_alg={k: 1000.0 * np.asarray(v["run_max_compute_time_s"], dtype=np.float64) for k, v in per_alg.items()},
        outdir=args.outdir,
    )
    make_three_plots(
        metric_key="safety_violation_indicator",
        title_root="Safety-Violation Indicator Across Runs",
        x_label="Safety-violation indicator",
        per_alg={k: np.asarray(v["safety_violation"], dtype=np.float64) for k, v in per_alg.items()},
        outdir=args.outdir,
    )
    make_three_plots(
        metric_key="collision_indicator",
        title_root="Collision Indicator Across Runs",
        x_label="Collision indicator",
        per_alg={k: np.asarray(v["collision"], dtype=np.float64) for k, v in per_alg.items()},
        outdir=args.outdir,
    )

    print("\n=== Paper Table (rows=metrics, cols=algorithms) ===", flush=True)
    print(table.to_string(index=False), flush=True)
    print(f"\n[OK] wrote: {out_table}", flush=True)
    print(f"[OK] wrote: {out_compact}", flush=True)
    print(f"[OK] wrote: {out_raw}", flush=True)
    print(f"[OK] plots saved in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
