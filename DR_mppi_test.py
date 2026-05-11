#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DR_mppi.py (Torch)
- Uses DR_MPPI from mppi_class.py (Torch-only)
- Keeps the SAME loop/plot logic as your CuPy script
- No CuPy import here

Run:
  python3 DR_mppi.py
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import matplotlib.image as mpimg
import torch

from mppi_class import DR_MPPI

Tensor = torch.Tensor


# ============================================================
# Utility & Dynamics (CPU)
# ============================================================

def angle_wrap(th: float) -> float:
    return (th + np.pi) % (2 * np.pi) - np.pi


def set_img_pose(img_artist, x, y, phi, length_along_heading, width_lateral, ax):
    L = float(length_along_heading) * 2.0
    W = float(width_lateral) * 2.0
    img_artist.set_extent([-L / 2.0, L / 2.0, -W / 2.0, W / 2.0])
    tr = Affine2D().rotate(phi).translate(x, y) + ax.transData
    img_artist.set_transform(tr)


def adaptive_follow_speed(
    obs_df: pd.DataFrame,
    ego_x: float,
    lane_y: float,
    ego_v: float,
    cruise_v: float,
    lane_tol: float,
    min_gap: float,
    time_headway: float,
    gap_gain: float,
    ego_radius: float,
    obs_radius: float,
    extra_margin: float = 0.0,
) -> float:
    if len(obs_df) == 0:
        return float(cruise_v)

    xs = obs_df["x"].to_numpy(dtype=float)
    ys = obs_df["y"].to_numpy(dtype=float)
    vs = obs_df["v"].to_numpy(dtype=float)

    dx = xs - float(ego_x)
    same_lane = np.abs(ys - float(lane_y)) <= float(lane_tol)
    ahead = dx > 0.0
    lead_mask = same_lane & ahead

    if not np.any(lead_mask):
        return float(cruise_v)

    lead_indices = np.flatnonzero(lead_mask)
    lead_idx = int(lead_indices[np.argmin(dx[lead_mask])])

    lead_dx = float(dx[lead_idx])
    lead_v = float(vs[lead_idx])
    clearance = float(ego_radius + obs_radius + extra_margin)
    gap = max(0.0, lead_dx - clearance)
    safe_gap = float(min_gap) + float(time_headway) * max(float(ego_v), 0.0)

    target_v = lead_v + float(gap_gain) * (gap - safe_gap)
    target_v = float(np.clip(target_v, 0.0, float(cruise_v)))

    if gap <= safe_gap:
        target_v = min(target_v, lead_v)

    return target_v


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


def diffdrive_dynamics_cpu(x, u, dt, v_min=0.0, v_max=10.0, w_max=np.deg2rad(180.0)):
    px, py, psi = float(x[0]), float(x[1]), float(x[2])
    v, w = float(u[0]), float(u[1])

    v = np.clip(v, v_min, v_max)
    w = np.clip(w, -w_max, w_max)

    px += dt * v * np.cos(psi)
    py += dt * v * np.sin(psi)
    psi = angle_wrap(psi + dt * w)

    return np.array([px, py, psi], dtype=float)


# ============================================================
# Torch dynamics & costs (generic, batched)
# ============================================================

def dyn_diffdrive(X: Tensor, U: Tensor, dt: float, w_max: float = float(np.deg2rad(180.0)), **_kwargs) -> Tensor:
    px, py, psi = X[:, 0], X[:, 1], X[:, 2]
    v, w = U[:, 0], U[:, 1]
    w = torch.clamp(w, -w_max, w_max)

    px = px + dt * v * torch.cos(psi)
    py = py + dt * v * torch.sin(psi)
    psi = (psi + dt * w + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.stack([px, py, psi], dim=1)


def running_cost_lane_obs(
    X: Tensor, U: Tensor, t: int,
    ref: Tensor,
    Q: Tensor, R: Tensor,
    O_mean: Tensor | None = None,
    radii: Tensor | None = None,
    obs_w: float = 0.0,   # keep 0 for DR/CVaR feasibility-only (avoid double counting)
    y_min: Tensor | None = None,
    y_max: Tensor | None = None,
    boundary_w: float = 0.0,
    **_kwargs,
) -> Tensor:
    e = X - ref.unsqueeze(0)
    e[:, 2] = (e[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi
    cost = (e * e) @ Q + torch.sum((U * U) * R.unsqueeze(0), dim=1)

    if boundary_w != 0.0 and y_min is not None and y_max is not None:
        y = X[:, 1]
        below_violation = torch.clamp(y_min - y, min=0.0)
        above_violation = torch.clamp(y - y_max, min=0.0)
        cost = cost + float(boundary_w) * (below_violation * below_violation + above_violation * above_violation)

    # optional obstacle penalty (normally disabled when using feasibility filter)
    if obs_w != 0.0 and O_mean is not None and radii is not None and O_mean.shape[1] > 0:
        O_t = O_mean[t]  # (K,2)
        diff = X[:, :2].unsqueeze(1) - O_t.unsqueeze(0)  # (M,K,2)
        dist = torch.linalg.norm(diff, dim=-1)           # (M,K)
        viol = torch.clamp(radii.unsqueeze(0) - dist, min=0.0)
        cost = cost + float(obs_w) * torch.sum(viol * viol, dim=1)

    return cost


def terminal_cost_track(
    X: Tensor, t_final: int,
    ref: Tensor, Qf: Tensor,
    **_kwargs,
) -> Tensor:
    e = X - ref.unsqueeze(0)
    e[:, 2] = (e[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi
    return (e * e) @ Qf


# ============================================================
# CSV Loader (CPU)
# ============================================================

class MovingObstacleCSV:
    def __init__(
        self,
        csv_path: str,
        time_col="t",
        id_col="id",
        x_col="x",
        y_col="y",
        phi_col="phi",
        v_col="v",
        a_col="a",
        phi_dot_col="phi_dot",
        x_offset=0.0,
        y_offset=0.0,
        pos_scale=1.0,
        vel_scale=1.0,
        acc_scale=1.0,
        time_round=6,
    ):
        self.df = pd.read_csv(csv_path)

        required = [time_col, id_col, x_col, y_col, phi_col, v_col, a_col, phi_dot_col]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}. Found: {list(self.df.columns)}")

        self.time_col = time_col
        self.id_col = id_col
        self.x_col = x_col
        self.y_col = y_col
        self.phi_col = phi_col
        self.v_col = v_col
        self.a_col = a_col
        self.phi_dot_col = phi_dot_col

        df = self.df.copy()
        df[x_col] = (df[x_col].astype(float) - float(x_offset)) * float(pos_scale)
        df[y_col] = (df[y_col].astype(float) - float(y_offset)) * float(pos_scale)
        df[v_col] = df[v_col].astype(float) * float(vel_scale)
        df[a_col] = df[a_col].astype(float) * float(acc_scale)

        df["_tkey"] = np.round(df[time_col].astype(float), time_round)

        self.df = df
        self.times = np.array(sorted(df["_tkey"].unique()), dtype=float)

        if len(self.times) < 2:
            raise ValueError("Not enough distinct time samples in CSV to compute dt.")

    def nearest_time(self, t_query: float) -> float:
        idx = int(np.argmin(np.abs(self.times - float(t_query))))
        return float(self.times[idx])

    def obstacles_now(self, t_query: float) -> pd.DataFrame:
        tkey = self.nearest_time(t_query)
        return self.df[self.df["_tkey"] == tkey].copy()

    @staticmethod
    def predict_horizon_from_state(x0, y0, phi0, v0, a, phi_dot, dt, T):
        x = float(x0)
        y = float(y0)
        phi = float(phi0)
        v = float(v0)

        traj = np.zeros((T, 2), dtype=float)
        for j in range(T):
            traj[j, 0] = x
            traj[j, 1] = y

            x = x + dt * v * np.cos(phi)
            y = y + dt * v * np.sin(phi)
            phi = angle_wrap(phi + dt * float(phi_dot))
            v = v + dt * float(a)

        return traj

    def build_prediction_for_mppi(self, obs_df: pd.DataFrame, dt: float, T: int, max_obs: int):
        if len(obs_df) == 0:
            return (
                np.array([], dtype=int),
                np.zeros((T, 0, 2), dtype=float),
                dict(ids=np.array([], dtype=int), xy=np.zeros((0, 2)), phi=np.array([])),
            )

        if len(obs_df) > max_obs:
            obs_df = obs_df.iloc[:max_obs].copy()

        ids = obs_df[self.id_col].to_numpy(dtype=int)
        xs = obs_df[self.x_col].to_numpy(dtype=float)
        ys = obs_df[self.y_col].to_numpy(dtype=float)
        phis = obs_df[self.phi_col].to_numpy(dtype=float)
        vs = obs_df[self.v_col].to_numpy(dtype=float)
        accs = obs_df[self.a_col].to_numpy(dtype=float)
        phidots = obs_df[self.phi_dot_col].to_numpy(dtype=float)

        K = len(ids)
        O_mean = np.zeros((T, K, 2), dtype=float)
        for k in range(K):
            O_mean[:, k, :] = self.predict_horizon_from_state(
                xs[k], ys[k], phis[k], vs[k], accs[k], phidots[k], dt, T
            )

        draw_pack = dict(ids=ids, xy=np.stack([xs, ys], axis=1), phi=phis)
        return ids, O_mean, draw_pack


# ============================================================
# Main Simulation
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DR-MPPI car simulation.")
    parser.add_argument("--save", action="store_true", help="Save replay data for plot_car.py")
    args = parser.parse_args()

    np.random.seed(3)

    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "Data/moderately_aggressive_lane_changes.csv")

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

    dt = float(np.round(np.min(np.diff(obs_csv.times)), 6))
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"Invalid dt computed from CSV: dt={dt}")

    # Core controller params matched to mppi_1.py
    T = 20
    M = 1200
    lam = 2.0
    sigma = np.array([2.0, 1.04], dtype=np.float32)
    u_min = np.array([0.0, -np.deg2rad(180.0)], dtype=np.float32)
    u_max = np.array([5.0, np.deg2rad(180.0)], dtype=np.float32)

    Q = np.array([0.001, 5., 0.001], dtype=np.float32)
    Qf = np.array([0.5, 50.0, 1.0], dtype=np.float32)
    R = np.array([0.004, 0.004], dtype=np.float32)

    # DR-CVaR settings (autotuned)
    cvar_alpha = 0.95
    cvar_N = 10
    obs_pos_sigma = (0.25 * pos_scale * 0.633807192247849, 0.25 * pos_scale * 1.15817673668582)
    dr_eps_cvar = 0.005487046696714133
    obs_noise_mode = "per_step"   # "static" or "per_step"
    dyn_w_max = float(np.deg2rad(180.0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mppi = DR_MPPI(
        dt=dt, T=T, M=M, lam=lam,
        noise_sigma=sigma,
        u_min=u_min, u_max=u_max,
        dynamics=dyn_diffdrive,
        running_cost=running_cost_lane_obs,
        terminal_cost=terminal_cost_track,
        device=device,
        dtype=torch.float32,
        cvar_alpha=cvar_alpha,
        cvar_N=cvar_N,
        obs_pos_sigma=obs_pos_sigma,
        obs_noise_mode=obs_noise_mode,
        dr_eps_cvar=dr_eps_cvar,
        dyn_kwargs=dict(w_max=dyn_w_max),
        cost_kwargs=dict(
            ref=None,
            Q=None,
            R=None,
            Qf=None,
            O_mean=None,
            radii=None,
            obs_w=0.0,
            y_min=None,
            y_max=None,
            boundary_w=0.0,
        ),
        verbose=False,
    )

    print("[DR_MPPI] device =", mppi.device)

    # Lane setup
    lane_psi = 0.0
    L_ref = 5.0
    v_cruise = 5.0
    v_des = v_cruise
    u_blend_v = 1

    ROAD_CENTER = 1.0
    LANE_W = 0.80
    y_divider = ROAD_CENTER
    y_bottom = ROAD_CENTER - LANE_W
    y_top = ROAD_CENTER + LANE_W
    y_divider_top_1 = y_top
    y_divider_top_2 = y_top + LANE_W
    y_top_outer = y_top + 2.0 * LANE_W
    lane_y = y_divider + 0.5 * LANE_W
    y_min_bound = 0.
    y_max_bound = y_divider_top_1
    boundary_w = 1e4

    # put constant weights/bounds on GPU once
    Q_t = torch.as_tensor(Q, device=mppi.device, dtype=mppi.dtype)
    R_t = torch.as_tensor(R, device=mppi.device, dtype=mppi.dtype)
    Qf_t = torch.as_tensor(Qf, device=mppi.device, dtype=mppi.dtype)
    y_min_t = torch.as_tensor(y_min_bound, device=mppi.device, dtype=mppi.dtype)
    y_max_t = torch.as_tensor(y_max_bound, device=mppi.device, dtype=mppi.dtype)

    x_mppi = np.array([0.0, lane_y, 0.0], dtype=np.float32)

    # Geometry
    ego_length = 4.5 * pos_scale
    ego_width = 1.8 * pos_scale
    obs_length = ego_length
    obs_width = ego_width

    ego_radius = 0.5 * np.sqrt(ego_length**2 + ego_width**2)
    obs_radius = ego_radius
    extra_margin = 0.0
    lead_lane_tol = 0.35
    follow_min_gap = 0.6
    follow_time_headway = 0.8
    follow_gap_gain = 0.8
    v_des_smoothing = 0.35
    ego_v = 0.0

    N_SHOW = 60
    N_SHOW = int(min(max(1, N_SHOW), M))
    max_obs_draw = 20

    x_ahead = 18.0
    x_behind = 10.0
    y_halfspan = 2.8

    save_dir = os.path.join(base_dir, "plot")
    save_path = os.path.join(save_dir, "drmppi_car_simulation.npz")

    steps = min(900, len(obs_csv.times))
    sim_time = np.full((steps,), np.nan, dtype=np.float32)
    solve_ms = np.full((steps,), np.nan, dtype=np.float32)
    ego_speed_hist = np.full((steps,), np.nan, dtype=np.float32)
    v_des_hist = np.full((steps,), np.nan, dtype=np.float32)
    min_dist_hist = np.full((steps,), np.nan, dtype=np.float32)
    safety_hist = np.zeros((steps,), dtype=np.int32)
    collision_hist = np.zeros((steps,), dtype=np.int32)
    X_hist = np.full((steps, 3), np.nan, dtype=np.float32)
    X_path = np.full((steps + 1, 2), np.nan, dtype=np.float32)
    X_path[0, :] = x_mppi[:2]
    pred_nominal_xy = np.full((steps, T + 1, 2), np.nan, dtype=np.float32)
    pred_samples_xy = np.full((steps, T + 1, N_SHOW, 2), np.nan, dtype=np.float32)
    obs_xy = np.full((steps, max_obs_draw, 2), np.nan, dtype=np.float32)
    obs_phi = np.full((steps, max_obs_draw), np.nan, dtype=np.float32)
    obs_ids = np.full((steps, max_obs_draw), -1, dtype=np.int32)
    K_hist = np.zeros((steps,), dtype=np.int32)
    xlim_hist = np.full((steps, 2), np.nan, dtype=np.float32)
    ylim_hist = np.full((steps, 2), np.nan, dtype=np.float32)

    # Solve-time stats (min/max)
    min_solve = float("inf")
    max_solve = 0.0
    safety_threshold = float(ego_radius + obs_radius + extra_margin)
    collision_threshold = float(ego_radius + obs_radius)
    ever_safety_violated = False
    ever_collided = False
    run_min_dist = float("inf")

    for k in range(steps):
        t_now = float(obs_csv.times[k])
        obs_now = obs_csv.obstacles_now(t_now)
        sim_time[k] = t_now

        dx = obs_now["x"].to_numpy(float) - float(x_mppi[0])
        dy = obs_now["y"].to_numpy(float) - float(x_mppi[1])
        mask = (dx > -x_behind) & (dx < x_ahead + 10.0) & (np.abs(dy) < (y_halfspan + 2.0))
        obs_now = obs_now[mask].copy()

        if len(obs_now) > 0:
            dist2 = (obs_now["x"] - float(x_mppi[0]))**2 + (obs_now["y"] - float(x_mppi[1]))**2
            obs_now = obs_now.iloc[np.argsort(dist2.to_numpy())]
            obs_now = obs_now.iloc[:max_obs_draw].copy()

        v_target = adaptive_follow_speed(
            obs_now,
            ego_x=float(x_mppi[0]),
            lane_y=lane_y,
            ego_v=ego_v,
            cruise_v=v_cruise,
            lane_tol=lead_lane_tol,
            min_gap=follow_min_gap,
            time_headway=follow_time_headway,
            gap_gain=follow_gap_gain,
            ego_radius=ego_radius,
            obs_radius=obs_radius,
            extra_margin=extra_margin,
        )
        v_des = float((1.0 - v_des_smoothing) * v_des + v_des_smoothing * v_target)

        ids, O_mean, draw_pack = obs_csv.build_prediction_for_mppi(obs_df=obs_now, dt=dt, T=T, max_obs=max_obs_draw)
        K = len(ids)

        if K > 0:
            radii_for_mppi = np.full((K,), obs_radius + ego_radius + extra_margin, dtype=np.float32)
        else:
            radii_for_mppi = np.array([], dtype=np.float32)

        ref = np.array([float(x_mppi[0]) + L_ref, lane_y, lane_psi], dtype=np.float32)

        # ---- update cost kwargs on GPU (this is REQUIRED; DR_MPPI reads O_mean/radii from here) ----
        mppi.cost_kwargs["ref"] = torch.as_tensor(ref, device=mppi.device, dtype=mppi.dtype)
        mppi.cost_kwargs["Q"] = Q_t
        mppi.cost_kwargs["R"] = R_t
        mppi.cost_kwargs["Qf"] = Qf_t
        mppi.cost_kwargs["y_min"] = y_min_t
        mppi.cost_kwargs["y_max"] = y_max_t
        mppi.cost_kwargs["boundary_w"] = boundary_w

        if O_mean is not None and O_mean.shape[1] > 0:
            mppi.cost_kwargs["O_mean"] = torch.as_tensor(O_mean, device=mppi.device, dtype=mppi.dtype)
            mppi.cost_kwargs["radii"] = torch.as_tensor(radii_for_mppi, device=mppi.device, dtype=mppi.dtype)
        else:
            mppi.cost_kwargs["O_mean"] = None
            mppi.cost_kwargs["radii"] = None

        # ---- timing (CUDA is async) ----
        if mppi.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        U, Xsamp = mppi.plan(
            x_mppi,
            return_samples=True,
            n_show=N_SHOW,
            show_seed=0,
        )

        if mppi.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        solve_time = t1 - t0
        solve_ms[k] = float(solve_time * 1000.0)
        min_solve = min(min_solve, solve_time)
        max_solve = max(max_solve, solve_time)

        # prediction rollout (CPU) for plotting
        x_pred = x_mppi.copy()
        pred_nominal_xy[k, 0, 0] = float(x_pred[0])
        pred_nominal_xy[k, 0, 1] = float(x_pred[1])
        for t in range(T):
            x_pred = diffdrive_dynamics_cpu(
                x_pred, U[t], dt, v_max=float(u_max[0]), w_max=dyn_w_max
            ).astype(np.float32)
            pred_nominal_xy[k, t + 1, 0] = float(x_pred[0])
            pred_nominal_xy[k, t + 1, 1] = float(x_pred[1])

        # rollout fan
        if Xsamp is not None:
            ns = min(N_SHOW, Xsamp.shape[1])
            pred_samples_xy[k, :, :ns, :] = Xsamp[:, :ns, :2].astype(np.float32)

        # apply first control
        u0 = U[0].copy()
        u0[0] = np.clip(u_blend_v * u0[0] + (1.0 - u_blend_v) * v_des, 0.0, float(u_max[0]))
        u0[0] = min(float(u0[0]), float(v_des))
        x_mppi = diffdrive_dynamics_cpu(
            x_mppi, u0, dt, v_max=float(u_max[0]), w_max=dyn_w_max
        ).astype(np.float32)
        ego_v = float(u0[0])
        ego_speed_hist[k] = ego_v
        v_des_hist[k] = float(v_des)

        k_next = min(k + 1, steps - 1)
        t_next = float(obs_csv.times[k_next])
        obs_xy_eval = all_obstacles_xy(obs_csv, t_next)
        dmin = min_dist_to_obstacles(x_mppi[:2], obs_xy_eval)
        step_safety_violated = bool(np.isfinite(dmin) and dmin < safety_threshold)
        step_collided = bool(np.isfinite(dmin) and dmin < collision_threshold)
        min_dist_hist[k] = float(dmin)
        safety_hist[k] = int(step_safety_violated)
        collision_hist[k] = int(step_collided)
        ever_safety_violated = ever_safety_violated or step_safety_violated
        ever_collided = ever_collided or step_collided
        if np.isfinite(dmin):
            run_min_dist = min(run_min_dist, float(dmin))

        print(
            f"[DR_MPPI] solve={solve_time*1000:8.2f} ms "
            f"| dmin_next={dmin:6.3f} "
            f"| safety={int(step_safety_violated)} "
            f"| collided={int(step_collided)}"
        )

        # shift nominal controls (warm start)
        mppi.U[:-1] = mppi.U[1:]
        mppi.U[:, 0] = np.minimum(mppi.U[:, 0], v_des)
        mppi.U[-1] = np.array([v_des, 0.0], dtype=np.float32)
        mppi.U_cpu = mppi.U  # keep internal consistent

        x_left = float(x_mppi[0] - 9.0)
        x_right = float(x_mppi[0] + 9.0)
        y_low = float(y_bottom - 0.2)
        y_high = float(y_top_outer + 0.2)
        xlim_hist[k, :] = [x_left, x_right]
        ylim_hist[k, :] = [y_low, y_high]

        X_hist[k, :] = x_mppi
        X_path[k + 1, :] = x_mppi[:2]
        K_hist[k] = K
        if K > 0:
            kk = min(K, max_obs_draw)
            obs_xy[k, :kk, :] = draw_pack["xy"][:kk, :2].astype(np.float32)
            obs_phi[k, :kk] = draw_pack["phi"][:kk].astype(np.float32)
            obs_ids[k, :kk] = draw_pack["ids"][:kk].astype(np.int32)

    run_min_dist_print = run_min_dist if np.isfinite(run_min_dist) else float("nan")
    print(
        f"[DR_MPPI] final_status "
        f"| min_dist={run_min_dist_print:.3f} "
        f"| safety_violation={int(ever_safety_violated)} "
        f"| collided={int(ever_collided)}"
    )

    if args.save:
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(
            save_path,
            method=np.array("drmppi"),
            dt=np.array(dt, dtype=np.float32),
            T=np.array(T, dtype=np.int32),
            M=np.array(M, dtype=np.int32),
            N_SHOW=np.array(N_SHOW, dtype=np.int32),
            max_obs_draw=np.array(max_obs_draw, dtype=np.int32),
            lane_y=np.array(lane_y, dtype=np.float32),
            y_bottom=np.array(y_bottom, dtype=np.float32),
            y_top=np.array(y_top, dtype=np.float32),
            y_divider=np.array(y_divider, dtype=np.float32),
            y_divider_top_1=np.array(y_divider_top_1, dtype=np.float32),
            y_divider_top_2=np.array(y_divider_top_2, dtype=np.float32),
            y_top_outer=np.array(y_top_outer, dtype=np.float32),
            ego_length=np.array(ego_length, dtype=np.float32),
            ego_width=np.array(ego_width, dtype=np.float32),
            obs_length=np.array(obs_length, dtype=np.float32),
            obs_width=np.array(obs_width, dtype=np.float32),
            sim_time=sim_time,
            solve_ms=solve_ms,
            ego_speed_hist=ego_speed_hist,
            v_des_hist=v_des_hist,
            min_dist_hist=min_dist_hist,
            safety_hist=safety_hist,
            collision_hist=collision_hist,
            X_hist=X_hist,
            X_path=X_path,
            pred_nominal_xy=pred_nominal_xy,
            pred_samples_xy=pred_samples_xy,
            obs_xy=obs_xy,
            obs_phi=obs_phi,
            obs_ids=obs_ids,
            K_hist=K_hist,
            xlim_hist=xlim_hist,
            ylim_hist=ylim_hist,
        )
        print(f"[SAVE] wrote {save_path}")
