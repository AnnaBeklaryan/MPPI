#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RA_mppi.py (Torch)
- Uses RA_MPPI from mppi_class.py (Torch-only)
- Keeps the SAME loop/plot logic as your CuPy script
- No CuPy import here

Run:
    python3 RA_mppi.py  --obs-update-steps 20 --scenario 2 --save

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

from mppi_class import RA_MPPI

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


def angle_wrap_array(th: np.ndarray) -> np.ndarray:
    return (np.asarray(th, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def heading_from_xy(xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=float)
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    psi = np.unwrap(np.arctan2(dy, dx))
    return angle_wrap_array(psi)


def build_path_from_waypoints(waypoints_xy: np.ndarray, ds: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(waypoints_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        raise ValueError("waypoints_xy must be shape (N,2) with N >= 2")

    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = float(s[-1])
    if total_len <= 0.0:
        raise ValueError("Waypoint path length must be positive.")

    n_samples = max(int(np.ceil(total_len / max(float(ds), 1e-3))) + 1, pts.shape[0])
    sq = np.linspace(0.0, total_len, n_samples)
    x = np.interp(sq, s, pts[:, 0])
    y = np.interp(sq, s, pts[:, 1])
    xy = np.column_stack([x, y])
    psi = heading_from_xy(xy)
    return np.column_stack([xy, psi]).astype(np.float32), sq.astype(np.float32)
def build_roundabout_waypoints(center_x: float, center_y: float, radius: float = 2.05) -> np.ndarray:
    # Scenario-2 ego reference should follow the lower black route:
    # straight from the left, dip below the island on the lower roundabout arc,
    # then continue straight to the right.
    straight_y = center_y - 0.42
    approach = np.array(
        [
            [0.0, straight_y],
            [2.0, straight_y],
            [3.6, straight_y],
            [4.05, straight_y],
        ],
        dtype=np.float32,
    )

    arc_angles_deg = np.array(
        [194.0, 220.0, 250.0, 270.0, 290.0, 320.0, 346.0],
        dtype=np.float32,
    )
    arc = np.column_stack(
        [
            center_x + radius * np.cos(np.deg2rad(arc_angles_deg)),
            center_y + radius * np.sin(np.deg2rad(arc_angles_deg)),
        ]
    ).astype(np.float32)

    exit_lane = np.array(
        [
            [8.25, straight_y],
            [9.8, straight_y],
            [11.6, straight_y],
            [13.8, straight_y],
            [16.0, straight_y],
            [18.2, straight_y],
        ],
        dtype=np.float32,
    )

    return np.vstack([approach, arc, exit_lane]).astype(np.float32)


def reference_from_path(
    x_now: np.ndarray,
    path_xypsi: np.ndarray,
    path_s: np.ndarray,
    lookahead_dist: float,
    last_nearest_idx: int,
    backward_slack: int = 12,
) -> tuple[np.ndarray, int, int]:
    xy = np.asarray(x_now[:2], dtype=float)
    start_idx = max(int(last_nearest_idx) - int(backward_slack), 0)
    local_xy = path_xypsi[start_idx:, :2]
    if local_xy.shape[0] == 0:
        final_idx = path_xypsi.shape[0] - 1
        return path_xypsi[final_idx].copy(), final_idx, final_idx

    d2 = np.sum((local_xy - xy[None, :]) ** 2, axis=1)
    nearest_idx = start_idx + int(np.argmin(d2))
    target_s = min(float(path_s[-1]), float(path_s[nearest_idx]) + float(lookahead_dist))
    target_idx = int(np.searchsorted(path_s, target_s, side="left"))
    target_idx = min(max(target_idx, nearest_idx), path_xypsi.shape[0] - 1)
    return path_xypsi[target_idx].copy(), nearest_idx, target_idx


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
    boundary_k: float = 50.0,
    boundary_d: float = 0.05,
    path_xy: Tensor | None = None,
    path_corridor_radius: float = 0.45,
    path_boundary_w: float = 0.0,
    path_boundary_k: float = 35.0,
    **_kwargs,
) -> Tensor:
    e = X - ref.unsqueeze(0)
    e[:, 2] = (e[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi
    cost = (e * e) @ Q + torch.sum((U * U) * R.unsqueeze(0), dim=1)

    # Sigmoid boundary barrier for straight-road y-limits.
    # This replaces the old quadratic violation cost:
    #   max(y_min - y, 0)^2 + max(y - y_max, 0)^2
    # with a smooth barrier that starts rising before the boundary.
    if boundary_w != 0.0 and y_min is not None and y_max is not None:
        y = X[:, 1]
        k_sig = float(boundary_k)
        d_buf = float(boundary_d)
        lower_barrier = torch.sigmoid(k_sig * (y_min + d_buf - y))
        upper_barrier = torch.sigmoid(k_sig * (y - y_max + d_buf))
        cost = cost + float(boundary_w) * (lower_barrier + upper_barrier)

    # Sigmoid path-corridor boundary for the roundabout scenario.
    # The valid driving corridor is defined around the saved reference path,
    # which works for both straight and curved/roundabout parts.
    if path_boundary_w != 0.0 and path_xy is not None:
        ego_xy = X[:, :2]
        path_points = path_xy.to(device=X.device, dtype=X.dtype)
        dx_path = ego_xy[:, None, 0] - path_points[None, :, 0]
        dy_path = ego_xy[:, None, 1] - path_points[None, :, 1]
        dist_to_path = torch.sqrt(dx_path * dx_path + dy_path * dy_path + 1e-9)
        min_dist_to_path = torch.min(dist_to_path, dim=1).values
        path_barrier = torch.sigmoid(
            float(path_boundary_k) * (min_dist_to_path - float(path_corridor_radius))
        )
        cost = cost + float(path_boundary_w) * path_barrier

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
    parser = argparse.ArgumentParser(description="Run RA-MPPI car simulation.")
    parser.add_argument("--save", action="store_true", help="Save replay data for plot_car.py")
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2],
        default=1,
        help="Scenario 1 keeps the current straight road. Scenario 2 uses a roundabout waypoint sequence.",
    )
    parser.add_argument(
        "--obs-update-steps",
        type=int,
        default=1,
        help=(
            "Read a fresh obstacle CSV sample every N control steps and hold that "
            "same obstacle observation between reads. Control dt is unchanged."
        ),
    )
    parser.add_argument(
        "--obs-update-dt",
        type=float,
        default=None,
        help=(
            "Optional observation update period in seconds. When set, this overrides "
            "--obs-update-steps and uses the nearest available control tick, so control "
            "dt is still unchanged."
        ),
    )
    parser.add_argument(
        "--extra-sim-time",
        type=float,
        default=8.0,
        help=(
            "Optional extra simulation time in seconds after the last CSV timestamp. "
            "Beyond the CSV range, the last available obstacle sample is held."
        ),
    )
    args = parser.parse_args()

    np.random.seed(3)

    base_dir = os.path.dirname(__file__)
    csv_name = "obstacle_data.csv" if int(args.scenario) == 1 else "obstacle_data_2.csv"
    csv_path = os.path.join(base_dir, "Data", csv_name)
    print(f"[OBS] scenario={int(args.scenario)} | csv={csv_name}")

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

    # Controller still runs at the CSV/control timestep, e.g. 0.04 s.
    # The options below only change how often a fresh obstacle measurement is read.
    obs_update_steps = max(1, int(args.obs_update_steps))
    obs_update_dt = None if args.obs_update_dt is None else float(args.obs_update_dt)
    extra_sim_time = float(args.extra_sim_time)
    if obs_update_dt is not None and not (np.isfinite(obs_update_dt) and obs_update_dt > 0.0):
        raise ValueError(f"--obs-update-dt must be positive, got {args.obs_update_dt}")
    if not (np.isfinite(extra_sim_time) and extra_sim_time >= 0.0):
        raise ValueError(f"--extra-sim-time must be >= 0, got {args.extra_sim_time}")

    csv_t_start = float(obs_csv.times[0])
    csv_t_end = float(obs_csv.times[-1])
    sim_steps_total = max(
        len(obs_csv.times),
        int(np.ceil((csv_t_end + extra_sim_time - csv_t_start) / dt - 1e-9)) + 1,
    )
    sim_time_grid = csv_t_start + dt * np.arange(sim_steps_total, dtype=float)

    if obs_update_dt is None:
        print(
            f"[OBS] control_dt={dt:.6f} s | fresh CSV obstacle read every "
            f"{obs_update_steps} control step(s) = {obs_update_steps * dt:.6f} s; "
            "held between reads"
        )
    else:
        print(
            f"[OBS] control_dt={dt:.6f} s | requested fresh CSV obstacle read period "
            f"{obs_update_dt:.6f} s; using nearest control tick and holding between reads"
        )

    print(f"[PLOT] saved obstacle animation data uses true CSV samples every control dt={dt:.6f} s")
    print(
        f"[SIM] csv_time_range=[{csv_t_start:.3f}, {csv_t_end:.3f}] s | "
        f"run_time_range=[{sim_time_grid[0]:.3f}, {sim_time_grid[-1]:.3f}] s"
    )
    if sim_time_grid[-1] > csv_t_end + 1e-9:
        print("[SIM] past the last CSV timestamp, the last available obstacle sample is held.")

    # Core controller params matched to mppi_1.py
    T = 20
    M = 1200
    lam = 2.0
    sigma = np.array([2.0, 1.04], dtype=np.float32)
    u_min = np.array([0.0, -np.deg2rad(180.0)], dtype=np.float32)
    u_max = np.array([5.0, np.deg2rad(180.0)], dtype=np.float32)

    Q = np.array([0.001, 0.01, 0.001], dtype=np.float32)
    Qf = np.array([0.5, 50.0, 1.0], dtype=np.float32)
    R = np.array([0.004, 0.004], dtype=np.float32)

    # RA-CVaR settings (autotuned)
    cvar_alpha = 0.95
    cvar_N = 10
    obs_pos_sigma = (0.07, 0.07)
    obs_noise_mode = "per_step"   # "static" or "per_step"
    risk_cost_A = 10.0
    risk_cost_Cu = 0.0
    dyn_w_max = float(np.deg2rad(180.0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mppi = RA_MPPI(
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
        risk_cost_A=risk_cost_A,
        risk_cost_Cu=risk_cost_Cu,
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
            boundary_k=50.0,
            boundary_d=0.05,
            path_xy=None,
            path_corridor_radius=0.45,
            path_boundary_w=0.0,
            path_boundary_k=35.0,
        ),
        verbose=False,
    )

    print("[RA_MPPI] device =", mppi.device)

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

    # Sigmoid boundary-barrier tuning for scenario 1 straight-road limits.
    boundary_w = 500.0
    boundary_k = 50.0
    boundary_d = 0.05

    # Sigmoid path-corridor boundary tuning for scenario 2 roundabout.
    path_boundary_w = 0.0
    path_boundary_k = 35.0
    path_corridor_radius = 0.45

    # put constant weights/bounds on GPU once
    Q_t = torch.as_tensor(Q, device=mppi.device, dtype=mppi.dtype)
    R_t = torch.as_tensor(R, device=mppi.device, dtype=mppi.dtype)
    Qf_t = torch.as_tensor(Qf, device=mppi.device, dtype=mppi.dtype)
    y_min_t = torch.as_tensor(y_min_bound, device=mppi.device, dtype=mppi.dtype)
    y_max_t = torch.as_tensor(y_max_bound, device=mppi.device, dtype=mppi.dtype)
    scenario = int(args.scenario)
    lane_psi = 0.0
    L_ref = 5.0
    v_des = 5.0
    u_blend_v = 1
    ref_path_xypsi = None
    ref_path_s = None
    ref_nearest_idx = 0
    roundabout_center_x = np.nan
    roundabout_center_y = np.nan

    if scenario == 1:
        x_mppi = np.array([0.0, lane_y, 0.0], dtype=np.float32)
    else:
        roundabout_center_x = 6.1
        roundabout_center_y = lane_y
        L_ref = 1.6
        v_des = 2.8
        boundary_w = 0.0
        path_boundary_w = 300.0
        ref_waypoints_xy = build_roundabout_waypoints(roundabout_center_x, roundabout_center_y)
        ref_path_xypsi, ref_path_s = build_path_from_waypoints(ref_waypoints_xy, ds=0.05)
        x_mppi = ref_path_xypsi[0].astype(np.float32)

    # Geometry
    ego_length = 4.5 * pos_scale
    ego_width = 1.8 * pos_scale
    obs_length = ego_length
    obs_width = ego_width

    ego_radius = 0.5 * np.sqrt(ego_length**2 + ego_width**2)
    obs_radius = ego_radius
    extra_margin = 0.0

    N_SHOW = 60
    N_SHOW = int(min(max(1, N_SHOW), M))
    max_obs_draw = 20

    x_ahead = 18.0
    x_behind = 4.0
    y_halfspan = 2.8

    save_dir = os.path.join(base_dir, "plot")
    save_path = os.path.join(save_dir, "ramppi_car_simulation.npz")

    steps = min(900, len(sim_time_grid))
    sim_time_grid = sim_time_grid[:steps]
    sim_time = np.full((steps,), np.nan, dtype=np.float32)
    solve_ms = np.full((steps,), np.nan, dtype=np.float32)
    X_hist = np.full((steps, 3), np.nan, dtype=np.float32)
    X_path = np.full((steps + 1, 2), np.nan, dtype=np.float32)
    X_path[0, :] = x_mppi[:2]
    pred_nominal_xy = np.full((steps, T + 1, 2), np.nan, dtype=np.float32)
    pred_samples_xy = np.full((steps, T + 1, N_SHOW, 2), np.nan, dtype=np.float32)
    # obs_xy/obs_phi/obs_ids are for plotting and are refreshed from the true CSV
    # at every control step, so the animation remains at dt=0.04 even when the
    # controller uses low-rate/held obstacle observations.
    obs_xy = np.full((steps, max_obs_draw, 2), np.nan, dtype=np.float32)
    obs_phi = np.full((steps, max_obs_draw), np.nan, dtype=np.float32)
    obs_ids = np.full((steps, max_obs_draw), -1, dtype=np.int32)
    K_hist = np.zeros((steps,), dtype=np.int32)

    # These arrays save the low-rate observation actually used by MPPI. They are
    # useful for debugging risk handling, but plot_car.py can keep using obs_xy.
    obs_used_xy = np.full((steps, max_obs_draw, 2), np.nan, dtype=np.float32)
    obs_used_phi = np.full((steps, max_obs_draw), np.nan, dtype=np.float32)
    obs_used_ids = np.full((steps, max_obs_draw), -1, dtype=np.int32)
    K_used_hist = np.zeros((steps,), dtype=np.int32)
    xlim_hist = np.full((steps, 2), np.nan, dtype=np.float32)
    ylim_hist = np.full((steps, 2), np.nan, dtype=np.float32)
    obs_read_time = np.full((steps,), np.nan, dtype=np.float32)
    obs_is_fresh = np.zeros((steps,), dtype=np.int8)

    # Zero-order-hold obstacle observation state.
    held_obs_df = None
    held_obs_t = float("nan")
    last_obs_update_k = -10**9
    next_obs_update_t = float(sim_time_grid[0])

    # Solve-time stats (min/max)
    min_solve = float("inf")
    max_solve = 0.0
    safety_threshold = float(ego_radius + obs_radius + extra_margin)
    collision_threshold = float(ego_radius + obs_radius)
    ever_safety_violated = False
    ever_collided = False
    run_min_dist = float("inf")

    for k in range(steps):
        t_now = float(sim_time_grid[k])
        sim_time[k] = t_now

        if obs_update_dt is None:
            need_fresh_obs = (held_obs_df is None) or ((k - last_obs_update_k) >= obs_update_steps)
        else:
            need_fresh_obs = (held_obs_df is None) or (t_now + 1e-9 >= next_obs_update_t)

        if need_fresh_obs:
            new_obs_t = float(obs_csv.nearest_time(t_now))
            held_obs_df = obs_csv.obstacles_now(t_now)
            last_obs_update_k = k
            obs_is_fresh[k] = int((not np.isfinite(held_obs_t)) or (abs(new_obs_t - held_obs_t) > 1e-9))
            held_obs_t = new_obs_t

            if obs_update_dt is not None:
                while next_obs_update_t <= t_now + 1e-9:
                    next_obs_update_t += obs_update_dt

        # This is the obstacle observation available to the controller.
        # Between fresh reads, it is intentionally held constant.
        obs_ctrl_now = held_obs_df.copy()
        obs_read_time[k] = held_obs_t
        obs_age = float(t_now - held_obs_t)

        dx = obs_ctrl_now["x"].to_numpy(float) - float(x_mppi[0])
        dy = obs_ctrl_now["y"].to_numpy(float) - float(x_mppi[1])
        if scenario == 1:
            mask = (dx > -x_behind) & (dx < x_ahead + 10.0) & (np.abs(dy) < (y_halfspan + 2.0))
        else:
            mask = (dx * dx + dy * dy) < float(9.5**2)
        obs_ctrl_now = obs_ctrl_now[mask].copy()

        if len(obs_ctrl_now) > 0:
            dist2 = (obs_ctrl_now["x"] - float(x_mppi[0]))**2 + (obs_ctrl_now["y"] - float(x_mppi[1]))**2
            obs_ctrl_now = obs_ctrl_now.iloc[np.argsort(dist2.to_numpy())]
            obs_ctrl_now = obs_ctrl_now.iloc[:max_obs_draw].copy()

        ids, O_mean, draw_pack_ctrl = obs_csv.build_prediction_for_mppi(obs_df=obs_ctrl_now, dt=dt, T=T, max_obs=max_obs_draw)
        K_used = len(ids)

        if K_used > 0:
            radii_for_mppi = np.full((K_used,), obs_radius + ego_radius + extra_margin, dtype=np.float32)
        else:
            radii_for_mppi = np.array([], dtype=np.float32)

        if scenario == 1:
            ref = np.array([float(x_mppi[0]) + L_ref, lane_y, lane_psi], dtype=np.float32)
        else:
            ref, ref_nearest_idx, _ref_target_idx = reference_from_path(
                x_now=x_mppi,
                path_xypsi=ref_path_xypsi,
                path_s=ref_path_s,
                lookahead_dist=L_ref,
                last_nearest_idx=ref_nearest_idx,
            )
            ref = np.asarray(ref, dtype=np.float32)

        # ---- update cost kwargs on GPU (this is REQUIRED; RA_MPPI reads O_mean/radii from here) ----
        mppi.cost_kwargs["ref"] = torch.as_tensor(ref, device=mppi.device, dtype=mppi.dtype)
        mppi.cost_kwargs["Q"] = Q_t
        mppi.cost_kwargs["R"] = R_t
        mppi.cost_kwargs["Qf"] = Qf_t
        mppi.cost_kwargs["y_min"] = y_min_t
        mppi.cost_kwargs["y_max"] = y_max_t
        mppi.cost_kwargs["boundary_w"] = boundary_w
        mppi.cost_kwargs["boundary_k"] = boundary_k
        mppi.cost_kwargs["boundary_d"] = boundary_d

        if scenario == 2 and ref_path_xypsi is not None:
            mppi.cost_kwargs["path_xy"] = torch.as_tensor(
                ref_path_xypsi[:, :2],
                device=mppi.device,
                dtype=mppi.dtype,
            )
            mppi.cost_kwargs["path_boundary_w"] = path_boundary_w
            mppi.cost_kwargs["path_boundary_k"] = path_boundary_k
            mppi.cost_kwargs["path_corridor_radius"] = path_corridor_radius
        else:
            mppi.cost_kwargs["path_xy"] = None
            mppi.cost_kwargs["path_boundary_w"] = 0.0

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
        x_mppi = diffdrive_dynamics_cpu(
            x_mppi, u0, dt, v_max=float(u_max[0]), w_max=dyn_w_max
        ).astype(np.float32)

        k_next = min(k + 1, steps - 1)
        t_next = float(sim_time_grid[k_next])
        obs_xy_eval = all_obstacles_xy(obs_csv, t_next)
        dmin = min_dist_to_obstacles(x_mppi[:2], obs_xy_eval)
        step_safety_violated = bool(np.isfinite(dmin) and dmin < safety_threshold)
        step_collided = bool(np.isfinite(dmin) and dmin < collision_threshold)
        ever_safety_violated = ever_safety_violated or step_safety_violated
        ever_collided = ever_collided or step_collided
        if np.isfinite(dmin):
            run_min_dist = min(run_min_dist, float(dmin))

        print(
            f"[RA_MPPI] solve={solve_time*1000:8.2f} ms "
            f"| obs_t={held_obs_t:7.3f} "
            f"| obs_age={obs_age:5.3f}s "
            f"| fresh={int(obs_is_fresh[k])} "
            f"| dmin_next={dmin:6.3f} "
            f"| safety={int(step_safety_violated)} "
            f"| collided={int(step_collided)}"
        )

        # shift nominal controls (warm start)
        mppi.U[:-1] = mppi.U[1:]
        mppi.U[-1] = np.array([v_des, 0.0], dtype=np.float32)
        mppi.U_cpu = mppi.U  # keep internal consistent

        if scenario == 1:
            x_left = float(x_mppi[0] - 9.0)
            x_right = float(x_mppi[0] + 9.0)
            y_low = float(y_bottom - 0.2)
            y_high = float(y_top_outer + 0.2)
        else:
            x_left = float(x_mppi[0] - 6.5)
            x_right = float(x_mppi[0] + 6.5)
            y_low = float(x_mppi[1] - 3.2)
            y_high = float(x_mppi[1] + 3.2)
        xlim_hist[k, :] = [x_left, x_right]
        ylim_hist[k, :] = [y_low, y_high]

        X_hist[k, :] = x_mppi
        X_path[k + 1, :] = x_mppi[:2]

        # Plotting data: always use the true CSV obstacle state at this 0.04-s
        # control/animation frame, not the held observation used by the controller.
        obs_plot_now = obs_csv.obstacles_now(t_now)
        dx_plot = obs_plot_now["x"].to_numpy(float) - float(x_mppi[0])
        dy_plot = obs_plot_now["y"].to_numpy(float) - float(x_mppi[1])
        if scenario == 1:
            mask_plot = (dx_plot > -x_behind) & (dx_plot < x_ahead + 10.0) & (np.abs(dy_plot) < (y_halfspan + 2.0))
        else:
            mask_plot = (dx_plot * dx_plot + dy_plot * dy_plot) < float(9.5**2)
        obs_plot_now = obs_plot_now[mask_plot].copy()

        if len(obs_plot_now) > 0:
            dist2_plot = (obs_plot_now["x"] - float(x_mppi[0]))**2 + (obs_plot_now["y"] - float(x_mppi[1]))**2
            obs_plot_now = obs_plot_now.iloc[np.argsort(dist2_plot.to_numpy())]
            obs_plot_now = obs_plot_now.iloc[:max_obs_draw].copy()

        plot_ids, _, draw_pack_plot = obs_csv.build_prediction_for_mppi(
            obs_df=obs_plot_now, dt=dt, T=1, max_obs=max_obs_draw
        )
        K_plot = len(plot_ids)
        K_hist[k] = K_plot
        if K_plot > 0:
            kk = min(K_plot, max_obs_draw)
            obs_xy[k, :kk, :] = draw_pack_plot["xy"][:kk, :2].astype(np.float32)
            obs_phi[k, :kk] = draw_pack_plot["phi"][:kk].astype(np.float32)
            obs_ids[k, :kk] = draw_pack_plot["ids"][:kk].astype(np.int32)

        # Debug data: save the held/low-rate observation actually used by MPPI.
        K_used_hist[k] = K_used
        if K_used > 0:
            kk_used = min(K_used, max_obs_draw)
            obs_used_xy[k, :kk_used, :] = draw_pack_ctrl["xy"][:kk_used, :2].astype(np.float32)
            obs_used_phi[k, :kk_used] = draw_pack_ctrl["phi"][:kk_used].astype(np.float32)
            obs_used_ids[k, :kk_used] = draw_pack_ctrl["ids"][:kk_used].astype(np.int32)

    run_min_dist_print = run_min_dist if np.isfinite(run_min_dist) else float("nan")
    print(
        f"[RA_MPPI] final_status "
        f"| min_dist={run_min_dist_print:.3f} "
        f"| safety_violation={int(ever_safety_violated)} "
        f"| collided={int(ever_collided)}"
    )

    if args.save:
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(
            save_path,
            method=np.array("ramppi"),
            scenario=np.array(scenario, dtype=np.int32),
            dt=np.array(dt, dtype=np.float32),
            obs_update_steps=np.array(obs_update_steps, dtype=np.int32),
            obs_update_dt=np.array(np.nan if obs_update_dt is None else obs_update_dt, dtype=np.float32),
            obs_read_time=obs_read_time,
            obs_is_fresh=obs_is_fresh,
            obs_used_xy=obs_used_xy,
            obs_used_phi=obs_used_phi,
            obs_used_ids=obs_used_ids,
            K_used_hist=K_used_hist,
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
            roundabout_center_x=np.array(roundabout_center_x, dtype=np.float32),
            roundabout_center_y=np.array(roundabout_center_y, dtype=np.float32),
            ego_length=np.array(ego_length, dtype=np.float32),
            ego_width=np.array(ego_width, dtype=np.float32),
            obs_length=np.array(obs_length, dtype=np.float32),
            obs_width=np.array(obs_width, dtype=np.float32),
            ref_path_xy=np.asarray(ref_path_xypsi[:, :2], dtype=np.float32) if ref_path_xypsi is not None else np.zeros((0, 2), dtype=np.float32),
            sim_time=sim_time,
            solve_ms=solve_ms,
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
