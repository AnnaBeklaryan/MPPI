# -*- coding: utf-8 -*-
"""
FULL COMPLETE SCRIPT: RA-MPPI (Torch) with CVaR feasibility filter + your lane/obstacle scenario + plotting

This script ASSUMES you have:
  - mppi_class.py  (contains the RA_MPPI class I provided: CVaR feasibility filter, obstacle sampling, etc.)
  - obstacle_data.csv
  - Data/car_ego.png and Data/car_obs1.png..car_obs14.png

Key points:
- Uses RA_MPPI from mppi_class.py
- Cost functions are generic torch functions
- CVaR feasibility uses O_mean and radii from mppi_ra.cost_kwargs (as in the class)

Run:
  python RA_mppi.py
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
from typing import Optional

from mppi_class import RA_MPPI

Tensor = torch.Tensor


# ============================================================
# Utility & Dynamics (CPU)
# ============================================================

def angle_wrap(th):
    return (th + np.pi) % (2 * np.pi) - np.pi


def set_img_pose(img_artist, x, y, phi, length_along_heading, width_lateral, ax):
    L = float(length_along_heading) * 2
    W = float(width_lateral) * 2
    img_artist.set_extent([-L / 2.0, L / 2.0, -W / 2.0, W / 2.0])
    tr = Affine2D().rotate(phi).translate(x, y) + ax.transData
    img_artist.set_transform(tr)


def diffdrive_dynamics_cpu(x, u, dt, v_min=0.0, v_max=10.0, w_max=np.deg2rad(180.0)):
    px, py, psi = x
    v, w = u
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be finite and > 0. Got dt={dt}")
    v = np.clip(v, v_min, v_max)
    w = np.clip(w, -w_max, w_max)
    px += dt * v * np.cos(psi)
    py += dt * v * np.sin(psi)
    psi = angle_wrap(psi + dt * w)
    return np.array([px, py, psi], dtype=float)


# ============================================================
# Torch dynamics & costs (batched)
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
    O_mean: Optional[Tensor] = None,
    radii: Optional[Tensor] = None,
    obs_w: float = 0.0,   # IMPORTANT: set to 0.0 when using CVaR feasibility to avoid double counting
    **_kwargs,
) -> Tensor:
    # tracking + control
    e = X - ref.unsqueeze(0)
    e[:, 2] = (e[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi
    cost = (e * e) @ Q + torch.sum((U * U) * R.unsqueeze(0), dim=1)

    # OPTIONAL obstacle penalty (usually keep 0 if CVaR already handles feasibility)
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
            raise ValueError("Not enough timestamps in CSV to compute dt.")

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
    args = parser.parse_args()

    np.random.seed(3)

    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "Data/obstacle_data.csv")

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
        raise ValueError(f"Computed dt is invalid: dt={dt}")

    # MPPI params (matched to DR_mppi.py)
    T = 10
    M = 200
    lam = 2.0
    sigma = np.array([2.0, 1.04], dtype=np.float32)

    u_min = np.array([0.0, -np.deg2rad(180.0)], dtype=np.float32)
    u_max = np.array([10.0, np.deg2rad(180.0)], dtype=np.float32)

    Q = np.array([0.001, 0.001, 0.001], dtype=np.float32)
    Qf = np.array([0.5, 10.0, 1.0], dtype=np.float32)
    R = np.array([1e-5, 1e-5], dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CVaR settings (matched to DR_mppi.py)
    cvar_alpha = 0.95
    cvar_N = 6
    obs_pos_sigma = (0.1, 0.1)
    obs_noise_mode = "per_step"   # "static" or "per_step"
    I = 1
    dyn_w_max = float(np.deg2rad(234.08634709537696))

    # IMPORTANT:
    # When using CVaR feasibility, keep obs_w in running_cost = 0.0 (avoid double counting).
    obs_w_cost = 0.0



    mppi_ra = RA_MPPI(
        dt=dt, T=T, M=M, lam=lam,
        noise_sigma=sigma,
        u_min=u_min, u_max=u_max,
        dynamics=dyn_diffdrive,
        running_cost=running_cost_lane_obs,
        terminal_cost=terminal_cost_track,
        device=device,
        dtype=torch.float32,
        dyn_kwargs=dict(w_max=dyn_w_max),
        cost_kwargs=dict(ref=None, Q=None, R=None, Qf=None, O_mean=None, radii=None, obs_w=0.0),
        I=I,
        verbose=False,
        cvar_alpha=cvar_alpha,
        cvar_N=cvar_N,
        obs_pos_sigma=obs_pos_sigma,
        obs_noise_mode=obs_noise_mode,
    )
    print("[RA_MPPI] device =", mppi_ra.device)

    # Put weights on GPU once
    Q_t  = torch.as_tensor(Q,  device=mppi_ra.device, dtype=mppi_ra.dtype)
    R_t  = torch.as_tensor(R,  device=mppi_ra.device, dtype=mppi_ra.dtype)
    Qf_t = torch.as_tensor(Qf, device=mppi_ra.device, dtype=mppi_ra.dtype)

    lane_psi = 0.0
    L_ref = 14.0
    v_des = 5.0
    u_blend_v = 0.5

    ROAD_CENTER = 1.0
    LANE_W = 0.70

    y_divider = ROAD_CENTER
    y_bottom  = ROAD_CENTER - LANE_W
    y_top     = ROAD_CENTER + LANE_W
    y_divider_top_1 = y_top
    y_divider_top_2 = y_top + LANE_W
    y_top_outer = y_top + 2.0 * LANE_W
    lane_y = y_divider + 0.5 * LANE_W

    x_mppi = np.array([0.0, lane_y, 0.0], dtype=np.float32)

    ego_length = 4.5 * pos_scale
    ego_width  = 1.8 * pos_scale

    obs_radius = 0.5 * np.sqrt(ego_length**2 + ego_width**2)
    ego_radius = obs_radius
    extra_margin = 0.0

    N_SHOW = 60
    N_SHOW = int(min(max(1, N_SHOW), M))
    max_obs_draw = 20

    x_ahead = 18.0
    x_behind = 4.0
    y_halfspan = 2.8

    save_dir = os.path.join(here, "plot")
    save_path = os.path.join(save_dir, "ramppi_car_simulation.npz")

    steps = min(900, len(obs_csv.times))
    sim_time = np.full((steps,), np.nan, dtype=np.float32)
    solve_ms = np.full((steps,), np.nan, dtype=np.float32)
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

        ids, O_mean_np, draw_pack = obs_csv.build_prediction_for_mppi(
            obs_df=obs_now,
            dt=dt,
            T=T,
            max_obs=max_obs_draw,
        )

        K = len(ids)
        if K > 0:
            radii_np = np.full((K,), obs_radius + ego_radius + extra_margin, dtype=np.float32)
        else:
            radii_np = np.array([], dtype=np.float32)

        ref_np = np.array([float(x_mppi[0]) + L_ref, lane_y, lane_psi], dtype=np.float32)

        # update kwargs on GPU
        mppi_ra.cost_kwargs["ref"] = torch.as_tensor(ref_np, device=mppi_ra.device, dtype=mppi_ra.dtype)
        mppi_ra.cost_kwargs["Q"]   = Q_t
        mppi_ra.cost_kwargs["R"]   = R_t
        mppi_ra.cost_kwargs["Qf"]  = Qf_t
        mppi_ra.cost_kwargs["obs_w"] = obs_w_cost

        if O_mean_np is not None and O_mean_np.shape[1] > 0:
            mppi_ra.cost_kwargs["O_mean"] = torch.as_tensor(O_mean_np, device=mppi_ra.device, dtype=mppi_ra.dtype)
            mppi_ra.cost_kwargs["radii"]  = torch.as_tensor(radii_np,  device=mppi_ra.device, dtype=mppi_ra.dtype)
        else:
            mppi_ra.cost_kwargs["O_mean"] = None
            mppi_ra.cost_kwargs["radii"]  = None

        # correct timing for CUDA (async)
        if mppi_ra.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        U, Xsamp = mppi_ra.plan(
            x_mppi,
            return_samples=True,
            n_show=N_SHOW,
            show_seed=0,
            return_debug=False,
        )

        if mppi_ra.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        print(f"[RA_MPPI Torch CVaR] solve_time = {t1 - t0:.6f} s | K={K} | alpha={cvar_alpha} N={cvar_N}")
        solve_ms[k] = float((t1 - t0) * 1000.0)

        U = np.nan_to_num(U, nan=0.0, posinf=u_max[0], neginf=u_min[0]).astype(np.float32)

        # prediction line (CPU sim)
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
        u0 = np.nan_to_num(u0, nan=0.0, posinf=u_max[0], neginf=u_min[0])

        x_mppi = diffdrive_dynamics_cpu(
            x_mppi, u0, dt, v_max=float(u_max[0]), w_max=dyn_w_max
        ).astype(np.float32)

        if not np.all(np.isfinite(x_mppi)):
            print(f"[ERROR] Non-finite ego state at step {k}, t={t_now:.3f}: x_mppi={x_mppi}, u0={u0}")
            break

        # warm-start shift
        mppi_ra.U[:-1] = mppi_ra.U[1:]
        mppi_ra.U[-1] = np.array([v_des, 0.0], dtype=np.float32)
        mppi_ra.U_cpu = mppi_ra.U  # keep internal consistent

        CAM_X_AHEAD  = 9.0
        CAM_X_BEHIND = 9.0
        CAM_Y_HALF   = 2.0

        x_left  = float(x_mppi[0] - CAM_X_BEHIND)
        x_right = float(x_mppi[0] + CAM_X_AHEAD)
        y_low   = float(y_bottom - 0.2)
        y_high  = float(y_top_outer + 0.2)
        xlim_hist[k, :] = [x_left, x_right]
        ylim_hist[k, :] = [y_low, y_high]

        if not (np.isfinite(x_left) and np.isfinite(x_right) and np.isfinite(y_low) and np.isfinite(y_high)):
            print(f"[WARN] Skipping axis update due to non-finite limits: "
                  f"x=({x_left},{x_right}) y=({y_low},{y_high}) x_mppi={x_mppi}")
            break

        X_hist[k, :] = x_mppi
        X_path[k + 1, :] = x_mppi[:2]
        K_hist[k] = K
        if K > 0:
            kk = min(K, max_obs_draw)
            obs_xy[k, :kk, :] = draw_pack["xy"][:kk, :2].astype(np.float32)
            obs_phi[k, :kk] = draw_pack["phi"][:kk].astype(np.float32)
            obs_ids[k, :kk] = draw_pack["ids"][:kk].astype(np.int32)

    if args.save:
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(
            save_path,
            method=np.array("ramppi"),
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
            obs_length=np.array(ego_length, dtype=np.float32),
            obs_width=np.array(ego_width, dtype=np.float32),
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
