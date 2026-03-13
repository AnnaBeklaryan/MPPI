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
    **_kwargs,
) -> Tensor:
    e = X - ref.unsqueeze(0)
    e[:, 2] = (e[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi
    cost = (e * e) @ Q + torch.sum((U * U) * R.unsqueeze(0), dim=1)

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
    csv_path = os.path.join(base_dir, "Data/features_dir2_5_10s.csv")

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

    # Safety params (from dr_mppi_safety_autotune_best_v2.json)
    T = 15
    M = 1024
    lam = 0.21739523859491372
    sigma = np.array([1.0134190864010681, np.deg2rad(32.067670133051664)], dtype=np.float32)
    u_min = np.array([0.0, -np.deg2rad(79.43091743290539)], dtype=np.float32)
    u_max = np.array([9.463706172331404, np.deg2rad(79.43091743290539)], dtype=np.float32)

    Q = np.array([1.9280837822534018, 19.28173868384513, 2.313205033116019], dtype=np.float32)
    Qf = np.array([7.172263131706658, 40.67035738033066, 6.350029066872338], dtype=np.float32)
    # With latest mppi_class.py, control cost weights are injected from Sigma^{-1}.
    R = np.array([1.7360501398582697, 0.4531768513639747], dtype=np.float32)

    # DR-CVaR settings (autotuned)
    cvar_alpha = 0.9306786750644642
    cvar_N = 8
    obs_pos_sigma = (0.30202412775601417, 0.2656071582397748)
    dr_eps_cvar = 0.019725268835558512
    obs_noise_mode = "static"   # "static" or "per_step"
    I = 2
    dyn_w_max = float(np.deg2rad(140.8915900964214))

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
        I=I,
        cvar_alpha=cvar_alpha,
        cvar_N=cvar_N,
        obs_pos_sigma=obs_pos_sigma,
        obs_noise_mode=obs_noise_mode,
        dr_eps_cvar=dr_eps_cvar,
        dyn_kwargs=dict(w_max=dyn_w_max),
        cost_kwargs=dict(ref=None, Q=None, R=None, Qf=None, O_mean=None, radii=None, obs_w=0.0),
        verbose=False,
    )

    print("[DR_MPPI] device =", mppi.device)

    # put constant weights on GPU once
    Q_t  = torch.as_tensor(Q,  device=mppi.device, dtype=mppi.dtype)
    R_t  = torch.as_tensor(R,  device=mppi.device, dtype=mppi.dtype)
    Qf_t = torch.as_tensor(Qf, device=mppi.device, dtype=mppi.dtype)

    # Lane setup
    lane_psi = 0.0
    L_ref = 18.62342430668639
    v_des = 6.6963753392438115
    u_blend_v = 0.4702057631212527

    ROAD_CENTER = 1.0
    LANE_W = 0.70
    y_divider = ROAD_CENTER
    y_bottom = ROAD_CENTER - LANE_W
    y_top = ROAD_CENTER + LANE_W
    lane_y = y_divider + 0.5 * LANE_W

    x_mppi = np.array([0.0, lane_y, 0.0], dtype=np.float32)

    # Geometry
    ego_length = 4.5 * pos_scale
    ego_width = 1.8 * pos_scale
    obs_length = ego_length
    obs_width = ego_width

    ego_radius = 0.5 * np.sqrt(ego_length**2 + ego_width**2)
    obs_radius = ego_radius
    extra_margin = 0.06907085734601331

    N_SHOW = 60
    N_SHOW = int(min(max(1, N_SHOW), M))
    max_obs_draw = 20

    x_ahead = 18.0
    x_behind = 4.0
    y_halfspan = 2.8

    save_dir = os.path.join(base_dir, "plot")
    save_path = os.path.join(save_dir, "drmppi_car_simulation.npz")

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

    # Solve-time stats (min/max)
    min_solve = float("inf")
    max_solve = 0.0

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

        print( f"[DR_MPPI] solve={solve_time*1000:8.2f} ms " )

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

        # shift nominal controls (warm start)
        mppi.U[:-1] = mppi.U[1:]
        mppi.U[-1] = np.array([v_des, 0.0], dtype=np.float32)
        mppi.U_cpu = mppi.U  # keep internal consistent

        x_left = float(x_mppi[0] - 9.0)
        x_right = float(x_mppi[0] + 9.0)
        y_low = float(lane_y - 2.0)
        y_high = float(lane_y + 2.0)
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
            ego_length=np.array(ego_length, dtype=np.float32),
            ego_width=np.array(ego_width, dtype=np.float32),
            obs_length=np.array(obs_length, dtype=np.float32),
            obs_width=np.array(obs_width, dtype=np.float32),
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
