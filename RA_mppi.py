# -*- coding: utf-8 -*-
"""
FULL COMPLETE SCRIPT: RA-MPPI (Torch) with CVaR feasibility filter + your lane/obstacle scenario + plotting

This script ASSUMES you have:
  - mppi_class.py  (contains the RA_MPPI class I provided: CVaR feasibility filter, obstacle sampling, etc.)
  - features_dir2_5_10s.csv
  - Data/car_ego.png and Data/car_obs1.png..car_obs14.png

Key points:
- Uses RA_MPPI from mppi_class.py
- Cost functions are generic torch functions
- CVaR feasibility uses O_mean and radii from mppi_ra.cost_kwargs (as in the class)

Run:
  python RA_mppi.py
"""

import os
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
    np.random.seed(3)

    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "Data/features_dir2_5_10s.csv")

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

    # MPPI params
    T = 30
    M = 1200
    lam = 2.0
    sigma = np.array([0.35, np.deg2rad(20.0)], dtype=np.float32)

    u_min = np.array([0.0, -np.deg2rad(180.0)], dtype=np.float32)
    u_max = np.array([10.0,  np.deg2rad(180.0)], dtype=np.float32)

    Q  = np.array([0.2, 4.0, 0.6], dtype=np.float32)
    Qf = np.array([0.5, 10.0, 1.0], dtype=np.float32)
    R  = np.array([0.0001, 0.5], dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CVaR settings (MATCH your CuPy logic)
    cvar_alpha = 0.9
    cvar_N = 64
    obs_pos_sigma = (0.25, 0.25)     # same meaning as your CuPy code
    noise_mode = "static"            # "static" or "per_step"

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
        dyn_kwargs=dict(w_max=float(np.deg2rad(180.0))),
        cost_kwargs=dict(ref=None, Q=None, R=None, Qf=None, O_mean=None, radii=None, obs_w=0.0),
        I=1,
        verbose=False,
        cvar_alpha=0.9,
        cvar_N=64,
        obs_pos_sigma=(0.25, 0.25),
        obs_noise_mode="static",   # <-- FIXED NAME
    )
    print("[RA_MPPI] device =", mppi_ra.device)

    # Put weights on GPU once
    Q_t  = torch.as_tensor(Q,  device=mppi_ra.device, dtype=mppi_ra.dtype)
    R_t  = torch.as_tensor(R,  device=mppi_ra.device, dtype=mppi_ra.dtype)
    Qf_t = torch.as_tensor(Qf, device=mppi_ra.device, dtype=mppi_ra.dtype)

    lane_psi = 0.0
    L_ref = 14.0
    v_des = 5.0

    ROAD_CENTER = 1.0
    LANE_W = 0.70

    y_divider = ROAD_CENTER
    y_bottom  = ROAD_CENTER - LANE_W
    y_top     = ROAD_CENTER + LANE_W
    lane_y = y_divider + 0.5 * LANE_W

    x_mppi = np.array([0.0, lane_y, 0.0], dtype=np.float32)

    ego_length = 4.5 * pos_scale
    ego_width  = 1.8 * pos_scale

    obs_radius = 0.5 * np.sqrt(ego_length**2 + ego_width**2)
    ego_radius = obs_radius
    extra_margin = 0.0

    # sprites
    car_ego_img = mpimg.imread(os.path.join(here, "Data/car_ego.png"))
    obs_sprite_paths = [os.path.join(here, f"Data/car_obs{i}.png") for i in range(1, 15)]
    obs_sprite_imgs = []
    for p in obs_sprite_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing obstacle sprite: {p}")
        obs_sprite_imgs.append(mpimg.imread(p))
    NUM_SPRITES = len(obs_sprite_imgs)

    id_to_sprite = {}
    next_sprite = 0

    # ========================================================
    # Visualization
    # ========================================================
    plt.ion()
    fig, ax = plt.subplots(figsize=(14., 5.0))
    fig.subplots_adjust(right=0.80)

    ax.set_facecolor("#646464")
    ax.set_aspect("auto")
    ax.set_xlabel("x [scaled m]")
    ax.set_ylabel("y [scaled m]")

    ax.axhline(y_bottom,  color="white", linewidth=3.0, zorder=2)
    ax.axhline(y_top,     color="white", linewidth=3.0, zorder=2)
    ax.axhline(y_divider, color="white", linewidth=2.0, linestyle=(0, (12, 12)),
               alpha=0.9, zorder=2)

    ax.plot([x_mppi[0] - 100, x_mppi[0] + 500], [lane_y, lane_y],
            "--", linewidth=1.5, color="#abbac6", alpha=0.8, label="Lane center")

    N_SHOW = 60
    N_SHOW = int(min(max(1, N_SHOW), M))
    sample_lines = []
    for _ in range(N_SHOW):
        ln, = ax.plot([], [], lw=1.1, color="#9ee2e9", alpha=0.12, zorder=1)
        sample_lines.append(ln)

    path_line, = ax.plot([], [], lw=2.6, color="#67bde2", label="Ego path")
    pred_line, = ax.plot([], [], lw=2.2, color="#36ff0e", alpha=0.95, label="MPPI prediction", zorder=3)

    ego_img_artist = ax.imshow(
        car_ego_img,
        extent=[-0.5, 0.5, -0.5, 0.5],
        zorder=6,
    )

    max_obs_draw = 20
    obs_imgs = []
    for _ in range(max_obs_draw):
        im = ax.imshow(
            obs_sprite_imgs[0],
            extent=[-0.5, 0.5, -0.5, 0.5],
            zorder=4,
            visible=False,
        )
        obs_imgs.append(im)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)

    x_ahead = 18.0
    x_behind = 4.0
    y_halfspan = 2.8

    xs, ys = [float(x_mppi[0])], [float(x_mppi[1])]
    steps = min(900, len(obs_csv.times))

    for k in range(steps):
        t_now = float(obs_csv.times[k])
        obs_now = obs_csv.obstacles_now(t_now)

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

        U = np.nan_to_num(U, nan=0.0, posinf=u_max[0], neginf=u_min[0]).astype(np.float32)

        # prediction line (CPU sim)
        x_pred = x_mppi.copy()
        pred_xs = [float(x_pred[0])]
        pred_ys = [float(x_pred[1])]
        for t in range(T):
            x_pred = diffdrive_dynamics_cpu(x_pred, U[t], dt, v_max=float(u_max[0])).astype(np.float32)
            pred_xs.append(float(x_pred[0]))
            pred_ys.append(float(x_pred[1]))
        pred_line.set_data(pred_xs, pred_ys)

        # rollout fan
        if Xsamp is not None:
            ns = min(N_SHOW, Xsamp.shape[1])
            for i in range(ns):
                sample_lines[i].set_data(Xsamp[:, i, 0], Xsamp[:, i, 1])
            for i in range(ns, N_SHOW):
                sample_lines[i].set_data([], [])

        # apply first control
        u0 = U[0].copy()
        u0[0] = np.clip(0.4 * u0[0] + 0.6 * v_des, 0.0, float(u_max[0]))
        u0 = np.nan_to_num(u0, nan=0.0, posinf=u_max[0], neginf=u_min[0])

        x_mppi = diffdrive_dynamics_cpu(x_mppi, u0, dt, v_max=float(u_max[0])).astype(np.float32)

        if not np.all(np.isfinite(x_mppi)):
            print(f"[ERROR] Non-finite ego state at step {k}, t={t_now:.3f}: x_mppi={x_mppi}, u0={u0}")
            break

        # warm-start shift
        mppi_ra.U[:-1] = mppi_ra.U[1:]
        mppi_ra.U[-1] = np.array([v_des, 0.0], dtype=np.float32)
        mppi_ra.U_cpu = mppi_ra.U  # keep internal consistent

        xs.append(float(x_mppi[0]))
        ys.append(float(x_mppi[1]))
        path_line.set_data(xs, ys)

        set_img_pose(ego_img_artist, x_mppi[0], x_mppi[1], x_mppi[2], ego_length, ego_width, ax)

        # draw obstacles
        for i in range(max_obs_draw):
            if i < K:
                obs_id = int(draw_pack["ids"][i])
                if obs_id not in id_to_sprite:
                    id_to_sprite[obs_id] = next_sprite
                    next_sprite = (next_sprite + 1) % NUM_SPRITES
                sprite_idx = id_to_sprite[obs_id]
                obs_imgs[i].set_data(obs_sprite_imgs[sprite_idx])
                obs_imgs[i].set_visible(True)
                set_img_pose(
                    obs_imgs[i],
                    draw_pack["xy"][i, 0],
                    draw_pack["xy"][i, 1],
                    float(draw_pack["phi"][i]),
                    ego_length,
                    ego_width,
                    ax
                )
            else:
                obs_imgs[i].set_visible(False)

        CAM_X_AHEAD  = 9.0
        CAM_X_BEHIND = 9.0
        CAM_Y_HALF   = 2.0

        x_left  = float(x_mppi[0] - CAM_X_BEHIND)
        x_right = float(x_mppi[0] + CAM_X_AHEAD)
        y_low   = float(lane_y - CAM_Y_HALF)
        y_high  = float(lane_y + CAM_Y_HALF)

        if np.isfinite(x_left) and np.isfinite(x_right) and np.isfinite(y_low) and np.isfinite(y_high):
            ax.set_xlim(x_left, x_right)
            ax.set_ylim(y_low, y_high)
        else:
            print(f"[WARN] Skipping axis update due to non-finite limits: "
                  f"x=({x_left},{x_right}) y=({y_low},{y_high}) x_mppi={x_mppi}")
            break

        ax.set_title(f"RA_MPPI Torch CVaR | t={t_now:.2f}s | K={K} | solve={t1 - t0:.3f}s")
        plt.pause(0.01)

    plt.ioff()
    plt.show()