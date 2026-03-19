#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRA_mppi.py (Torch)
- Torch port of the previous CuPy DRA-MPPI script
- Keeps the same simulation/CSV/plot loop style
- No CuPy dependency
"""

import os
import argparse
import time
from typing import Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.transforms import Affine2D


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
# DRA-MPPI (Torch)
# ============================================================

class DRA_MPPI:
    def __init__(
        self,
        dt,
        T,
        M,
        lam,
        sigma,
        Q,
        R,
        Qf,
        u_min,
        u_max,
        I=1,
        sigma_cp=0.05,
        Nmc=400,
        omega_soft=10.0,
        omega_hard=1000.0,
        obs_pos_sigma=(0.25, 0.25),
        mc_chunk=50,
        seed=3,
        device="cpu",
        dtype=torch.float32,
    ):
        self.dt = float(dt)
        self.T = int(T)
        self.M = int(M)
        self.lam = float(lam)
        self.I = int(I)

        self.device = torch.device(device)
        self.dtype = dtype

        self.sigma = torch.as_tensor(np.asarray(sigma, dtype=np.float32).reshape(2,), device=self.device, dtype=self.dtype)
        self.Q = torch.as_tensor(np.asarray(Q, dtype=np.float32).reshape(3,), device=self.device, dtype=self.dtype)
        self.R = torch.as_tensor(np.asarray(R, dtype=np.float32).reshape(2,), device=self.device, dtype=self.dtype)
        self.Qf = torch.as_tensor(np.asarray(Qf, dtype=np.float32).reshape(3,), device=self.device, dtype=self.dtype)
        self.u_min = torch.as_tensor(np.asarray(u_min, dtype=np.float32).reshape(2,), device=self.device, dtype=self.dtype)
        self.u_max = torch.as_tensor(np.asarray(u_max, dtype=np.float32).reshape(2,), device=self.device, dtype=self.dtype)

        self.U = np.zeros((self.T, 2), dtype=np.float32)
        self.U_cpu = self.U

        self.sigma_cp = float(sigma_cp)
        self.Nmc = int(Nmc)
        self.omega_soft = float(omega_soft)
        self.omega_hard = float(omega_hard)
        self.obs_pos_sigma = torch.as_tensor(np.asarray(obs_pos_sigma, dtype=np.float32).reshape(2,), device=self.device, dtype=self.dtype)
        self.mc_chunk = int(mc_chunk)
        self._u_min_112 = self.u_min.view(1, 1, 2)
        self._u_max_112 = self.u_max.view(1, 1, 2)
        self._sigma_cp_t = torch.tensor(self.sigma_cp, device=self.device, dtype=self.dtype)
        self._omega_soft_t = torch.tensor(self.omega_soft, device=self.device, dtype=self.dtype)
        self._omega_hard_t = torch.tensor(self.omega_hard, device=self.device, dtype=self.dtype)

        self._rng_eps = torch.Generator(device=self.device)
        self._rng_eps.manual_seed(int(seed))

    @staticmethod
    def _angle_wrap_arr(th: Tensor) -> Tensor:
        return (th + torch.pi) % (2 * torch.pi) - torch.pi

    def dyn_step(self, X: Tensor, U_t: Tensor) -> Tensor:
        px = X[:, 0]
        py = X[:, 1]
        psi = X[:, 2]

        v = torch.clamp(U_t[:, 0], self.u_min[0], self.u_max[0])
        w = torch.clamp(U_t[:, 1], self.u_min[1], self.u_max[1])

        px = px + self.dt * v * torch.cos(psi)
        py = py + self.dt * v * torch.sin(psi)
        psi = self._angle_wrap_arr(psi + self.dt * w)
        return torch.stack([px, py, psi], dim=1)

    def running_cost(self, X: Tensor, U_t: Tensor, ref: Tensor) -> Tensor:
        e = X - ref.unsqueeze(0)
        e[:, 2] = self._angle_wrap_arr(e[:, 2])
        c_track = (e * e) @ self.Q
        c_u = torch.sum((U_t * U_t) * self.R.unsqueeze(0), dim=1)
        return c_track + c_u

    def terminal_cost(self, X: Tensor, ref: Tensor) -> Tensor:
        e = X - ref.unsqueeze(0)
        e[:, 2] = self._angle_wrap_arr(e[:, 2])
        return (e * e) @ self.Qf

    @torch.no_grad()
    def _dra_risk_cost_over_time(self, X_hist_xy: Tensor, O_mean: Optional[Tensor], radii: Tensor) -> Tensor:
        T, M, _ = X_hist_xy.shape
        if O_mean is None or O_mean.shape[1] == 0:
            return torch.zeros(M, device=self.device, dtype=self.dtype)

        K = int(O_mean.shape[1])
        r = float(torch.max(radii).item()) if K > 0 else 0.0
        if r <= 0.0:
            return torch.zeros(M, device=self.device, dtype=self.dtype)
        r2 = r * r

        sig = self.obs_pos_sigma
        sigx = max(float(sig[0].item()), 1e-6)
        sigy = max(float(sig[1].item()), 1e-6)
        norm_const = torch.tensor(1.0 / (2.0 * np.pi * sigx * sigy), device=self.device, dtype=self.dtype)

        risk_cost = torch.zeros(M, device=self.device, dtype=self.dtype)

        for t in range(T):
            xr = X_hist_xy[t, :, 0]
            yr = X_hist_xy[t, :, 1]

            xlow = torch.min(xr) - r
            xhigh = torch.max(xr) + r
            ylow = torch.min(yr) - r
            yhigh = torch.max(yr) + r

            dx = torch.clamp(xhigh - xlow, min=1e-6)
            dy = torch.clamp(yhigh - ylow, min=1e-6)
            area = dx * dy

            u = torch.rand((self.Nmc,), device=self.device, dtype=self.dtype, generator=self._rng_eps)
            v = torch.rand((self.Nmc,), device=self.device, dtype=self.dtype, generator=self._rng_eps)
            xj = xlow + u * dx
            yj = ylow + v * dy
            cell_area = area / float(self.Nmc)

            mu = O_mean[t, :, :]
            mux = mu[:, 0].unsqueeze(1)
            muy = mu[:, 1].unsqueeze(1)

            dxo = (xj.unsqueeze(0) - mux) / sigx
            dyo = (yj.unsqueeze(0) - muy) / sigy
            expo = -0.5 * (dxo * dxo + dyo * dyo)
            pdf = norm_const * torch.exp(expo)

            p_mass = torch.clamp(pdf * cell_area, 0.0, 0.999)
            prod_term = torch.prod(1.0 - p_mass, dim=0)
            pjoint_mass = 1.0 - prod_term

            Phat = torch.zeros(M, device=self.device, dtype=self.dtype)
            for j0 in range(0, self.Nmc, self.mc_chunk):
                j1 = min(self.Nmc, j0 + self.mc_chunk)
                xchunk = xj[j0:j1].unsqueeze(0)
                ychunk = yj[j0:j1].unsqueeze(0)
                dxm = xchunk - xr.unsqueeze(1)
                dym = ychunk - yr.unsqueeze(1)
                inside = (dxm * dxm + dym * dym) <= r2
                Phat += inside.to(self.dtype) @ pjoint_mass[j0:j1]

            Phat = torch.clamp(Phat, 0.0, 1.0)
            risk_cost += self._omega_soft_t * Phat
            risk_cost += self._omega_hard_t * (Phat > self._sigma_cp_t).to(self.dtype)

        return risk_cost

    @torch.no_grad()
    def plan(
        self,
        x0_cpu,
        ref_cpu,
        O_mean_cpu,
        radii_cpu,
        eps_cpu=None,
        return_samples=False,
        n_show=60,
        show_seed=0,
    ):
        x0 = torch.as_tensor(np.asarray(x0_cpu, dtype=np.float32), device=self.device, dtype=self.dtype)
        ref = torch.as_tensor(np.asarray(ref_cpu, dtype=np.float32), device=self.device, dtype=self.dtype)
        O_mean = None if O_mean_cpu is None else torch.as_tensor(np.asarray(O_mean_cpu, dtype=np.float32), device=self.device, dtype=self.dtype)
        radii = torch.as_tensor(np.asarray(radii_cpu, dtype=np.float32), device=self.device, dtype=self.dtype)

        U = torch.as_tensor(self.U, device=self.device, dtype=self.dtype).clone()
        sigma = self.sigma.reshape(1, 1, 2)

        last_Xsamp_hist = None

        for _ in range(self.I):
            if eps_cpu is None:
                eps = torch.randn((self.M, self.T, 2), device=self.device, dtype=self.dtype, generator=self._rng_eps) * sigma
            else:
                eps = torch.as_tensor(np.asarray(eps_cpu, dtype=np.float32), device=self.device, dtype=self.dtype) * sigma

            eps[0, :, :] = 0.0
            U_roll = torch.clamp(U.unsqueeze(0) + eps, self._u_min_112, self._u_max_112)

            J = torch.zeros(self.M, device=self.device, dtype=self.dtype)
            X = x0.unsqueeze(0).repeat(self.M, 1)
            X_hist_xy = torch.zeros((self.T, self.M, 2), device=self.device, dtype=self.dtype)

            if return_samples:
                ns = int(min(max(1, int(n_show)), self.M))
                g_show = torch.Generator(device="cpu")
                g_show.manual_seed(int(show_seed))
                sample_idx = torch.randperm(self.M, generator=g_show)[:ns].to(self.device)
                Xsamp_hist = torch.zeros((self.T + 1, ns, 2), device=self.device, dtype=self.dtype)
                Xsamp_hist[0] = X[sample_idx, :2]
            else:
                sample_idx = None
                Xsamp_hist = None

            for t in range(self.T):
                U_t = U_roll[:, t, :]
                X = self.dyn_step(X, U_t)

                X_hist_xy[t] = X[:, :2]
                if return_samples:
                    Xsamp_hist[t + 1] = X[sample_idx, :2]

                J += self.running_cost(X, U_t, ref)

            J += self.terminal_cost(X, ref)
            J += self._dra_risk_cost_over_time(X_hist_xy, O_mean, radii)

            J = torch.where(torch.isfinite(J), J, torch.full_like(J, torch.inf))
            rho = torch.min(J)
            w = torch.exp(-(J - rho) / self.lam)
            w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
            w_sum = torch.sum(w)
            if float(w_sum.item()) < 1e-12:
                break
            w = w / w_sum

            dU = torch.einsum("m,mtd->td", w, eps)
            U = torch.clamp(U + dU, self.u_min, self.u_max)

            if return_samples:
                last_Xsamp_hist = Xsamp_hist

        U_cpu = U.detach().cpu().numpy().astype(np.float32)
        self.U = U_cpu
        self.U_cpu = U_cpu

        if return_samples:
            Xsamp_cpu = None
            if last_Xsamp_hist is not None:
                Xsamp_cpu = last_Xsamp_hist.detach().cpu().numpy().astype(np.float32)
            return U_cpu, Xsamp_cpu

        return U_cpu


# ============================================================
# Main Simulation
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DRA-MPPI car simulation.")
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

    # Core controller params matched to mppi_1.py
    T = 10
    M = 1200
    lam = 2.0
    sigma = np.array([2.0, 1.04], dtype=np.float32)
    u_min = np.array([0.0, -np.deg2rad(180.0)], dtype=np.float32)
    u_max = np.array([10.0, np.deg2rad(180.0)], dtype=np.float32)

    Q = np.array([0.001, 0.001, 0.001], dtype=np.float32)
    Qf = np.array([0.5, 10.0, 1.0], dtype=np.float32)
    R = np.array([1e-5, 1e-5], dtype=np.float32)

    sigma_cp = 0.05
    Nmc = 20000
    omega_soft = 10.0
    omega_hard = 1000.0
    obs_pos_sigma = (0.25 * pos_scale, 0.25 * pos_scale)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mppi = DRA_MPPI(
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
        sigma_cp=sigma_cp,
        Nmc=Nmc,
        omega_soft=omega_soft,
        omega_hard=omega_hard,
        obs_pos_sigma=obs_pos_sigma,
        mc_chunk=1000,
        seed=3,
        device=device,
        dtype=torch.float32,
    )
    print("[DRA_MPPI] device =", mppi.device)

    lane_psi = 0.0
    L_ref = 14.0
    v_des = 5.0
    u_blend_v = 0.4

    ROAD_CENTER = 1.0
    LANE_W = 0.70
    y_divider = ROAD_CENTER
    y_bottom = ROAD_CENTER - LANE_W
    y_top = ROAD_CENTER + LANE_W
    lane_y = y_divider + 0.5 * LANE_W

    x_mppi = np.array([0.0, lane_y, 0.0], dtype=np.float32)

    ego_length = 4.5 * pos_scale
    ego_width = 1.8 * pos_scale
    obs_length = ego_length
    obs_width = ego_width

    ego_radius = 0.5 * np.sqrt(ego_length**2 + ego_width**2)
    obs_radius = ego_radius
    extra_margin = 0.0

    N_SHOW = int(min(max(1, 60), M))
    max_obs_draw = 20

    x_ahead = 18.0
    x_behind = 4.0
    y_halfspan = 2.8

    save_dir = os.path.join(base_dir, "plot")
    save_path = os.path.join(save_dir, "dramppi_car_simulation.npz")

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
            dist2 = (obs_now["x"] - float(x_mppi[0])) ** 2 + (obs_now["y"] - float(x_mppi[1])) ** 2
            obs_now = obs_now.iloc[np.argsort(dist2.to_numpy())]
        obs_now = obs_now.iloc[:max_obs_draw].copy()

        ids, O_mean, draw_pack = obs_csv.build_prediction_for_mppi(
            obs_df=obs_now,
            dt=dt,
            T=T,
            max_obs=max_obs_draw,
        )

        K = len(ids)
        if K > 0:
            radii_for_mppi = np.full((K,), obs_radius + ego_radius + extra_margin, dtype=np.float32)
        else:
            radii_for_mppi = np.array([], dtype=np.float32)

        ref = np.array([float(x_mppi[0]) + L_ref, lane_y, lane_psi], dtype=np.float32)

        if mppi.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        U, Xsamp = mppi.plan(
            x_mppi,
            ref,
            O_mean,
            radii_for_mppi,
            return_samples=True,
            n_show=N_SHOW,
            show_seed=0,
        )

        if mppi.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        solve_time = t1 - t0
        solve_ms[k] = float(solve_time * 1000.0)
        print(f"[DRA_MPPI] solve={solve_time*1000:8.2f} ms")

        x_pred = x_mppi.copy()
        pred_nominal_xy[k, 0, 0] = float(x_pred[0])
        pred_nominal_xy[k, 0, 1] = float(x_pred[1])
        for t in range(T):
            x_pred = diffdrive_dynamics_cpu(x_pred, U[t], dt, v_max=float(u_max[0])).astype(np.float32)
            pred_nominal_xy[k, t + 1, 0] = float(x_pred[0])
            pred_nominal_xy[k, t + 1, 1] = float(x_pred[1])

        if Xsamp is not None:
            ns = min(N_SHOW, Xsamp.shape[1])
            pred_samples_xy[k, :, :ns, :] = Xsamp[:, :ns, :2].astype(np.float32)

        u0 = U[0].copy()
        u0[0] = np.clip(u_blend_v * u0[0] + (1.0 - u_blend_v) * v_des, 0.0, float(u_max[0]))
        x_mppi = diffdrive_dynamics_cpu(x_mppi, u0, dt, v_max=float(u_max[0])).astype(np.float32)

        mppi.U[:-1] = mppi.U[1:]
        mppi.U[-1] = np.array([v_des, 0.0], dtype=np.float32)
        mppi.U_cpu = mppi.U

        xlim_hist[k, :] = [float(x_mppi[0] - 9.0), float(x_mppi[0] + 9.0)]
        ylim_hist[k, :] = [float(lane_y - 2.0), float(lane_y + 2.0)]

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
            method=np.array("dramppi"),
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
