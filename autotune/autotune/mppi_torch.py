# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import matplotlib.image as mpimg
import torch
from typing import Callable, Optional, Tuple, Union, Dict, Any

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


def diffdrive_dynamics(x, u, dt, v_min=0.0, v_max=10.0, w_max=np.deg2rad(180.0)):
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

def dyn_diffdrive(X: Tensor, U: Tensor, dt: float, w_max: float = float(np.deg2rad(180.0))) -> Tensor:
    # X: (M,3) [px,py,psi]  U: (M,2) [v,w]
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
    obs_w: float = 5e3,
    **_kwargs,   # <-- IMPORTANT: swallow extra keys like Qf
) -> Tensor:
    e = X - ref.unsqueeze(0)
    e[:, 2] = (e[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi
    cost = (e * e) @ Q + torch.sum((U * U) * R.unsqueeze(0), dim=1)

    if O_mean is not None and radii is not None and O_mean.shape[1] > 0:
        O_t = O_mean[t]
        diff = X[:, :2].unsqueeze(1) - O_t.unsqueeze(0)
        dist = torch.linalg.norm(diff, dim=-1)
        viol = torch.clamp(radii.unsqueeze(0) - dist, min=0.0)
        cost = cost + float(obs_w) * torch.sum(viol * viol, dim=1)

    return cost


def terminal_cost_track(
    X: Tensor, t_final: int,
    ref: Tensor, Qf: Tensor,
    **_kwargs,   # <-- swallow extra keys like Q, R, O_mean, radii, ...
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

        draw_pack = dict(
            ids=ids,
            xy=np.stack([xs, ys], axis=1),
            phi=phis,
        )
        return ids, O_mean, draw_pack


# ============================================================
# Generic MPPI Torch
# ============================================================

class MPPI_Torch_Generic:
    def __init__(
        self,
        dt: float,
        T: int,
        M: int,
        lam: float,
        noise_sigma: Union[np.ndarray, Tensor],  # (nu,) diag OR (nu,nu) covariance
        u_min: Union[np.ndarray, Tensor],        # (nu,)
        u_max: Union[np.ndarray, Tensor],        # (nu,)
        dynamics: Callable[..., Tensor],
        running_cost: Callable[..., Tensor],
        terminal_cost: Callable[..., Tensor],
        I: int = 1,
        exp_clip: float = 80.0,
        weight_floor: float = 1e-12,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
        dyn_kwargs: Optional[Dict[str, Any]] = None,
        cost_kwargs: Optional[Dict[str, Any]] = None,
        u_init: Optional[np.ndarray] = None,  # (T,nu)
        perturbation_cost: float = 0.0,
    ):
        self.dt = float(dt)
        self.T = int(T)
        self.M = int(M)
        self.lam = float(lam)
        self.I = int(I)
        self.verbose = bool(verbose)

        if not (np.isfinite(self.dt) and self.dt > 0):
            raise ValueError(f"dt must be finite and > 0. Got dt={self.dt}")
        if not (np.isfinite(self.lam) and self.lam > 0):
            raise ValueError(f"lam must be finite and > 0. Got lam={self.lam}")
        if self.T <= 0 or self.M <= 0:
            raise ValueError(f"T and M must be positive. Got T={self.T}, M={self.M}")

        self.exp_clip = float(exp_clip)
        self.weight_floor = float(weight_floor)
        self.perturbation_cost = float(perturbation_cost)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

        self.dynamics = dynamics
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost

        self.dyn_kwargs = dict(dyn_kwargs or {})
        self.cost_kwargs = dict(cost_kwargs or {})

        self.u_min = torch.as_tensor(np.asarray(u_min, dtype=np.float32).reshape(-1), device=self.device, dtype=self.dtype)
        self.u_max = torch.as_tensor(np.asarray(u_max, dtype=np.float32).reshape(-1), device=self.device, dtype=self.dtype)
        if self.u_min.shape != self.u_max.shape:
            raise ValueError(f"u_min shape {self.u_min.shape} != u_max shape {self.u_max.shape}")
        self.nu = int(self.u_min.numel())

        # noise sigma
        if isinstance(noise_sigma, torch.Tensor):
            sig_t = noise_sigma.to(device=self.device, dtype=self.dtype)
        else:
            sig_t = torch.as_tensor(np.asarray(noise_sigma, dtype=np.float32), device=self.device, dtype=self.dtype)

        if sig_t.ndim == 1:
            if sig_t.numel() != self.nu:
                raise ValueError(f"noise_sigma diag must have shape (nu,), got {tuple(sig_t.shape)} with nu={self.nu}")
            self.noise_mode = "diag"
            self.noise_std = sig_t.reshape(1, 1, self.nu)
            self.noise_L = None
        elif sig_t.ndim == 2:
            if sig_t.shape != (self.nu, self.nu):
                raise ValueError(f"noise_sigma cov must be (nu,nu), got {tuple(sig_t.shape)} with nu={self.nu}")
            self.noise_mode = "cov"
            jitter = 1e-6 * torch.eye(self.nu, device=self.device, dtype=self.dtype)
            self.noise_L = torch.linalg.cholesky(sig_t + jitter)
            self.noise_std = None
        else:
            raise ValueError(f"noise_sigma must be (nu,) or (nu,nu), got shape {tuple(sig_t.shape)}")

        # nominal controls on CPU (compat with your sim loop)
        if u_init is None:
            self.U_cpu = np.zeros((self.T, self.nu), dtype=np.float32)
        else:
            u_init = np.asarray(u_init, dtype=np.float32)
            if u_init.shape != (self.T, self.nu):
                raise ValueError(f"u_init must be (T,nu)={(self.T,self.nu)}, got {u_init.shape}")
            self.U_cpu = u_init.copy()

        # Alias to match your old code style: mppi.U
        self.U = self.U_cpu

    def _sample_eps(self, generator: Optional[torch.Generator] = None) -> Tensor:
        if self.noise_mode == "diag":
            z = torch.randn((self.M, self.T, self.nu), device=self.device, dtype=self.dtype, generator=generator)
            return z * self.noise_std
        else:
            z = torch.randn((self.M, self.T, self.nu), device=self.device, dtype=self.dtype, generator=generator)
            return z @ self.noise_L.T

    def _to_torch(self, arr, shape=None) -> Optional[Tensor]:
        if arr is None:
            return None
        if isinstance(arr, torch.Tensor):
            t = arr.to(device=self.device, dtype=self.dtype)
        else:
            t = torch.as_tensor(np.asarray(arr, dtype=np.float32), device=self.device, dtype=self.dtype)
        if shape is not None and tuple(t.shape) != tuple(shape):
            raise ValueError(f"Bad shape: expected {shape}, got {tuple(t.shape)}")
        return t

    @torch.no_grad()
    def plan(
        self,
        x0: Union[np.ndarray, Tensor],        # (nx,)
        return_samples: bool = False,
        n_show: int = 60,
        show_seed: int = 0,
        eps: Optional[Union[np.ndarray, Tensor]] = None,  # standard normal (M,T,nu)
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # x0
        if isinstance(x0, torch.Tensor):
            x0_t = x0.to(device=self.device, dtype=self.dtype).reshape(-1)
        else:
            x0_t = torch.as_tensor(np.asarray(x0, dtype=np.float32).reshape(-1), device=self.device, dtype=self.dtype)
        nx = int(x0_t.numel())

        # U
        U = torch.as_tensor(self.U_cpu, device=self.device, dtype=self.dtype)

        # sample idx for plotting
        if return_samples:
            ns = int(min(max(1, n_show), self.M))
            g_show = torch.Generator(device=self.device)
            g_show.manual_seed(int(show_seed))
            sample_idx = torch.randperm(self.M, generator=g_show, device=self.device)[:ns]
        else:
            ns = 0
            sample_idx = None

        for it in range(self.I):
            # eps
            if eps is None:
                eps_t = self._sample_eps(generator=None)
            else:
                if isinstance(eps, torch.Tensor):
                    z = eps.to(device=self.device, dtype=self.dtype)
                else:
                    z = torch.as_tensor(np.asarray(eps, dtype=np.float32), device=self.device, dtype=self.dtype)
                if z.shape != (self.M, self.T, self.nu):
                    raise ValueError(f"eps must be (M,T,nu)={(self.M,self.T,self.nu)}, got {tuple(z.shape)}")
                if self.noise_mode == "diag":
                    eps_t = z * self.noise_std
                else:
                    eps_t = z @ self.noise_L.T

            J = torch.zeros((self.M,), device=self.device, dtype=self.dtype)
            X = x0_t.unsqueeze(0).repeat(self.M, 1)

            if return_samples:
                X_hist = torch.zeros((self.T + 1, ns, nx), device=self.device, dtype=self.dtype)
                X_hist[0] = X.index_select(0, sample_idx)
            else:
                X_hist = None

            for t in range(self.T):
                U_t = torch.clamp(U[t].unsqueeze(0) + eps_t[:, t, :], self.u_min, self.u_max)

                X = self.dynamics(X, U_t, self.dt, **self.dyn_kwargs)
                if X.shape != (self.M, nx):
                    raise ValueError(f"dynamics must return (M,nx)={(self.M,nx)}, got {tuple(X.shape)}")

                if return_samples:
                    X_hist[t + 1] = X.index_select(0, sample_idx)

                c = self.running_cost(X, U_t, t, **self.cost_kwargs)
                if c.shape != (self.M,):
                    raise ValueError(f"running_cost must return (M,), got {tuple(c.shape)}")
                J = J + c

                if self.perturbation_cost != 0.0:
                    J = J + self.perturbation_cost * torch.sum(eps_t[:, t, :] ** 2, dim=1)

            ct = self.terminal_cost(X, self.T, **self.cost_kwargs)
            if ct.shape != (self.M,):
                raise ValueError(f"terminal_cost must return (M,), got {tuple(ct.shape)}")
            J = J + ct

            J = torch.nan_to_num(
                J,
                nan=torch.tensor(float("inf"), device=self.device, dtype=self.dtype),
                posinf=torch.tensor(float("inf"), device=self.device, dtype=self.dtype),
                neginf=torch.tensor(float("inf"), device=self.device, dtype=self.dtype),
            )

            rho = torch.min(J)
            z = -(J - rho) / self.lam
            z = torch.clamp(z, -self.exp_clip, self.exp_clip)
            w = torch.exp(z)
            w_sum = torch.sum(w)
            w_sum_val = float(w_sum.item())

            if (not np.isfinite(w_sum_val)) or (w_sum_val < self.weight_floor):
                if self.verbose:
                    print(f"[WARN] weight collapse at iter {it}: w_sum={w_sum_val}")
                continue

            w = w / w_sum

            for t in range(self.T):
                dU = torch.sum(w.unsqueeze(1) * eps_t[:, t, :], dim=0)
                U[t] = torch.clamp(U[t] + dU, self.u_min, self.u_max)

        U_cpu = U.detach().cpu().numpy().astype(np.float32)
        U_cpu = np.nan_to_num(U_cpu, nan=0.0)

        # store back
        self.U_cpu = U_cpu
        self.U = self.U_cpu  # keep alias

        if return_samples and X_hist is not None:
            Xsamp_cpu = X_hist.detach().cpu().numpy().astype(np.float32)
            return U_cpu, Xsamp_cpu

        return U_cpu, None


# ============================================================
# Main Simulation
# ============================================================

if __name__ == "__main__":
    np.random.seed(3)

    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "obstacle_data.csv")

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

    mppi = MPPI_Torch_Generic(
        dt=dt, T=T, M=M, lam=lam,
        noise_sigma=sigma,
        u_min=u_min, u_max=u_max,
        dynamics=dyn_diffdrive,
        running_cost=running_cost_lane_obs,
        terminal_cost=terminal_cost_track,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dyn_kwargs=dict(w_max=float(np.deg2rad(180.0))),
        cost_kwargs=dict(
            ref=None, Q=None, R=None, Qf=None,
            O_mean=None, radii=None, obs_w=5e3
        ),
        I=1,
        verbose=False
    )

    print("[MPPI] device =", mppi.device)

    # Pre-move constant weights to GPU once (DON'T redo every loop)
    Q_t  = torch.as_tensor(Q,  device=mppi.device, dtype=mppi.dtype)
    R_t  = torch.as_tensor(R,  device=mppi.device, dtype=mppi.dtype)
    Qf_t = torch.as_tensor(Qf, device=mppi.device, dtype=mppi.dtype)

    lane_psi = 0.0
    L_ref = 14.0
    v_des = 5.0

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

    obs_length = ego_length
    obs_width  = ego_width

    ego_radius = 0.5 * np.sqrt(ego_length**2 + ego_width**2)
    obs_radius = ego_radius
    extra_margin = 0.0

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
    ax.axhline(y_divider, color="white", linewidth=2.0, linestyle=(0, (12, 12)),
               alpha=0.9, zorder=2)
    ax.axhline(y_divider_top_1, color="white", linewidth=2.0, linestyle=(0, (12, 12)),
               alpha=0.9, zorder=2)
    ax.axhline(y_divider_top_2, color="white", linewidth=2.0, linestyle=(0, (12, 12)),
               alpha=0.9, zorder=2)
    ax.axhline(y_top_outer, color="white", linewidth=3.0, zorder=2)

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

        # Update cost kwargs on GPU (fast enough; main bottleneck is pandas anyway)
        mppi.cost_kwargs["ref"] = torch.as_tensor(ref_np, device=mppi.device, dtype=mppi.dtype)
        mppi.cost_kwargs["Q"]   = Q_t
        mppi.cost_kwargs["R"]   = R_t
        mppi.cost_kwargs["Qf"]  = Qf_t

        if O_mean_np is not None and O_mean_np.shape[1] > 0:
            mppi.cost_kwargs["O_mean"] = torch.as_tensor(O_mean_np, device=mppi.device, dtype=mppi.dtype)
            mppi.cost_kwargs["radii"]  = torch.as_tensor(radii_np, device=mppi.device, dtype=mppi.dtype)
        else:
            mppi.cost_kwargs["O_mean"] = None
            mppi.cost_kwargs["radii"]  = None

        # Time correctly (CUDA is async)
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

        print(f"[MPPI Torch] solve_time = {t1 - t0:.6f} s | K={K}")
        U = np.nan_to_num(U, nan=0.0, posinf=u_max[0], neginf=u_min[0]).astype(np.float32)

        # Prediction line (CPU sim)
        x_pred = x_mppi.copy()
        pred_xs = [float(x_pred[0])]
        pred_ys = [float(x_pred[1])]
        for t in range(T):
            x_pred = diffdrive_dynamics(x_pred, U[t], dt, v_max=float(u_max[0])).astype(np.float32)
            pred_xs.append(float(x_pred[0]))
            pred_ys.append(float(x_pred[1]))
        pred_line.set_data(pred_xs, pred_ys)

        # Sample rollout fan
        if Xsamp is not None:
            ns = min(N_SHOW, Xsamp.shape[1])
            for i in range(ns):
                sample_lines[i].set_data(Xsamp[:, i, 0], Xsamp[:, i, 1])
            for i in range(ns, N_SHOW):
                sample_lines[i].set_data([], [])

        # Apply first control + mild speed smoothing
        u0 = U[0].copy()
        u0[0] = np.clip(0.4 * u0[0] + 0.6 * v_des, 0.0, float(u_max[0]))
        u0 = np.nan_to_num(u0, nan=0.0, posinf=u_max[0], neginf=u_min[0])

        x_mppi = diffdrive_dynamics(x_mppi, u0, dt, v_max=float(u_max[0])).astype(np.float32)

        if not np.all(np.isfinite(x_mppi)):
            print(f"[ERROR] Non-finite ego state at step {k}, t={t_now:.3f}: x_mppi={x_mppi}, u0={u0}")
            break

        # Shift warm-start controls (IMPORTANT: use mppi.U, not a missing attribute)
        mppi.U[:-1] = mppi.U[1:]
        mppi.U[-1] = np.array([v_des, 0.0], dtype=np.float32)
        mppi.U_cpu = mppi.U  # keep internal storage consistent too

        xs.append(float(x_mppi[0])); ys.append(float(x_mppi[1]))
        path_line.set_data(xs, ys)

        set_img_pose(ego_img_artist, x_mppi[0], x_mppi[1], x_mppi[2], ego_length, ego_width, ax)

        # Draw obstacles
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
                    obs_length,
                    obs_width,
                    ax
                )
            else:
                obs_imgs[i].set_visible(False)

        CAM_X_AHEAD  = 9.0
        CAM_X_BEHIND = 9.0
        CAM_Y_HALF   = 2.0

        x_left  = float(x_mppi[0] - CAM_X_BEHIND)
        x_right = float(x_mppi[0] + CAM_X_AHEAD)
        y_low   = float(y_bottom - 0.2)
        y_high  = float(y_top_outer + 0.2)

        if np.isfinite(x_left) and np.isfinite(x_right) and np.isfinite(y_low) and np.isfinite(y_high):
            ax.set_xlim(x_left, x_right)
            ax.set_ylim(y_low, y_high)
        else:
            print(f"[WARN] Skipping axis update due to non-finite limits: "
                  f"x=({x_left},{x_right}) y=({y_low},{y_high}) x_mppi={x_mppi}")
            break

        ax.set_title(f"MPPI Torch Generic | t={t_now:.2f}s | K={K} | solve={t1 - t0:.3f}s")
        plt.pause(0.01)

    plt.ioff()
    plt.show()
