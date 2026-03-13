#!/usr/bin/env python3
"""
Buffered DRA-MPPI Crazyflie outer-loop simulation (Torch).

- Minimum-snap reference trajectory (3D)
- Static cylinder obstacles (soft exponential cost, z-gated)
- One moving spherical obstacle (mean prediction + uncertainty)
- DRA-MPPI risk term via Monte-Carlo collision probability with moving obstacle uncertainty

This version:
- Runs full simulation and stores state/obstacle/solve-time buffers
- Saves buffers to .npz for offline plotting/replay
"""

from __future__ import annotations
import argparse
import time
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

# -----------------------------
# Helpers
# -----------------------------
def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------
# Minimum-snap trajectory (7th order) via KKT
# -----------------------------
def poly_derivative_vector(order, deriv, t, xp=np):
    v = xp.zeros(order + 1, dtype=float)
    for i in range(deriv, order + 1):
        coef = 1.0
        for k in range(deriv):
            coef *= (i - k)
        v[i] = coef * (t ** (i - deriv))
    return v


def snap_cost_Q(order, T, deriv=4):
    Q = np.zeros((order + 1, order + 1), dtype=float)
    for i in range(deriv, order + 1):
        for j in range(deriv, order + 1):
            ci = 1.0
            for k in range(deriv):
                ci *= (i - k)
            cj = 1.0
            for k in range(deriv):
                cj *= (j - k)
            power = (i - deriv) + (j - deriv)
            Q[i, j] = ci * cj * (T ** (power + 1)) / (power + 1)
    return Q


def min_snap_1d(waypoints, seg_times):
    wp = np.asarray(waypoints, dtype=float).ravel()
    n_seg = len(wp) - 1
    if n_seg < 1:
        raise ValueError("Need at least 2 waypoints.")

    order = 7
    n_coef = order + 1
    n_var = n_seg * n_coef

    Q = np.zeros((n_var, n_var), dtype=float)
    for k in range(n_seg):
        Qk = snap_cost_Q(order, seg_times[k], deriv=4)
        sl = slice(k*n_coef, (k+1)*n_coef)
        Q[sl, sl] = Qk

    m_con = 2*n_seg + 6 + 3*(n_seg-1)
    A = np.zeros((m_con, n_var), dtype=float)
    b = np.zeros((m_con,), dtype=float)
    row = 0

    for k in range(n_seg):
        T = seg_times[k]
        sl = slice(k*n_coef, (k+1)*n_coef)
        v0 = poly_derivative_vector(order, 0, 0.0)
        vT = poly_derivative_vector(order, 0, T)

        A[row, sl] = v0; b[row] = wp[k];   row += 1
        A[row, sl] = vT; b[row] = wp[k+1]; row += 1

    sl0 = slice(0, n_coef)
    for d in (1, 2, 3):
        A[row, sl0] = poly_derivative_vector(order, d, 0.0)
        b[row] = 0.0
        row += 1

    slL = slice((n_seg-1)*n_coef, n_seg*n_coef)
    TL = seg_times[-1]
    for d in (1, 2, 3):
        A[row, slL] = poly_derivative_vector(order, d, TL)
        b[row] = 0.0
        row += 1

    for k in range(n_seg - 1):
        Tk = seg_times[k]
        slA = slice(k*n_coef, (k+1)*n_coef)
        slB = slice((k+1)*n_coef, (k+2)*n_coef)
        for d in (1, 2, 3):
            A[row, slA] = poly_derivative_vector(order, d, Tk)
            A[row, slB] = -poly_derivative_vector(order, d, 0.0)
            b[row] = 0.0
            row += 1

    KKT = np.zeros((n_var + m_con, n_var + m_con), dtype=float)
    KKT[:n_var, :n_var] = Q
    KKT[:n_var, n_var:] = A.T
    KKT[n_var:, :n_var] = A
    rhs = np.zeros((n_var + m_con,), dtype=float)
    rhs[n_var:] = b

    sol = np.linalg.solve(KKT, rhs)
    c = sol[:n_var]
    return c.reshape(n_seg, n_coef)


@dataclass
class MinSnapTraj:
    waypoints: np.ndarray
    seg_times: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    cz: np.ndarray
    total_time: float

    def eval(self, t: float):
        t = float(np.clip(t, 0.0, self.total_time))
        acc = 0.0
        for k, Tk in enumerate(self.seg_times):
            if t <= acc + Tk or k == len(self.seg_times) - 1:
                tau = t - acc
                break
            acc += Tk

        p_basis = poly_derivative_vector(7, 0, tau)
        v_basis = poly_derivative_vector(7, 1, tau)

        px = float(p_basis @ self.cx[k]); vx = float(v_basis @ self.cx[k])
        py = float(p_basis @ self.cy[k]); vy = float(v_basis @ self.cy[k])
        pz = float(p_basis @ self.cz[k]); vz = float(v_basis @ self.cz[k])
        return np.array([px, py, pz]), np.array([vx, vy, vz])


def build_min_snap_3d(waypoints_xyz, avg_speed=1.5):
    wps = np.asarray(waypoints_xyz, dtype=float)
    dif = wps[1:] - wps[:-1]
    dist = np.linalg.norm(dif, axis=1)
    seg_times = np.maximum(0.25, dist / max(1e-6, avg_speed))

    cx = min_snap_1d(wps[:, 0], seg_times)
    cy = min_snap_1d(wps[:, 1], seg_times)
    cz = min_snap_1d(wps[:, 2], seg_times)
    return MinSnapTraj(wps, seg_times, cx, cy, cz, float(seg_times.sum()))


# ============================================================
# Torch batched dynamics & costs
# ============================================================
def _wrap_pi_torch(a: torch.Tensor) -> torch.Tensor:
    return (a + torch.pi) % (2.0 * torch.pi) - torch.pi


def _z_body_world_torch(phi: torch.Tensor, theta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    cphi = torch.cos(phi); sphi = torch.sin(phi)
    cth  = torch.cos(theta); sth = torch.sin(theta)
    cpsi = torch.cos(psi); spsi = torch.sin(psi)
    zx = cpsi*sth*cphi + spsi*sphi
    zy = spsi*sth*cphi - cpsi*sphi
    zz = cth*cphi
    return torch.stack([zx, zy, zz], dim=-1)


def quad_dyn_step(X: torch.Tensor, U: torch.Tensor, dt: float, m: float, g: float, **_kw) -> torch.Tensor:
    # X: (M,9) [px,py,pz,vx,vy,vz,psi,phi,theta], U: (M,4) [T,phi_cmd,theta_cmd,psidot]
    p = X[:, 0:3]
    v = X[:, 3:6]
    psi = X[:, 6]
    phi = X[:, 7]
    theta = X[:, 8]

    Tcmd = U[:, 0]
    phi_cmd = U[:, 1]
    theta_cmd = U[:, 2]
    psidot = U[:, 3]

    tau_phi = float(_kw.get("tau_phi", 0.14))
    tau_theta = float(_kw.get("tau_theta", 0.14))
    ang_max = float(_kw.get("ang_max", math.radians(25.0)))
    phi_rate_max = float(_kw.get("phi_rate_max", math.radians(250.0)))
    theta_rate_max = float(_kw.get("theta_rate_max", math.radians(250.0)))

    phi_dot = (phi_cmd - phi) / max(1e-6, tau_phi)
    theta_dot = (theta_cmd - theta) / max(1e-6, tau_theta)
    phi_dot = torch.clamp(phi_dot, -phi_rate_max, phi_rate_max)
    theta_dot = torch.clamp(theta_dot, -theta_rate_max, theta_rate_max)
    phi_next = torch.clamp(phi + float(dt) * phi_dot, -ang_max, ang_max)
    theta_next = torch.clamp(theta + float(dt) * theta_dot, -ang_max, ang_max)

    zb = _z_body_world_torch(phi, theta, psi)
    gvec = torch.tensor([0.0, 0.0, float(g)], device=X.device, dtype=X.dtype)
    a = (Tcmd[:, None] / float(m)) * zb - gvec[None, :]

    v_next = v + float(dt) * a
    p_next = p + float(dt) * v_next
    psi_next = _wrap_pi_torch(psi + float(dt) * psidot)
    return torch.cat([p_next, v_next, psi_next[:, None], phi_next[:, None], theta_next[:, None]], dim=1)


def running_cost_quad(
    X: torch.Tensor, U: torch.Tensor, t: int,
    ref_seq: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
    # cylinders
    cyl_cx: torch.Tensor | None = None,
    cyl_cy: torch.Tensor | None = None,
    cyl_r: torch.Tensor | None = None,
    cyl_zmin: torch.Tensor | None = None,
    cyl_zmax: torch.Tensor | None = None,
    w_cyl: float = 350.0,
    cyl_safety_margin: float = 0.25,
    cyl_alpha: float = 10.0,
    # moving mean (3D) soft cost
    obs_mean_3d: torch.Tensor | None = None,   # (T+1,3)
    w_moving: float = 800.0,
    moving_r: float = 0.35,
    moving_safety_margin: float = 0.50,
    moving_alpha: float = 12.0,
    U_nom: torch.Tensor | None = None,  # (T,4)
    Rd: torch.Tensor | None = None,     # (4,)
    **_kw
) -> torch.Tensor:
    ref = ref_seq[t + 1]  # (4,)
    e_pos = X[:, 0:3] - ref[None, 0:3]
    e_psi = _wrap_pi_torch(X[:, 6] - ref[3]).unsqueeze(1)
    e = torch.cat([e_pos, e_psi], dim=1)

    J = torch.sum((e * e) * Q[None, :], dim=1) + torch.sum((U * U) * R[None, :], dim=1)
    if (U_nom is not None) and (Rd is not None):
        du = U - U_nom[t][None, :]
        J = J + torch.sum((du * du) * Rd[None, :], dim=1)

    # cylinders: exp(-alpha * signed_dist) with z-gate
    if cyl_cx is not None and cyl_cx.numel() > 0:
        px = X[:, 0:1]
        py = X[:, 1:2]
        pz = X[:, 2:3]

        dx = px - cyl_cx[None, :]
        dy = py - cyl_cy[None, :]
        d_xy = torch.sqrt(dx * dx + dy * dy)
        signed = d_xy - (cyl_r[None, :] + float(cyl_safety_margin))

        inside_z = (pz >= cyl_zmin[None, :]) & (pz <= cyl_zmax[None, :])
        pen = torch.exp(-float(cyl_alpha) * signed)
        pen = torch.where(inside_z, pen, torch.zeros_like(pen))
        J = J + float(w_cyl) * torch.sum(pen, dim=1)

    # moving mean soft cost (true 3D dist)
    if obs_mean_3d is not None:
        if obs_mean_3d.ndim == 2:
            obs = obs_mean_3d[t + 1]  # (3,)
            d = X[:, 0:3] - obs[None, :]
            dist = torch.sqrt(torch.sum(d * d, dim=1))
            signed = dist - (float(moving_r) + float(moving_safety_margin))
            pen = torch.exp(-float(moving_alpha) * signed)
            J = J + float(w_moving) * pen
        elif obs_mean_3d.ndim == 3:
            obs = obs_mean_3d[t + 1]  # (K,3)
            d = X[:, None, 0:3] - obs[None, :, :]
            dist = torch.sqrt(torch.sum(d * d, dim=2))
            signed = dist - (float(moving_r) + float(moving_safety_margin))
            pen = torch.exp(-float(moving_alpha) * signed)
            J = J + float(w_moving) * torch.sum(pen, dim=1)

    return J


def terminal_cost_quad(X: torch.Tensor, t_final: int, ref_seq: torch.Tensor, Qf: torch.Tensor, **_kw) -> torch.Tensor:
    ref = ref_seq[-1]
    e_pos = X[:, 0:3] - ref[None, 0:3]
    e_psi = _wrap_pi_torch(X[:, 6] - ref[3]).unsqueeze(1)
    e = torch.cat([e_pos, e_psi], dim=1)
    return torch.sum((e * e) * Qf[None, :], dim=1)


# ============================================================
# DRA-MPPI wrapper (Torch)
# ============================================================
@dataclass
class DRAParams:
    dt: float = 0.02
    horizon_steps: int = 60
    rollouts: int = 1096
    iterations: int = 1
    lam: float = 1.0

    # bounds
    ang_max: float = math.radians(25.0)
    yawrate_max: float = math.radians(200.0)
    tau_phi: float = 0.14
    tau_theta: float = 0.14
    phi_rate_max: float = math.radians(250.0)
    theta_rate_max: float = math.radians(250.0)

    # control noise std (diag): [thrust, phi, theta, yawrate]
    sigma: np.ndarray | None = None

    # cylinder soft cost
    w_cyl: float = 350.0
    cyl_safety_margin: float = 0.25
    cyl_alpha: float = 10.0

    # moving mean soft cost
    w_moving: float = 800.0
    moving_r: float = 0.35
    moving_safety_margin: float = 0.50
    moving_alpha: float = 12.0

    # DRA risk terms
    sigma_cp: float = 0.05
    Nmc: int = 3000
    omega_soft: float = 10.0
    omega_hard: float = 1000.0
    obs_pos_sigma_xyz: tuple[float, float, float] = (0.25, 0.25, 0.25)
    mc_chunk: int = 512

    # drone radius included in safe radius for DRA collision sphere
    drone_radius: float = 0.25
    # control-smoothing weight around nominal sequence
    Rd_u: tuple[float, float, float, float] = (0.0, 3.0, 3.0, 0.5)


class DRA_MPPI:
    def __init__(
        self,
        dt: float,
        T: int,
        M: int,
        lam: float,
        noise_sigma: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
        dynamics,
        running_cost,
        terminal_cost,
        I: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        dyn_kwargs: dict | None = None,
        cost_kwargs: dict | None = None,
        sigma_cp: float = 0.05,
        Nmc: int = 3000,
        omega_soft: float = 10.0,
        omega_hard: float = 1000.0,
        obs_pos_sigma_xyz: tuple[float, float, float] = (0.25, 0.25, 0.25),
        verbose: bool = False,
    ):
        self.dt = float(dt)
        self.T = int(T)
        self.M = int(M)
        self.lam = float(lam)
        self.I = int(I)
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = bool(verbose)

        self.noise_sigma = torch.as_tensor(np.asarray(noise_sigma, dtype=np.float32), device=self.device, dtype=self.dtype)
        self.nu = int(self.noise_sigma.numel())
        self.u_min = torch.as_tensor(np.asarray(u_min, dtype=np.float32), device=self.device, dtype=self.dtype).reshape(self.nu)
        self.u_max = torch.as_tensor(np.asarray(u_max, dtype=np.float32), device=self.device, dtype=self.dtype).reshape(self.nu)
        self._u_min_112 = self.u_min.view(1, 1, self.nu)
        self._u_max_112 = self.u_max.view(1, 1, self.nu)

        self.dynamics = dynamics
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.dyn_kwargs = dict(dyn_kwargs or {})
        self.cost_kwargs = dict(cost_kwargs or {})

        self.sigma_cp = float(sigma_cp)
        self.Nmc = int(Nmc)
        self.omega_soft = float(omega_soft)
        self.omega_hard = float(omega_hard)
        self.obs_pos_sigma_xyz = torch.as_tensor(np.asarray(obs_pos_sigma_xyz, dtype=np.float32), device=self.device, dtype=self.dtype).reshape(3)
        self._sigma_cp_t = torch.tensor(self.sigma_cp, device=self.device, dtype=self.dtype)
        self._omega_soft_t = torch.tensor(self.omega_soft, device=self.device, dtype=self.dtype)
        self._omega_hard_t = torch.tensor(self.omega_hard, device=self.device, dtype=self.dtype)

        self.U_cpu = np.zeros((self.T, self.nu), dtype=np.float32)
        self.U = self.U_cpu
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(3)

    @torch.no_grad()
    def _dra_risk_cost_over_time(self, X_hist_xyz: torch.Tensor, obs_mean_3d: torch.Tensor, safe_R: float, mc_chunk: int) -> torch.Tensor:
        T, M, _ = X_hist_xyz.shape
        if obs_mean_3d is None or obs_mean_3d.numel() == 0 or safe_R <= 0.0:
            return torch.zeros(M, device=self.device, dtype=self.dtype)
        if obs_mean_3d.ndim == 2:
            obs_all = obs_mean_3d.unsqueeze(1)  # (T+1,1,3)
        elif obs_mean_3d.ndim == 3:
            obs_all = obs_mean_3d  # (T+1,K,3)
        else:
            raise ValueError(f"obs_mean_3d must be (T+1,3) or (T+1,K,3), got shape={tuple(obs_mean_3d.shape)}")
        K = int(obs_all.shape[1])

        sx = max(float(self.obs_pos_sigma_xyz[0].item()), 1e-6)
        sy = max(float(self.obs_pos_sigma_xyz[1].item()), 1e-6)
        sz = max(float(self.obs_pos_sigma_xyz[2].item()), 1e-6)
        norm_const = torch.tensor(1.0 / (((2.0 * np.pi) ** 1.5) * sx * sy * sz), device=self.device, dtype=self.dtype)
        r2 = float(safe_R * safe_R)

        risk_cost = torch.zeros(M, device=self.device, dtype=self.dtype)
        for t in range(T):
            pts = X_hist_xyz[t]  # (M,3)
            px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]

            xlow = torch.min(px) - safe_R
            xhigh = torch.max(px) + safe_R
            ylow = torch.min(py) - safe_R
            yhigh = torch.max(py) + safe_R
            zlow = torch.min(pz) - safe_R
            zhigh = torch.max(pz) + safe_R

            dx = torch.clamp(xhigh - xlow, min=1e-6)
            dy = torch.clamp(yhigh - ylow, min=1e-6)
            dz = torch.clamp(zhigh - zlow, min=1e-6)
            vol = dx * dy * dz

            ux = torch.rand((self.Nmc,), device=self.device, dtype=self.dtype, generator=self._rng)
            uy = torch.rand((self.Nmc,), device=self.device, dtype=self.dtype, generator=self._rng)
            uz = torch.rand((self.Nmc,), device=self.device, dtype=self.dtype, generator=self._rng)
            xj = xlow + ux * dx
            yj = ylow + uy * dy
            zj = zlow + uz * dz
            cell_vol = vol / float(self.Nmc)

            for k in range(K):
                mu = obs_all[t + 1, k]  # (3,)
                dxo = (xj - mu[0]) / sx
                dyo = (yj - mu[1]) / sy
                dzo = (zj - mu[2]) / sz
                pdf = norm_const * torch.exp(-0.5 * (dxo * dxo + dyo * dyo + dzo * dzo))
                pjoint_mass = torch.clamp(pdf * cell_vol, 0.0, 0.999)

                Phat = torch.zeros(M, device=self.device, dtype=self.dtype)
                for j0 in range(0, self.Nmc, mc_chunk):
                    j1 = min(self.Nmc, j0 + mc_chunk)
                    dxm = xj[j0:j1].unsqueeze(0) - px.unsqueeze(1)
                    dym = yj[j0:j1].unsqueeze(0) - py.unsqueeze(1)
                    dzm = zj[j0:j1].unsqueeze(0) - pz.unsqueeze(1)
                    inside = (dxm * dxm + dym * dym + dzm * dzm) <= r2
                    Phat += inside.to(self.dtype) @ pjoint_mass[j0:j1]

                Phat = torch.clamp(Phat, 0.0, 1.0)
                risk_cost += self._omega_soft_t * Phat
                risk_cost += self._omega_hard_t * (Phat > self._sigma_cp_t).to(self.dtype)

        return risk_cost

    @torch.no_grad()
    def plan(self, x0_np: np.ndarray, return_samples: bool = False, n_show: int = 60, show_seed: int = 0):
        x0 = torch.as_tensor(np.asarray(x0_np, dtype=np.float32).reshape(-1), device=self.device, dtype=self.dtype)
        U = torch.as_tensor(self.U_cpu, device=self.device, dtype=self.dtype).clone()
        sigma = self.noise_sigma.view(1, 1, self.nu)

        obs_mean_3d = self.cost_kwargs.get("obs_mean_3d", None)
        moving_r = float(self.cost_kwargs.get("moving_r", 0.0))
        moving_margin = float(self.cost_kwargs.get("moving_safety_margin", 0.0))
        drone_radius = float(self.cost_kwargs.get("drone_radius", 0.0))
        safe_R = moving_r + moving_margin + drone_radius
        mc_chunk = int(self.cost_kwargs.get("mc_chunk", 512))

        if return_samples:
            ns = int(min(max(1, int(n_show)), self.M))
            g = torch.Generator(device=self.device)
            g.manual_seed(int(show_seed))
            sample_idx = torch.randperm(self.M, generator=g, device=self.device)[:ns]
        else:
            ns = 0
            sample_idx = None

        for _ in range(self.I):
            eps = torch.randn((self.M, self.T, self.nu), device=self.device, dtype=self.dtype, generator=self._rng) * sigma
            eps[0, :, :] = 0.0
            U_roll = torch.clamp(U.unsqueeze(0) + eps, self._u_min_112, self._u_max_112)

            J = torch.zeros((self.M,), device=self.device, dtype=self.dtype)
            X = x0.unsqueeze(0).repeat(self.M, 1)
            X_hist_xyz = torch.zeros((self.T, self.M, 3), device=self.device, dtype=self.dtype)
            if return_samples:
                nx = int(x0.numel())
                Xsamp = torch.zeros((self.T + 1, ns, nx), device=self.device, dtype=self.dtype)
                Xsamp[0] = X.index_select(0, sample_idx)
            else:
                Xsamp = None

            for t in range(self.T):
                U_t = U_roll[:, t, :]
                X = self.dynamics(X, U_t, self.dt, **self.dyn_kwargs)
                X_hist_xyz[t] = X[:, 0:3]
                J += self.running_cost(X, U_t, t=t, **self.cost_kwargs)
                if return_samples:
                    Xsamp[t + 1] = X.index_select(0, sample_idx)

            J += self.terminal_cost(X, t_final=self.T, **self.cost_kwargs)
            if obs_mean_3d is not None:
                J += self._dra_risk_cost_over_time(X_hist_xyz, obs_mean_3d, safe_R=safe_R, mc_chunk=mc_chunk)

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

        U_cpu = U.detach().cpu().numpy().astype(np.float32)
        self.U_cpu = U_cpu
        self.U = self.U_cpu
        Xsamp_cpu = None
        if return_samples and Xsamp is not None:
            Xsamp_cpu = Xsamp.detach().cpu().numpy().astype(np.float32)
        return U_cpu, Xsamp_cpu


class TorchDRAMPPIQuadOuter:
    def __init__(self, mass=0.028, g=9.81, params: DRAParams | None = None, cylinders=None, device=None):
        self.m = float(mass)
        self.g = float(g)
        self.p = params if params is not None else DRAParams()

        hover = self.m * self.g
        T_min = 0.0
        T_max = 2.0 * hover

        if self.p.sigma is None:
            self.p.sigma = np.array(
                [0.06 * hover, math.radians(1.3), math.radians(1.3), math.radians(8.0)],
                dtype=np.float32,
            )

        self.dt = float(self.p.dt)
        self.T = int(self.p.horizon_steps)
        self.M = int(self.p.rollouts)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.u_min = np.array([T_min, -self.p.ang_max, -self.p.ang_max, -self.p.yawrate_max], dtype=np.float32)
        self.u_max = np.array([T_max, self.p.ang_max, self.p.ang_max, self.p.yawrate_max], dtype=np.float32)

        sigma_np = np.asarray(self.p.sigma, dtype=np.float32).reshape(4,)
        self.R_np = 1.0 / np.maximum(sigma_np ** 2, 1e-12).astype(np.float32)
        self.Rd_np = np.asarray(self.p.Rd_u, dtype=np.float32).reshape(4,)

        cyl = cylinders if cylinders is not None else []
        if len(cyl) > 0:
            self.cyl_cx = torch.tensor([c["cx"] for c in cyl], device=self.device, dtype=torch.float32)
            self.cyl_cy = torch.tensor([c["cy"] for c in cyl], device=self.device, dtype=torch.float32)
            self.cyl_r = torch.tensor([c["r"] for c in cyl], device=self.device, dtype=torch.float32)
            self.cyl_zmin = torch.tensor([c.get("zmin", -1e9) for c in cyl], device=self.device, dtype=torch.float32)
            self.cyl_zmax = torch.tensor([c.get("zmax", 1e9) for c in cyl], device=self.device, dtype=torch.float32)
        else:
            self.cyl_cx = self.cyl_cy = self.cyl_r = self.cyl_zmin = self.cyl_zmax = None

        self.mppi = DRA_MPPI(
            dt=self.dt,
            T=self.T,
            M=self.M,
            lam=float(self.p.lam),
            noise_sigma=np.asarray(self.p.sigma, dtype=np.float32),
            u_min=self.u_min,
            u_max=self.u_max,
            dynamics=quad_dyn_step,
            running_cost=running_cost_quad,
            terminal_cost=terminal_cost_quad,
            I=int(self.p.iterations),
            device=self.device,
            dtype=torch.float32,
            dyn_kwargs=dict(
                m=self.m,
                g=self.g,
                tau_phi=self.p.tau_phi,
                tau_theta=self.p.tau_theta,
                ang_max=self.p.ang_max,
                phi_rate_max=self.p.phi_rate_max,
                theta_rate_max=self.p.theta_rate_max,
            ),
            cost_kwargs=dict(
                ref_seq=None,
                Q=None,
                Qf=None,
                R=None,
                obs_mean_3d=None,
                cyl_cx=None,
                cyl_cy=None,
                cyl_r=None,
                cyl_zmin=None,
                cyl_zmax=None,
                w_cyl=float(self.p.w_cyl),
                cyl_safety_margin=float(self.p.cyl_safety_margin),
                cyl_alpha=float(self.p.cyl_alpha),
                w_moving=float(self.p.w_moving),
                moving_r=float(self.p.moving_r),
                moving_safety_margin=float(self.p.moving_safety_margin),
                moving_alpha=float(self.p.moving_alpha),
                U_nom=None,
                Rd=None,
                drone_radius=float(self.p.drone_radius),
                mc_chunk=int(self.p.mc_chunk),
            ),
            sigma_cp=float(self.p.sigma_cp),
            Nmc=int(self.p.Nmc),
            omega_soft=float(self.p.omega_soft),
            omega_hard=float(self.p.omega_hard),
            obs_pos_sigma_xyz=tuple(self.p.obs_pos_sigma_xyz),
            verbose=False,
        )

        self.mppi.U_cpu[:] = 0.0
        self.mppi.U_cpu[:, 0] = hover
        self.mppi.U = self.mppi.U_cpu

    def _predict_nominal_xyz(self, x0_np, U_cpu):
        x = torch.as_tensor(np.asarray(x0_np, dtype=np.float32).reshape(1, 9), device=self.device, dtype=torch.float32)
        pred = np.zeros((self.T + 1, 3), dtype=np.float32)
        pred[0] = np.asarray(x0_np, dtype=np.float32).reshape(9,)[0:3]
        dyn = self.mppi.dyn_kwargs
        for t in range(self.T):
            u_t = torch.as_tensor(U_cpu[t].reshape(1, 4), device=self.device, dtype=torch.float32)
            x = quad_dyn_step(
                x, u_t, self.dt, self.m, self.g,
                tau_phi=dyn.get("tau_phi", self.p.tau_phi),
                tau_theta=dyn.get("tau_theta", self.p.tau_theta),
                ang_max=dyn.get("ang_max", self.p.ang_max),
                phi_rate_max=dyn.get("phi_rate_max", self.p.phi_rate_max),
                theta_rate_max=dyn.get("theta_rate_max", self.p.theta_rate_max),
            )
            pred[t + 1] = x[0, 0:3].detach().cpu().numpy().astype(np.float32)
        return pred

    def plan(
        self,
        x0_np,
        ref_seq_np,
        Q_np,
        Qf_np,
        obs_seq_np,
        return_predictions: bool = False,
        n_pred: int = 20,
        include_nominal_prediction: bool = True,
    ):
        ref_seq_t = torch.as_tensor(ref_seq_np, device=self.device, dtype=torch.float32)
        Q_t = torch.as_tensor(Q_np, device=self.device, dtype=torch.float32)
        Qf_t = torch.as_tensor(Qf_np, device=self.device, dtype=torch.float32)
        R_t = torch.as_tensor(self.R_np, device=self.device, dtype=torch.float32)
        obs_mean_3d_t = torch.as_tensor(obs_seq_np, device=self.device, dtype=torch.float32)

        ck = self.mppi.cost_kwargs
        ck["ref_seq"] = ref_seq_t
        ck["Q"] = Q_t
        ck["Qf"] = Qf_t
        ck["R"] = R_t
        ck["obs_mean_3d"] = obs_mean_3d_t
        ck["U_nom"] = torch.as_tensor(self.mppi.U_cpu, device=self.device, dtype=torch.float32)
        ck["Rd"] = torch.as_tensor(self.Rd_np, device=self.device, dtype=torch.float32)

        if self.cyl_cx is not None:
            ck["cyl_cx"] = self.cyl_cx
            ck["cyl_cy"] = self.cyl_cy
            ck["cyl_r"] = self.cyl_r
            ck["cyl_zmin"] = self.cyl_zmin
            ck["cyl_zmax"] = self.cyl_zmax
        else:
            ck["cyl_cx"] = ck["cyl_cy"] = ck["cyl_r"] = ck["cyl_zmin"] = ck["cyl_zmax"] = None

        U_cpu, Xsamp = self.mppi.plan(
            x0_np,
            return_samples=bool(return_predictions),
            n_show=int(n_pred),
            show_seed=0,
        )
        u0 = U_cpu[0].copy()
        pred_samples_xyz = None
        pred_nominal_xyz = None
        if return_predictions:
            if Xsamp is not None:
                pred_samples_xyz = np.asarray(Xsamp[:, :, 0:3], dtype=np.float32)
            if include_nominal_prediction:
                pred_nominal_xyz = self._predict_nominal_xyz(x0_np, U_cpu)
        self.mppi.U_cpu[:-1] = self.mppi.U_cpu[1:]
        self.mppi.U_cpu[-1] = self.mppi.U_cpu[-2]
        self.mppi.U = self.mppi.U_cpu
        if return_predictions:
            return u0, pred_samples_xyz, pred_nominal_xyz
        return u0


# ============================================================
# Buffered simulation (plot after run)
# ============================================================
def simulate(save_dir: str | None = None):
    waypoints = np.array([
        [ 2.5,  2.0, 0.0],
        [ 0.0,  3.5, 2.0],
        [-3.0,  1.5, 4.5],
        [-2.0, -2.5, 3.0],
        [ 2.0, -3.0, 1.0],
        [ 3.0,  0.0, 0.5],
        [ 2.5,  2.0, 0.0],
    ], dtype=float)

    waypoints_1 = np.array([
        [ 2.5,  2.0, 0.0],
        [ 3.0,  0.0, 0.5],
        [ 2.0, -3.0, 1.0],
        [-2.0, -2.5, 3.0],
        [-3.0,  1.5, 4.5],
        [ 0.0,  3.5, 2.0],
        [ 2.5,  2.0, 0.0],
    ], dtype=float)
    waypoints_2 = np.array([
        [ 2.5,  2.0, 0.0],
        [-2.0, -2.5, 3.0],
        [ 0.0,  3.5, 2.0],
        [ 3.0,  0.0, 0.5],
        [ 2.5,  2.0, 0.0],
    ], dtype=float)

    traj = build_min_snap_3d(waypoints, avg_speed=1.8)

    cylinders = [
        {"cx":  0.5, "cy":  1.0, "r": 0.6, "zmin": 0.0, "zmax": 4.5},
        {"cx": -1.8, "cy": -0.5, "r": 0.7, "zmin": 0.0, "zmax": 5.0},
        {"cx":  1.5, "cy": -1.8, "r": 0.5, "zmin": 0.0, "zmax": 3.0},
        {"cx": -1.0, "cy":  4.0, "r": 0.7, "zmin": 0.0, "zmax": 4.5},
    ]

    moving_trajs = [
        build_min_snap_3d(waypoints_1, avg_speed=1.8),
        build_min_snap_3d(waypoints_2, avg_speed=1.8),
    ]

    params = DRAParams(
        dt=0.028677838561381404,
        horizon_steps=35,
        rollouts=1096,
        iterations=1,
        lam=0.9447651256582109,

        ang_max=math.radians(28.533048677493525),
        yawrate_max=math.radians(125.002671219858),
        tau_phi=0.22445102088241203,
        tau_theta=0.19182828711184713,
        phi_rate_max=math.radians(186.91775798517736),
        theta_rate_max=math.radians(250.74834372828798),

        sigma=np.array(
            [
                0.03860791271607059,
                math.radians(6.072834867326699),
                math.radians(7.139534744261536),
                math.radians(9.50181217517332),
            ],
            dtype=np.float32,
        ),

        w_cyl=510.6988597095838,
        cyl_safety_margin=0.2150292356967072,
        cyl_alpha=8.72155294537059,

        w_moving=376.5800114691624,
        moving_r=0.3045201563139591,
        moving_safety_margin=0.4224283788684363,
        moving_alpha=15.033309531167399,

        sigma_cp=0.05,
        Nmc=2500,
        omega_soft=10.0,
        omega_hard=1000.0,
        obs_pos_sigma_xyz=(0.1569012991377936, 0.18206816632027703, 0.18206816632027703),
        mc_chunk=512,

        drone_radius=0.3662153322325755,
        Rd_u=(0.9046852667672367, 2.390585391824126, 2.8043325852902052, 1.0686213306251247),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctrl = TorchDRAMPPIQuadOuter(mass=0.028, g=9.81, params=params, cylinders=cylinders, device=device)
    print("Planner backend:", device)

    # tracking weights (yaw disabled)
    Q = np.array([84.46141326914871, 49.474546616417555, 61.1121046345, 0.0], dtype=np.float32)
    Qf = np.array([303.4373360699704, 193.59494682617972, 231.06814617684867, 0.0], dtype=np.float32)

    dt = params.dt
    sim_T = max([traj.total_time] + [mt.total_time for mt in moving_trajs])
    steps = int(sim_T / dt) + 1

    # initial state
    p0, _ = traj.eval(0.0)
    x = np.zeros(9, dtype=float)
    x[0:3] = p0
    x[7] = 0.0
    x[8] = 0.0

    # for yaw reference smoothing
    last_yaw_ref = float(x[6])

    def yaw_follow_from_vel(v, fallback):
        vx, vy = float(v[0]), float(v[1])
        if vx*vx + vy*vy < 1e-6:
            return fallback
        return math.atan2(vy, vx)

    lead_time = 1.724533520574184

    # Precompute curves for reference drawing
    tt = np.linspace(0.0, traj.total_time, 600)
    ref_curve = np.array([traj.eval(float(ti))[0] for ti in tt])

    obs_curves = []
    for moving_traj in moving_trajs:
        tt2 = np.linspace(0.0, moving_traj.total_time, 600)
        obs_curve = np.array([moving_traj.eval(float(ti))[0] for ti in tt2])
        obs_curves.append(obs_curve)

    # axis limits for post-run plot
    all_pts = np.vstack([ref_curve] + obs_curves)
    mins = all_pts.min(axis=0) - 1.0
    maxs = all_pts.max(axis=0) + 1.0

    # logs / buffers
    X_path = np.zeros((steps, 3), dtype=float)
    X_hist = np.zeros((steps, 9), dtype=float)
    n_obs = len(moving_trajs)
    obs_path = np.zeros((steps, n_obs, 3), dtype=float)
    n_pred_plot = min(8, ctrl.M)
    pred_stride = 1
    pred_samples_xyz = np.full((steps, ctrl.T + 1, n_pred_plot, 3), np.nan, dtype=np.float32)
    pred_nominal_xyz = np.full((steps, ctrl.T + 1, 3), np.nan, dtype=np.float32)
    solve_ms = np.zeros((steps,), dtype=float)
    U_applied = np.zeros((steps, 4), dtype=float)
    sim_time = np.arange(steps, dtype=float) * dt

    for i in range(steps):
        t = i * dt

        # build ref_seq (T+1,4)
        ref_seq = np.zeros((ctrl.T + 1, 4), dtype=float)

        ref_seq[0, 0:3], v0 = traj.eval(min(t, traj.total_time))
        psi0_raw = wrap_pi(yaw_follow_from_vel(v0, last_yaw_ref))
        ref_seq[0, 3] = last_yaw_ref + wrap_pi(psi0_raw - last_yaw_ref)

        for k in range(1, ctrl.T + 1):
            tk = min(traj.total_time, t + k * dt)
            pk, vk = traj.eval(tk)
            psi_raw = wrap_pi(yaw_follow_from_vel(vk, ref_seq[k-1, 3]))
            prev = ref_seq[k-1, 3]
            ref_seq[k, 0:3] = pk
            ref_seq[k, 3] = prev + wrap_pi(psi_raw - prev)

        last_yaw_ref = float(ref_seq[1, 3])

        # moving obstacle mean prediction seq (T+1,n_obs,3)
        obs_seq = np.zeros((ctrl.T + 1, n_obs, 3), dtype=float)
        for j, moving_traj in enumerate(moving_trajs):
            for k in range(ctrl.T + 1):
                tk = min(moving_traj.total_time, t + lead_time + k * dt)
                op, _ = moving_traj.eval(tk)
                obs_seq[k, j] = op

        obs_path[i] = obs_seq[0]

        # plan (time it correctly on cuda)
        if ctrl.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        collect_pred = (i % pred_stride) == 0
        if collect_pred:
            u, pred_s_xyz, pred_nom_xyz = ctrl.plan(
                x,
                ref_seq,
                Q,
                Qf,
                obs_seq,
                return_predictions=True,
                n_pred=n_pred_plot,
                include_nominal_prediction=True,
            )
        else:
            u = ctrl.plan(x, ref_seq, Q, Qf, obs_seq, return_predictions=False)
            pred_s_xyz = None
            pred_nom_xyz = None

        if ctrl.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        solve_ms[i] = ms
        if collect_pred and pred_s_xyz is not None:
            ns = min(n_pred_plot, int(pred_s_xyz.shape[1]))
            pred_samples_xyz[i, :, :ns, :] = pred_s_xyz[:, :ns, :]
        elif i > 0:
            pred_samples_xyz[i] = pred_samples_xyz[i - 1]
        if collect_pred and pred_nom_xyz is not None:
            pred_nominal_xyz[i] = pred_nom_xyz
        elif i > 0:
            pred_nominal_xyz[i] = pred_nominal_xyz[i - 1]

        # print solve time online (don’t spam too hard)
        if (i % 10) == 0:
            print(f"[step {i:04d}] solve_ms={ms:.2f}")

        # apply same numpy sim dynamics
        Tcmd, phi_cmd, theta_cmd, yawrate = map(float, u)
        # bounds
        Tcmd = clamp(Tcmd, ctrl.u_min[0], ctrl.u_max[0])
        phi_cmd = clamp(phi_cmd, -params.ang_max, params.ang_max)
        theta_cmd = clamp(theta_cmd, -params.ang_max, params.ang_max)
        yawrate = clamp(yawrate, -params.yawrate_max, params.yawrate_max)
        U_applied[i] = [Tcmd, phi_cmd, theta_cmd, yawrate]

        phi = float(x[7])
        theta = float(x[8])
        cphi = math.cos(phi); sphi = math.sin(phi)
        cth  = math.cos(theta); sth = math.sin(theta)
        cpsi = math.cos(x[6]); spsi = math.sin(x[6])
        zb = np.array([
            cpsi*sth*cphi + spsi*sphi,
            spsi*sth*cphi - cpsi*sphi,
            cth*cphi
        ], dtype=float)

        a = (Tcmd / ctrl.m) * zb - np.array([0.0, 0.0, ctrl.g])
        phi_dot = (phi_cmd - phi) / max(1e-6, params.tau_phi)
        theta_dot = (theta_cmd - theta) / max(1e-6, params.tau_theta)
        phi_dot = clamp(phi_dot, -params.phi_rate_max, params.phi_rate_max)
        theta_dot = clamp(theta_dot, -params.theta_rate_max, params.theta_rate_max)
        x[3:6] = x[3:6] + dt * a
        x[0:3] = x[0:3] + dt * x[3:6]
        x[6] = wrap_pi(x[6] + dt * yawrate)
        x[7] = clamp(x[7] + dt * phi_dot, -params.ang_max, params.ang_max)
        x[8] = clamp(x[8] + dt * theta_dot, -params.ang_max, params.ang_max)

        X_path[i] = x[0:3]
        X_hist[i] = x.copy()

    run_data = {
        "method": "dramppi",
        "dt": float(dt),
        "sim_time": sim_time,
        "X_path": X_path,
        "X_hist": X_hist,
        "obs_path": obs_path,
        "U_applied": U_applied,
        "solve_ms": solve_ms,
        "ref_curve": ref_curve,
        "obs_curves": np.asarray(obs_curves, dtype=float),
        "mins": mins,
        "maxs": maxs,
        "pred_samples_xyz": pred_samples_xyz,
        "pred_nominal_xyz": pred_nominal_xyz,
        "cylinders": np.asarray(
            [[c["cx"], c["cy"], c["r"], c.get("zmin", 0.0), c.get("zmax", 1.0)] for c in cylinders],
            dtype=float,
        ),
    }
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "dramppi_simulation.npz")
        np.savez_compressed(out_path, **run_data)
        print(f"Saved simulation data to {out_path}")
    return run_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DRA-MPPI Crazyflie simulation.")
    parser.add_argument("--save", action="store_true", help="Save simulation .npz to the plot directory.")
    args = parser.parse_args()
    save_dir = os.path.join(os.path.dirname(__file__), "plot") if args.save else None
    simulate(save_dir=save_dir)
