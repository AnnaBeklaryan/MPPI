#!/usr/bin/env python3
"""
Buffered Torch-MPPI Crazyflie-like quadrotor outer-loop simulation with:
- minimum-snap drone reference trajectory
- static cylinder obstacles
- ONE moving obstacle (sphere) predicted time-ahead

Planner:
- Uses MPPI from mppi_class.py (Torch)

Control: u = [T, phi_cmd, theta_cmd, psi_dot_cmd]
State:   x = [px,py,pz, vx,vy,vz, psi]

This version:
- Runs simulation and stores data in buffers
- Saves buffers to .npz for offline plotting/replay
- Prints solve time during simulation
"""

from __future__ import annotations
import argparse
import time
import math
import os
from dataclasses import dataclass

import numpy as np

import torch
from mppi_class import MPPI  # your Torch MPPI


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
        t = float(clamp(t, 0.0, self.total_time))
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
    if wps.ndim != 2 or wps.shape[1] != 3 or wps.shape[0] < 2:
        raise ValueError("waypoints_xyz must be (m,3), m>=2")

    dif = wps[1:] - wps[:-1]
    dist = np.linalg.norm(dif, axis=1)
    seg_times = np.maximum(0.25, dist / max(1e-6, avg_speed))

    cx = min_snap_1d(wps[:, 0], seg_times)
    cy = min_snap_1d(wps[:, 1], seg_times)
    cz = min_snap_1d(wps[:, 2], seg_times)
    return MinSnapTraj(wps, seg_times, cx, cy, cz, float(seg_times.sum()))


# ============================================================
# Torch dynamics & costs
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


def quad_dyn_step(X, U, dt, m, g, **_kwargs):
    # X: (M,9) [px,py,pz,vx,vy,vz,psi,phi,theta]
    p = X[:, 0:3]
    v = X[:, 3:6]
    psi = X[:, 6]
    phi = X[:, 7]
    theta = X[:, 8]

    Tcmd = U[:, 0]
    phi_cmd = U[:, 1]
    theta_cmd = U[:, 2]
    psidot = U[:, 3]

    tau_phi = float(_kwargs.get("tau_phi", 0.14))
    tau_theta = float(_kwargs.get("tau_theta", 0.14))
    ang_max = float(_kwargs.get("ang_max", math.radians(25.0)))
    phi_rate_max = float(_kwargs.get("phi_rate_max", math.radians(250.0)))
    theta_rate_max = float(_kwargs.get("theta_rate_max", math.radians(250.0)))

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
    X, U, t,
    ref_seq, Q, R,
    cyl_cx=None, cyl_cy=None, cyl_r=None, cyl_zmin=None, cyl_zmax=None,
    w_cyl=350.0, cyl_safety_margin=0.25, cyl_alpha=10.0,
    obs_seq=None,
    w_moving=1200.0, moving_r=0.35, moving_safety_margin=0.50, moving_alpha=12.0,
    U_nom=None, Rd=None,
    **_kwargs
):
    ref = ref_seq[t + 1]  # (4,)
    e_pos = X[:, 0:3] - ref[None, 0:3]
    e_psi = _wrap_pi_torch(X[:, 6] - ref[3]).unsqueeze(1)
    e = torch.cat([e_pos, e_psi], dim=1)

    J = torch.sum((e * e) * Q[None, :], dim=1) + torch.sum((U * U) * R[None, :], dim=1)
    if (U_nom is not None) and (Rd is not None):
        du = U - U_nom[t][None, :]
        J = J + torch.sum((du * du) * Rd[None, :], dim=1)

    if cyl_cx is not None and cyl_cx.numel() > 0:
        px = X[:, 0:1]; py = X[:, 1:2]; pz = X[:, 2:3]
        dx = px - cyl_cx[None, :]
        dy = py - cyl_cy[None, :]
        d_xy = torch.sqrt(dx * dx + dy * dy)
        signed = d_xy - (cyl_r[None, :] + float(cyl_safety_margin))
        inside_z = (pz >= cyl_zmin[None, :]) & (pz <= cyl_zmax[None, :])
        pen = torch.exp(-float(cyl_alpha) * signed)
        pen = torch.where(inside_z, pen, torch.zeros_like(pen))
        J = J + float(w_cyl) * torch.sum(pen, dim=1)

    if obs_seq is not None:
        # single moving obstacle: (T+1,3)
        # multiple moving obstacles: (T+1,K,3)
        if obs_seq.ndim == 2:
            obs = obs_seq[t + 1]  # (3,)
            d = X[:, 0:3] - obs[None, :]
            dist = torch.sqrt(torch.sum(d * d, dim=1))
            signed = dist - (float(moving_r) + float(moving_safety_margin))
            pen = torch.exp(-float(moving_alpha) * signed)
            J = J + float(w_moving) * pen
        elif obs_seq.ndim == 3:
            obs = obs_seq[t + 1]  # (K,3)
            d = X[:, None, 0:3] - obs[None, :, :]  # (M,K,3)
            dist = torch.sqrt(torch.sum(d * d, dim=2))  # (M,K)
            signed = dist - (float(moving_r) + float(moving_safety_margin))
            pen = torch.exp(-float(moving_alpha) * signed)
            J = J + float(w_moving) * torch.sum(pen, dim=1)

    return J


def terminal_cost_quad(X, t_final, ref_seq, Qf, **_kwargs):
    ref = ref_seq[-1]
    e_pos = X[:, 0:3] - ref[None, 0:3]
    e_psi = _wrap_pi_torch(X[:, 6] - ref[3]).unsqueeze(1)
    e = torch.cat([e_pos, e_psi], dim=1)
    return torch.sum((e * e) * Qf[None, :], dim=1)


# -----------------------------
# MPPI wrapper (Torch)
# -----------------------------
@dataclass
class MPPIParams:
    dt: float = 0.02
    horizon_steps: int = 40
    rollouts: int = 500
    iterations: int = 2
    lam: float = 1.0
    sigma: np.ndarray | None = None
    T_min: float = 0.0
    T_max: float = 0.0
    # ang_max: float = math.radians(25.0)
    ang_max: float = math.radians(40.0)
    yawrate_max: float = math.radians(200.0)
    tau_phi: float = 0.14
    tau_theta: float = 0.14
    phi_rate_max: float = math.radians(250.0)
    theta_rate_max: float = math.radians(250.0)

    w_cyl: float = 350.0
    cyl_safety_margin: float = 0.25
    cyl_alpha: float = 10.0

    w_moving: float = 1200.0
    moving_r: float = 0.35
    moving_safety_margin: float = 0.50
    moving_alpha: float = 12.0
    Rd_u: tuple[float, float, float, float] = (0.0, 3.0, 3.0, 0.5)


class TorchMPPIQuadOuter:
    def __init__(self, mass=0.028, g=9.81, params: MPPIParams | None = None, cylinders=None, device=None):
        self.m = float(mass)
        self.g = float(g)
        self.p = params if params is not None else MPPIParams()

        hover = self.m * self.g
        if self.p.T_max <= 0.0:
            self.p.T_max = 2.0 * hover
        if self.p.T_min < 0.0:
            self.p.T_min = 0.0

        if self.p.sigma is None:
            self.p.sigma = np.array([0.15*hover, math.radians(8.0), math.radians(8.0), math.radians(40.0)], dtype=np.float32)
            

        self.u_min = np.array([self.p.T_min, -self.p.ang_max, -self.p.ang_max, -self.p.yawrate_max], dtype=np.float32)
        self.u_max = np.array([self.p.T_max,  self.p.ang_max,  self.p.ang_max,  self.p.yawrate_max], dtype=np.float32)

        sigma_np = np.asarray(self.p.sigma, dtype=np.float32).reshape(4,)
        self.R_np = 1.0 / np.maximum(sigma_np ** 2, 1e-12).astype(np.float32)

        self.Rd_np = np.asarray(self.p.Rd_u, dtype=np.float32).reshape(4,)

        self.dt = float(self.p.dt)
        self.T = int(self.p.horizon_steps)
        self.M = int(self.p.rollouts)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.cyl = cylinders if cylinders is not None else []
        if len(self.cyl) > 0:
            cx = torch.tensor([c["cx"] for c in self.cyl], device=self.device, dtype=torch.float32)
            cy = torch.tensor([c["cy"] for c in self.cyl], device=self.device, dtype=torch.float32)
            r  = torch.tensor([c["r"]  for c in self.cyl], device=self.device, dtype=torch.float32)
            zmin = torch.tensor([c.get("zmin", -1e9) for c in self.cyl], device=self.device, dtype=torch.float32)
            zmax = torch.tensor([c.get("zmax",  1e9) for c in self.cyl], device=self.device, dtype=torch.float32)
            self.cyl_t = (cx, cy, r, zmin, zmax)
        else:
            self.cyl_t = None

        self.mppi = MPPI(
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
                ref_seq=None, Q=None, Qf=None, R=None, obs_seq=None,
                cyl_cx=None, cyl_cy=None, cyl_r=None, cyl_zmin=None, cyl_zmax=None,
                w_cyl=self.p.w_cyl, cyl_safety_margin=self.p.cyl_safety_margin, cyl_alpha=self.p.cyl_alpha,
                w_moving=self.p.w_moving, moving_r=self.p.moving_r,
                moving_safety_margin=self.p.moving_safety_margin, moving_alpha=self.p.moving_alpha,
                U_nom=None, Rd=None,
            ),
            verbose=False,
        )

        # warm-start: hover thrust
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
        return_predictions=False,
        n_pred=20,
        include_nominal_prediction=True,
    ):
        ref_seq_t = torch.as_tensor(ref_seq_np, device=self.device, dtype=torch.float32)
        Q_t  = torch.as_tensor(Q_np,  device=self.device, dtype=torch.float32)
        Qf_t = torch.as_tensor(Qf_np, device=self.device, dtype=torch.float32)
        R_t  = torch.as_tensor(self.R_np, device=self.device, dtype=torch.float32)
        obs_seq_t = torch.as_tensor(obs_seq_np, device=self.device, dtype=torch.float32) if obs_seq_np is not None else None

        ck = self.mppi.cost_kwargs
        ck["ref_seq"] = ref_seq_t
        ck["Q"] = Q_t
        ck["Qf"] = Qf_t
        ck["R"] = R_t
        ck["obs_seq"] = obs_seq_t
        ck["U_nom"] = torch.as_tensor(self.mppi.U_cpu, device=self.device, dtype=torch.float32)
        ck["Rd"] = torch.as_tensor(self.Rd_np, device=self.device, dtype=torch.float32)

        if self.cyl_t is not None:
            cx, cy, r, zmin, zmax = self.cyl_t
            ck["cyl_cx"] = cx; ck["cyl_cy"] = cy; ck["cyl_r"] = r; ck["cyl_zmin"] = zmin; ck["cyl_zmax"] = zmax
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

        # warm-start shift
        self.mppi.U_cpu[:-1] = self.mppi.U_cpu[1:]
        self.mppi.U_cpu[-1] = self.mppi.U_cpu[-2]
        self.mppi.U = self.mppi.U_cpu
        if return_predictions:
            return u0, pred_samples_xyz, pred_nominal_xyz
        return u0


# -----------------------------
# Buffered Simulation / Data Export
# -----------------------------
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

    # waypoints_2 = np.array([
    #     [ 2.5,  2.0, 0.0],
    #     [ 0.0,  3.5, 2.0],
    #     [-3.0,  1.5, 4.5],
    #     [-2.0, -2.5, 3.0],
    #     [ 2.0, -3.0, 1.0],
    #     [ 3.0,  0.0, 0.5],
    #     [ 2.5,  2.0, 0.0],
    # ], dtype=float)

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

    moving_waypoint_sets = [waypoints_1, waypoints_2]

    moving_trajs = [
        build_min_snap_3d(moving_waypoint_sets[i], avg_speed=1.8)
        for i in range(2)
    ]
    params = MPPIParams(
        dt=0.028677838561381404,
        horizon_steps=35,
        rollouts=1096,
        iterations=1,
        lam=0.9447651256582109,
        sigma=np.array(
            [
                0.03860791271607059,
                math.radians(6.072834867326699),
                math.radians(7.139534744261536),
                math.radians(9.50181217517332),
            ],
            dtype=np.float32,
        ),
        ang_max=math.radians(28.533048677493525),
        yawrate_max=math.radians(125.002671219858),
        tau_phi=0.22445102088241203,
        tau_theta=0.19182828711184713,
        phi_rate_max=math.radians(186.91775798517736),
        theta_rate_max=math.radians(250.74834372828798),
        w_cyl=510.6988597095838,
        cyl_safety_margin=0.2150292356967072,
        cyl_alpha=8.72155294537059,
        w_moving=376.5800114691624,
        moving_r=0.3045201563139591,
        moving_safety_margin=0.4224283788684363,
        moving_alpha=15.033309531167399,
        Rd_u=(0.9046852667672367, 2.390585391824126, 2.8043325852902052, 1.0686213306251247),
    )

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctrl = TorchMPPIQuadOuter(mass=0.028, g=9.81, params=params, cylinders=cylinders, device=device)
    print("Planner backend:", device)

    Q = np.array([121.54191793531642, 120.7320008579965, 143.93311001418877, 0.0], dtype=np.float32)
    Qf = np.array([
        4.24444323202221 * 121.54191793531642,
        2.2842848307817354 * 120.7320008579965,
        3.7565634665191308 * 143.93311001418877,
        0.0,
    ], dtype=np.float32)
    dt = params.dt
    sim_T = max([traj.total_time] + [mt.total_time for mt in moving_trajs])
    steps = int(sim_T / dt) + 1

    # initial state
    p0, _ = traj.eval(0.0)
    x = np.zeros(9, dtype=float)
    x[0:3] = p0
    x[3:6] = 0.0
    x[6] = 0.0
    x[7] = 0.0
    x[8] = 0.0

    last_yaw_ref = float(x[6])

    def yaw_follow_from_vel(v, fallback):
        vx, vy = float(v[0]), float(v[1])
        if vx*vx + vy*vy < 1e-6:
            return fallback
        return math.atan2(vy, vx)

    lead_time = 1.5495997771078638

    # precompute curves for drawing
    tt = np.linspace(0.0, traj.total_time, 600)
    ref_curve = np.zeros((len(tt), 3))
    for j, tj in enumerate(tt):
        p, _ = traj.eval(float(tj))
        ref_curve[j] = p

    obs_curves = []
    for moving_traj in moving_trajs:
        tt2 = np.linspace(0.0, moving_traj.total_time, 600)
        obs_curve = np.zeros((len(tt2), 3))
        for j, tj in enumerate(tt2):
            p, _ = moving_traj.eval(float(tj))
            obs_curve[j] = p
        obs_curves.append(obs_curve)

    all_pts = np.vstack([ref_curve] + obs_curves)
    mins = all_pts.min(axis=0) - 1.0
    maxs = all_pts.max(axis=0) + 1.0
    X_path = np.zeros((steps, 3), dtype=float)
    X_hist = np.zeros((steps, 9), dtype=float)
    n_obs = len(moving_trajs)
    obs_path = np.zeros((steps, n_obs, 3), dtype=float)
    n_pred_plot = min(8, ctrl.M)
    pred_stride = 1
    pred_samples_xyz = np.full((steps, ctrl.T + 1, n_pred_plot, 3), np.nan, dtype=np.float32)
    pred_nominal_xyz = np.full((steps, ctrl.T + 1, 3), np.nan, dtype=np.float32)
    U_applied = np.zeros((steps, 4), dtype=float)
    solve_ms = np.zeros((steps,), dtype=float)
    sim_time = np.arange(steps, dtype=float) * dt

    for i in range(steps):
        t = i * dt

        ref_seq = np.zeros((ctrl.T + 1, 4), dtype=float)
        ref_seq[0, 0:3], v0 = traj.eval(min(t, traj.total_time))
        psi0_raw = wrap_pi(yaw_follow_from_vel(v0, last_yaw_ref))
        ref_seq[0, 3] = last_yaw_ref + wrap_pi(psi0_raw - last_yaw_ref)

        for k in range(1, ctrl.T + 1):
            tk = min(traj.total_time, t + k * dt)
            pk, vk = traj.eval(tk)
            psi_raw = wrap_pi(yaw_follow_from_vel(vk, ref_seq[k - 1, 3]))
            prev = ref_seq[k - 1, 3]
            ref_seq[k, 3] = prev + wrap_pi(psi_raw - prev)
            ref_seq[k, 0:3] = pk

        last_yaw_ref = float(ref_seq[1, 3])

        obs_seq = np.zeros((ctrl.T + 1, n_obs, 3), dtype=float)
        for j, moving_traj in enumerate(moving_trajs):
            for k in range(ctrl.T + 1):
                tk = min(moving_traj.total_time, t + lead_time + k * dt)
                op, _ = moving_traj.eval(tk)
                obs_seq[k, j] = op

        obs_path[i] = obs_seq[0]

        if ctrl.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # LAG HOTSPOT:
        # Enabling prediction collection triggers sample extraction + GPU->CPU transfer.
        # Lower frequency (increase pred_stride) to reduce solve-time overhead.
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
        solve_ms[i] = (time.perf_counter() - t0) * 1000.0
        # LAG HOTSPOT:
        # Storing full prediction buffers each step increases memory traffic.
        if collect_pred and pred_s_xyz is not None:
            ns = min(n_pred_plot, int(pred_s_xyz.shape[1]))
            pred_samples_xyz[i, :, :ns, :] = pred_s_xyz[:, :ns, :]
        elif i > 0:
            pred_samples_xyz[i] = pred_samples_xyz[i - 1]
        if collect_pred and pred_nom_xyz is not None:
            pred_nominal_xyz[i] = pred_nom_xyz
        elif i > 0:
            pred_nominal_xyz[i] = pred_nominal_xyz[i - 1]

        if (i % 10) == 0:
            print(f"[step {i:04d}] solve_ms={solve_ms[i]:.2f}")

        Tcmd, phi_cmd, theta_cmd, yawrate = map(float, u)
        Tcmd = clamp(Tcmd, ctrl.p.T_min, ctrl.p.T_max)
        phi_cmd = clamp(phi_cmd, -ctrl.p.ang_max, ctrl.p.ang_max)
        theta_cmd = clamp(theta_cmd, -ctrl.p.ang_max, ctrl.p.ang_max)
        yawrate = clamp(yawrate, -ctrl.p.yawrate_max, ctrl.p.yawrate_max)
        U_applied[i] = [Tcmd, phi_cmd, theta_cmd, yawrate]

        phi = float(x[7])
        theta = float(x[8])
        cphi = math.cos(phi); sphi = math.sin(phi)
        cth = math.cos(theta); sth = math.sin(theta)
        cpsi = math.cos(x[6]); spsi = math.sin(x[6])
        zb = np.array([
            cpsi*sth*cphi + spsi*sphi,
            spsi*sth*cphi - cpsi*sphi,
            cth*cphi,
        ], dtype=float)
        a = (Tcmd / ctrl.m) * zb - np.array([0.0, 0.0, ctrl.g])
        phi_dot = (phi_cmd - phi) / max(1e-6, ctrl.p.tau_phi)
        theta_dot = (theta_cmd - theta) / max(1e-6, ctrl.p.tau_theta)
        phi_dot = clamp(phi_dot, -ctrl.p.phi_rate_max, ctrl.p.phi_rate_max)
        theta_dot = clamp(theta_dot, -ctrl.p.theta_rate_max, ctrl.p.theta_rate_max)

        x[3:6] = x[3:6] + dt * a
        x[0:3] = x[0:3] + dt * x[3:6]
        x[6] = wrap_pi(x[6] + dt * yawrate)
        x[7] = clamp(x[7] + dt * phi_dot, -ctrl.p.ang_max, ctrl.p.ang_max)
        x[8] = clamp(x[8] + dt * theta_dot, -ctrl.p.ang_max, ctrl.p.ang_max)

        X_path[i] = x[0:3]
        X_hist[i] = x.copy()

    run_data = {
        "method": "mppi",
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
        out_path = os.path.join(save_dir, "mppi_simulation.npz")
        np.savez_compressed(out_path, **run_data)
        print(f"Saved simulation data to {out_path}")
    return run_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MPPI Crazyflie simulation.")
    parser.add_argument("--save", action="store_true", help="Save simulation .npz to the plot directory.")
    args = parser.parse_args()
    save_dir = os.path.join(os.path.dirname(__file__), "plot") if args.save else None
    simulate(save_dir=save_dir)
