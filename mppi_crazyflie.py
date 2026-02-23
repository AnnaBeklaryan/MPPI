#!/usr/bin/env python3
"""
ONLINE (real-time) Torch-MPPI Crazyflie-like quadrotor outer-loop simulation with:
- minimum-snap drone reference trajectory
- static cylinder obstacles
- ONE moving obstacle (sphere) predicted time-ahead

Planner:
- Uses MPPI from mppi_class.py (Torch)

Control: u = [T, phi_cmd, theta_cmd, psi_dot_cmd]
State:   x = [px,py,pz, vx,vy,vz, psi]

This version:
- Simulates ONLINE inside the animation callback (no "simulate then plot")
- Prints solve time online
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import torch
from mppi_class import MPPI  # your Torch MPPI


# -----------------------------
# Helpers
# -----------------------------
def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rotation_matrix_zyx(phi, theta, psi):
    cphi = math.cos(phi); sphi = math.sin(phi)
    cth  = math.cos(theta); sth = math.sin(theta)
    cpsi = math.cos(psi); spsi = math.sin(psi)

    Rz = np.array([[cpsi, -spsi, 0.0],
                   [spsi,  cpsi, 0.0],
                   [0.0,   0.0,  1.0]])
    Ry = np.array([[ cth, 0.0, sth],
                   [0.0, 1.0, 0.0],
                   [-sth, 0.0, cth]])
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cphi, -sphi],
                   [0.0, sphi,  cphi]])
    return Rz @ Ry @ Rx


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


# -----------------------------
# Plot cylinders
# -----------------------------
def plot_cylinders(ax, cylinders, n_theta=40, n_z=8):
    for c in cylinders:
        cx, cy, r = c["cx"], c["cy"], c["r"]
        zmin = c.get("zmin", 0.0)
        zmax = c.get("zmax", 1.0)

        theta = np.linspace(0, 2*np.pi, n_theta)
        z = np.linspace(zmin, zmax, n_z)
        Theta, Z = np.meshgrid(theta, z)

        X = cx + r * np.cos(Theta)
        Y = cy + r * np.sin(Theta)

        ax.plot_wireframe(X, Y, Z, linewidth=0.6, alpha=0.5)


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
    p = X[:, 0:3]
    v = X[:, 3:6]
    psi = X[:, 6]

    Tcmd = U[:, 0]
    phi = U[:, 1]
    theta = U[:, 2]
    psidot = U[:, 3]

    zb = _z_body_world_torch(phi, theta, psi)
    gvec = torch.tensor([0.0, 0.0, float(g)], device=X.device, dtype=X.dtype)
    a = (Tcmd[:, None] / float(m)) * zb - gvec[None, :]

    v_next = v + float(dt) * a
    p_next = p + float(dt) * v_next
    psi_next = _wrap_pi_torch(psi + float(dt) * psidot)
    return torch.cat([p_next, v_next, psi_next[:, None]], dim=1)


def running_cost_quad(
    X, U, t,
    ref_seq, Q, R,
    cyl_cx=None, cyl_cy=None, cyl_r=None, cyl_zmin=None, cyl_zmax=None,
    w_cyl=350.0, cyl_safety_margin=0.25, cyl_alpha=10.0,
    obs_seq=None,
    w_moving=1200.0, moving_r=0.35, moving_safety_margin=0.50, moving_alpha=12.0,
    **_kwargs
):
    ref = ref_seq[t + 1]  # (4,)
    e_pos = X[:, 0:3] - ref[None, 0:3]
    e_psi = _wrap_pi_torch(X[:, 6] - ref[3]).unsqueeze(1)
    e = torch.cat([e_pos, e_psi], dim=1)

    J = torch.sum((e * e) * Q[None, :], dim=1) + torch.sum((U * U) * R[None, :], dim=1)

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
        obs = obs_seq[t + 1]  # (3,)
        d = X[:, 0:3] - obs[None, :]
        dist = torch.sqrt(torch.sum(d * d, dim=1))
        signed = dist - (float(moving_r) + float(moving_safety_margin))
        pen = torch.exp(-float(moving_alpha) * signed)
        J = J + float(w_moving) * pen

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
    horizon_steps: int = 60
    rollouts: int = 2048
    iterations: int = 2
    lam: float = 1.0
    sigma: np.ndarray | None = None
    T_min: float = 0.0
    T_max: float = 0.0
    #ang_max: float = math.radians(25.0)
    ang_max = math.radians(40.0)
    yawrate_max: float = math.radians(200.0)

    w_cyl: float = 350.0
    cyl_safety_margin: float = 0.25
    cyl_alpha: float = 10.0

    w_moving: float = 1200.0
    moving_r: float = 0.35
    moving_safety_margin: float = 0.50
    moving_alpha: float = 12.0


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

        self.R_np = np.array(
            [1.0/(hover**2),
             1.0/(self.p.ang_max**2),
             1.0/(self.p.ang_max**2),
             1.0/(self.p.yawrate_max**2)],
            dtype=np.float32
        )

        self.R_np *= 0.2   

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
            dyn_kwargs=dict(m=self.m, g=self.g),
            cost_kwargs=dict(
                ref_seq=None, Q=None, Qf=None, R=None, obs_seq=None,
                cyl_cx=None, cyl_cy=None, cyl_r=None, cyl_zmin=None, cyl_zmax=None,
                w_cyl=self.p.w_cyl, cyl_safety_margin=self.p.cyl_safety_margin, cyl_alpha=self.p.cyl_alpha,
                w_moving=self.p.w_moving, moving_r=self.p.moving_r,
                moving_safety_margin=self.p.moving_safety_margin, moving_alpha=self.p.moving_alpha,
            ),
            verbose=False,
        )

        # warm-start: hover thrust
        self.mppi.U_cpu[:] = 0.0
        self.mppi.U_cpu[:, 0] = hover
        self.mppi.U = self.mppi.U_cpu

    def plan(self, x0_np, ref_seq_np, Q_np, Qf_np, obs_seq_np):
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

        if self.cyl_t is not None:
            cx, cy, r, zmin, zmax = self.cyl_t
            ck["cyl_cx"] = cx; ck["cyl_cy"] = cy; ck["cyl_r"] = r; ck["cyl_zmin"] = zmin; ck["cyl_zmax"] = zmax
        else:
            ck["cyl_cx"] = ck["cyl_cy"] = ck["cyl_r"] = ck["cyl_zmin"] = ck["cyl_zmax"] = None

        U_cpu, _ = self.mppi.plan(x0_np, return_samples=False)
        u0 = U_cpu[0].copy()

        # warm-start shift
        self.mppi.U_cpu[:-1] = self.mppi.U_cpu[1:]
        self.mppi.U_cpu[-1] = self.mppi.U_cpu[-2]
        self.mppi.U = self.mppi.U_cpu
        return u0


# -----------------------------
# ONLINE Simulation / Visualization
# -----------------------------
def simulate_online():
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

    traj = build_min_snap_3d(waypoints, avg_speed=1.8)

    cylinders = [
        {"cx":  0.5, "cy":  1.0, "r": 0.6, "zmin": 0.0, "zmax": 4.5},
        {"cx": -1.8, "cy": -0.5, "r": 0.7, "zmin": 0.0, "zmax": 5.0},
        {"cx":  1.5, "cy": -1.8, "r": 0.5, "zmin": 0.0, "zmax": 3.0},
        {"cx": -1.0, "cy":  4.0, "r": 0.7, "zmin": 0.0, "zmax": 4.5},
    ]

    offset = np.array([0.40, -0.30, 0.00], dtype=float)
    moving_traj = build_min_snap_3d(waypoints_1 + offset, avg_speed=1.8)

    params = MPPIParams(
        dt=0.02,
        horizon_steps=60,
        rollouts=2048,
        iterations=2,
        lam=1.0,
        w_cyl=350.0,
        cyl_safety_margin=0.25,
        cyl_alpha=10.0,
        w_moving=1200.0,
        moving_r=0.35,
        moving_safety_margin=0.50,
        moving_alpha=12.0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctrl = TorchMPPIQuadOuter(mass=0.028, g=9.81, params=params, cylinders=cylinders, device=device)
    print("Planner backend:", device)

    Q  = np.array([30.0, 30.0, 40.0, 0.0], dtype=np.float32)
    Qf = np.array([120.0,120.0,160.0,0.0], dtype=np.float32)


    Q  = np.array([80.0, 80.0, 120.0, 0.0], dtype=np.float32)
    Qf = np.array([200.0, 200.0, 300.0, 0.0], dtype=np.float32)
    dt = params.dt
    sim_T = max(traj.total_time, moving_traj.total_time)
    steps = int(sim_T / dt) + 1

    # initial state
    p0, _ = traj.eval(0.0)
    x = np.zeros(7, dtype=float)
    x[0:3] = p0
    x[3:6] = 0.0
    x[6] = 0.0

    # logs that grow online (keep last N for speed)
    max_tail = 3000
    X_tail = []  # list of (3,)
    obs_tail = []
    U_tail = []
    solve_tail = []

    last_yaw_ref = float(x[6])

    def yaw_follow_from_vel(v, fallback):
        vx, vy = float(v[0]), float(v[1])
        if vx*vx + vy*vy < 1e-6:
            return fallback
        return math.atan2(vy, vx)

    lead_time = 1.0

    # ---------- plot setup ----------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("ONLINE Torch MPPI — Static Cylinders + Moving Obstacle")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Altitude")

    # reference curve (static)
    tt = np.linspace(0.0, traj.total_time, 600)
    ref_curve = np.zeros((len(tt), 3))
    for j, tj in enumerate(tt):
        p, _ = traj.eval(float(tj))
        ref_curve[j] = p
    ax.plot(ref_curve[:, 0], ref_curve[:, 1], ref_curve[:, 2], linestyle=":", linewidth=2)

    # moving obstacle curve (static)
    tt2 = np.linspace(0.0, moving_traj.total_time, 600)
    obs_curve = np.zeros((len(tt2), 3))
    for j, tj in enumerate(tt2):
        p, _ = moving_traj.eval(float(tj))
        obs_curve[j] = p
    ax.plot(obs_curve[:, 0], obs_curve[:, 1], obs_curve[:, 2], linestyle="--", linewidth=1)

    plot_cylinders(ax, cylinders)

    traj_line, = ax.plot([], [], [], linewidth=2)
    obs_pt, = ax.plot([], [], [], marker="o", markersize=7, linestyle="")

    arm_len = 0.35
    arm1, = ax.plot([], [], [], linewidth=2)
    arm2, = ax.plot([], [], [], linewidth=2)
    center_pt, = ax.plot([], [], [], marker="o", markersize=4, linestyle="")
    txt = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)

    all_pts = np.vstack([ref_curve, obs_curve])
    mins = all_pts.min(axis=0) - 1.0
    maxs = all_pts.max(axis=0) + 1.0
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    def set_segment(line, p_center, direction, half_len):
        p0 = p_center - half_len * direction
        p1 = p_center + half_len * direction
        line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        line.set_3d_properties([p0[2], p1[2]])

    # ---------- online step ----------
    def step_once(frame_idx: int):
        nonlocal x, last_yaw_ref

        if frame_idx >= steps:
            return traj_line, obs_pt, arm1, arm2, center_pt, txt

        t = frame_idx * dt

        # Build ref sequence (T+1,4)
        ref_seq = np.zeros((ctrl.T + 1, 4), dtype=float)
        ref_seq[0, 0:3], v0 = traj.eval(min(t, traj.total_time))
        psi0_raw = wrap_pi(yaw_follow_from_vel(v0, last_yaw_ref))
        ref_seq[0, 3] = last_yaw_ref + wrap_pi(psi0_raw - last_yaw_ref)

        for k in range(1, ctrl.T + 1):
            tk = min(traj.total_time, t + k * dt)
            pk, vk = traj.eval(tk)
            psi_raw = wrap_pi(yaw_follow_from_vel(vk, ref_seq[k-1, 3]))
            prev = ref_seq[k-1, 3]
            ref_seq[k, 3] = prev + wrap_pi(psi_raw - prev)
            ref_seq[k, 0:3] = pk

        last_yaw_ref = float(ref_seq[1, 3])

        # Moving obstacle predicted seq (T+1,3)
        obs_seq = np.zeros((ctrl.T + 1, 3), dtype=float)
        for k in range(ctrl.T + 1):
            tk = min(moving_traj.total_time, t + lead_time + k * dt)
            op, _ = moving_traj.eval(tk)
            obs_seq[k] = op

        # MPPI solve (timed)
        if ctrl.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        u = ctrl.plan(x, ref_seq, Q, Qf, obs_seq)
        if ctrl.device.type == "cuda":
            torch.cuda.synchronize()
        solve_s = time.perf_counter() - t0

        # Print solve time ONLINE (you can throttle if you want)
        print(f"[step {frame_idx:04d}] t={t:6.2f}s | solve={solve_s:7.2f} ms")

        # Apply one sim step (numpy)
        Tcmd, phi_cmd, theta_cmd, yawrate = u
        Tcmd = clamp(float(Tcmd), ctrl.p.T_min, ctrl.p.T_max)
        phi_cmd = clamp(float(phi_cmd), -ctrl.p.ang_max, ctrl.p.ang_max)
        theta_cmd = clamp(float(theta_cmd), -ctrl.p.ang_max, ctrl.p.ang_max)
        yawrate = clamp(float(yawrate), -ctrl.p.yawrate_max, ctrl.p.yawrate_max)

        cphi = math.cos(phi_cmd); sphi = math.sin(phi_cmd)
        cth  = math.cos(theta_cmd); sth = math.sin(theta_cmd)
        cpsi = math.cos(x[6]); spsi = math.sin(x[6])
        zb = np.array([cpsi*sth*cphi + spsi*sphi,
                       spsi*sth*cphi - cpsi*sphi,
                       cth*cphi], dtype=float)
        a = (Tcmd / ctrl.m) * zb - np.array([0.0, 0.0, ctrl.g])

        x[3:6] = x[3:6] + dt * a
        x[0:3] = x[0:3] + dt * x[3:6]
        x[6] = wrap_pi(x[6] + dt * yawrate)

        # tail logs
        X_tail.append(x[0:3].copy())
        obs_tail.append(obs_seq[0].copy())
        U_tail.append(np.array([Tcmd, phi_cmd, theta_cmd, yawrate], dtype=float))
        solve_tail.append(float(solve_s))

        if len(X_tail) > max_tail:
            X_tail.pop(0); obs_tail.pop(0); U_tail.pop(0); solve_tail.pop(0)

        # update plot
        P = np.asarray(X_tail)
        traj_line.set_data(P[:, 0], P[:, 1])
        traj_line.set_3d_properties(P[:, 2])

        op = obs_tail[-1]
        obs_pt.set_data([op[0]], [op[1]])
        obs_pt.set_3d_properties([op[2]])

        phi = float(U_tail[-1][1])
        theta = float(U_tail[-1][2])
        psi = float(x[6])
        p = x[0:3]

        Rm = rotation_matrix_zyx(phi, theta, psi)
        xb = Rm[:, 0]; yb = Rm[:, 1]

        d1 = xb + yb
        d2 = xb - yb
        d1 = d1 / (np.linalg.norm(d1) + 1e-12)
        d2 = d2 / (np.linalg.norm(d2) + 1e-12)

        set_segment(arm1, p, d1, arm_len)
        set_segment(arm2, p, d2, arm_len)

        center_pt.set_data([p[0]], [p[1]])
        center_pt.set_3d_properties([p[2]])

        txt.set_text(f"t={t:.2f}s | solve={solve_s:.3f} s")
        return traj_line, obs_pt, arm1, arm2, center_pt, txt

    ani = FuncAnimation(
        fig, step_once,
        frames=steps,
        interval=int(1000 * dt),   # real-time pacing attempt
        blit=False,
        repeat=False,
    )
    plt.show()


if __name__ == "__main__":
    simulate_online()