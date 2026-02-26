#!/usr/bin/env python3
"""
Buffered DR-MPPI Crazyflie outer-loop simulation (Torch) using DR_MPPI from mppi_class.py.

- Minimum-snap reference trajectory (3D)
- Static cylinder obstacles (soft exponential cost, z-gated)
- One moving spherical obstacle (mean prediction + uncertainty)
- DR-MPPI feasibility filter via CVaR on XY distance to moving obstacle:
    g = R_safe_xy - ||(x,y) - (ox,oy)||

Buffered execution:
- Run full simulation first: build ref/obs sequences -> DR_MPPI.plan() -> apply u0
- Store state/obstacle/solve-time history in buffers
- Plot only after simulation finishes

Requires:
- torch
- matplotlib with 3D enabled (fix your matplotlib first)
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mppi_class import DR_MPPI  # uses CVaR feasibility + DR correction


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
        obs = obs_mean_3d[t + 1]  # (3,)
        d = X[:, 0:3] - obs[None, :]
        dist = torch.sqrt(torch.sum(d * d, dim=1))
        signed = dist - (float(moving_r) + float(moving_safety_margin))
        pen = torch.exp(-float(moving_alpha) * signed)
        J = J + float(w_moving) * pen

    return J


def terminal_cost_quad(X: torch.Tensor, t_final: int, ref_seq: torch.Tensor, Qf: torch.Tensor, **_kw) -> torch.Tensor:
    ref = ref_seq[-1]
    e_pos = X[:, 0:3] - ref[None, 0:3]
    e_psi = _wrap_pi_torch(X[:, 6] - ref[3]).unsqueeze(1)
    e = torch.cat([e_pos, e_psi], dim=1)
    return torch.sum((e * e) * Qf[None, :], dim=1)


# ============================================================
# DR-MPPI wrapper using mppi_class.DR_MPPI
# ============================================================
@dataclass
class DRParams:
    dt: float = 0.02
    horizon_steps: int = 60
    rollouts: int = 1096
    iterations: int = 2
    lam: float = 1.0

    # bounds
    ang_max: float = math.radians(25.0)
    yawrate_max: float = math.radians(200.0)
    tau_phi: float = 0.14
    tau_theta: float = 0.14
    phi_rate_max: float = math.radians(250.0)
    theta_rate_max: float = math.radians(250.0)

    # control noise std (diag)
    # (T thrust, phi, theta, yawrate)
    # if None -> auto set from hover
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

    # DR feasibility (XY CVaR)
    cvar_alpha: float = 0.85
    cvar_N: int = 64
    obs_pos_sigma_xy: tuple[float, float] = (0.25, 0.25)
    noise_mode: str = "static"
    dr_eps_cvar: float = 0.0

    # drone radius included in safe radius for feasibility
    drone_radius: float = 0.25
    # control-smoothing weight around nominal sequence
    Rd_u: tuple[float, float, float, float] = (0.0, 3.0, 3.0, 0.5)


class TorchDRMPPIQuadOuter:
    def __init__(self, mass=0.028, g=9.81, params: DRParams | None = None, cylinders=None, device=None):
        self.m = float(mass)
        self.g = float(g)
        self.p = params if params is not None else DRParams()

        hover = self.m * self.g
        T_min = 0.0
        T_max = 2.0 * hover

        if self.p.sigma is None:
            self.p.sigma = np.array(
                [0.06*hover, math.radians(1.3), math.radians(1.3), math.radians(8.0)],
                dtype=np.float32
            )

        self.dt = float(self.p.dt)
        self.T = int(self.p.horizon_steps)
        self.M = int(self.p.rollouts)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # bounds
        self.u_min = np.array([T_min, -self.p.ang_max, -self.p.ang_max, -self.p.yawrate_max], dtype=np.float32)
        self.u_max = np.array([T_max,  self.p.ang_max,  self.p.ang_max,  self.p.yawrate_max], dtype=np.float32)

        # control effort weights (normalize)
        self.R_np = np.array(
            [1.0/(hover**2),
             1.0/(self.p.ang_max**2),
             1.0/(self.p.ang_max**2),
             1.0/(self.p.yawrate_max**2)],
            dtype=np.float32
        )
        self.Rd_np = np.asarray(self.p.Rd_u, dtype=np.float32).reshape(4,)

        # cylinders -> torch
        cyl = cylinders if cylinders is not None else []
        if len(cyl) > 0:
            self.cyl_cx = torch.tensor([c["cx"] for c in cyl], device=self.device, dtype=torch.float32)
            self.cyl_cy = torch.tensor([c["cy"] for c in cyl], device=self.device, dtype=torch.float32)
            self.cyl_r  = torch.tensor([c["r"]  for c in cyl], device=self.device, dtype=torch.float32)
            self.cyl_zmin = torch.tensor([c.get("zmin", -1e9) for c in cyl], device=self.device, dtype=torch.float32)
            self.cyl_zmax = torch.tensor([c.get("zmax",  1e9) for c in cyl], device=self.device, dtype=torch.float32)
        else:
            self.cyl_cx = self.cyl_cy = self.cyl_r = self.cyl_zmin = self.cyl_zmax = None

        # DR_MPPI instance
        self.mppi = DR_MPPI(
            dt=self.dt,
            T=self.T,
            M=self.M,
            lam=float(self.p.lam),
            noise_sigma=np.asarray(self.p.sigma, dtype=np.float32),   # control noise
            u_min=self.u_min,
            u_max=self.u_max,
            dynamics=quad_dyn_step,
            running_cost=running_cost_quad,
            terminal_cost=terminal_cost_quad,
            I=int(self.p.iterations),
            device=self.device,
            dtype=torch.float32,
            verbose=False,
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
                # updated each call:
                ref_seq=None, Q=None, Qf=None, R=None,
                obs_mean_3d=None,

                # cylinders:
                cyl_cx=None, cyl_cy=None, cyl_r=None, cyl_zmin=None, cyl_zmax=None,
                w_cyl=float(self.p.w_cyl),
                cyl_safety_margin=float(self.p.cyl_safety_margin),
                cyl_alpha=float(self.p.cyl_alpha),

                # moving mean soft cost:
                w_moving=float(self.p.w_moving),
                moving_r=float(self.p.moving_r),
                moving_safety_margin=float(self.p.moving_safety_margin),
                moving_alpha=float(self.p.moving_alpha),
                U_nom=None,
                Rd=None,

                # DR feasibility inputs used internally by DR_MPPI:
                O_mean=None,   # (T,K,2)
                radii=None,    # (K,)
            ),
            cvar_alpha=float(self.p.cvar_alpha),
            cvar_N=int(self.p.cvar_N),
            obs_pos_sigma=np.asarray(self.p.obs_pos_sigma_xy, dtype=np.float32),
            obs_noise_mode=str(self.p.noise_mode),
            dr_eps_cvar=float(self.p.dr_eps_cvar),
        )

        # warm-start
        self.mppi.U_cpu[:] = 0.0
        self.mppi.U_cpu[:, 0] = hover
        self.mppi.U = self.mppi.U_cpu

    def plan(self, x0_np, ref_seq_np, Q_np, Qf_np, obs_seq_np):
        # tensors
        ref_seq_t = torch.as_tensor(ref_seq_np, device=self.device, dtype=torch.float32)  # (T+1,4)
        Q_t  = torch.as_tensor(Q_np,  device=self.device, dtype=torch.float32)
        Qf_t = torch.as_tensor(Qf_np, device=self.device, dtype=torch.float32)
        R_t  = torch.as_tensor(self.R_np, device=self.device, dtype=torch.float32)

        obs_mean_3d_t = torch.as_tensor(obs_seq_np, device=self.device, dtype=torch.float32)  # (T+1,3)

        # DR feasibility uses XY only:
        # O_mean: (T,K,2) with T steps aligned to X_hist (t=0..T-1) in DR_MPPI
        # We use obs_seq[1:T+1] as mean at future steps
        O_mean = obs_mean_3d_t[1:self.T+1, :2].unsqueeze(1)  # (T,1,2)
        safe_R = float(self.p.moving_r + self.p.moving_safety_margin + self.p.drone_radius)
        radii = torch.tensor([safe_R], device=self.device, dtype=torch.float32)  # (1,)

        ck = self.mppi.cost_kwargs
        ck["ref_seq"] = ref_seq_t
        ck["Q"] = Q_t
        ck["Qf"] = Qf_t
        ck["R"] = R_t
        ck["obs_mean_3d"] = obs_mean_3d_t
        ck["U_nom"] = torch.as_tensor(self.mppi.U_cpu, device=self.device, dtype=torch.float32)
        ck["Rd"] = torch.as_tensor(self.Rd_np, device=self.device, dtype=torch.float32)

        ck["O_mean"] = O_mean
        ck["radii"] = radii

        if self.cyl_cx is not None:
            ck["cyl_cx"] = self.cyl_cx
            ck["cyl_cy"] = self.cyl_cy
            ck["cyl_r"] = self.cyl_r
            ck["cyl_zmin"] = self.cyl_zmin
            ck["cyl_zmax"] = self.cyl_zmax
        else:
            ck["cyl_cx"] = ck["cyl_cy"] = ck["cyl_r"] = ck["cyl_zmin"] = ck["cyl_zmax"] = None

        # solve
        U_cpu, _ = self.mppi.plan(x0_np, return_samples=False)
        u0 = U_cpu[0].copy()

        # shift warm-start
        self.mppi.U_cpu[:-1] = self.mppi.U_cpu[1:]
        self.mppi.U_cpu[-1] = self.mppi.U_cpu[-2]
        self.mppi.U = self.mppi.U_cpu
        return u0


# ============================================================
# Buffered simulation (plot after run)
# ============================================================
def simulate():
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

    # moving obstacle
    offset = np.array([0.40, -0.30, 0.00], dtype=float)
    moving_waypoints = waypoints_1 + offset
    moving_traj = build_min_snap_3d(moving_waypoints, avg_speed=1.8)

    params = DRParams(
        dt=0.02365431268834492,
        horizon_steps=50,
        rollouts=2048,
        iterations=2,
        lam=0.8444146922964824,

        ang_max=math.radians(42.52350986190698),
        yawrate_max=math.radians(173.74212275188287),
        tau_phi=0.19171151474569642,
        tau_theta=0.16905177236157454,
        phi_rate_max=math.radians(302.76732088763026),
        theta_rate_max=math.radians(205.07368428551388),

        sigma=np.array(
            [
                0.043029010625575535,              # sigma_T_ratio * hover
                math.radians(4.954352217068673),
                math.radians(7.702984740619588),
                math.radians(24.490391108169387),
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

        cvar_alpha=0.7790712559865822,
        cvar_N=48,
        obs_pos_sigma_xy=(0.1569012991377936, 0.18206816632027703),
        noise_mode="per_step",
        dr_eps_cvar=0.030026822645753973,

        drone_radius=0.3662153322325755,
        Rd_u=(0.1391209429786332, 0.17400953848855316, 1.3542346730954953, 0.9273697619931647),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctrl = TorchDRMPPIQuadOuter(mass=0.028, g=9.81, params=params, cylinders=cylinders, device=device)
    print("Planner backend:", device)

    # tracking weights (yaw disabled)
    Q = np.array([70.83921979768184, 101.7158401270657, 24.804057074906098, 0.0], dtype=np.float32)
    Qf = np.array([161.80431672271496, 242.31023361428362, 85.67477925065472, 0.0], dtype=np.float32)

    dt = params.dt
    sim_T = max(traj.total_time, moving_traj.total_time)
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

    lead_time = 1.704014105868615  # seconds (autotuned)

    # Precompute curves for reference drawing
    tt = np.linspace(0.0, traj.total_time, 600)
    ref_curve = np.array([traj.eval(float(ti))[0] for ti in tt])

    tt2 = np.linspace(0.0, moving_traj.total_time, 600)
    obs_curve = np.array([moving_traj.eval(float(ti))[0] for ti in tt2])

    # axis limits for post-run plot
    all_pts = np.vstack([ref_curve, obs_curve])
    mins = all_pts.min(axis=0) - 1.0
    maxs = all_pts.max(axis=0) + 1.0

    # logs / buffers
    X_path = np.zeros((steps, 3), dtype=float)
    X_hist = np.zeros((steps, 9), dtype=float)
    obs_path = np.zeros((steps, 3), dtype=float)
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

        # moving obstacle mean prediction seq (T+1,3)
        obs_seq = np.zeros((ctrl.T + 1, 3), dtype=float)
        for k in range(ctrl.T + 1):
            tk = min(moving_traj.total_time, t + lead_time + k * dt)
            op, _ = moving_traj.eval(tk)
            obs_seq[k] = op

        obs_path[i] = obs_seq[0]

        # plan (time it correctly on cuda)
        if ctrl.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        u = ctrl.plan(x, ref_seq, Q, Qf, obs_seq)

        if ctrl.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        solve_ms[i] = ms

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

    # ---- playback animation after simulation has completed ----
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Torch DR-MPPI Playback (buffered run)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Altitude")

    ax.plot(ref_curve[:, 0], ref_curve[:, 1], ref_curve[:, 2], linestyle=":", linewidth=2, label="Reference")
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], s=40, label="Waypoints")
    ax.plot(obs_curve[:, 0], obs_curve[:, 1], obs_curve[:, 2], linestyle="--", linewidth=1, label="Obstacle nominal")

    plot_cylinders(ax, cylinders)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    traj_line, = ax.plot([], [], [], linewidth=2, label="Drone path")
    obs_trace, = ax.plot([], [], [], linewidth=1.5, label="Obstacle used")
    obs_pt, = ax.plot([], [], [], marker="o", markersize=7, linestyle="")
    center_pt, = ax.plot([], [], [], marker="o", markersize=4, linestyle="")
    arm1, = ax.plot([], [], [], linewidth=2)
    arm2, = ax.plot([], [], [], linewidth=2)
    time_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)
    solve_text = ax.text2D(0.03, 0.91, "", transform=ax.transAxes)

    arm_len = 0.35
    def set_segment(line, p_center, direction, half_len):
        p0 = p_center - half_len * direction
        p1 = p_center + half_len * direction
        line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        line.set_3d_properties([p0[2], p1[2]])

    def rotation_matrix_zyx(phi, theta, psi):
        cphi = math.cos(phi); sphi = math.sin(phi)
        cth  = math.cos(theta); sth = math.sin(theta)
        cpsi = math.cos(psi); spsi = math.sin(psi)
        Rz = np.array([[cpsi, -spsi, 0.0], [spsi, cpsi, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cth, 0.0, sth], [0.0, 1.0, 0.0], [-sth, 0.0, cth]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cphi, -sphi], [0.0, sphi, cphi]])
        return Rz @ Ry @ Rx

    def init_anim():
        traj_line.set_data([], [])
        traj_line.set_3d_properties([])
        obs_trace.set_data([], [])
        obs_trace.set_3d_properties([])
        obs_pt.set_data([], [])
        obs_pt.set_3d_properties([])
        center_pt.set_data([], [])
        center_pt.set_3d_properties([])
        arm1.set_data([], [])
        arm1.set_3d_properties([])
        arm2.set_data([], [])
        arm2.set_3d_properties([])
        time_text.set_text("")
        solve_text.set_text("")
        return traj_line, obs_trace, obs_pt, center_pt, arm1, arm2, time_text, solve_text

    def update_anim(i):
        traj_line.set_data(X_path[:i + 1, 0], X_path[:i + 1, 1])
        traj_line.set_3d_properties(X_path[:i + 1, 2])
        obs_trace.set_data(obs_path[:i + 1, 0], obs_path[:i + 1, 1])
        obs_trace.set_3d_properties(obs_path[:i + 1, 2])
        obs_pt.set_data([obs_path[i, 0]], [obs_path[i, 1]])
        obs_pt.set_3d_properties([obs_path[i, 2]])

        p = X_hist[i, 0:3]
        psi = X_hist[i, 6]
        phi = X_hist[i, 7]
        theta = X_hist[i, 8]
        Rm = rotation_matrix_zyx(phi, theta, psi)
        xb = Rm[:, 0]
        yb = Rm[:, 1]
        d1 = xb + yb
        d2 = xb - yb
        d1 = d1 / (np.linalg.norm(d1) + 1e-12)
        d2 = d2 / (np.linalg.norm(d2) + 1e-12)
        set_segment(arm1, p, d1, arm_len)
        set_segment(arm2, p, d2, arm_len)
        center_pt.set_data([p[0]], [p[1]])
        center_pt.set_3d_properties([p[2]])

        time_text.set_text(f"Time = {sim_time[i]:.2f} s")
        solve_text.set_text(f"Solve = {solve_ms[i]:.1f} ms")
        return traj_line, obs_trace, obs_pt, center_pt, arm1, arm2, time_text, solve_text

    _ani = FuncAnimation(
        fig,
        update_anim,
        init_func=init_anim,
        frames=steps,
        interval=int(1000 * dt),
        blit=False,
        repeat=False,
    )

    ax.legend(loc="best")

    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.plot(sim_time, solve_ms, linewidth=1.0)
    ax2.set_title("DR-MPPI Solve Time per Step")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Solve [ms]")
    ax2.grid(True, alpha=0.3)

    fig3, ax3 = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    labels = ["Thrust [N]", "Roll cmd [rad]", "Pitch cmd [rad]", "Yaw-rate cmd [rad/s]"]
    for j in range(4):
        ax3[j].plot(sim_time, U_applied[:, j], linewidth=1.0)
        ax3[j].set_ylabel(labels[j])
        ax3[j].grid(True, alpha=0.3)
    ax3[-1].set_xlabel("Time [s]")
    fig3.suptitle("Applied Control Inputs (Buffered)")

    plt.show()

    # summary at end
    print(
        "Solve time (ms) -> "
        f"min: {solve_ms.min():.2f}, "
        f"median: {np.median(solve_ms):.2f}, "
        f"mean: {solve_ms.mean():.2f}, "
        f"max: {solve_ms.max():.2f}"
    )
if __name__ == "__main__":
    simulate()
