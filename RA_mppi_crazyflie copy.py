#!/usr/bin/env python3
"""
Buffered Torch-MPPI Crazyflie-like quadrotor outer-loop simulation with:
- minimum-snap drone reference trajectory
- ONE moving obstacle (sphere) predicted time-ahead

Planner:
- Uses RA_MPPI from mppi_class.py (Torch)

Control: u = [roll_c, pitch_c, yaw_c, thrust]
State:   x = [px,py,pz, vx,vy,vz, roll, pitch, yaw]

This version:
- Runs simulation and stores data in buffers
- Saves buffers to .npz for offline plotting/replay
- Prints solve time during simulation

python3 "RA_mppi_crazyflie copy.py"  --obs-update-steps 15 --steps 400 --save

"""

from __future__ import annotations
import argparse

import time
import math
import os
from dataclasses import dataclass

import numpy as np

import torch
from mppi_class import RA_MPPI


# -----------------------------
# Helpers
# -----------------------------
def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class SimpleCrazyflieReference:
    start_position: np.ndarray
    dt: float = 0.03
    trajectory_type: str = "tilted_figure8"
    hover_begin: float = 0.0
    cycle_time: float = 40.0
    trajectory_speed: float = 0.6
    speed_ramp: float = 0.1
    reverse_direction: bool = False
    speed_schedule: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self):
        self.start_position = np.asarray(self.start_position, dtype=float).reshape(3)
        self.speed_schedule_np = None
        if self.speed_schedule is not None:
            schedule = np.asarray(self.speed_schedule, dtype=float)
            if schedule.ndim != 2 or schedule.shape[1] != 2 or schedule.shape[0] < 1:
                raise ValueError("speed_schedule must be ((time_s, speed), ...)")
            order = np.argsort(schedule[:, 0])
            schedule = schedule[order]
            schedule[:, 0] = np.maximum(schedule[:, 0], 0.0)
            schedule[0, 0] = 0.0
            self.speed_schedule_np = schedule
        if self.trajectory_type == "tilted_figure8" and self.speed_schedule_np is None:
            self.cycle_time = (2.0 * math.pi) / max(1e-6, float(self.trajectory_speed))
        self.total_time = float(self.hover_begin + self.cycle_time)

    @staticmethod
    def _square_progress(t: float, side_length: float, speed: float):
        segment_time = side_length / speed
        cycle_time = 4.0 * segment_time
        t_cycle = t % cycle_time

        if t_cycle < segment_time:
            s = t_cycle / segment_time
            pos_uv = np.array([side_length * s, 0.0], dtype=float)
            vel_uv = np.array([speed, 0.0], dtype=float)
        elif t_cycle < 2.0 * segment_time:
            s = (t_cycle - segment_time) / segment_time
            pos_uv = np.array([side_length, side_length * s], dtype=float)
            vel_uv = np.array([0.0, speed], dtype=float)
        elif t_cycle < 3.0 * segment_time:
            s = (t_cycle - 2.0 * segment_time) / segment_time
            pos_uv = np.array([side_length * (1.0 - s), side_length], dtype=float)
            vel_uv = np.array([-speed, 0.0], dtype=float)
        else:
            s = (t_cycle - 3.0 * segment_time) / segment_time
            pos_uv = np.array([0.0, side_length * (1.0 - s)], dtype=float)
            vel_uv = np.array([0.0, -speed], dtype=float)

        return pos_uv, vel_uv

    @staticmethod
    def _yaw_follow_from_vel(v: np.ndarray, fallback: float = 0.0) -> float:
        vx, vy = float(v[0]), float(v[1])
        if vx * vx + vy * vy < 1e-6:
            return fallback
        return math.atan2(vy, vx)

    def _phase_and_speed(self, t: float):
        if self.speed_schedule_np is None:
            return float(self.trajectory_speed) * t, float(self.trajectory_speed)

        schedule = self.speed_schedule_np
        phase = 0.0
        last_t = 0.0
        speed = float(schedule[0, 1])
        for next_t, next_speed in schedule[1:]:
            next_t = float(next_t)
            if t <= next_t:
                phase += speed * max(0.0, t - last_t)
                return phase, speed
            phase += speed * max(0.0, next_t - last_t)
            last_t = next_t
            speed = float(next_speed)
        phase += speed * max(0.0, t - last_t)
        return phase, speed

    def _pos_vel(self, t: float):
        if t < self.hover_begin:
            return self.start_position.copy(), np.zeros(3, dtype=float)

        t = float(t - self.hover_begin) % max(1e-9, float(self.cycle_time))
        direction = -1.0 if self.reverse_direction else 1.0

        if self.trajectory_type == "tilted_figure8":
            amp_x = 0.5
            amp_y = 0.2
            amp_z = 0.0
            phase, omega = self._phase_and_speed(t)
            s = direction * phase
            p = self.start_position + np.array(
                [
                    amp_x * math.sin(s),
                    amp_y * math.sin(2.0 * s),
                    amp_z * math.sin(s),
                ],
                dtype=float,
            )
            v = direction * np.array(
                [
                    amp_x * omega * math.cos(s),
                    2.0 * amp_y * omega * math.cos(2.0 * s),
                    amp_z * omega * math.cos(s),
                ],
                dtype=float,
            )
        elif self.trajectory_type == "tilted_square":
            side_length = 1.0
            speed = self.trajectory_speed
            pos_uv, vel_uv = self._square_progress(t, side_length, speed)
            dir_1 = np.array([1.0, 0.0, 0.0], dtype=float)
            dir_2 = np.array([0.0, 1.0, 0.6], dtype=float)
            dir_2 = dir_2 / np.linalg.norm(dir_2)
            p = self.start_position + pos_uv[0] * dir_1 + pos_uv[1] * dir_2
            v = direction * (vel_uv[0] * dir_1 + vel_uv[1] * dir_2)
        elif self.trajectory_type == "point":
            p = self.start_position.copy()
            v = np.zeros(3, dtype=float)
        else:
            raise ValueError(f"Unknown trajectory_type: {self.trajectory_type}")

        return p, v

    def eval(self, t: float, yaw_fallback: float = 0.0):
        p, v = self._pos_vel(t)
        yaw = self._yaw_follow_from_vel(v, yaw_fallback)
        return np.array(
            [p[0], p[1], p[2], v[0], v[1], v[2], 0.0, 0.0, yaw],
            dtype=float,
        )


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


def _z_body_world_torch(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    croll = torch.cos(roll); sroll = torch.sin(roll)
    cpitch = torch.cos(pitch); spitch = torch.sin(pitch)
    cyaw = torch.cos(yaw); syaw = torch.sin(yaw)
    zx = cyaw * spitch * croll + syaw * sroll
    zy = syaw * spitch * croll - cyaw * sroll
    zz = cpitch * croll
    return torch.stack([zx, zy, zz], dim=-1)


def quad_dyn_step(X, U, dt, m, g, **_kwargs):
    # X: (M,9) [px,py,pz,vx,vy,vz,roll,pitch,yaw]
    p = X[:, 0:3]
    v = X[:, 3:6]
    roll = X[:, 6]
    pitch = X[:, 7]
    yaw = X[:, 8]

    roll_c = U[:, 0]
    pitch_c = U[:, 1]
    yaw_c = U[:, 2]
    thrust = U[:, 3]

    tau_roll = float(_kwargs.get("tau_roll", 0.14))
    tau_pitch = float(_kwargs.get("tau_pitch", 0.14))
    tau_yaw = float(_kwargs.get("tau_yaw", 0.14))

    roll_dot = (roll_c - roll) / max(1e-6, tau_roll)
    pitch_dot = (pitch_c - pitch) / max(1e-6, tau_pitch)
    yaw_dot = (yaw_c - yaw) / max(1e-6, tau_yaw)

    zb = _z_body_world_torch(roll, pitch, yaw)
    gvec = torch.tensor([0.0, 0.0, float(g)], device=X.device, dtype=X.dtype)
    a = (thrust[:, None] / float(m)) * zb - gvec[None, :]

    v_next = v + float(dt) * a
    p_next = p + float(dt) * v_next
    roll_next = roll + float(dt) * roll_dot
    pitch_next = pitch + float(dt) * pitch_dot
    yaw_next = yaw + float(dt) * yaw_dot
    return torch.cat([p_next, v_next, roll_next[:, None], pitch_next[:, None], yaw_next[:, None]], dim=1)


def running_cost_quad(
    X, U, t,
    ref_seq, Q, R,
    drone_radius=0.0,
    obs_seq=None,
    moving_r=0.35, moving_safety_margin=0.50, moving_alpha=12.0,
    U_nom=None, Rd=None,
    **_kwargs
):
    ref = ref_seq[t + 1]  # (9,) [pos, vel, roll, pitch, yaw]
    e = X - ref[None, :]
    e[:, 6:9] = _wrap_pi_torch(e[:, 6:9])

    J = torch.sum((e * e) * Q[None, :], dim=1) + torch.sum((U * U) * R[None, :], dim=1)
    if (U_nom is not None) and (Rd is not None):
        du = U - U_nom[t][None, :]
        J = J + torch.sum((du * du) * Rd[None, :], dim=1)


    return J


def terminal_cost_quad(
    X,
    t_final,
    ref_seq,
    Qf,
    drone_radius=0.0,
    obs_seq=None,
    moving_r=0.35,
    moving_safety_margin=0.50,
    moving_alpha=12.0,
    **_kwargs,
):
    ref = ref_seq[-1]
    e = X - ref[None, :]
    e[:, 6:9] = _wrap_pi_torch(e[:, 6:9])
    J = torch.sum((e * e) * Qf[None, :], dim=1)


    return J


# -----------------------------
# RA-MPPI wrapper (Torch)
# -----------------------------
@dataclass
class RAMPPIParams:
    dt: float = 0.02
    horizon_steps: int = 40
    rollouts: int = 500
    lam: float = 1.0
    sigma: np.ndarray | None = None
    R_u: tuple[float, float, float, float] | None = None
    T_min: float = 0.0
    T_max: float = 0.0
    # ang_max: float = math.radians(25.0)
    ang_max: float = math.radians(40.0)
    yaw_max: float = math.radians(200.0)
    tau_roll: float = 0.14
    tau_pitch: float = 0.14
    tau_yaw: float = 0.14

    drone_radius: float = 0.25

    moving_r: float = 0.35
    moving_safety_margin: float = 0.50
    moving_alpha: float = 12.0
    obs_pos_sigma_xyz: tuple[float, float, float] = (0.01, 0.01, 0.01)
    noise_mode: str = "per_step"
    cvar_alpha: float = 0.95
    cvar_N: int = 50
    risk_cost_A: float = 0.0
    risk_cost_Cu: float = 0.0
    filter_infeasible_rollouts: bool = False
    Rd_u: tuple[float, float, float, float] = (3.0, 3.0, 0.5, 0.0)


class TorchRAMPPIQuadOuter:
    def __init__(self, mass=0.028, g=9.81, params: RAMPPIParams | None = None, device=None):
        self.m = float(mass)
        self.g = float(g)
        self.p = params if params is not None else RAMPPIParams()

        hover = self.m * self.g
        if self.p.T_max <= 0.0:
            self.p.T_max = 2.0 * hover
        if self.p.T_min < 0.0:
            self.p.T_min = 0.0

        if self.p.sigma is None:
            self.p.sigma = np.array([math.radians(8.0), math.radians(8.0), math.radians(40.0), 0.15*hover], dtype=np.float32)
            

        self.u_min = np.array([-self.p.ang_max, -self.p.ang_max, -self.p.yaw_max, self.p.T_min], dtype=np.float32)
        self.u_max = np.array([self.p.ang_max, self.p.ang_max, self.p.yaw_max, self.p.T_max], dtype=np.float32)

        sigma_np = np.asarray(self.p.sigma, dtype=np.float32).reshape(4,)
        if self.p.R_u is None:
            self.R_np = 1.0 / np.maximum(sigma_np ** 2, 1e-12).astype(np.float32)
        else:
            self.R_np = np.asarray(self.p.R_u, dtype=np.float32).reshape(4,)

        self.Rd_np = np.asarray(self.p.Rd_u, dtype=np.float32).reshape(4,)

        self.dt = float(self.p.dt)
        self.T = int(self.p.horizon_steps)
        self.M = int(self.p.rollouts)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.mppi = RA_MPPI(
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
            device=self.device,
            dtype=torch.float32,
            dyn_kwargs=dict(
                m=self.m,
                g=self.g,
                tau_roll=self.p.tau_roll,
                tau_pitch=self.p.tau_pitch,
                tau_yaw=self.p.tau_yaw,
            ),
            cost_kwargs=dict(
                ref_seq=None, Q=None, Qf=None, R=None, obs_seq=None,
                O_mean=None, radii=None,
                drone_radius=self.p.drone_radius,
                moving_r=self.p.moving_r,
                moving_safety_margin=self.p.moving_safety_margin, moving_alpha=self.p.moving_alpha,
                U_nom=None, Rd=None,
            ),
            cvar_alpha=float(self.p.cvar_alpha),
            cvar_N=int(self.p.cvar_N),
            obs_pos_sigma=np.asarray(self.p.obs_pos_sigma_xyz, dtype=np.float32),
            obs_noise_mode=str(self.p.noise_mode),
            risk_cost_A=float(self.p.risk_cost_A),
            risk_cost_Cu=float(self.p.risk_cost_Cu),
            filter_infeasible_rollouts=bool(self.p.filter_infeasible_rollouts),
            verbose=False,
        )

        # warm-start: hover thrust
        self.mppi.U_cpu[:] = 0.0
        self.mppi.U_cpu[:, 3] = hover
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
                tau_roll=dyn.get("tau_roll", self.p.tau_roll),
                tau_pitch=dyn.get("tau_pitch", self.p.tau_pitch),
                tau_yaw=dyn.get("tau_yaw", self.p.tau_yaw),
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
        if obs_seq_t is not None:
            if obs_seq_t.ndim == 2:
                O_mean = obs_seq_t[1:self.T+1, 0:3].unsqueeze(1)  # (T,1,3)
                K = 1
            elif obs_seq_t.ndim == 3:
                O_mean = obs_seq_t[1:self.T+1, :, 0:3]  # (T,K,3)
                K = int(obs_seq_t.shape[1])
            else:
                raise ValueError(f"obs_seq_np must be (T+1,3) or (T+1,K,3), got shape={tuple(obs_seq_t.shape)}")
            safe_R = float(self.p.moving_r + self.p.moving_safety_margin + self.p.drone_radius)
            radii = torch.full((K,), safe_R, device=self.device, dtype=torch.float32)
        else:
            O_mean = None
            radii = None

        ck = self.mppi.cost_kwargs
        ck["ref_seq"] = ref_seq_t
        ck["Q"] = Q_t
        ck["Qf"] = Qf_t
        ck["R"] = R_t
        ck["obs_seq"] = obs_seq_t
        ck["O_mean"] = O_mean
        ck["radii"] = radii
        ck["U_nom"] = torch.as_tensor(self.mppi.U_cpu, device=self.device, dtype=torch.float32)
        ck["Rd"] = torch.as_tensor(self.Rd_np, device=self.device, dtype=torch.float32)

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
def simulate(
    save_dir: str | None = None,
    obs_update_steps: int = 1,
    use_gpu: bool = True,
    sim_steps: int | None = None,
):
    trajectory_type = "tilted_figure8"
    trajectory_center = np.array([0.0, -0.8, 2.4], dtype=float)
    ego_ref_speed = 0.6
    obstacle_ref_speed = 0.5
    moving_speed_schedules = [
        ((0.0, 0.50), (3.0, 0.80), (6.0, 0.35), (9.0, 0.70), (12.0, 0.45)),
        ((0.0, 0.45), (2.5, 0.30), (5.5, 0.85), (8.5, 0.40), (11.5, 0.75)),
    ]
    traj = SimpleCrazyflieReference(
        start_position=trajectory_center,
        dt=0.02,
        trajectory_type=trajectory_type,
        hover_begin=0.0,
        trajectory_speed=ego_ref_speed,
    )

    moving_trajs = [
        SimpleCrazyflieReference(
            start_position=trajectory_center,
            dt=0.03,
            trajectory_type=trajectory_type,
            hover_begin=0.0,
            trajectory_speed=obstacle_ref_speed,
            reverse_direction=True,
            speed_schedule=moving_speed_schedules[0],
        ),
        SimpleCrazyflieReference(
            start_position=trajectory_center,
            dt=0.03,
            trajectory_type=trajectory_type,
            hover_begin=0.0,
            trajectory_speed=obstacle_ref_speed,
            reverse_direction=True,
            speed_schedule=moving_speed_schedules[1],
        ),
    ]
    n_moving_obstacles = len(moving_trajs)
    moving_time_offsets = np.array(
        [0.0, moving_trajs[1].total_time * 0.25],
        dtype=float,
    )

    params = RAMPPIParams(
        dt=0.02,
        horizon_steps=25,
        rollouts=2000,
        lam=5,
        sigma=np.array(
            [0.1, 0.1, 0.01, 0.0807464837],
            dtype=np.float32,
        ),
        T_max=0.4776,
        ang_max=math.radians(20),
        yaw_max=math.radians(20),
        tau_roll=0.22,
        tau_pitch=0.19,
        tau_yaw=0.22,
        drone_radius=0.04,
        moving_r=0.04,
        moving_safety_margin=0.0,
        moving_alpha=12.0,
        obs_pos_sigma_xyz=(0.01, 0.01, 0.01),
        noise_mode="per_step",
        cvar_alpha=0.95,
        cvar_N=60,
        risk_cost_A=5000,
        risk_cost_Cu=0.0,
        filter_infeasible_rollouts=False,
        R_u=(5.7, 8.7, 3.1, 0.01),
        Rd_u=(5.7, 8.7, 3.1, 0.01),
    )

    
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    ctrl = TorchRAMPPIQuadOuter(mass=0.028, g=9.81, params=params, device=device)
    print(f"Planner backend: {device} (cuda_available={int(torch.cuda.is_available())})")

    # [x, y, z, vx, vy, vz, roll, pitch, yaw]
    Q = np.array([
        1008, 1000, 1008,
        25, 30, 13,
        0.01, 0.01, 0.01,
    ], dtype=np.float32)
    Qf = np.array([
        1008, 1000, 1008,
        25, 30, 13,
        0.01, 0.01, 0.01,
    ], dtype=np.float32)
    dt = params.dt
    default_sim_T = max(
        [traj.total_time]
        + [
            mt.total_time + float(moving_time_offsets[j])
            for j, mt in enumerate(moving_trajs)
        ]
    )
    steps = int(default_sim_T / dt) + 1 if sim_steps is None else max(1, int(sim_steps))
    sim_T = (steps - 1) * dt

    # initial state
    ref0 = traj.eval(0.0)
    x = np.zeros(9, dtype=float)
    x[0:3] = ref0[0:3]
    x[3:6] = 0.0
    x[6] = 0.0
    x[7] = 0.0
    x[8] = 0.0

    last_yaw_ref = float(x[8])

    def eval_ego_loop(t_raw: float):
        if t_raw <= 0.0:
            return traj.eval(0.0, yaw_fallback=last_yaw_ref)
        loop_T = max(1e-9, float(traj.total_time))
        return traj.eval(float(t_raw) % loop_T, yaw_fallback=last_yaw_ref)

    def eval_moving_loop(moving_traj: SimpleCrazyflieReference, t_raw: float):
        loop_T = max(1e-9, float(moving_traj.total_time))
        ref = moving_traj.eval(float(t_raw) % loop_T)
        return ref[0:3], ref[3:6]

    lead_time = 1.5495997771078638
    moving_initial_advance = max(0.0, float(np.max(moving_time_offsets)) - lead_time + 3.0)

    # precompute curves for drawing
    tt = np.linspace(0.0, traj.total_time, 600)
    ref_curve = np.array([traj.eval(float(tj))[0:3] for tj in tt])

    obs_curves = []
    for moving_traj in moving_trajs:
        tt2 = np.linspace(0.0, moving_traj.total_time, 600)
        obs_curve = np.array([moving_traj.eval(float(tj))[0:3] for tj in tt2])
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
    min_obstacle_dist = np.full((steps,), np.nan, dtype=float)
    safety_flags = np.zeros((steps,), dtype=np.int8)
    collision_flags = np.zeros((steps,), dtype=np.int8)
    sim_time = np.arange(steps, dtype=float) * dt
    drone_radius = float(getattr(params, "drone_radius", 0.3662153322325755))
    moving_margin = float(getattr(params, "moving_safety_margin", getattr(params, "moving_margin", 0.0)))
    moving_collision_radius = float(params.moving_r + drone_radius)
    moving_safe_radius = float(params.moving_r + moving_margin + drone_radius)

    def collision_log_for_position(p, obs_now):
        obs_points = np.asarray(obs_now, dtype=float).reshape(-1, 3)
        move_dists = np.linalg.norm(obs_points - p[None, :], axis=1) if obs_points.size else np.array([np.inf])
        min_dist = float(np.min(move_dists))
        safety = bool(np.any(move_dists < moving_safe_radius))
        collision = bool(np.any(move_dists < moving_collision_radius))
        return min_dist, safety, collision
    obs_update_steps = max(1, int(obs_update_steps))
    held_obs_seq = None
    last_obs_update_i = -10**9
    obs_pos_sigma_xyz = np.asarray(params.obs_pos_sigma_xyz, dtype=float).reshape(1, 1, 3)
    ever_safety = False
    ever_collision = False
    first_collision_step = None

    for i in range(steps):
        t = i * dt

        # build ref_seq (T+1,9): [x, y, z, vx, vy, vz, roll, pitch, yaw]
        ref_seq = np.zeros((ctrl.T + 1, 9), dtype=float)
        ref_seq[0] = eval_ego_loop(t)
        ref_seq[0, 8] = last_yaw_ref + wrap_pi(float(ref_seq[0, 8]) - last_yaw_ref)

        for k in range(1, ctrl.T + 1):
            tk = t + k * dt
            prev = float(ref_seq[k - 1, 8])
            ref_seq[k] = traj.eval(tk % max(1e-9, float(traj.total_time)), yaw_fallback=prev)
            ref_seq[k, 8] = prev + wrap_pi(float(ref_seq[k, 8]) - prev)

        last_yaw_ref = float(ref_seq[1, 8])

        obs_seq = np.zeros((ctrl.T + 1, n_obs, 3), dtype=float)
        for j, moving_traj in enumerate(moving_trajs):
            for k in range(ctrl.T + 1):
                tk = t + lead_time + moving_initial_advance + k * dt - float(moving_time_offsets[j])
                op, _ = eval_moving_loop(moving_traj, tk)
                obs_seq[k, j] = op

        obs_path[i] = obs_seq[0]
        true_obs_seq = obs_seq
        if held_obs_seq is None or (i - last_obs_update_i) >= obs_update_steps:
            obs_noise = np.random.normal(
                loc=0.0,
                scale=obs_pos_sigma_xyz,
                size=true_obs_seq.shape,
            )
            held_obs_seq = true_obs_seq + obs_noise
            last_obs_update_i = i
        obs_seq = held_obs_seq

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

        roll_c, pitch_c, yaw_c, thrust = map(float, u)
        roll_c = clamp(roll_c, -ctrl.p.ang_max, ctrl.p.ang_max)
        pitch_c = clamp(pitch_c, -ctrl.p.ang_max, ctrl.p.ang_max)
        yaw_c = clamp(yaw_c, -ctrl.p.yaw_max, ctrl.p.yaw_max)
        thrust = clamp(thrust, ctrl.p.T_min, ctrl.p.T_max)
        U_applied[i] = [roll_c, pitch_c, yaw_c, thrust]

        roll = float(x[6])
        pitch = float(x[7])
        yaw = float(x[8])
        croll = math.cos(roll); sroll = math.sin(roll)
        cpitch = math.cos(pitch); spitch = math.sin(pitch)
        cyaw = math.cos(yaw); syaw = math.sin(yaw)
        zb = np.array([
            cyaw * spitch * croll + syaw * sroll,
            syaw * spitch * croll - cyaw * sroll,
            cpitch * croll,
        ], dtype=float)
        a = (thrust / ctrl.m) * zb - np.array([0.0, 0.0, ctrl.g])
        roll_dot = (roll_c - roll) / max(1e-6, ctrl.p.tau_roll)
        pitch_dot = (pitch_c - pitch) / max(1e-6, ctrl.p.tau_pitch)
        yaw_dot = (yaw_c - yaw) / max(1e-6, ctrl.p.tau_yaw)

        x[3:6] = x[3:6] + dt * a
        x[0:3] = x[0:3] + dt * x[3:6]
        x[6] = x[6] + dt * roll_dot
        x[7] = x[7] + dt * pitch_dot
        x[8] = x[8] + dt * yaw_dot

        X_path[i] = x[0:3]
        X_hist[i] = x.copy()
        dmin, hit_safety, hit_collision = collision_log_for_position(x[0:3], true_obs_seq[0])
        min_obstacle_dist[i] = dmin
        safety_flags[i] = int(hit_safety)
        collision_flags[i] = int(hit_collision)
        if hit_safety:
            ever_safety = True
        if hit_collision and not ever_collision:
            ever_collision = True
            first_collision_step = i
            print(f"[RA_MPPI COLLISION] step={i:04d} t={t:.2f}s dmin={dmin:.3f} pos={x[0:3]}")
        if (i % 10) == 0:
            print(
                f"[RA_MPPI collision] step={i:04d} t={t:.2f}s "
                f"dmin={dmin:.3f} safety={int(hit_safety)} collision={int(hit_collision)}"
            )

    print(
        f"[RA_MPPI summary] min_dist={np.nanmin(min_obstacle_dist):.3f} "
        f"safety={int(ever_safety)} collision={int(ever_collision)} "
        f"collision_steps={int(np.count_nonzero(collision_flags))} "
        f"first_collision_step={first_collision_step}"
    )

    run_data = {
        "method": "ramppi",
        "trajectory_type": np.array(trajectory_type),
        "dt": float(dt),
        "sim_time": sim_time,
        "sim_steps": np.array(steps, dtype=np.int32),
        "sim_duration": np.array(sim_T, dtype=float),
        "X_path": X_path,
        "X_hist": X_hist,
        "obs_path": obs_path,
        "U_applied": U_applied,
        "solve_ms": solve_ms,
        "min_obstacle_dist": min_obstacle_dist,
        "safety_flags": safety_flags,
        "collision_flags": collision_flags,
        "first_collision_step": np.array(-1 if first_collision_step is None else first_collision_step, dtype=np.int32),
        "collision_points": X_path[collision_flags.astype(bool)],
        "moving_collision_radius": np.array(moving_collision_radius, dtype=float),
        "moving_safe_radius": np.array(moving_safe_radius, dtype=float),
        "drone_radius": np.array(drone_radius, dtype=float),
        "obs_pos_sigma_xyz": np.asarray(params.obs_pos_sigma_xyz, dtype=float),
        "noise_mode": np.array(params.noise_mode),
        "cvar_alpha": np.array(params.cvar_alpha, dtype=float),
        "cvar_N": np.array(params.cvar_N, dtype=np.int32),
        "risk_cost_A": np.array(params.risk_cost_A, dtype=float),
        "risk_cost_Cu": np.array(params.risk_cost_Cu, dtype=float),
        "filter_infeasible_rollouts": np.array(params.filter_infeasible_rollouts, dtype=bool),
        "obs_update_steps": np.array(obs_update_steps, dtype=np.int32),
        "ref_curve": ref_curve,
        "obs_curves": np.asarray(obs_curves, dtype=float),
        "moving_speed_schedules": np.asarray(moving_speed_schedules, dtype=float),
        "mins": mins,
        "maxs": maxs,
        "pred_samples_xyz": pred_samples_xyz,
        "pred_nominal_xyz": pred_nominal_xyz,
    }
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "ramppi_simulation.npz")
        np.savez_compressed(out_path, **run_data)
        print(f"Saved simulation data to {out_path}")
    return run_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RA-MPPI Crazyflie simulation.")
    parser.add_argument("--save", action="store_true", help="Save simulation .npz to the plot directory.")
    parser.add_argument("--obs-update-steps", type=int, default=15, help="Read fresh moving-obstacle observations every N control steps and hold between reads.")
    parser.add_argument("--steps", type=int, default=520, help="Number of simulation control steps. The ego and moving paths loop for this many steps.")
    parser.add_argument("--use_gpu", "--cuda", dest="use_gpu", action="store_true", default=True, help="Use CUDA when available (default).")
    parser.add_argument("--cpu", dest="use_gpu", action="store_false", help="Force CPU even when CUDA is available.")
    args = parser.parse_args()
    save_dir = os.path.join(os.path.dirname(__file__), "plot") if args.save else None
    simulate(save_dir=save_dir, obs_update_steps=args.obs_update_steps, use_gpu=args.use_gpu, sim_steps=args.steps)
