#!/usr/bin/env python3
"""Physical Crazyflie figure-eight flight using RA-MPPI and Vicon avoidance.

Vicon rigid body ``cf_1`` supplies the controlled vehicle state and ``cf_2`` is
a moving obstacle controlled by the separate ``obstacle.py`` process. This
process never commands cf_2: it only reads its measured Vicon state for MPPI
avoidance. A farther point on the risk-aware optimized prediction is streamed
to cf_1.
"""

from __future__ import annotations
import argparse
import time
import math
import os
from queue import Empty
from threading import Lock, Thread
from dataclasses import dataclass

import numpy as np

import torch
from mppi_class import RA_MPPI
from mocap_hl_commander_sim import figure8 as MOCAP_FIGURE8

import motioncapture
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

# Edit this value directly: 1.0 = normal speed, 0.5 = half speed.
SPEED_SCALE = 0.5


# -----------------------------
# Helpers
# -----------------------------
def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def set_led_ring(cf, red, green, blue):
    """Set the LED Ring Deck to a solid RGB color."""
    cf.param.set_value("ring.effect", "7")
    cf.param.set_value("ring.solidRed", str(int(red)))
    cf.param.set_value("ring.solidGreen", str(int(green)))
    cf.param.set_value("ring.solidBlue", str(int(blue)))


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
            amp_x = 1.0
            amp_y = 0.4
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


class MocapPoly4DFigure8Reference:
    """Evaluate the exact Poly4D figure-eight from mocap_hl_commander.py."""

    def __init__(self, trajectory, center):
        self.trajectory = tuple(tuple(float(value) for value in row) for row in trajectory)
        self.center = np.asarray(center, dtype=float).reshape(3)
        self.segment_durations = np.asarray(
            [row[0] for row in self.trajectory], dtype=float
        )
        self.segment_ends = np.cumsum(self.segment_durations)
        self.total_time = float(self.segment_ends[-1])

    @staticmethod
    def _poly_value(coefficients, t):
        return sum(coefficient * t**power for power, coefficient in enumerate(coefficients))

    @staticmethod
    def _poly_derivative(coefficients, t):
        return sum(
            power * coefficient * t ** (power - 1)
            for power, coefficient in enumerate(coefficients)
            if power > 0
        )

    def eval(self, t, yaw_fallback=0.0):
        del yaw_fallback
        loop_t = float(t) % self.total_time
        segment_index = int(np.searchsorted(self.segment_ends, loop_t, side="right"))
        segment_index = min(segment_index, len(self.trajectory) - 1)
        segment_start = 0.0 if segment_index == 0 else self.segment_ends[segment_index - 1]
        local_t = loop_t - float(segment_start)
        row = self.trajectory[segment_index]

        x_coefficients = row[1:9]
        y_coefficients = row[9:17]
        z_coefficients = row[17:25]
        yaw_coefficients = row[25:33]
        position = self.center + np.array(
            [
                self._poly_value(x_coefficients, local_t),
                self._poly_value(y_coefficients, local_t),
                self._poly_value(z_coefficients, local_t),
            ],
            dtype=float,
        )
        velocity = np.array(
            [
                self._poly_derivative(x_coefficients, local_t),
                self._poly_derivative(y_coefficients, local_t),
                self._poly_derivative(z_coefficients, local_t),
            ],
            dtype=float,
        )
        yaw = self._poly_value(yaw_coefficients, local_t)
        return np.array(
            [
                position[0], position[1], position[2],
                velocity[0], velocity[1], velocity[2],
                0.0, 0.0, yaw,
            ],
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


def _state_error_cost_np(x, ref, Q):
    e = np.asarray(x, dtype=float) - np.asarray(ref, dtype=float)
    e[6:9] = np.array([wrap_pi(float(v)) for v in e[6:9]], dtype=float)
    return float(np.sum((e * e) * np.asarray(Q, dtype=float)))


def _cvar_tail_mean_np(values, alpha):
    vals = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if vals.size == 0:
        return 0.0
    tail_n = max(1, int(math.ceil((1.0 - float(alpha)) * vals.size)))
    return float(np.mean(vals[-tail_n:]))


def _executed_cvar_risk_cost_np(x, obs_now, params, rng):
    if obs_now is None or float(params.risk_cost_A) <= 0.0:
        return 0.0
    obs_points = np.asarray(obs_now, dtype=float).reshape(-1, 3)
    if obs_points.size == 0:
        return 0.0
    sigma = np.asarray(
        params.obs_pos_sigma_xyz, dtype=float
    ).reshape(1, 1, 3)
    samples = obs_points[:, None, :] + rng.normal(
        loc=0.0,
        scale=sigma,
        size=(obs_points.shape[0], int(params.cvar_N), 3),
    )
    safe_radius = float(
        params.moving_r
        + params.moving_safety_margin
        + params.drone_radius
    )
    distances = np.linalg.norm(
        samples - np.asarray(x, dtype=float)[None, None, 0:3],
        axis=-1,
    )
    violations = safe_radius - distances
    cvars = np.array(
        [
            _cvar_tail_mean_np(values, params.cvar_alpha)
            for values in violations
        ],
        dtype=float,
    )
    cvar_max = float(np.max(cvars)) if cvars.size else 0.0
    if cvar_max > float(params.risk_cost_Cu):
        return float(params.risk_cost_A) * cvar_max
    return 0.0


def _executed_tracking_stage_cost_np(x, u, ref, Q, R, u_nom, Rd):
    u = np.asarray(u, dtype=float).reshape(4)
    total = _state_error_cost_np(x, ref, Q)
    total += float(np.sum((u * u) * np.asarray(R, dtype=float).reshape(4)))
    if u_nom is not None and Rd is not None:
        du = u - np.asarray(u_nom, dtype=float).reshape(4)
        total += float(np.sum((du * du) * np.asarray(Rd, dtype=float).reshape(4)))
    return float(total)


def _executed_full_stage_cost_np(
    x, u, ref, Q, R, u_nom, Rd, params, obs_now, rng
):
    total = _executed_tracking_stage_cost_np(x, u, ref, Q, R, u_nom, Rd)
    total += _executed_cvar_risk_cost_np(x, obs_now, params, rng)
    return float(total)


# -----------------------------
# RA-MPPI wrapper (Torch)
# -----------------------------
@dataclass
class RAMPPIParams:
    dt: float = 0.05
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
    cvar_N: int = 10
    risk_cost_A: float = 2000.0
    risk_cost_Cu: float = 0.0
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
            obs_pos_sigma=np.asarray(
                self.p.obs_pos_sigma_xyz, dtype=np.float32
            ),
            obs_noise_mode=str(self.p.noise_mode),
            risk_cost_A=float(self.p.risk_cost_A),
            risk_cost_Cu=float(self.p.risk_cost_Cu),
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
        return_nominal_prediction=False,
    ):
        ref_seq_t = torch.as_tensor(ref_seq_np, device=self.device, dtype=torch.float32)
        Q_t  = torch.as_tensor(Q_np,  device=self.device, dtype=torch.float32)
        Qf_t = torch.as_tensor(Qf_np, device=self.device, dtype=torch.float32)
        R_t  = torch.as_tensor(self.R_np, device=self.device, dtype=torch.float32)
        obs_seq_t = torch.as_tensor(obs_seq_np, device=self.device, dtype=torch.float32) if obs_seq_np is not None else None
        if obs_seq_t is not None:
            if obs_seq_t.ndim == 2:
                O_mean = obs_seq_t[1:self.T + 1, 0:3].unsqueeze(1)
                obstacle_count = 1
            elif obs_seq_t.ndim == 3:
                O_mean = obs_seq_t[1:self.T + 1, :, 0:3]
                obstacle_count = int(obs_seq_t.shape[1])
            else:
                raise ValueError(
                    "obs_seq_np must have shape (T+1,3) or (T+1,K,3), "
                    f"got {tuple(obs_seq_t.shape)}"
                )
            safe_radius = float(
                self.p.moving_r
                + self.p.moving_safety_margin
                + self.p.drone_radius
            )
            radii = torch.full(
                (obstacle_count,),
                safe_radius,
                device=self.device,
                dtype=torch.float32,
            )
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
        if include_nominal_prediction and (
            return_predictions or return_nominal_prediction
        ):
            pred_nominal_xyz = self._predict_nominal_xyz(x0_np, U_cpu)

        # warm-start shift
        self.mppi.U_cpu[:-1] = self.mppi.U_cpu[1:]
        self.mppi.U_cpu[-1] = self.mppi.U_cpu[-2]
        self.mppi.U = self.mppi.U_cpu
        if return_predictions:
            return u0, pred_samples_xyz, pred_nominal_xyz
        if return_nominal_prediction:
            return u0, pred_nominal_xyz
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
        ((0.0, 0.50), (0.50, 2.70), (0.62, 0.50), (1.10, 2.50), (1.22, 0.50), (1.70, 2.80), (1.82, 0.50), (2.35, 2.60), (2.47, 0.50), (3.05, 2.70), (3.18, 0.50), (3.75, 2.50), (3.88, 0.50), (4.45, 2.80), (4.57, 0.50), (5.15, 2.60), (5.28, 0.50), (5.90, 2.70), (6.02, 0.50), (6.60, 2.50), (6.72, 0.50), (7.30, 2.80), (7.43, 0.50), (8.05, 2.60), (8.18, 0.50), (8.80, 2.70), (8.92, 0.50), (9.50, 2.50), (9.62, 0.50), (10.20, 2.80), (10.32, 0.50), (10.82, 2.60), (10.94, 0.50), (11.42, 2.70), (11.54, 0.50), (12.12, 2.50), (12.24, 0.50), (12.82, 2.80), (12.94, 0.50), (13.52, 2.60), (13.64, 0.50), (14.22, 2.70), (14.34, 0.50), (14.92, 2.50), (15.04, 0.50)),
        ((0.0, 0.50), (0.35, 2.60), (0.48, 0.50), (0.95, 2.80), (1.07, 0.50), (1.55, 2.50), (1.68, 0.50), (2.15, 2.70), (2.27, 0.50), (2.85, 2.60), (2.98, 0.50), (3.50, 2.80), (3.62, 0.50), (4.20, 2.50), (4.33, 0.50), (4.90, 2.70), (5.03, 0.50), (6.07, 2.60), (6.19, 0.50), (6.77, 2.80), (6.90, 0.50), (7.47, 2.50), (7.59, 0.50), (8.22, 2.70), (8.35, 0.50), (8.92, 2.60), (9.04, 0.50), (9.62, 2.80), (9.75, 0.50), (10.32, 2.50), (10.44, 0.50), (11.07, 2.70), (11.20, 0.50), (11.77, 2.60), (11.89, 0.50), (12.47, 2.80), (12.60, 0.50), (13.17, 2.50), (13.29, 0.50), (13.92, 2.70), (14.05, 0.50), (14.62, 2.60), (14.74, 0.50), (14.87, 2.80), (14.99, 0.50)),
    ]
    traj = SimpleCrazyflieReference(
        start_position=trajectory_center,
        dt=0.05,
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
        [0.0, moving_trajs[1].total_time * 0.09],
        dtype=float,
    )

    params = RAMPPIParams(
        dt=0.02,
        horizon_steps=40,
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
        drone_radius=0.044,
        moving_r=0.04,
        moving_safety_margin=0.0,
        moving_alpha=12.0,
        obs_pos_sigma_xyz=(0.01, 0.01, 0.01),
        noise_mode="per_step",
        cvar_alpha=0.95,
        cvar_N=50,
        risk_cost_A=3000.0,
        risk_cost_Cu=0.0,
        R_u=(5.7, 8.7, 3.1, 0.01),
        Rd_u=(5.7, 8.7, 3.1, 0.01),
    )

    
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    ctrl = TorchRAMPPIQuadOuter(
        mass=0.028, g=9.81, params=params, device=device
    )
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
    moving_initial_advance = max(0.0, float(np.max(moving_time_offsets)) - lead_time + 5.3)

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
    full_stage_cost = np.zeros((steps,), dtype=float)
    tracking_stage_cost = np.zeros((steps,), dtype=float)
    terminal_full_cost = 0.0
    terminal_tracking_cost = 0.0
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
        u_nom0 = np.asarray(ctrl.mppi.U_cpu[0], dtype=float).copy()
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
        tracking_stage_cost[i] = _executed_tracking_stage_cost_np(
            x=x,
            u=U_applied[i],
            ref=ref_seq[1],
            Q=Q,
            R=ctrl.R_np,
            u_nom=u_nom0,
            Rd=ctrl.Rd_np,
        )
        full_stage_cost[i] = _executed_full_stage_cost_np(
            x=x,
            u=U_applied[i],
            ref=ref_seq[1],
            Q=Q,
            R=ctrl.R_np,
            u_nom=u_nom0,
            Rd=ctrl.Rd_np,
            params=params,
            obs_now=obs_seq[1],
        )
        dmin, hit_safety, hit_collision = collision_log_for_position(x[0:3], true_obs_seq[0])
        min_obstacle_dist[i] = dmin
        safety_flags[i] = int(hit_safety)
        collision_flags[i] = int(hit_collision)
        if hit_safety:
            ever_safety = True
        if hit_collision and not ever_collision:
            ever_collision = True
            first_collision_step = i
            print(f"[MPPI COLLISION] step={i:04d} t={t:.2f}s dmin={dmin:.3f} pos={x[0:3]}")
        if (i % 10) == 0:
            print(
                f"[MPPI collision] step={i:04d} t={t:.2f}s "
                f"dmin={dmin:.3f} safety={int(hit_safety)} collision={int(hit_collision)}"
            )

    terminal_full_cost = _state_error_cost_np(x, ref_seq[1], Qf)
    terminal_tracking_cost = _state_error_cost_np(x, ref_seq[1], Qf)
    total_full_cost = float(np.sum(full_stage_cost) + terminal_full_cost)
    total_tracking_cost = float(np.sum(tracking_stage_cost) + terminal_tracking_cost)
    print(
        f"[MPPI summary] min_dist={np.nanmin(min_obstacle_dist):.3f} "
        f"safety={int(ever_safety)} collision={int(ever_collision)} "
        f"collision_steps={int(np.count_nonzero(collision_flags))} "
        f"first_collision_step={first_collision_step}"
    )

    run_data = {
        "method": "ra_mppi",
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
        "full_stage_cost": full_stage_cost,
        "tracking_stage_cost": tracking_stage_cost,
        "terminal_full_cost": np.array(terminal_full_cost, dtype=float),
        "total_full_cost": np.array(total_full_cost, dtype=float),
        "terminal_tracking_cost": np.array(terminal_tracking_cost, dtype=float),
        "total_tracking_cost": np.array(total_tracking_cost, dtype=float),
        "min_obstacle_dist": min_obstacle_dist,
        "safety_flags": safety_flags,
        "collision_flags": collision_flags,
        "first_collision_step": np.array(-1 if first_collision_step is None else first_collision_step, dtype=np.int32),
        "collision_points": X_path[collision_flags.astype(bool)],
        "moving_collision_radius": np.array(moving_collision_radius, dtype=float),
        "moving_safe_radius": np.array(moving_safe_radius, dtype=float),
        "drone_radius": np.array(drone_radius, dtype=float),
        "obs_pos_sigma_xyz": np.asarray(params.obs_pos_sigma_xyz, dtype=float),
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
        out_path = os.path.join(save_dir, "mppi_simulation.npz")
        np.savez_compressed(out_path, **run_data)
        print(f"Saved simulation data to {out_path}")
    return run_data


# -----------------------------
# Physical flight
# -----------------------------
class ViconTracker(Thread):
    """Collect filtered states for cf_1 and cf_2 from one Vicon connection."""

    def __init__(self, host, ego_name="cf_1", obstacle_name="cf_2", velocity_alpha=0.25):
        super().__init__(daemon=True)
        self.host = host
        self.ego_name = ego_name
        self.obstacle_name = obstacle_name
        self.velocity_alpha = float(velocity_alpha)
        self.on_ego_pose = None
        self.error = None
        self._running = True
        self._lock = Lock()
        self._states = {}
        self.start()

    def close(self):
        self._running = False

    @staticmethod
    def _quat_to_euler(quat):
        qx, qy, qz, qw = map(float, (quat.x, quat.y, quat.z, quat.w))
        sinr = 2.0 * (qw * qx + qy * qz)
        cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr, cosr)
        sinp = clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = math.asin(sinp)
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny, cosy)
        return np.array([roll, pitch, yaw], dtype=float)

    def _update(self, name, obj, stamp):
        pos = np.asarray(obj.position, dtype=float).reshape(3)
        if not np.all(np.isfinite(pos)):
            return
        quat = obj.rotation
        euler = self._quat_to_euler(quat)
        with self._lock:
            old = self._states.get(name)
            vel = np.zeros(3, dtype=float)
            if old is not None:
                sample_dt = stamp - old["stamp"]
                if 1e-4 < sample_dt < 1.0:
                    raw_vel = (pos - old["position"]) / sample_dt
                    vel = (
                        self.velocity_alpha * raw_vel
                        + (1.0 - self.velocity_alpha) * old["velocity"]
                    )
            self._states[name] = {
                "position": pos,
                "velocity": vel,
                "euler": euler,
                "quat": quat,
                "stamp": stamp,
            }
        if name == self.ego_name and self.on_ego_pose is not None:
            self.on_ego_pose(pos, quat)

    def snapshot(self, name):
        with self._lock:
            state = self._states.get(name)
            if state is None:
                return None
            return {
                "position": state["position"].copy(),
                "velocity": state["velocity"].copy(),
                "euler": state["euler"].copy(),
                "stamp": float(state["stamp"]),
            }

    def wait_for_body(self, name, timeout):
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if self.error is not None:
                raise RuntimeError("Vicon thread failed") from self.error
            state = self.snapshot(name)
            if state is not None:
                return state
            time.sleep(0.02)
        raise TimeoutError(f"No Vicon data received for rigid body {name!r}")

    def run(self):
        try:
            mc = motioncapture.connect("vicon", {"hostname": self.host})
            while self._running:
                mc.waitForNextFrame()
                stamp = time.monotonic()
                for name in (self.ego_name, self.obstacle_name):
                    obj = mc.rigidBodies.get(name)
                    if obj is not None:
                        self._update(name, obj, stamp)
        except Exception as exc:
            self.error = exc
            self._running = False


def wait_for_position_estimator(scf, timeout=20.0):
    log_config = LogConfig(name="Kalman Variance", period_in_ms=500)
    for variable in ("kalman.varPX", "kalman.varPY", "kalman.varPZ"):
        log_config.add_variable(variable, "float")
    histories = {name: [1000.0] * 10 for name in ("kalman.varPX", "kalman.varPY", "kalman.varPZ")}
    deadline = time.monotonic() + float(timeout)
    with SyncLogger(scf, log_config) as logger:
        sample_count = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    "Crazyflie position estimator did not provide "
                    "converged variance data"
                )
            try:
                log_entry = logger._queue.get(
                    timeout=min(0.5, remaining)
                )
            except Empty:
                continue
            if log_entry == SyncLogger.DISCONNECT_EVENT:
                raise ConnectionError(
                    "Crazyflie disconnected during estimator setup"
                )
            _, data, _ = log_entry
            sample_count += 1
            for name, history in histories.items():
                history.append(float(data[name]))
                history.pop(0)
            ranges = {
                name: max(values) - min(values)
                for name, values in histories.items()
            }
            print(
                "Estimator sample "
                f"{sample_count}: "
                f"var=({float(data['kalman.varPX']):.6f}, "
                f"{float(data['kalman.varPY']):.6f}, "
                f"{float(data['kalman.varPZ']):.6f}) "
                f"range=({ranges['kalman.varPX']:.6f}, "
                f"{ranges['kalman.varPY']:.6f}, "
                f"{ranges['kalman.varPZ']:.6f})"
            )
            if all(value < 0.001 for value in ranges.values()):
                print("Crazyflie estimator converged.")
                return


def make_extpose_forwarder(cf, rate_hz=50.0):
    """Create a thread-safe, rate-limited external-pose callback."""
    minimum_period = 1.0 / float(rate_hz)
    callback_lock = Lock()
    last_send_time = [0.0]

    def forward(pos, quat):
        now = time.monotonic()
        with callback_lock:
            if now - last_send_time[0] < minimum_period:
                return
            last_send_time[0] = now
        cf.extpos.send_extpose(
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(quat.x),
            float(quat.y),
            float(quat.z),
            float(quat.w),
        )

    return forward


def configure_estimator(scf):
    cf = scf.cf
    # Match mocap_hl_commander.py:
    # adjust_orientation_sensitivity() followed by activate_kalman_estimator().
    cf.param.set_value("locSrv.extQuatStdDev", 8.0e-3)
    cf.param.set_value("stabilizer.estimator", "2")
    cf.param.set_value("locSrv.extQuatStdDev", 0.06)
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    wait_for_position_estimator(scf)


def make_reference_horizon(
    traj,
    elapsed,
    horizon_steps,
    dt,
    previous_yaw,
    trajectory_duration=None,
    speed_scale=1.0,
):
    ref_seq = np.zeros((horizon_steps + 1, 9), dtype=float)
    yaw = float(previous_yaw)
    loop_time = max(1e-9, float(traj.total_time))
    if trajectory_duration is None:
        trajectory_duration = loop_time
    for k in range(horizon_steps + 1):
        absolute_time = min(
            float(speed_scale) * (elapsed + k * dt),
            float(trajectory_duration),
        )
        if absolute_time >= float(trajectory_duration):
            sample_time = loop_time
        else:
            sample_time = absolute_time % loop_time
        ref_seq[k] = traj.eval(sample_time, yaw_fallback=yaw)
        ref_seq[k, 3:6] *= float(speed_scale)
        ref_seq[k, 8] = yaw + wrap_pi(float(ref_seq[k, 8]) - yaw)
        yaw = float(ref_seq[k, 8])
    return ref_seq


def make_obstacle_horizon(tracker, now, horizon_steps, dt, stale_timeout):
    """Return None until cf_2 is seen; predict fresh cf_2 data at constant velocity."""
    obstacle = tracker.snapshot(tracker.obstacle_name)
    if obstacle is None or now - obstacle["stamp"] > float(stale_timeout):
        return None
    times = np.arange(horizon_steps + 1, dtype=float)[:, None] * float(dt)
    return obstacle["position"][None, :] + times * obstacle["velocity"][None, :]


def predict_next_position(ctrl, state, control):
    with torch.no_grad():
        x_t = torch.as_tensor(
            np.asarray(state, dtype=np.float32).reshape(1, 9),
            device=ctrl.device,
            dtype=torch.float32,
        )
        u_t = torch.as_tensor(
            np.asarray(control, dtype=np.float32).reshape(1, 4),
            device=ctrl.device,
            dtype=torch.float32,
        )
        x_next = quad_dyn_step(
            x_t, u_t, ctrl.dt, ctrl.m, ctrl.g,
            tau_roll=ctrl.p.tau_roll,
            tau_pitch=ctrl.p.tau_pitch,
            tau_yaw=ctrl.p.tau_yaw,
        )
        return x_next[0, 0:3].detach().cpu().numpy().astype(float)


def save_physical_flight_log(flight_log, output_dir):
    if not flight_log["time"]:
        print("No flight samples were collected; nothing to save.")
        return None, None

    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    data_path = os.path.join(
        output_dir, f"ra_mppi_physical_flight_{timestamp}.npz"
    )
    plot_path = os.path.join(
        output_dir, f"ra_mppi_physical_flight_{timestamp}.png"
    )
    arrays = {
        "time": np.asarray(flight_log["time"], dtype=float),
        "desired_position": np.asarray(flight_log["desired_position"], dtype=float),
        "actual_position": np.asarray(flight_log["actual_position"], dtype=float),
        "commanded_position": np.asarray(flight_log["commanded_position"], dtype=float),
        "obstacle_position": np.asarray(flight_log["obstacle_position"], dtype=float),
        "obstacle_active": np.asarray(flight_log["obstacle_active"], dtype=np.int8),
        "solve_ms": np.asarray(flight_log["solve_ms"], dtype=float),
    }
    arrays["position_error"] = np.linalg.norm(
        arrays["actual_position"] - arrays["desired_position"], axis=1
    )
    np.savez_compressed(data_path, **arrays)

    matplotlib_config_dir = os.path.join("/tmp", "mppi_matplotlib_cache")
    os.makedirs(matplotlib_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_config_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(14, 9))
    try:
        path_axis = figure.add_subplot(2, 2, 1, projection="3d")
        has_3d_axis = True
    except ValueError:
        path_axis = figure.add_subplot(2, 2, 1)
        has_3d_axis = False
    desired = arrays["desired_position"]
    actual = arrays["actual_position"]
    commanded = arrays["commanded_position"]
    obstacle = arrays["obstacle_position"]
    active = arrays["obstacle_active"].astype(bool)

    if has_3d_axis:
        path_axis.plot(
            desired[:, 0], desired[:, 1], desired[:, 2],
            "k--", linewidth=2.0, label="Desired Poly4D",
        )
        path_axis.plot(
            actual[:, 0], actual[:, 1], actual[:, 2],
            color="tab:blue", linewidth=1.5, label="Actual cf_1 (Vicon)",
        )
        path_axis.scatter(
            commanded[::10, 0], commanded[::10, 1], commanded[::10, 2],
            color="tab:green", s=8, alpha=0.6, label="MPPI position targets",
        )
        if np.any(active):
            path_axis.plot(
                obstacle[active, 0], obstacle[active, 1], obstacle[active, 2],
                color="tab:red", linewidth=1.2, label="Actual cf_2 (Vicon)",
            )
        path_axis.set_zlabel("Z [m]")
        path_axis.set_title("Desired and actual 3D flight paths")
    else:
        path_axis.plot(
            desired[:, 0], desired[:, 1],
            "k--", linewidth=2.0, label="Desired Poly4D",
        )
        path_axis.plot(
            actual[:, 0], actual[:, 1],
            color="tab:blue", linewidth=1.5, label="Actual cf_1 (Vicon)",
        )
        path_axis.scatter(
            commanded[::10, 0], commanded[::10, 1],
            color="tab:green", s=8, alpha=0.6, label="MPPI position targets",
        )
        if np.any(active):
            path_axis.plot(
                obstacle[active, 0], obstacle[active, 1],
                color="tab:red", linewidth=1.2, label="Actual cf_2 (Vicon)",
            )
        path_axis.set_aspect("equal", adjustable="datalim")
        path_axis.set_title("Desired and actual XY flight paths")
    path_axis.set_xlabel("X [m]")
    path_axis.set_ylabel("Y [m]")
    path_axis.grid(True, alpha=0.3)
    path_axis.legend(loc="best")

    labels = ("X", "Y", "Z")
    colors = ("tab:blue", "tab:orange", "tab:green")
    time_axis = arrays["time"]
    for index in range(3):
        axis = figure.add_subplot(2, 2, index + 2)
        axis.plot(
            time_axis, desired[:, index], "k--", linewidth=1.5,
            label=f"Desired {labels[index]}",
        )
        axis.plot(
            time_axis, actual[:, index], color=colors[index], linewidth=1.2,
            label=f"Actual {labels[index]}",
        )
        axis.plot(
            time_axis, commanded[:, index], color="tab:green",
            linewidth=0.8, alpha=0.7, label=f"Commanded {labels[index]}",
        )
        axis.set_xlabel("Time [s]")
        axis.set_ylabel(f"{labels[index]} [m]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    figure.suptitle(
        f"Physical RA-MPPI flight — mean position error "
        f"{np.mean(arrays['position_error']):.3f} m"
    )
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    print(f"Saved physical flight data to {data_path}")
    print(f"Saved physical flight plot to {plot_path}")
    return data_path, plot_path


def run_physical_mppi_sequence(
    args,
    traj,
    ctrl,
    q,
    qf,
    tracker,
    ego_scf,
    center,
    flight_log,
):
    """Fly the ego with MPPI while reading the obstacle only from Vicon."""
    dt = float(args.period)
    speed_scale = float(SPEED_SCALE)
    trajectory_duration = int(args.cycles) * float(traj.total_time)
    flight_duration = trajectory_duration / speed_scale + 10.0 * dt

    cf = ego_scf.cf
    commander = cf.high_level_commander
    ego_airborne = False
    collision_threshold = float(
        args.drone_radius + args.obstacle_radius
    )
    minimum_distance_seen = float("inf")
    collision_detected = False

    try:
        # Allow the external-pose callback to feed a few samples first.
        time.sleep(0.5)
        print(f"Waiting for {args.ego_body!r} estimator...")
        configure_estimator(ego_scf)

        print(f"Taking {args.ego_body!r} off to {center[2]:.2f} m...")
        commander.takeoff(float(center[2]), float(args.takeoff_duration))
        ego_airborne = True
        time.sleep(float(args.takeoff_duration) + 1.0)

        print(f"Positioning {args.ego_body!r} at the trajectory center...")
        commander.go_to(
            float(center[0]),
            float(center[1]),
            float(center[2]),
            0.0,
            float(args.initial_go_to_duration),
            relative=False,
        )
        time.sleep(float(args.initial_go_to_duration) + 0.2)

        print(
            f"RA-MPPI flight started on {ctrl.device}: "
            f"{args.ego_body!r}=RA-MPPI; "
            f"{args.obstacle_body!r}=Vicon-only obstacle. "
            f"Tracking {int(args.cycles)} cycle(s), followed by 10 final "
            "reference steps. "
            "Press Ctrl-C to land the ego."
        )
        start_time = time.monotonic()
        next_tick = start_time
        next_command_time = start_time
        command_period = 1.0 / float(args.command_rate)
        last_yaw_ref = 0.0
        last_commanded_target = center.copy()
        last_commanded_yaw = 0.0
        obstacle_was_active = False
        print(
            "Measured collision threshold: "
            f"{collision_threshold:.3f} m"
        )
        step = 0

        try:
            while True:
                now = time.monotonic()
                if tracker.error is not None:
                    raise RuntimeError("Vicon thread stopped") from tracker.error

                ego = tracker.snapshot(args.ego_body)
                obstacle = tracker.snapshot(args.obstacle_body)
                if (
                    ego is None
                    or now - ego["stamp"] > float(args.ego_timeout)
                ):
                    raise RuntimeError(
                        f"Lost fresh Vicon data for {args.ego_body!r}"
                    )
                if (
                    obstacle is None
                    or now - obstacle["stamp"] > float(args.obstacle_timeout)
                ):
                    raise RuntimeError(
                        f"Lost fresh Vicon data for {args.obstacle_body!r}"
                    )

                elapsed = now - start_time
                if elapsed >= flight_duration:
                    print("Figure-eight complete; landing ego...")
                    break

                x = np.zeros(9, dtype=float)
                x[0:3] = ego["position"]
                x[3:6] = ego["velocity"]
                x[6:9] = ego["euler"]
                ref_seq = make_reference_horizon(
                    traj,
                    elapsed,
                    ctrl.T,
                    dt,
                    last_yaw_ref,
                    trajectory_duration=trajectory_duration,
                    speed_scale=speed_scale,
                )
                last_yaw_ref = float(ref_seq[1, 8])

                # The avoidance prediction starts from measured Vicon cf_2
                # state, rather than assuming it followed its command exactly.
                obs_seq = make_obstacle_horizon(
                    tracker, now, ctrl.T, dt, args.obstacle_timeout
                )
                if obs_seq is None:
                    raise RuntimeError(
                        f"No valid obstacle horizon for "
                        f"{args.obstacle_body!r}"
                    )
                if not obstacle_was_active:
                    print(
                        f"Active obstacles: 1 ({args.obstacle_body} acquired)"
                    )
                    obstacle_was_active = True

                command_due = now >= next_command_time
                solve_start = time.perf_counter()
                if command_due:
                    _, nominal_xyz = ctrl.plan(
                        x,
                        ref_seq,
                        q,
                        qf,
                        obs_seq,
                        return_predictions=False,
                        return_nominal_prediction=True,
                    )
                    lookahead_index = min(
                        int(args.lookahead_steps), ctrl.T
                    )
                    target = np.asarray(
                        nominal_xyz[lookahead_index], dtype=float
                    )
                else:
                    ctrl.plan(
                        x,
                        ref_seq,
                        q,
                        qf,
                        obs_seq,
                        return_predictions=False,
                    )
                    target = last_commanded_target
                solve_ms = (time.perf_counter() - solve_start) * 1000.0
                if not np.all(np.isfinite(target)):
                    raise RuntimeError(
                        "MPPI produced a non-finite next position"
                    )

                if command_due:
                    yaw_index = min(int(args.lookahead_steps), ctrl.T)
                    last_commanded_target = target.copy()
                    last_commanded_yaw = float(ref_seq[yaw_index, 8])
                    while next_command_time <= now:
                        next_command_time += command_period

                cf.commander.send_position_setpoint(
                    float(last_commanded_target[0]),
                    float(last_commanded_target[1]),
                    float(last_commanded_target[2]),
                    math.degrees(last_commanded_yaw),
                )

                distance = float(
                    np.linalg.norm(x[0:3] - obstacle["position"])
                )
                minimum_distance_seen = min(
                    minimum_distance_seen, distance
                )
                if (
                    distance <= collision_threshold
                    and not collision_detected
                ):
                    collision_detected = True
                    print(
                        "\n"
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                        f"COLLISION FLAG: {args.ego_body} and "
                        f"{args.obstacle_body} collided\n"
                        f"step={step}, time={elapsed:.3f} s, "
                        f"distance={distance:.3f} m, "
                        f"threshold={collision_threshold:.3f} m\n"
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    )
                flight_log["time"].append(float(elapsed))
                flight_log["desired_position"].append(
                    ref_seq[0, 0:3].copy()
                )
                flight_log["actual_position"].append(x[0:3].copy())
                flight_log["commanded_position"].append(target.copy())
                flight_log["obstacle_position"].append(
                    obstacle["position"].copy()
                )
                flight_log["obstacle_active"].append(1)
                flight_log["solve_ms"].append(float(solve_ms))

                if step % 10 == 0:
                    print(
                        f"[step {step:05d}] solve={solve_ms:.1f} ms "
                        f"distance={distance:.3f} "
                        f"collision={int(collision_detected)} "
                        f"actual=({x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}) "
                        f"obstacle=({obstacle['position'][0]:.3f}, "
                        f"{obstacle['position'][1]:.3f}, "
                        f"{obstacle['position'][2]:.3f}) "
                        f"target=({target[0]:.3f}, "
                        f"{target[1]:.3f}, {target[2]:.3f})"
                    )

                step += 1
                next_tick += dt
                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.monotonic()
        except KeyboardInterrupt:
            print("Landing requested for ego...")
    finally:
        minimum_text = (
            f"{minimum_distance_seen:.3f} m"
            if math.isfinite(minimum_distance_seen)
            else "not available"
        )
        print(
            "Collision summary: "
            f"collided={int(collision_detected)}, "
            f"minimum_distance={minimum_text}, "
            f"threshold={collision_threshold:.3f} m"
        )
        # Release low-level position setpoints before high-level landing.
        try:
            cf.commander.send_notify_setpoint_stop()
        except Exception as exc:
            print(f"Could not release position setpoint mode: {exc}")
        time.sleep(0.1)

        if ego_airborne:
            try:
                commander.land(0.0, float(args.land_duration))
            except Exception as exc:
                print(f"Could not command {args.ego_body} landing: {exc}")
        if ego_airborne:
            time.sleep(float(args.land_duration) + 0.5)
        try:
            commander.stop()
        except Exception:
            pass


def run_physical_flight(args):
    dt = float(args.period)
    if dt <= 0.0:
        raise ValueError("--period must be positive")
    if float(args.command_rate) <= 0.0:
        raise ValueError("--command-rate must be positive")
    if int(args.lookahead_steps) < 1:
        raise ValueError("--lookahead-steps must be at least 1")
    if int(args.horizon) < int(args.lookahead_steps):
        raise ValueError("--horizon must be at least --lookahead-steps")
    if float(SPEED_SCALE) <= 0.0:
        raise ValueError("SPEED_SCALE must be positive")
    center = np.array([args.center_x, args.center_y, args.center_z], dtype=float)
    traj = MocapPoly4DFigure8Reference(MOCAP_FIGURE8, center=center)
    if int(args.cycles) < 1:
        raise ValueError("--cycles must be at least 1")
    params = RAMPPIParams(
        dt=dt,
        horizon_steps=int(args.horizon),
        rollouts=int(args.rollouts),
        lam=5.0,
        sigma=np.array([0.1, 0.1, 0.01, 0.0807464837], dtype=np.float32),
        T_max=0.4776,
        ang_max=math.radians(20.0),
        yaw_max=math.radians(20.0),
        tau_roll=0.22,
        tau_pitch=0.19,
        tau_yaw=0.22,
        drone_radius=float(args.drone_radius),
        moving_r=float(args.obstacle_radius),
        moving_safety_margin=float(args.safety_margin),
        moving_alpha=12.0,
        obs_pos_sigma_xyz=(0.01, 0.01, 0.01),
        noise_mode="per_step",
        cvar_alpha=0.95,
        cvar_N=100,
        risk_cost_A=2000.0,
        risk_cost_Cu=0.0,
        R_u=(5.7, 8.7, 3.1, 0.01),
        Rd_u=(5.7, 8.7, 3.1, 0.01),
    )
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    print(
        "[RA_MPPI] "
        f"alpha={params.cvar_alpha:.3f}, samples={params.cvar_N}"
    )
    ctrl = TorchRAMPPIQuadOuter(
        mass=0.028, g=9.81, params=params, device=device
    )
    q = np.array([1008, 1000, 1008, 25, 30, 13, 0.01, 0.01, 0.01], dtype=np.float32)
    qf = q.copy()

    tracker = ViconTracker(
        args.vicon_host,
        ego_name=args.ego_body,
        obstacle_name=args.obstacle_body,
    )
    flight_log = {
        "time": [],
        "desired_position": [],
        "actual_position": [],
        "commanded_position": [],
        "obstacle_position": [],
        "obstacle_active": [],
        "solve_ms": [],
    }
    try:
        print(f"Waiting for Vicon rigid body {args.ego_body!r}...")
        tracker.wait_for_body(args.ego_body, args.vicon_timeout)
        print(f"Waiting for Vicon rigid body {args.obstacle_body!r}...")
        tracker.wait_for_body(args.obstacle_body, args.vicon_timeout)
        cflib.crtp.init_drivers()
        ego_uri = uri_helper.uri_from_env(default=args.uri)
        print(f"Connecting ego {args.ego_body!r} on {ego_uri}...")
        with SyncCrazyflie(
            ego_uri, cf=Crazyflie(rw_cache="./cache")
        ) as ego_scf:
            set_led_ring(ego_scf.cf, 0, 0, 255)
            print("Ego LED ring: blue")
            tracker.on_ego_pose = make_extpose_forwarder(
                ego_scf.cf, rate_hz=50.0
            )
            try:
                run_physical_mppi_sequence(
                    args,
                    traj,
                    ctrl,
                    q,
                    qf,
                    tracker,
                    ego_scf,
                    center,
                    flight_log,
                )
            finally:
                tracker.on_ego_pose = None
    finally:
        tracker.close()
        tracker.join(timeout=1.0)
        if not args.no_save:
            try:
                save_physical_flight_log(flight_log, args.output_dir)
            except Exception as exc:
                print(f"Could not save physical flight plot: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Fly physical cf_1 with RA-MPPI and read cf_2 obstacle data from "
            "Vicon. Run obstacle.py separately to fly cf_2."
        )
    )
    parser.add_argument("--uri", default="radio://0/80/2M/E7E7E7E701")
    parser.add_argument("--vicon-host", default="169.254.134.206")
    parser.add_argument("--ego-body", default="cf_1")
    parser.add_argument("--obstacle-body", default="cf_2")
    parser.add_argument("--center-x", type=float, default=0.0)
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--center-z", type=float, default=0.5)
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Number of figure-eight cycles before automatic landing.",
    )
    parser.set_defaults(
        period=0.02,
        command_rate=20.0,
        lookahead_steps=25,
        horizon=25,
        rollouts=2000,
        obstacle_radius=0.044,
        drone_radius=0.046,
        safety_margin=0.10,
        obstacle_weight=50.0,
        obstacle_timeout=0.25,
        ego_timeout=0.25,
        vicon_timeout=10.0,
        takeoff_duration=3.0,
        initial_go_to_duration=3.0,
        land_duration=2.0,
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "plot"),
        help="Directory for post-flight .npz data and .png plots.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save post-flight data or plots.",
    )
    parser.add_argument("--use-gpu", "--cuda", dest="use_gpu", action="store_true", default=True)
    parser.add_argument("--cpu", dest="use_gpu", action="store_false")
    run_physical_flight(parser.parse_args())
