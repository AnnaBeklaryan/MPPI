#!/usr/bin/env python3
"""
Autotune DR_mppi_crazyflie.py MPPI parameters with SHADE.

Tuned parameters:
  - Q, with Qf forced to Q
  - R_u, with Rd_u forced to R_u
  - horizon_steps
  - lam
  - sigma

python3 autotune.py --steps 120 --rollouts 1200 --epoch 100 --pop-size 20 --json-out t
uned_dr_cf.json --use_gpu
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

try:
    from mealpy import FloatVar, SHADE
except ModuleNotFoundError:
    FloatVar = None
    SHADE = None

from DR_mppi_crazyflie import (
    DRParams,
    SimpleCrazyflieReference,
    TorchDRMPPIQuadOuter,
    clamp,
    wrap_pi,
)


BASE_PARAM = {
    "Q": [
        1008.6549649,
        1000.1692188,
        1008.5572027,
        25.6443361,
        30.0,
        1.39585633,
        0.01,
        0.01,
        0.01,
    ],
    "R_u": [5.79609690, 8.72231596, 3.1, 0.01],
    "horizon_steps": 10,
    "lam": 3.29599886,
    "sigma": [0.1, 0.1, 0.01, 0.0807464837],
}


def _local_bounds(base: dict, scale: float = 0.5) -> tuple[dict, dict]:
    """Build positive local search bounds around the currently working params."""
    scale = float(max(0.05, scale))

    def bounds_list(values, min_floor):
        lower = []
        upper = []
        for raw in values:
            v = float(raw)
            if v <= min_floor * 2.0:
                lower.append(float(min_floor))
                upper.append(float(max(min_floor * 10.0, v + scale)))
            else:
                lower.append(float(max(min_floor, v * (1.0 - scale))))
                upper.append(float(v * (1.0 + scale)))
        return lower, upper

    q_lb, q_ub = bounds_list(base["Q"], 0.01)
    r_lb, r_ub = bounds_list(base["R_u"], 0.01)
    sigma_lb, sigma_ub = bounds_list(base["sigma"], 0.001)
    h = int(base["horizon_steps"])
    lam = float(base["lam"])
    lower = {
        "Q": q_lb,
        "R_u": r_lb,
        "horizon_steps": max(6, h - 4),
        "lam": max(0.05, lam * (1.0 - scale)),
        "sigma": sigma_lb,
    }
    upper = {
        "Q": q_ub,
        "R_u": r_ub,
        "horizon_steps": h + 6,
        "lam": lam * (1.0 + scale),
        "sigma": sigma_ub,
    }
    return lower, upper


class DRMppiCrazyflieTune:
    def __init__(
        self,
        upper_bound: dict | None = None,
        lower_bound: dict | None = None,
        local_scale: float = 0.5,
        steps: int = 120,
        rollouts: int = 600,
        use_gpu: bool = False,
        seed: int = 0,
    ):
        self.state_dim = 9
        self.action_dim = 4
        self.steps = int(steps)
        self.rollouts = int(rollouts)
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self.rng = np.random.default_rng(seed)

        if upper_bound is None:
            lower_bound, upper_bound = _local_bounds(BASE_PARAM, scale=local_scale)
        if lower_bound is None:
            lower_bound, _ = _local_bounds(BASE_PARAM, scale=local_scale)

        self.upper_bound_dict = upper_bound
        self.lower_bound_dict = lower_bound
        self.upper_bound = self.decode_param(upper_bound)
        self.lower_bound = self.decode_param(lower_bound)

    @staticmethod
    def decode_param(param: dict) -> list[float]:
        x: list[float] = []
        for value in param.values():
            if isinstance(value, list):
                x.extend(value)
            else:
                x.append(value)
        return x

    def encode_param(self, x) -> dict:
        x = np.asarray(x, dtype=float).ravel()
        q_end = self.state_dim
        r_end = q_end + self.action_dim
        h_idx = r_end
        lam_idx = h_idx + 1
        sigma_start = lam_idx + 1
        sigma_end = sigma_start + self.action_dim

        Q = np.asarray(x[:q_end], dtype=np.float32)
        R_u = np.asarray(x[q_end:r_end], dtype=np.float32)
        horizon_steps = int(np.clip(round(float(x[h_idx])), 1, 200))
        lam = float(max(1e-6, x[lam_idx]))
        sigma = np.asarray(x[sigma_start:sigma_end], dtype=np.float32)

        return {
            "Q": Q,
            "Qf": Q.copy(),
            "R_u": R_u,
            "Rd_u": R_u.copy(),
            "horizon_steps": horizon_steps,
            "lam": lam,
            "sigma": sigma,
        }

    @staticmethod
    def _z_body_world(roll: float, pitch: float, yaw: float) -> np.ndarray:
        croll = math.cos(roll)
        sroll = math.sin(roll)
        cpitch = math.cos(pitch)
        spitch = math.sin(pitch)
        cyaw = math.cos(yaw)
        syaw = math.sin(yaw)
        return np.array(
            [
                cyaw * spitch * croll + syaw * sroll,
                syaw * spitch * croll - cyaw * sroll,
                cpitch * croll,
            ],
            dtype=float,
        )

    @staticmethod
    def _build_ref_seq(traj, t: float, dt: float, horizon: int, last_yaw_ref: float):
        ref_seq = np.zeros((horizon + 1, 9), dtype=float)
        loop_T = max(1e-9, float(traj.total_time))
        ref_seq[0] = traj.eval(t % loop_T, yaw_fallback=last_yaw_ref)
        ref_seq[0, 8] = last_yaw_ref + wrap_pi(float(ref_seq[0, 8]) - last_yaw_ref)
        for k in range(1, horizon + 1):
            prev = float(ref_seq[k - 1, 8])
            ref_seq[k] = traj.eval((t + k * dt) % loop_T, yaw_fallback=prev)
            ref_seq[k, 8] = prev + wrap_pi(float(ref_seq[k, 8]) - prev)
        return ref_seq

    @staticmethod
    def _build_obs_seq(moving_trajs, moving_time_offsets, t: float, dt: float, horizon: int, lead_time: float):
        n_obs = len(moving_trajs)
        obs_seq = np.zeros((horizon + 1, n_obs, 3), dtype=float)
        for j, moving_traj in enumerate(moving_trajs):
            loop_T = max(1e-9, float(moving_traj.total_time))
            for k in range(horizon + 1):
                tk = t + lead_time + k * dt - float(moving_time_offsets[j])
                obs_seq[k, j] = moving_traj.eval(tk % loop_T)[0:3]
        return obs_seq

    def _make_scene(self, dt: float):
        trajectory_type = "tilted_figure8"
        trajectory_center = np.array([0.0, 0.0, 1.0], dtype=float)
        ego_ref_speed = 0.4
        obstacle_ref_speed = 0.4

        traj = SimpleCrazyflieReference(
            start_position=trajectory_center,
            dt=dt,
            trajectory_type=trajectory_type,
            hover_begin=0.0,
            trajectory_speed=ego_ref_speed,
        )
        moving_trajs = [
            SimpleCrazyflieReference(
                start_position=trajectory_center,
                dt=dt,
                trajectory_type=trajectory_type,
                hover_begin=0.0,
                trajectory_speed=obstacle_ref_speed,
                reverse_direction=True,
            ),
            SimpleCrazyflieReference(
                start_position=trajectory_center,
                dt=dt,
                trajectory_type=trajectory_type,
                hover_begin=0.0,
                trajectory_speed=obstacle_ref_speed,
                reverse_direction=True,
            ),
        ]
        moving_time_offsets = np.array([0.0, moving_trajs[1].total_time * 0.25], dtype=float)
        return traj, moving_trajs, moving_time_offsets

    def obj_func(self, solution):
        p = self.encode_param(solution)
        dt = 0.02
        params = DRParams(
            dt=dt,
            horizon_steps=int(p["horizon_steps"]),
            rollouts=self.rollouts,
            lam=float(p["lam"]),
            ang_max=math.radians(50),
            yaw_max=math.radians(50),
            tau_roll=0.08,
            tau_pitch=0.08,
            tau_yaw=0.08,
            max_thrust=0.4776,
            sigma=np.asarray(p["sigma"], dtype=np.float32),
            w_cyl=80,
            cyl_safety_margin=0.215,
            cyl_alpha=5.0,
            moving_r=0.05,
            moving_safety_margin=0.05,
            cvar_alpha=0.95,
            cvar_N=60,
            obs_pos_sigma_xyz=(0.07, 0.07, 0.07),
            noise_mode="per_step",
            dr_eps_cvar=0.08,
            risk_cost_A=1000.0,
            risk_cost_Cu=0.0,
            w_terminal_obs=400.0,
            terminal_obs_alpha=8.0,
            drone_radius=0.05,
            R_u=tuple(float(v) for v in p["R_u"]),
            Rd_u=tuple(float(v) for v in p["Rd_u"]),
        )

        try:
            ctrl = TorchDRMPPIQuadOuter(mass=0.028, g=9.80665, params=params, cylinders=[], device=self.device)
            traj, moving_trajs, moving_time_offsets = self._make_scene(dt)
            Q = np.asarray(p["Q"], dtype=np.float32)
            Qf = np.asarray(p["Qf"], dtype=np.float32)
            eval_Q = np.asarray(BASE_PARAM["Q"], dtype=np.float32)

            x = np.zeros(9, dtype=float)
            x[0:3] = traj.eval(0.0)[0:3]
            last_yaw_ref = float(x[8])
            lead_time = 1.5495997771078638
            moving_collision_radius = float(params.moving_r + params.drone_radius)
            moving_safe_radius = float(params.moving_r + params.moving_safety_margin + params.drone_radius)

            total_cost = 0.0
            for i in range(self.steps):
                t = i * dt
                ref_seq = self._build_ref_seq(traj, t, dt, ctrl.T, last_yaw_ref)
                last_yaw_ref = float(ref_seq[min(1, ctrl.T), 8])
                obs_seq = self._build_obs_seq(moving_trajs, moving_time_offsets, t, dt, ctrl.T, lead_time)
                u = ctrl.plan(x, ref_seq, Q, Qf, obs_seq, return_predictions=False)

                roll_c, pitch_c, yaw_c, thrust = map(float, u)
                roll_c = clamp(roll_c, -params.ang_max, params.ang_max)
                pitch_c = clamp(pitch_c, -params.ang_max, params.ang_max)
                yaw_c = clamp(yaw_c, -params.yaw_max, params.yaw_max)
                thrust = clamp(thrust, ctrl.u_min[3], ctrl.u_max[3])

                roll, pitch, yaw = float(x[6]), float(x[7]), float(x[8])
                zb = self._z_body_world(roll, pitch, yaw)
                a = (thrust / ctrl.m) * zb - np.array([0.0, 0.0, ctrl.g])
                x[3:6] = x[3:6] + dt * a
                x[0:3] = x[0:3] + dt * x[3:6]
                x[6] = x[6] + dt * ((roll_c - roll) / max(1e-6, params.tau_roll))
                x[7] = x[7] + dt * ((pitch_c - pitch) / max(1e-6, params.tau_pitch))
                x[8] = x[8] + dt * ((yaw_c - yaw) / max(1e-6, params.tau_yaw))

                err = x - ref_seq[0]
                err[6:9] = np.array([wrap_pi(float(v)) for v in err[6:9]])
                total_cost += float(np.sum((err * err) * eval_Q))
                total_cost += 0.02 * float(np.sum(np.asarray(u, dtype=float) ** 2))

                dists = np.linalg.norm(obs_seq[0] - x[None, 0:3], axis=1)
                min_dist = float(np.min(dists))
                if min_dist < moving_safe_radius:
                    total_cost += 2500.0 * (moving_safe_radius - min_dist) ** 2
                if min_dist < moving_collision_radius:
                    total_cost += 50000.0 * (moving_collision_radius - min_dist + 1.0)

                if not np.all(np.isfinite(x)):
                    return 1e12

            return float(total_cost)
        except Exception as exc:
            print(f"[autotune] failed solution: {exc}")
            return 1e12


def shade_tune(
    steps: int = 120,
    rollouts: int = 600,
    epoch: int = 100,
    pop_size: int = 20,
    use_gpu: bool = False,
    local_scale: float = 0.5,
    log_file: str = "dr_crazyflie_shade_tune.log",
):
    if FloatVar is None or SHADE is None:
        raise ModuleNotFoundError(
            "mealpy is required for SHADE tuning. Install it in this environment, "
            "or run this script from the RL_LNN environment that already has mealpy."
        )

    tuner = DRMppiCrazyflieTune(
        steps=steps,
        rollouts=rollouts,
        use_gpu=use_gpu,
        local_scale=local_scale,
    )
    problem_dict = {
        "bounds": FloatVar(lb=tuner.lower_bound, ub=tuner.upper_bound),
        "obj_func": tuner.obj_func,
        "minmax": "min",
        "log_to": "file",
        "log_file": log_file,
    }
    model = SHADE.L_SHADE(epoch=int(epoch), pop_size=int(pop_size), miu_f=0.5, miu_cr=0.5)
    best = model.solve(problem_dict)
    best_param = tuner.encode_param(best.solution)
    printable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best_param.items()}
    print(f"results: {json.dumps(printable, indent=2)}")
    print(f"cost: {best.target.fitness}")
    return best_param, best.target.fitness


def main():
    parser = argparse.ArgumentParser(description="Tune DR_mppi_crazyflie.py MPPI parameters with SHADE.")
    parser.add_argument("--steps", type=int, default=120, help="Simulation steps per candidate.")
    parser.add_argument("--rollouts", type=int, default=600, help="MPPI rollouts per candidate.")
    parser.add_argument("--epoch", type=int, default=100, help="SHADE epochs.")
    parser.add_argument("--pop-size", type=int, default=20, help="SHADE population size.")
    parser.add_argument("--use_gpu", "--cuda", dest="use_gpu", action="store_true", default=False)
    parser.add_argument(
        "--local-scale",
        type=float,
        default=0.5,
        help="Local search width around current DR_mppi_crazyflie.py values, e.g. 0.25 means about +/-25%%.",
    )
    parser.add_argument("--log-file", type=str, default="dr_crazyflie_shade_tune.log")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save best parameters as JSON.")
    args = parser.parse_args()

    best_param, best_cost = shade_tune(
        steps=args.steps,
        rollouts=args.rollouts,
        epoch=args.epoch,
        pop_size=args.pop_size,
        use_gpu=args.use_gpu,
        local_scale=args.local_scale,
        log_file=args.log_file,
    )
    if args.json_out:
        out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best_param.items()}
        out["cost"] = float(best_cost)
        Path(args.json_out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
