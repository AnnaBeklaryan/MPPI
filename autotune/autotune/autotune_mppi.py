#!/usr/bin/env python3
"""
Auto-tune MPPI hyperparameters with MEALPY (L-SHADE) for your MPPI_Torch_Generic.

This version is wired for your diff-drive MPPI:
  state:  x = [px, py, psi]  -> state_dim = 3
  action: u = [v, w]         -> action_dim = 2

You must provide an env factory:  --env-factory my_env_module:make_env
Env must implement:
  - reset() -> obs (np.ndarray shape (3,))
  - step(u) -> obs, reward, done, info
  - dt attribute (float)

We tune:
  Q (3,), R (2,), T (int), noise_sigma (2,), lambda (scalar)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Callable, Any

import numpy as np
import torch
from mealpy import SHADE, FloatVar

from mppi_torch import (
    MPPI_Torch_Generic,
    dyn_diffdrive,
    running_cost_lane_obs,
    terminal_cost_track,
)


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Helper: dynamic import
# -------------------------
def load_callable(spec: str) -> Callable[[], Any]:
    """
    spec: "some_module:callable_name"
    returns: the callable object
    """
    if ":" not in spec:
        raise ValueError(f"--env-factory must be like module:callable, got: {spec!r}")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Could not find callable {fn_name!r} in module {mod_name!r}")
    return fn


# -------------------------
# Parameter encoding/decoding
# -------------------------
@dataclass(frozen=True)
class TuneSpec:
    # IMPORTANT: diff-drive state is 3D: [px, py, psi]
    state_dim: int = 3
    # action is 2D: [v, w]
    action_dim: int = 2

    def flat_size(self) -> int:
        # Q(state_dim) + R(action_dim) + T(1) + sigma(action_dim) + lambda(1)
        return self.state_dim + self.action_dim + 1 + self.action_dim + 1

    def decode_bounds(self, bounds: Dict[str, Any]) -> List[float]:
        """
        bounds example:
          {"Q":[...3...], "R":[...2...], "T":20, "noise_sigma":[...2...], "lambda":6.0}
        """
        Q = list(bounds["Q"])
        R = list(bounds["R"])
        T = float(bounds["T"])
        sigma = list(bounds["noise_sigma"])
        lam = float(bounds["lambda"])

        x = Q + R + [T] + sigma + [lam]
        if len(x) != self.flat_size():
            raise ValueError(f"Bounds vector has wrong length {len(x)} != {self.flat_size()}")
        return [float(v) for v in x]

    def encode_param(self, x: np.ndarray) -> Dict[str, Any]:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.flat_size():
            raise ValueError(f"Decision vector length {x.size} != {self.flat_size()}")

        # layout:
        # Q[0:3], R[3:5], T[5], sigma[6:8], lam[8]
        Q = x[: self.state_dim].tolist()
        kR = self.state_dim + self.action_dim
        R = x[self.state_dim : kR].tolist()

        T_raw = float(x[kR])
        T = int(np.clip(int(np.rint(T_raw)), 1, 10_000))

        sigma = x[kR + 1 : kR + 1 + self.action_dim].tolist()
        lam = float(x[kR + 1 + self.action_dim])

        # safety clamps
        lam = max(lam, 1e-9)
        sigma = [max(s, 1e-9) for s in sigma]
        R = [max(r, 0.0) for r in R]
        Q = [max(q, 0.0) for q in Q]

        return {"Q": Q, "R": R, "T": T, "noise_sigma": sigma, "lambda": lam}


# -------------------------
# Objective (fitness) function
# -------------------------
def rollout_cost(env_factory, spec: TuneSpec, x, steps: int, device):
    """
    Evaluate candidate x by running closed-loop MPPI with your MPPI_Torch_Generic.
    """
    params = spec.encode_param(x)

    env = env_factory()
    obs = env.reset()  # expected shape (3,)

    # Fixed constants (not tuned here)
    M = 1200
    lane_y = 1.0
    lane_psi = 0.0
    L_ref = 14.0
    v_des = 5.0

    # Tuned variables
    T = int(params["T"])
    lam = float(params["lambda"])
    sigma = np.asarray(params["noise_sigma"], dtype=np.float32)

    # Control bounds (fixed)
    u_min = np.asarray([0.0, -np.deg2rad(180.0)], dtype=np.float32)
    u_max = np.asarray([10.0,  np.deg2rad(180.0)], dtype=np.float32)

    mppi = MPPI_Torch_Generic(
        dt=float(env.dt),
        T=T,
        M=M,
        lam=lam,
        noise_sigma=sigma,
        u_min=u_min,
        u_max=u_max,
        dynamics=dyn_diffdrive,
        running_cost=running_cost_lane_obs,
        terminal_cost=terminal_cost_track,
        device=device,
        dyn_kwargs=dict(w_max=float(np.deg2rad(180.0))),
        cost_kwargs=dict(ref=None, Q=None, R=None, Qf=None, O_mean=None, radii=None, obs_w=5e3),
        I=1,
        verbose=False,
    )

    # weights must match state_dim=3 and action_dim=2
    Q_np = np.asarray(params["Q"], dtype=np.float32)          # (3,)
    R_np = np.asarray(params["R"], dtype=np.float32)          # (2,)
    Qf_np = np.asarray(2.0 * Q_np, dtype=np.float32)          # (3,) fallback

    Q = torch.as_tensor(Q_np, device=mppi.device, dtype=mppi.dtype)
    R = torch.as_tensor(R_np, device=mppi.device, dtype=mppi.dtype)
    Qf = torch.as_tensor(Qf_np, device=mppi.device, dtype=mppi.dtype)

    total_cost = 0.0

    for _k in range(int(steps)):
        x0 = np.asarray(obs, dtype=np.float32).reshape(3)

        # ref: lookahead in x, stick to lane center y, keep psi ~ 0
        ref_np = np.array([float(x0[0]) + L_ref, lane_y, lane_psi], dtype=np.float32)

        mppi.cost_kwargs["ref"] = torch.as_tensor(ref_np, device=mppi.device, dtype=mppi.dtype)
        mppi.cost_kwargs["Q"] = Q
        mppi.cost_kwargs["R"] = R
        mppi.cost_kwargs["Qf"] = Qf
        mppi.cost_kwargs["O_mean"] = None
        mppi.cost_kwargs["radii"] = None

        U, _ = mppi.plan(x0, return_samples=False)
        U = np.nan_to_num(U, nan=0.0, posinf=u_max[0], neginf=u_min[0]).astype(np.float32)

        u0 = U[0].copy()
        u0[0] = np.clip(0.4 * u0[0] + 0.6 * v_des, 0.0, float(u_max[0]))
        u0 = np.nan_to_num(u0, nan=0.0, posinf=u_max[0], neginf=u_min[0]).astype(np.float32)

        obs, _reward, done, _info = env.step(u0)

        # CPU cost consistent with running_cost (vector-weight form)
        e = np.asarray(obs, dtype=np.float32).reshape(3) - ref_np
        e[2] = (e[2] + np.pi) % (2 * np.pi) - np.pi
        stage = float(np.sum((e * e) * Q_np) + np.sum((u0 * u0) * R_np))
        total_cost += stage

        # warm start shift
        mppi.U[:-1] = mppi.U[1:]
        mppi.U[-1] = np.array([v_des, 0.0], dtype=np.float32)
        mppi.U_cpu = mppi.U

        if done:
            break

    return float(total_cost)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-factory", required=True,
                    help="Python import spec like 'my_env_module:make_env' that returns a fresh env instance.")
    ap.add_argument("--steps", type=int, default=400, help="Max rollout steps per evaluation.")
    ap.add_argument("--epoch", type=int, default=50, help="Optimizer epochs (generations).")
    ap.add_argument("--pop", type=int, default=20, help="Population size.")
    ap.add_argument("--min-f", type=float, default=0.5, help="Initial mean mutation factor for L-SHADE.")
    ap.add_argument("--min-cr", type=float, default=0.5, help="Initial mean crossover rate for L-SHADE.")
    ap.add_argument("--seed", type=int, default=1, help="Random seed.")
    ap.add_argument("--log-file", type=str, default="shade_tune.log", help="MEALPY log file path.")
    ap.add_argument("--save-json", type=str, default="best_mppi_params.json", help="Where to save best parameters.")
    args = ap.parse_args()

    set_seed(args.seed)

    # Mac (MPS) / CPU / CUDA: MPPI_Torch_Generic supports torch.device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    env_factory = load_callable(args.env_factory)

    # IMPORTANT: diff-drive spec
    spec = TuneSpec(state_dim=3, action_dim=2)

    # Bounds for diff-drive:
    # Q = [qx, qy, qpsi]
    # R = [rv, rw]
    upper = {
        "Q": [600.0, 600.0, 100.0],
        "R": [5.0, 5.0],
        "T": 40.0,
        "noise_sigma": [2.0, float(np.deg2rad(60.0))],
        "lambda": 10.0,
    }
    lower = {
        "Q": [0.001, 0.001, 0.001],
        "R": [0.00001, 0.00001],
        "T": 10.0,
        "noise_sigma": [0.001, float(np.deg2rad(1.0))],
        "lambda": 0.0001,
    }

    lb = spec.decode_bounds(lower)
    ub = spec.decode_bounds(upper)

    def obj(x: np.ndarray) -> float:
        return rollout_cost(env_factory, spec, x, steps=args.steps, device=device)

    problem = {
        "bounds": FloatVar(lb=lb, ub=ub),
        "obj_func": obj,
        "minmax": "min",
        "log_to": "file",
        "log_file": args.log_file,
    }

    model = SHADE.L_SHADE(epoch=args.epoch, pop_size=args.pop, miu_f=args.min_f, miu_cr=args.min_cr)
    best = model.solve(problem)

    best_param = spec.encode_param(best.solution)
    best_cost = float(best.target.fitness)

    out = {
        "best_cost": best_cost,
        "best_param": best_param,
        "decision_vector": [float(v) for v in np.asarray(best.solution).ravel().tolist()],
        "meta": {
            "steps": args.steps,
            "epoch": args.epoch,
            "pop": args.pop,
            "seed": args.seed,
            "device": str(device),
        },
    }

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== DONE ===")
    print("Best cost:", best_cost)
    print("Best params saved to:", os.path.abspath(args.save_json))
    print("Best params:", json.dumps(best_param, indent=2))


if __name__ == "__main__":
    main()