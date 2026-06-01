#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import torch
from car_geometry import rectangle_signed_distance_torch

Tensor = torch.Tensor


def _as_torch(x, device: torch.device, dtype: torch.dtype) -> Tensor | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device, dtype=dtype)


def _wrap_angle_t(th: Tensor) -> Tensor:
    return (th + torch.pi) % (2.0 * torch.pi) - torch.pi


def diffdrive_dynamics_cpu(
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    v_min: float = 0.0,
    v_max: float = 10.0,
    w_max: float = float(np.deg2rad(180.0)),
) -> np.ndarray:
    px, py, psi = float(x[0]), float(x[1]), float(x[2])
    v, w = float(u[0]), float(u[1])

    v = np.clip(v, v_min, v_max)
    w = np.clip(w, -w_max, w_max)

    px += dt * v * np.cos(psi)
    py += dt * v * np.sin(psi)
    psi = (psi + np.pi) % (2.0 * np.pi) - np.pi
    psi = (psi + dt * w + np.pi) % (2.0 * np.pi) - np.pi
    return np.array([px, py, psi], dtype=np.float32)


class DRA_MPPI:
    def __init__(
        self,
        dt: float,
        T: int,
        M: int,
        lam: float,
        sigma: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
        sigma_cp: float = 0.05,
        Nmc: int = 2000,
        omega_soft: float = 10.0,
        omega_hard: float = 1000.0,
        obs_pos_sigma: tuple[float, float] = (0.015, 0.015),
        mc_chunk: int = 1000,
        seed: int = 3,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.dt = float(dt)
        self.T = int(T)
        self.M = int(M)
        self.lam = float(lam)
        self.device = torch.device(device)
        self.dtype = dtype

        self.noise_sigma = _as_torch(np.asarray(sigma, dtype=np.float32), self.device, self.dtype).reshape(-1)
        self.Q = _as_torch(np.asarray(Q, dtype=np.float32), self.device, self.dtype).reshape(-1)
        self.R = _as_torch(np.asarray(R, dtype=np.float32), self.device, self.dtype).reshape(-1)
        self.Qf = _as_torch(np.asarray(Qf, dtype=np.float32), self.device, self.dtype).reshape(-1)
        self.u_min = _as_torch(np.asarray(u_min, dtype=np.float32), self.device, self.dtype).reshape(-1)
        self.u_max = _as_torch(np.asarray(u_max, dtype=np.float32), self.device, self.dtype).reshape(-1)

        self.nu = int(self.u_min.numel())
        self._u_min_112 = self.u_min.view(1, 1, self.nu)
        self._u_max_112 = self.u_max.view(1, 1, self.nu)

        self.sigma_cp = float(sigma_cp)
        self.Nmc = int(Nmc)
        self.omega_soft = float(omega_soft)
        self.omega_hard = float(omega_hard)
        self.obs_pos_sigma = _as_torch(np.asarray(obs_pos_sigma, dtype=np.float32), self.device, self.dtype).reshape(2)
        self.mc_chunk = int(mc_chunk)

        self._sigma_cp_t = torch.tensor(self.sigma_cp, device=self.device, dtype=self.dtype)
        self._omega_soft_t = torch.tensor(self.omega_soft, device=self.device, dtype=self.dtype)
        self._omega_hard_t = torch.tensor(self.omega_hard, device=self.device, dtype=self.dtype)

        self.U_cpu = np.zeros((self.T, self.nu), dtype=np.float32)
        self.U = self.U_cpu

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(int(seed))

        # Straight-road boundary settings.
        self.y_min = None
        self.y_max = None
        self.boundary_w = 0.0
        self.boundary_k = 50.0
        self.boundary_d = 0.05
        self._boundary_w_t = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Scenario-2 path corridor settings.
        self.path_xy = None
        self.path_boundary_w = 0.0
        self.path_boundary_k = 35.0
        self.path_corridor_radius = 0.45

    def _dyn_diffdrive(self, X: Tensor, U: Tensor) -> Tensor:
        px, py, psi = X[:, 0], X[:, 1], X[:, 2]
        v = torch.clamp(U[:, 0], self.u_min[0], self.u_max[0])
        w = torch.clamp(U[:, 1], self.u_min[1], self.u_max[1])

        px = px + self.dt * v * torch.cos(psi)
        py = py + self.dt * v * torch.sin(psi)
        psi = _wrap_angle_t(psi + self.dt * w)
        return torch.stack([px, py, psi], dim=1)

    def _running_cost(self, X: Tensor, U: Tensor, t: int, ref: Tensor) -> Tensor:
        del t

        e = X - ref.unsqueeze(0)
        e[:, 2] = _wrap_angle_t(e[:, 2])
        cost = (e * e) @ self.Q + torch.sum((U * U) * self.R.unsqueeze(0), dim=1)

        if self.boundary_w != 0.0 and self.y_min is not None and self.y_max is not None:
            y = X[:, 1]
            lower_barrier = torch.sigmoid(float(self.boundary_k) * (self.y_min + float(self.boundary_d) - y))
            upper_barrier = torch.sigmoid(float(self.boundary_k) * (y - self.y_max + float(self.boundary_d)))
            cost = cost + float(self.boundary_w) * (lower_barrier + upper_barrier)

        if self.path_boundary_w != 0.0 and self.path_xy is not None:
            path_points = self.path_xy.to(device=X.device, dtype=X.dtype)
            ego_xy = X[:, :2]
            dx_path = ego_xy[:, None, 0] - path_points[None, :, 0]
            dy_path = ego_xy[:, None, 1] - path_points[None, :, 1]
            dist_to_path = torch.sqrt(dx_path * dx_path + dy_path * dy_path + 1e-9)
            min_dist_to_path = torch.min(dist_to_path, dim=1).values
            path_barrier = torch.sigmoid(
                float(self.path_boundary_k) * (min_dist_to_path - float(self.path_corridor_radius))
            )
            cost = cost + float(self.path_boundary_w) * path_barrier

        return cost

    def _terminal_cost(self, X: Tensor, ref: Tensor) -> Tensor:
        e = X - ref.unsqueeze(0)
        e[:, 2] = _wrap_angle_t(e[:, 2])
        return (e * e) @ self.Qf

    @torch.no_grad()
    def _dra_risk_cost_over_time(
        self,
        X_hist_xy: Tensor,
        X_hist_phi: Tensor | None,
        O_mean: Tensor | None,
        radii: Tensor | None,
        O_phi: Tensor | None = None,
        ego_half_length: float | None = None,
        ego_half_width: float | None = None,
        obs_half_length: float | None = None,
        obs_half_width: float | None = None,
        obstacle_buffer: float = 0.0,
    ) -> Tensor:
        T, M, _ = X_hist_xy.shape
        if O_mean is None or O_mean.numel() == 0 or O_mean.shape[1] == 0:
            return torch.zeros((M,), device=self.device, dtype=self.dtype)

        O_mean = O_mean.to(device=self.device, dtype=self.dtype)
        K = int(O_mean.shape[1])
        use_rectangles = (
            X_hist_phi is not None
            and O_phi is not None
            and ego_half_length is not None
            and ego_half_width is not None
            and obs_half_length is not None
            and obs_half_width is not None
        )

        sx = max(float(self.obs_pos_sigma[0].item()), 1e-6)
        sy = max(float(self.obs_pos_sigma[1].item()), 1e-6)
        norm_const = torch.tensor(1.0 / (2.0 * np.pi * sx * sy), device=self.device, dtype=self.dtype)

        risk_cost = torch.zeros((M,), device=self.device, dtype=self.dtype)
        mc_chunk = max(1, int(self.mc_chunk))

        if use_rectangles:
            O_phi = O_phi.to(device=self.device, dtype=self.dtype)
            ego_bound_radius = float(np.hypot(float(ego_half_length), float(ego_half_width)))
            obs_bound_radius = float(np.hypot(float(obs_half_length), float(obs_half_width)))
            safe_R = ego_bound_radius + obs_bound_radius + float(obstacle_buffer)

            for t in range(T):
                pts = X_hist_xy[t]
                psi = X_hist_phi[t].to(device=self.device, dtype=self.dtype)
                px = pts[:, 0]
                py = pts[:, 1]

                for k in range(K):
                    mu = O_mean[t, k]
                    phi_k = float(O_phi[t, k].item())
                    xlow = torch.minimum(torch.min(px) - safe_R, mu[0] - 4.0 * sx - safe_R)
                    xhigh = torch.maximum(torch.max(px) + safe_R, mu[0] + 4.0 * sx + safe_R)
                    ylow = torch.minimum(torch.min(py) - safe_R, mu[1] - 4.0 * sy - safe_R)
                    yhigh = torch.maximum(torch.max(py) + safe_R, mu[1] + 4.0 * sy + safe_R)

                    dx_box = torch.clamp(xhigh - xlow, min=1e-6)
                    dy_box = torch.clamp(yhigh - ylow, min=1e-6)
                    area = dx_box * dy_box

                    ux = torch.rand((self.Nmc,), device=self.device, dtype=self.dtype, generator=self._rng)
                    uy = torch.rand((self.Nmc,), device=self.device, dtype=self.dtype, generator=self._rng)
                    xj = xlow + ux * dx_box
                    yj = ylow + uy * dy_box
                    cell_area = area / float(self.Nmc)

                    dxo = (xj - mu[0]) / sx
                    dyo = (yj - mu[1]) / sy
                    pdf = norm_const * torch.exp(-0.5 * (dxo * dxo + dyo * dyo))
                    pjoint_mass = torch.clamp(pdf * cell_area, 0.0, 0.999)

                    Phat = torch.zeros((M,), device=self.device, dtype=self.dtype)

                    for j0 in range(0, self.Nmc, mc_chunk):
                        j1 = min(self.Nmc, j0 + mc_chunk)
                        chunk_size = j1 - j0
                        obs_xy_chunk = torch.stack(
                            (
                                xj[j0:j1].unsqueeze(0).expand(M, chunk_size),
                                yj[j0:j1].unsqueeze(0).expand(M, chunk_size),
                            ),
                            dim=-1,
                        )
                        obs_phi_chunk = torch.full(
                            (M, chunk_size),
                            phi_k,
                            device=self.device,
                            dtype=self.dtype,
                        )
                        sep = rectangle_signed_distance_torch(
                            ego_xy=pts.unsqueeze(1).expand(M, chunk_size, 2),
                            ego_phi=psi.unsqueeze(1).expand(M, chunk_size),
                            ego_half_length=float(ego_half_length),
                            ego_half_width=float(ego_half_width),
                            obs_xy=obs_xy_chunk,
                            obs_phi=obs_phi_chunk,
                            obs_half_length=float(obs_half_length),
                            obs_half_width=float(obs_half_width),
                        )
                        inside = sep <= float(obstacle_buffer)
                        Phat += inside.to(self.dtype) @ pjoint_mass[j0:j1]

                    Phat = torch.clamp(Phat, 0.0, 1.0)
                    risk_cost += self._omega_soft_t * Phat
                    risk_cost += self._omega_hard_t * (Phat > self._sigma_cp_t).to(self.dtype)

            return risk_cost

        if radii is None or radii.numel() == 0:
            return risk_cost

        radii = radii.to(device=self.device, dtype=self.dtype).reshape(-1)
        for t in range(T):
            pts = X_hist_xy[t]
            px = pts[:, 0]
            py = pts[:, 1]

            for k in range(K):
                safe_R = float(radii[k].item())
                if safe_R <= 0.0:
                    continue

                mu = O_mean[t, k]
                xlow = torch.minimum(torch.min(px) - safe_R, mu[0] - 4.0 * sx - safe_R)
                xhigh = torch.maximum(torch.max(px) + safe_R, mu[0] + 4.0 * sx + safe_R)
                ylow = torch.minimum(torch.min(py) - safe_R, mu[1] - 4.0 * sy - safe_R)
                yhigh = torch.maximum(torch.max(py) + safe_R, mu[1] + 4.0 * sy + safe_R)

                dx_box = torch.clamp(xhigh - xlow, min=1e-6)
                dy_box = torch.clamp(yhigh - ylow, min=1e-6)
                area = dx_box * dy_box

                ux = torch.rand((self.Nmc,), device=self.device, dtype=self.dtype, generator=self._rng)
                uy = torch.rand((self.Nmc,), device=self.device, dtype=self.dtype, generator=self._rng)
                xj = xlow + ux * dx_box
                yj = ylow + uy * dy_box
                cell_area = area / float(self.Nmc)

                dxo = (xj - mu[0]) / sx
                dyo = (yj - mu[1]) / sy
                pdf = norm_const * torch.exp(-0.5 * (dxo * dxo + dyo * dyo))
                pjoint_mass = torch.clamp(pdf * cell_area, 0.0, 0.999)

                Phat = torch.zeros((M,), device=self.device, dtype=self.dtype)
                r2 = safe_R * safe_R

                for j0 in range(0, self.Nmc, mc_chunk):
                    j1 = min(self.Nmc, j0 + mc_chunk)
                    dxm = xj[j0:j1].unsqueeze(0) - px.unsqueeze(1)
                    dym = yj[j0:j1].unsqueeze(0) - py.unsqueeze(1)
                    inside = (dxm * dxm + dym * dym) <= r2
                    Phat += inside.to(self.dtype) @ pjoint_mass[j0:j1]

                Phat = torch.clamp(Phat, 0.0, 1.0)
                risk_cost += self._omega_soft_t * Phat
                risk_cost += self._omega_hard_t * (Phat > self._sigma_cp_t).to(self.dtype)

        return risk_cost

    @torch.no_grad()
    def plan(
        self,
        x0_cpu: np.ndarray,
        ref_cpu: np.ndarray,
        O_mean_cpu: np.ndarray | None = None,
        radii_cpu: np.ndarray | None = None,
        O_phi_cpu: np.ndarray | None = None,
        ego_half_length: float | None = None,
        ego_half_width: float | None = None,
        obs_half_length: float | None = None,
        obs_half_width: float | None = None,
        obstacle_buffer: float = 0.0,
        return_samples: bool = False,
        n_show: int = 60,
        show_seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        x0 = _as_torch(np.asarray(x0_cpu, dtype=np.float32).reshape(-1), self.device, self.dtype)
        ref = _as_torch(np.asarray(ref_cpu, dtype=np.float32).reshape(-1), self.device, self.dtype)
        O_mean = _as_torch(O_mean_cpu, self.device, self.dtype)
        radii = _as_torch(radii_cpu, self.device, self.dtype)
        O_phi = _as_torch(O_phi_cpu, self.device, self.dtype)

        U = _as_torch(self.U_cpu, self.device, self.dtype).clone()
        sigma = self.noise_sigma.view(1, 1, self.nu)

        if return_samples:
            ns = int(min(max(1, int(n_show)), self.M))
            g = torch.Generator(device=self.device)
            g.manual_seed(int(show_seed))
            sample_idx = torch.randperm(self.M, generator=g, device=self.device)[:ns]
        else:
            ns = 0
            sample_idx = None

        eps = torch.randn((self.M, self.T, self.nu), device=self.device, dtype=self.dtype, generator=self._rng) * sigma
        eps[0, :, :] = 0.0
        U_roll = torch.clamp(U.unsqueeze(0) + eps, self._u_min_112, self._u_max_112)

        J = torch.zeros((self.M,), device=self.device, dtype=self.dtype)
        X = x0.unsqueeze(0).repeat(self.M, 1)
        X_hist_xy = torch.zeros((self.T, self.M, 2), device=self.device, dtype=self.dtype)
        X_hist_phi = torch.zeros((self.T, self.M), device=self.device, dtype=self.dtype)

        if return_samples:
            nx = int(x0.numel())
            Xsamp = torch.zeros((self.T + 1, ns, nx), device=self.device, dtype=self.dtype)
            Xsamp[0] = X.index_select(0, sample_idx)
        else:
            Xsamp = None

        for t in range(self.T):
            U_t = U_roll[:, t, :]
            X = self._dyn_diffdrive(X, U_t)
            X_hist_xy[t] = X[:, :2]
            X_hist_phi[t] = X[:, 2]
            J += self._running_cost(X, U_t, t=t, ref=ref)

            if return_samples:
                Xsamp[t + 1] = X.index_select(0, sample_idx)

        J += self._terminal_cost(X, ref=ref)
        J += self._dra_risk_cost_over_time(
            X_hist_xy,
            X_hist_phi,
            O_mean,
            radii,
            O_phi=O_phi,
            ego_half_length=ego_half_length,
            ego_half_width=ego_half_width,
            obs_half_length=obs_half_length,
            obs_half_width=obs_half_width,
            obstacle_buffer=obstacle_buffer,
        )

        J = torch.where(torch.isfinite(J), J, torch.full_like(J, torch.inf))
        rho = torch.min(J)
        w = torch.exp(-(J - rho) / self.lam)
        w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
        w_sum = torch.sum(w)

        if float(w_sum.item()) >= 1e-12:
            w = w / w_sum
            dU = torch.einsum("m,mtd->td", w, eps)
            U = torch.clamp(U + dU, self.u_min, self.u_max)

        U_cpu = U.detach().cpu().numpy().astype(np.float32)
        self.U_cpu = U_cpu
        self.U = self.U_cpu

        if return_samples and Xsamp is not None:
            Xsamp_cpu = Xsamp.detach().cpu().numpy().astype(np.float32)
            return U_cpu, Xsamp_cpu

        return U_cpu
