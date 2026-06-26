# mppi_class.py
# -*- coding: utf-8 -*-
import numpy as np
import torch
from typing import Callable, Optional, Tuple, Union, Dict, Any

from car_geometry import rectangle_signed_distance_torch

Tensor = torch.Tensor


def _as_torch(x, device: torch.device, dtype: torch.dtype) -> Tensor:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device, dtype=dtype)


# ============================================================
# Plain MPPI (Torch, generic)
# ============================================================
class MPPI:
    """
    Generic MPPI in Torch.

    Provide:
      dynamics(X,U,dt,**dyn_kwargs)->X_next            (M,nx)
      running_cost(X,U,t,**cost_kwargs)->cost          (M,)
      terminal_cost(X,t_final,**cost_kwargs)->cost     (M,)
    """

    def __init__(
        self,
        dt: float,
        T: int,
        M: int,
        lam: float,
        noise_sigma: Union[np.ndarray, Tensor],  # (nu,) diag or (nu,nu) covariance
        u_min: Union[np.ndarray, Tensor],        # (nu,)
        u_max: Union[np.ndarray, Tensor],        # (nu,)
        dynamics: Callable[..., Tensor],
        running_cost: Callable[..., Tensor],
        terminal_cost: Callable[..., Tensor],
        exp_clip: float = 80.0,
        weight_floor: float = 1e-12,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
        dyn_kwargs: Optional[Dict[str, Any]] = None,
        cost_kwargs: Optional[Dict[str, Any]] = None,
        u_init: Optional[np.ndarray] = None,  # (T,nu)
    ):
        self.dt = float(dt)
        self.T = int(T)
        self.M = int(M)
        self.lam = float(lam)
        self.verbose = bool(verbose)

        if not (np.isfinite(self.dt) and self.dt > 0):
            raise ValueError(f"dt must be finite and >0, got {self.dt}")
        if not (np.isfinite(self.lam) and self.lam > 0):
            raise ValueError(f"lam must be finite and >0, got {self.lam}")

        self.exp_clip = float(exp_clip)
        self.weight_floor = float(weight_floor)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

        self.dynamics = dynamics
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost

        self.dyn_kwargs = dict(dyn_kwargs or {})
        self.cost_kwargs = dict(cost_kwargs or {})

        self.u_min = _as_torch(u_min, self.device, self.dtype).reshape(-1)
        self.u_max = _as_torch(u_max, self.device, self.dtype).reshape(-1)
        if self.u_min.shape != self.u_max.shape:
            raise ValueError("u_min/u_max shape mismatch")
        self.nu = int(self.u_min.numel())

        # ----- CONTROL NOISE (important: do not reuse this name for obstacle noise) -----
        sig = _as_torch(noise_sigma, self.device, self.dtype)
        if sig.ndim == 1:
            if sig.numel() != self.nu:
                raise ValueError("noise_sigma diag must be (nu,)")
            self.ctrl_noise_mode = "diag"
            self.noise_std = sig.reshape(1, 1, self.nu)
            self.noise_L = None
        elif sig.ndim == 2:
            if sig.shape != (self.nu, self.nu):
                raise ValueError("noise_sigma cov must be (nu,nu)")
            self.ctrl_noise_mode = "cov"
            jitter = 1e-6 * torch.eye(self.nu, device=self.device, dtype=self.dtype)
            cov = sig + jitter
            self.noise_L = torch.linalg.cholesky(cov)
            self.noise_std = None
        else:
            raise ValueError("noise_sigma must be (nu,) or (nu,nu)")

        if u_init is None:
            self.U_cpu = np.zeros((self.T, self.nu), dtype=np.float32)
        else:
            u_init = np.asarray(u_init, dtype=np.float32)
            if u_init.shape != (self.T, self.nu):
                raise ValueError("u_init must be (T,nu)")
            self.U_cpu = u_init.copy()

        # alias for your warm-start shifting code
        self.U = self.U_cpu

    def _sample_eps(self) -> Tensor:
        z = torch.randn((self.M, self.T, self.nu), device=self.device, dtype=self.dtype)
        if self.ctrl_noise_mode == "diag":
            return z * self.noise_std
        else:
            # cov mode: eps = z @ L^T
            return z @ self.noise_L.T

    @torch.no_grad()
    def plan(
        self,
        x0: Union[np.ndarray, Tensor],
        return_samples: bool = False,
        n_show: int = 60,
        show_seed: int = 0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        x0_t = _as_torch(np.asarray(x0, dtype=np.float32).reshape(-1), self.device, self.dtype)
        nx = int(x0_t.numel())
        U = _as_torch(self.U_cpu, self.device, self.dtype)  # (T,nu)

        if return_samples:
            ns = int(min(max(1, int(n_show)), self.M))
            g = torch.Generator(device=self.device)
            g.manual_seed(int(show_seed))
            sample_idx = torch.randperm(self.M, generator=g, device=self.device)[:ns]
        else:
            ns = 0
            sample_idx = None

        eps = self._sample_eps()
        J = torch.zeros((self.M,), device=self.device, dtype=self.dtype)

        X = x0_t.unsqueeze(0).repeat(self.M, 1)

        if return_samples:
            Xsamp = torch.zeros((self.T + 1, ns, nx), device=self.device, dtype=self.dtype)
            Xsamp[0] = X.index_select(0, sample_idx)
        else:
            Xsamp = None

        for t in range(self.T):
            U_t = torch.clamp(U[t].unsqueeze(0) + eps[:, t, :], self.u_min, self.u_max)
            X = self.dynamics(X, U_t, self.dt, **self.dyn_kwargs)

            c = self.running_cost(X, U_t, t, **self.cost_kwargs)
            J = J + c

            if return_samples:
                Xsamp[t + 1] = X.index_select(0, sample_idx)

        J = J + self.terminal_cost(X, self.T, **self.cost_kwargs)
        J = torch.nan_to_num(J, nan=torch.tensor(float("inf"), device=self.device, dtype=self.dtype))

        rho = torch.min(J)
        z = -(J - rho) / self.lam
        z = torch.clamp(z, -self.exp_clip, self.exp_clip)
        w = torch.exp(z)

        w_sum = torch.sum(w)
        if float(w_sum.item()) < self.weight_floor:
            if self.verbose:
                print(f"[WARN] weight collapse, w_sum={float(w_sum.item())}")
        else:
            w = w / w_sum

            for t in range(self.T):
                dU = torch.sum(w.unsqueeze(1) * eps[:, t, :], dim=0)
                U[t] = torch.clamp(U[t] + dU, self.u_min, self.u_max)

        U_cpu = U.detach().cpu().numpy().astype(np.float32)
        U_cpu = np.nan_to_num(U_cpu, nan=0.0)
        self.U_cpu = U_cpu
        self.U = self.U_cpu

        if return_samples and Xsamp is not None:
            return U_cpu, Xsamp.detach().cpu().numpy().astype(np.float32)
        return U_cpu, None


# ============================================================
# RA-MPPI with CVaR feasibility filter (Torch) — matches your CuPy logic
# ============================================================
class RA_MPPI(MPPI):
    """
    Same logic as your CuPy MPPI+CVaR:

      - Roll out M trajectories
      - Compute tracking/control/terminal costs J
      - Compute feasible mask using CVaR over N obstacle-noise samples:
          g = R - dist
          L = max_t g
          CVaR_alpha(L)
          cvar_max = max_obs CVaR
          feasible if cvar_max <= 0
      - weights ONLY feasible trajectories (infeasible weight 0)
      - if none feasible: choose best by min cvar_max and force it feasible

    Obstacle data comes from:
      self.cost_kwargs["O_mean"]  (T,K,2) or (T,K,3)
      self.cost_kwargs["O_phi"]   (T,K)
      self.cost_kwargs["radii"]   (K,)
    """

    def __init__(
        self,
        *args,
        cvar_alpha: float = 0.9,
        cvar_N: int = 64,
        obs_pos_sigma: Union[Tuple[float, float, float], np.ndarray, Tensor] = (0.25, 0.25, 0.25),
        obs_noise_mode: str = "static",  # "static" or "per_step"
        risk_cost_A: float = 0.0,
        risk_cost_Cu: float = 0.0,
        filter_infeasible_rollouts: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.cvar_alpha = float(cvar_alpha)
        self.cvar_N = int(cvar_N)

        # IMPORTANT: use a different name than control noise mode
        self.obs_noise_mode = str(obs_noise_mode)
        if self.obs_noise_mode not in ("static", "per_step"):
            raise ValueError("obs_noise_mode must be 'static' or 'per_step'")

        obs_pos_sigma_t = _as_torch(obs_pos_sigma, self.device, self.dtype).reshape(-1)
        if obs_pos_sigma_t.numel() not in (2, 3):
            raise ValueError("obs_pos_sigma must have 2 or 3 entries")
        if obs_pos_sigma_t.numel() == 2:
            obs_pos_sigma_t = torch.cat([obs_pos_sigma_t, torch.zeros(1, device=self.device, dtype=self.dtype)])
        self.obs_pos_sigma = obs_pos_sigma_t
        self.risk_cost_A = float(risk_cost_A)
        self.risk_cost_Cu = float(risk_cost_Cu)
        self.filter_infeasible_rollouts = bool(filter_infeasible_rollouts)
        if self.risk_cost_A < 0.0:
            raise ValueError("risk_cost_A must be >= 0")

    def _sample_obstacles(
        self,
        O_mean: Tensor,                          # (T,K,2) or (T,K,3)
        obs_noise_std: Optional[Tensor] = None,  # optional standard normal noise
    ) -> Tensor:
        """
        Returns O_samp: (T,K,N,D), where D is O_mean.shape[-1]

        obs_noise_mode:
          - static: one noise per (K,N) shared over time
          - per_step: different noise per (T,K,N)
        """
        T, K, obs_dim = O_mean.shape
        if obs_dim not in (2, 3):
            raise ValueError(f"O_mean last dimension must be 2 or 3, got {obs_dim}")
        N = self.cvar_N
        if K == 0:
            return torch.zeros((T, 0, N, obs_dim), device=self.device, dtype=self.dtype)

        sig = self.obs_pos_sigma[:obs_dim].reshape(1, 1, obs_dim)

        if self.obs_noise_mode == "static":
            if obs_noise_std is None:
                noise_kn = torch.randn((K, N, obs_dim), device=self.device, dtype=self.dtype)
            else:
                noise_kn = obs_noise_std.to(device=self.device, dtype=self.dtype)
                if noise_kn.shape != (K, N, obs_dim):
                    raise ValueError(f"obs_noise_std must be (K,N,{obs_dim}) got {tuple(noise_kn.shape)}")
            noise_kn = noise_kn * sig
            return O_mean[:, :, None, :] + noise_kn[None, :, :, :]
        else:
            if obs_noise_std is None:
                noise_tkn = torch.randn((T, K, N, obs_dim), device=self.device, dtype=self.dtype)
            else:
                noise_tkn = obs_noise_std.to(device=self.device, dtype=self.dtype)
                if noise_tkn.shape != (T, K, N, obs_dim):
                    raise ValueError(f"obs_noise_std must be (T,K,N,{obs_dim}) got {tuple(noise_tkn.shape)}")
            noise_tkn = noise_tkn * sig.reshape(1, 1, 1, obs_dim)
            return O_mean[:, :, None, :] + noise_tkn

    def _cvar_from_sorted(self, L_sorted: Tensor) -> Tensor:
        """
        Matches your CuPy formula.
        L_sorted shape: (..., N)
        returns: (...) CVaR
        """
        alpha = self.cvar_alpha
        N = int(L_sorted.shape[-1])
        if N < 1:
            raise ValueError("L_sorted must have a non-empty sample dimension")
        k = int(np.ceil(alpha * N))
        k_idx = k - 1
        tail_sum = torch.sum(L_sorted[..., k_idx:], dim=-1)
        frac_term = (k - alpha * N) * L_sorted[..., k_idx]
        return (tail_sum + frac_term) / (N * (1.0 - alpha))

    def cvar_feasible_mask(
        self,
        X_hist_xy: Tensor,           # (T,M,2) or (T,M,3)
        O_mean: Optional[Tensor],    # (T,K,2) or (T,K,3)
        radii: Optional[Tensor],     # (K,)
        obs_noise_std: Optional[Tensor] = None,
        X_hist_phi: Optional[Tensor] = None,  # (T,M)
        O_phi: Optional[Tensor] = None,       # (T,K)
        ego_half_length: Optional[float] = None,
        ego_half_width: Optional[float] = None,
        obs_half_length: Optional[float] = None,
        obs_half_width: Optional[float] = None,
        obstacle_buffer: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
          feasible: (M,) bool
          cvar_max: (M,) float
        """
        T, M, _ = X_hist_xy.shape
        if (O_mean is None) or (O_mean.numel() == 0) or (O_mean.shape[1] == 0):
            feasible = torch.ones((M,), device=self.device, dtype=torch.bool)
            cvar_max = torch.full((M,), -torch.inf, device=self.device, dtype=self.dtype)
            return feasible, cvar_max

        O_mean = O_mean.to(device=self.device, dtype=self.dtype)
        K = int(O_mean.shape[1])

        use_rectangles = (
            (X_hist_phi is not None)
            and (O_phi is not None)
            and (ego_half_length is not None)
            and (ego_half_width is not None)
            and (obs_half_length is not None)
            and (obs_half_width is not None)
        )

        if use_rectangles:
            O_phi = O_phi.to(device=self.device, dtype=self.dtype)
            O_samp = self._sample_obstacles(O_mean, obs_noise_std=obs_noise_std)  # (T,K,N,D)
            L = None
            for t in range(T):
                sep_t = rectangle_signed_distance_torch(
                    ego_xy=X_hist_xy[t][:, None, None, :2],
                    ego_phi=X_hist_phi[t][:, None, None],
                    ego_half_length=ego_half_length,
                    ego_half_width=ego_half_width,
                    obs_xy=O_samp[t][None, :, :, :2],
                    obs_phi=O_phi[t][None, :, None],
                    obs_half_length=obs_half_length,
                    obs_half_width=obs_half_width,
                )
                g_t = float(obstacle_buffer) - sep_t  # (M,K,N)
                L = g_t if L is None else torch.maximum(L, g_t)
        else:
            if radii is None:
                feasible = torch.ones((M,), device=self.device, dtype=torch.bool)
                cvar_max = torch.full((M,), -torch.inf, device=self.device, dtype=self.dtype)
                return feasible, cvar_max
            radii = radii.to(device=self.device, dtype=self.dtype).reshape(-1)
            O_samp = self._sample_obstacles(O_mean, obs_noise_std=obs_noise_std)  # (T,K,N,D)

            # dist: (T,M,K,N)
            diff = X_hist_xy[:, :, None, None, :] - O_samp[:, None, :, :, :]
            dist = torch.linalg.norm(diff, dim=-1)

            # g = R - dist
            Rk = radii.reshape(1, 1, K, 1)
            g = Rk - dist  # (T,M,K,N)

            # L = max_t g  -> (M,K,N)
            L = torch.amax(g, dim=0)

        Ls, _ = torch.sort(L, dim=-1)  # (M,K,N)

        cvar_per_obs = self._cvar_from_sorted(Ls)   # (M,K)
        cvar_max = torch.amax(cvar_per_obs, dim=1)  # (M,)
        feasible = (cvar_max <= 0.0)
        return feasible, cvar_max

    @torch.no_grad()
    def plan(
        self,
        x0: Union[np.ndarray, Tensor],
        return_samples: bool = False,
        n_show: int = 60,
        show_seed: int = 0,
        obs_noise_std: Optional[Union[np.ndarray, Tensor]] = None,
        return_debug: bool = False,
    ):
        x0_t = _as_torch(np.asarray(x0, dtype=np.float32).reshape(-1), self.device, self.dtype)
        U = _as_torch(self.U_cpu, self.device, self.dtype)

        if obs_noise_std is not None:
            obs_noise_std = _as_torch(obs_noise_std, self.device, self.dtype)

        # read obstacle data from cost_kwargs (same place you update it in your loop)
        O_mean = self.cost_kwargs.get("O_mean", None)
        O_phi = self.cost_kwargs.get("O_phi", None)
        radii = self.cost_kwargs.get("radii", None)
        ego_half_length = self.cost_kwargs.get("ego_half_length", None)
        ego_half_width = self.cost_kwargs.get("ego_half_width", None)
        obs_half_length = self.cost_kwargs.get("obs_half_length", None)
        obs_half_width = self.cost_kwargs.get("obs_half_width", None)
        obstacle_buffer = float(self.cost_kwargs.get("obstacle_buffer", 0.0))

        if return_samples:
            ns = int(min(max(1, int(n_show)), self.M))
            g = torch.Generator(device=self.device)
            g.manual_seed(int(show_seed))
            sample_idx = torch.randperm(self.M, generator=g, device=self.device)[:ns]
        else:
            ns = 0
            sample_idx = None

        debug = {}

        eps = self._sample_eps()
        J = torch.zeros((self.M,), device=self.device, dtype=self.dtype)

        X = x0_t.unsqueeze(0).repeat(self.M, 1)
        O_mean_for_dim = self.cost_kwargs.get("O_mean", None)
        pos_dim = 3 if (O_mean_for_dim is not None and int(O_mean_for_dim.shape[-1]) == 3 and x0_t.numel() >= 3) else 2
        X_hist_xy = torch.zeros((self.T, self.M, pos_dim), device=self.device, dtype=self.dtype)
        X_hist_phi = (
            torch.zeros((self.T, self.M), device=self.device, dtype=self.dtype)
            if x0_t.numel() >= 3
            else None
        )

        if return_samples:
            nx = int(x0_t.numel())
            Xsamp = torch.zeros((self.T + 1, ns, nx), device=self.device, dtype=self.dtype)
            Xsamp[0] = X.index_select(0, sample_idx)
        else:
            Xsamp = None

        for t in range(self.T):
            U_t = torch.clamp(U[t].unsqueeze(0) + eps[:, t, :], self.u_min, self.u_max)
            X = self.dynamics(X, U_t, self.dt, **self.dyn_kwargs)
            X_hist_xy[t] = X[:, :pos_dim]
            if X_hist_phi is not None:
                X_hist_phi[t] = X[:, 2]

            c = self.running_cost(X, U_t, t, **self.cost_kwargs)
            J = J + c

            if return_samples:
                Xsamp[t + 1] = X.index_select(0, sample_idx)

        J = J + self.terminal_cost(X, self.T, **self.cost_kwargs)
        J = torch.nan_to_num(J, nan=torch.tensor(float("inf"), device=self.device, dtype=self.dtype))

        feasible, cvar_max = self.cvar_feasible_mask(
            X_hist_xy,
            O_mean,
            radii,
            obs_noise_std=obs_noise_std,
            X_hist_phi=X_hist_phi,
            O_phi=O_phi,
            ego_half_length=ego_half_length,
            ego_half_width=ego_half_width,
            obs_half_length=obs_half_length,
            obs_half_width=obs_half_width,
            obstacle_buffer=obstacle_buffer,
        )
        risk_penalty = torch.zeros_like(J)
        if self.risk_cost_A > 0.0:
            risk_mask = cvar_max > self.risk_cost_Cu
            risk_penalty[risk_mask] = self.risk_cost_A * cvar_max[risk_mask]
        J_total = J + risk_penalty

        if not self.filter_infeasible_rollouts:
            feasible = torch.ones_like(feasible)
        elif not bool(torch.any(feasible).item()):
            best = int(torch.argmin(J_total).item())
            feasible = torch.zeros_like(feasible)
            feasible[best] = True

        Jf = J_total[feasible]
        rho = torch.min(Jf)

        w = torch.zeros_like(J_total)
        w[feasible] = torch.exp(-(Jf - rho) / self.lam)

        w_sum = torch.sum(w)
        w_sum_val = float(w_sum.item())
        feas_count = int(torch.sum(feasible).item())

        debug = {
            "feasible_count": feas_count,
            "w_sum": w_sum_val,
            "cvar_min": float(torch.min(cvar_max).item()),
            "cvar_med": float(torch.median(cvar_max).item()),
            "cvar_max": float(torch.max(cvar_max).item()),
            "risk_cost_max": float(torch.max(risk_penalty).item()),
            "risk_cost_mean": float(torch.mean(risk_penalty).item()),
        }

        if (not np.isfinite(w_sum_val)) or (w_sum_val < self.weight_floor):
            if self.verbose:
                print(f"[WARN] weight collapse: {debug}")
        else:
            w = w / w_sum

            for t in range(self.T):
                dU = torch.sum(w.unsqueeze(1) * eps[:, t, :], dim=0)
                U[t] = torch.clamp(U[t] + dU, self.u_min, self.u_max)

        U_cpu = U.detach().cpu().numpy().astype(np.float32)
        U_cpu = np.nan_to_num(U_cpu, nan=0.0)
        self.U_cpu = U_cpu
        self.U = self.U_cpu

        Xsamp_cpu = None
        if return_samples and Xsamp is not None:
            Xsamp_cpu = Xsamp.detach().cpu().numpy().astype(np.float32)

        if return_debug:
            return U_cpu, Xsamp_cpu, debug
        return U_cpu, Xsamp_cpu
    

class DR_MPPI(RA_MPPI):
    """
    DR-MPPI = RA_MPPI (CVaR feasibility filter) + DR correction on CVaR
    (Wasserstein-1 ball radius dr_eps_cvar).

    Matches your CuPy logic exactly:
      cvar_per_obs = cvar_per_obs + dr_eps_cvar * (1/(1-alpha))
    """

    def __init__(
        self,
        *args,
        dr_eps_cvar: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dr_eps_cvar = float(dr_eps_cvar)
        if self.dr_eps_cvar < 0.0:
            raise ValueError("dr_eps_cvar must be >= 0")

        if not (0.0 < float(self.cvar_alpha) < 1.0):
            raise ValueError("cvar_alpha must be in (0, 1)")
        if int(self.cvar_N) < 2:
            raise ValueError("cvar_N must be >= 2")

        # same as your code
        self._cvar_lip = 1.0 / (1.0 - float(self.cvar_alpha))

    @torch.no_grad()
    def cvar_feasible_mask(
        self,
        X_hist_xy: Tensor,           # (T,M,2) or (T,M,3)
        O_mean: Optional[Tensor],    # (T,K,2) or (T,K,3)
        radii: Optional[Tensor],     # (K,)
        obs_noise_std: Optional[Tensor] = None,
        X_hist_phi: Optional[Tensor] = None,  # (T,M)
        O_phi: Optional[Tensor] = None,       # (T,K)
        ego_half_length: Optional[float] = None,
        ego_half_width: Optional[float] = None,
        obs_half_length: Optional[float] = None,
        obs_half_width: Optional[float] = None,
        obstacle_buffer: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Same as RA_MPPI.cvar_feasible_mask, but:
          - uses component-wise distance calc (to avoid huge (..,D) diff tensor)
          - applies DR correction on cvar_per_obs
        Returns:
          feasible: (M,) bool
          cvar_max: (M,) float
        """
        T, M, _ = X_hist_xy.shape

        if (O_mean is None) or (O_mean.numel() == 0) or (O_mean.shape[1] == 0):
            feasible = torch.ones((M,), device=self.device, dtype=torch.bool)
            cvar_max = torch.full((M,), -torch.inf, device=self.device, dtype=self.dtype)
            return feasible, cvar_max

        O_mean = O_mean.to(device=self.device, dtype=self.dtype)
        K = int(O_mean.shape[1])

        use_rectangles = (
            (X_hist_phi is not None)
            and (O_phi is not None)
            and (ego_half_length is not None)
            and (ego_half_width is not None)
            and (obs_half_length is not None)
            and (obs_half_width is not None)
        )

        O_samp = self._sample_obstacles(O_mean, obs_noise_std=obs_noise_std)  # (T,K,N,D)
        N = int(O_samp.shape[2])
        if use_rectangles:
            O_phi = O_phi.to(device=self.device, dtype=self.dtype)
            L_steps = []
            for t in range(T):
                sep_t = rectangle_signed_distance_torch(
                    ego_xy=X_hist_xy[t][:, None, None, :2],
                    ego_phi=X_hist_phi[t][:, None, None],
                    ego_half_length=ego_half_length,
                    ego_half_width=ego_half_width,
                    obs_xy=O_samp[t][None, :, :, :2],
                    obs_phi=O_phi[t][None, :, None],
                    obs_half_length=obs_half_length,
                    obs_half_width=obs_half_width,
                )
                g_t = float(obstacle_buffer) - sep_t
                L_steps.append(g_t)
            L = torch.stack(L_steps, dim=0)                                 # (T,M,K,N)
        else:
            if radii is None:
                feasible = torch.ones((M,), device=self.device, dtype=torch.bool)
                cvar_max = torch.full((M,), -torch.inf, device=self.device, dtype=self.dtype)
                return feasible, cvar_max
            radii = radii.to(device=self.device, dtype=self.dtype).reshape(-1)

            # dist without allocating huge diff tensor:
            # X_hist_xy: (T,M,D), O_samp: (T,K,N,D)
            dx = X_hist_xy[:, :, None, None, 0] - O_samp[:, None, :, :, 0]  # (T,M,K,N)
            dy = X_hist_xy[:, :, None, None, 1] - O_samp[:, None, :, :, 1]  # (T,M,K,N)
            dist_sq = dx * dx + dy * dy
            if O_samp.shape[-1] == 3:
                dz = X_hist_xy[:, :, None, None, 2] - O_samp[:, None, :, :, 2]  # (T,M,K,N)
                dist_sq = dist_sq + dz * dz
            dist = torch.sqrt(dist_sq)                                      # (T,M,K,N)

            # g = R - dist
            Rk = radii.reshape(1, 1, K, 1)
            g = Rk - dist                                                  # (T,M,K,N)

            L = g

        # DR risk samples are all horizon-step / obstacle-noise losses.
        # No max over horizon is taken here.
        L = L.permute(1, 2, 0, 3).reshape(M, K, T * N)                  # (M,K,T*N)
        Ls, _ = torch.sort(L, dim=-1)                                   # (M,K,T*N)
        cvar_per_obs = self._cvar_from_sorted(Ls)                       # (M,K)

        # DR correction (exactly your CuPy)
        if self.dr_eps_cvar > 0.0:
            cvar_per_obs = cvar_per_obs + (self.dr_eps_cvar * self._cvar_lip)

        cvar_max = torch.amax(cvar_per_obs, dim=1)                     # (M,)
        feasible = (cvar_max <= 0.0)
        return feasible, cvar_max


# ============================================================
# DRA-MPPI with Monte Carlo joint collision-probability cost
# ============================================================
class DRA_MPPI(MPPI):
    """
    Dynamic Risk-Aware MPPI.

    This keeps MPPI's normal cost weighting, but augments each rollout cost
    with a Monte Carlo estimate of joint collision probability:

        C_risk = omega_soft * P_hat + omega_hard * 1{P_hat > sigma_cp}

    Obstacle means are read from cost_kwargs["obs_mean_3d"] or
    cost_kwargs["O_mean"]. Shapes supported:
        (T+1,K,D) or (T,K,D), with D in {2,3}

    The collision radius is taken from cost_kwargs["radii"] when present,
    otherwise from:
        moving_r + moving_safety_margin + drone_radius
    """

    def __init__(
        self,
        *args,
        sigma_cp: float = 0.05,
        Nmc: int = 2000,
        omega_soft: float = 10.0,
        omega_hard: float = 1000.0,
        obs_pos_sigma: Optional[Union[np.ndarray, Tuple[float, ...]]] = None,
        obs_pos_sigma_xyz: Optional[Union[np.ndarray, Tuple[float, ...]]] = None,
        seed: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sigma_cp = float(sigma_cp)
        self.Nmc = int(Nmc)
        self.omega_soft = float(omega_soft)
        self.omega_hard = float(omega_hard)
        if self.Nmc < 1:
            raise ValueError("Nmc must be >= 1")
        if self.sigma_cp < 0.0:
            raise ValueError("sigma_cp must be >= 0")

        sigma = obs_pos_sigma_xyz if obs_pos_sigma_xyz is not None else obs_pos_sigma
        if sigma is None:
            sigma = (0.015, 0.015, 0.015)
        self.obs_pos_sigma = _as_torch(np.asarray(sigma, dtype=np.float32), self.device, self.dtype).reshape(-1)
        if int(self.obs_pos_sigma.numel()) not in (2, 3):
            raise ValueError("obs_pos_sigma must have length 2 or 3")

        self._sigma_cp_t = torch.tensor(self.sigma_cp, device=self.device, dtype=self.dtype)
        self._omega_soft_t = torch.tensor(self.omega_soft, device=self.device, dtype=self.dtype)
        self._omega_hard_t = torch.tensor(self.omega_hard, device=self.device, dtype=self.dtype)
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(int(seed))

    def _sample_eps_dra(self) -> Tensor:
        z = torch.randn((self.M, self.T, self.nu), device=self.device, dtype=self.dtype, generator=self._rng)
        if self.ctrl_noise_mode == "diag":
            return z * self.noise_std
        return z @ self.noise_L.T

    def _get_obstacle_means_for_dra(self) -> Optional[Tensor]:
        obs = self.cost_kwargs.get("obs_mean_3d", None)
        if obs is None:
            obs = self.cost_kwargs.get("O_mean", None)
        if obs is None:
            return None
        obs_t = _as_torch(obs, self.device, self.dtype)
        if obs_t.ndim == 2:
            obs_t = obs_t.unsqueeze(1)
        if obs_t.ndim != 3:
            raise ValueError(f"DRA obstacle means must be (T,K,D) or (T+1,K,D), got {tuple(obs_t.shape)}")
        if int(obs_t.shape[-1]) not in (2, 3):
            raise ValueError(f"DRA obstacle dimension must be 2 or 3, got {int(obs_t.shape[-1])}")
        return obs_t

    def _safe_radius_for_dra(self, num_obstacles: int) -> float:
        _O_phi, ego_half_length, ego_half_width, obs_half_length, obs_half_width, obstacle_buffer = (
            self._rectangle_geometry_for_dra()
        )
        if (
            ego_half_length is not None
            and ego_half_width is not None
            and obs_half_length is not None
            and obs_half_width is not None
        ):
            ego_bound = float(np.hypot(float(ego_half_length), float(ego_half_width)))
            obs_bound = float(np.hypot(float(obs_half_length), float(obs_half_width)))
            return ego_bound + obs_bound + float(obstacle_buffer)

        radii = self.cost_kwargs.get("radii", None)
        if radii is not None:
            radii_t = _as_torch(radii, self.device, self.dtype).reshape(-1)
            if int(radii_t.numel()) == 1:
                return float(radii_t[0].item())
            if int(radii_t.numel()) == int(num_obstacles):
                return float(torch.max(radii_t).item())
            raise ValueError(f"radii must have length 1 or K={num_obstacles}, got {int(radii_t.numel())}")
        return float(self.cost_kwargs.get("moving_r", 0.0)) + float(
            self.cost_kwargs.get("moving_safety_margin", 0.0)
        ) + float(self.cost_kwargs.get("drone_radius", 0.0))

    def _rectangle_geometry_for_dra(self) -> tuple[Optional[Tensor], Optional[float], Optional[float], Optional[float], Optional[float], float]:
        O_phi = self.cost_kwargs.get("O_phi", None)
        O_phi_t = _as_torch(O_phi, self.device, self.dtype) if O_phi is not None else None
        return (
            O_phi_t,
            self.cost_kwargs.get("ego_half_length", None),
            self.cost_kwargs.get("ego_half_width", None),
            self.cost_kwargs.get("obs_half_length", None),
            self.cost_kwargs.get("obs_half_width", None),
            float(self.cost_kwargs.get("obstacle_buffer", 0.0)),
        )

    @torch.no_grad()
    def _dra_risk_cost_over_time(
        self,
        X_hist_pos: Tensor,
        obs_mean: Optional[Tensor],
        safe_R: float,
        mc_chunk: int,
        X_hist_phi: Optional[Tensor] = None,
    ) -> Tensor:
        T, M, D = X_hist_pos.shape
        if obs_mean is None or obs_mean.numel() == 0 or safe_R <= 0.0:
            return torch.zeros(M, device=self.device, dtype=self.dtype)

        if int(obs_mean.shape[0]) == T + 1:
            obs_all = obs_mean[1 : T + 1]
        elif int(obs_mean.shape[0]) == T:
            obs_all = obs_mean
        else:
            raise ValueError(f"Expected obstacle horizon T or T+1, got {int(obs_mean.shape[0])} for T={T}")

        K = int(obs_all.shape[1])
        O_phi, ego_half_length, ego_half_width, obs_half_length, obs_half_width, obstacle_buffer = (
            self._rectangle_geometry_for_dra()
        )
        use_rectangles = (
            D == 2
            and X_hist_phi is not None
            and O_phi is not None
            and ego_half_length is not None
            and ego_half_width is not None
            and obs_half_length is not None
            and obs_half_width is not None
        )
        if use_rectangles:
            if int(O_phi.shape[0]) == T + 1:
                O_phi = O_phi[1 : T + 1]
            elif int(O_phi.shape[0]) != T:
                raise ValueError(f"Expected O_phi horizon T or T+1, got {int(O_phi.shape[0])} for T={T}")

        sigma = self.obs_pos_sigma
        if int(sigma.numel()) < D:
            raise ValueError(f"obs_pos_sigma has length {int(sigma.numel())}, but obstacle dimension is {D}")
        sigma = torch.clamp(sigma[:D], min=1e-6)
        norm_const = torch.tensor(
            1.0 / (((2.0 * np.pi) ** (0.5 * D)) * float(torch.prod(sigma).item())),
            device=self.device,
            dtype=self.dtype,
        )
        r2 = float(safe_R * safe_R)
        mc_chunk = max(1, int(mc_chunk))

        risk_cost = torch.zeros(M, device=self.device, dtype=self.dtype)
        for t in range(T):
            pts = X_hist_pos[t]  # (M,D)
            lows = torch.min(pts, dim=0).values - safe_R
            highs = torch.max(pts, dim=0).values + safe_R
            widths = torch.clamp(highs - lows, min=1e-6)
            volume = torch.prod(widths)

            unit = torch.rand((self.Nmc, D), device=self.device, dtype=self.dtype, generator=self._rng)
            samples = lows.unsqueeze(0) + unit * widths.unsqueeze(0)
            cell_volume = volume / float(self.Nmc)

            obs_t = obs_all[t, :, :D]  # (K,D)
            delta = (samples[:, None, :] - obs_t[None, :, :]) / sigma.view(1, 1, D)
            pdf = norm_const * torch.exp(-0.5 * torch.sum(delta * delta, dim=-1))  # (Nmc,K)
            p_obs_mass = torch.clamp(pdf * cell_volume, 0.0, 0.999)
            p_joint_mass = 1.0 - torch.prod(1.0 - p_obs_mass, dim=1)  # (Nmc,)

            phat = torch.zeros(M, device=self.device, dtype=self.dtype)
            for j0 in range(0, self.Nmc, mc_chunk):
                j1 = min(self.Nmc, j0 + mc_chunk)
                if use_rectangles:
                    Jc = j1 - j0
                    inside_any = torch.zeros((M, Jc), device=self.device, dtype=torch.bool)
                    for k in range(K):
                        obs_xy = samples[j0:j1, :2].unsqueeze(0).expand(M, Jc, 2)
                        obs_phi = O_phi[t, k].expand(M, Jc)
                        sep = rectangle_signed_distance_torch(
                            ego_xy=pts[:, :2].unsqueeze(1).expand(M, Jc, 2),
                            ego_phi=X_hist_phi[t].unsqueeze(1).expand(M, Jc),
                            ego_half_length=float(ego_half_length),
                            ego_half_width=float(ego_half_width),
                            obs_xy=obs_xy,
                            obs_phi=obs_phi,
                            obs_half_length=float(obs_half_length),
                            obs_half_width=float(obs_half_width),
                        )
                        inside_any |= sep <= obstacle_buffer
                    inside = inside_any
                else:
                    diff = samples[j0:j1].unsqueeze(0) - pts.unsqueeze(1)  # (M,J,D)
                    inside = torch.sum(diff * diff, dim=-1) <= r2
                phat += inside.to(self.dtype) @ p_joint_mass[j0:j1]

            phat = torch.clamp(phat, 0.0, 1.0)
            risk_cost += self._omega_soft_t * phat
            risk_cost += self._omega_hard_t * (phat > self._sigma_cp_t).to(self.dtype)

        return risk_cost

    @torch.no_grad()
    def plan(
        self,
        x0: Union[np.ndarray, Tensor],
        return_samples: bool = False,
        n_show: int = 60,
        show_seed: int = 0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        x0_t = _as_torch(np.asarray(x0, dtype=np.float32).reshape(-1), self.device, self.dtype)
        nx = int(x0_t.numel())
        U = _as_torch(self.U_cpu, self.device, self.dtype).clone()

        if return_samples:
            ns = int(min(max(1, int(n_show)), self.M))
            g = torch.Generator(device=self.device)
            g.manual_seed(int(show_seed))
            sample_idx = torch.randperm(self.M, generator=g, device=self.device)[:ns]
        else:
            ns = 0
            sample_idx = None

        eps = self._sample_eps_dra()
        U_roll = torch.clamp(U.unsqueeze(0) + eps, self.u_min.view(1, 1, self.nu), self.u_max.view(1, 1, self.nu))
        J = torch.zeros((self.M,), device=self.device, dtype=self.dtype)
        X = x0_t.unsqueeze(0).repeat(self.M, 1)

        obs_mean = self._get_obstacle_means_for_dra()
        pos_dim = 3 if obs_mean is not None and int(obs_mean.shape[-1]) == 3 and nx >= 3 else 2
        X_hist_pos = torch.zeros((self.T, self.M, pos_dim), device=self.device, dtype=self.dtype)
        X_hist_phi = torch.zeros((self.T, self.M), device=self.device, dtype=self.dtype) if nx >= 3 else None

        if return_samples:
            Xsamp = torch.zeros((self.T + 1, ns, nx), device=self.device, dtype=self.dtype)
            Xsamp[0] = X.index_select(0, sample_idx)
        else:
            Xsamp = None

        for t in range(self.T):
            U_t = U_roll[:, t, :]
            X = self.dynamics(X, U_t, self.dt, **self.dyn_kwargs)
            X_hist_pos[t] = X[:, :pos_dim]
            if X_hist_phi is not None:
                X_hist_phi[t] = X[:, 2]
            J = J + self.running_cost(X, U_t, t=t, **self.cost_kwargs)
            if return_samples:
                Xsamp[t + 1] = X.index_select(0, sample_idx)

        J = J + self.terminal_cost(X, self.T, **self.cost_kwargs)
        if obs_mean is not None:
            safe_R = self._safe_radius_for_dra(num_obstacles=int(obs_mean.shape[1]))
            mc_chunk = int(self.cost_kwargs.get("mc_chunk", 512))
            J = J + self._dra_risk_cost_over_time(
                X_hist_pos,
                obs_mean,
                safe_R=safe_R,
                mc_chunk=mc_chunk,
                X_hist_phi=X_hist_phi,
            )

        J = torch.where(torch.isfinite(J), J, torch.full_like(J, torch.inf))
        rho = torch.min(J)
        z = torch.clamp(-(J - rho) / self.lam, -self.exp_clip, self.exp_clip)
        w = torch.exp(z)
        w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
        w_sum = torch.sum(w)
        if float(w_sum.item()) < self.weight_floor:
            if self.verbose:
                print(f"[WARN] DRA weight collapse, w_sum={float(w_sum.item())}")
        else:
            w = w / w_sum
            dU = torch.einsum("m,mtd->td", w, eps)
            U = torch.clamp(U + dU, self.u_min, self.u_max)

        U_cpu = U.detach().cpu().numpy().astype(np.float32)
        U_cpu = np.nan_to_num(U_cpu, nan=0.0)
        self.U_cpu = U_cpu
        self.U = self.U_cpu

        if return_samples and Xsamp is not None:
            return U_cpu, Xsamp.detach().cpu().numpy().astype(np.float32)
        return U_cpu, None
