# mppi_class.py
# -*- coding: utf-8 -*-
import numpy as np
import torch
from typing import Callable, Optional, Tuple, Union, Dict, Any

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
        I: int = 1,
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
        self.I = int(I)
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
            # Sigma is diagonal std; control weight uses diag(Sigma^{-1}) = 1 / sigma^2
            eps = torch.tensor(1e-12, device=self.device, dtype=self.dtype)
            self.ctrl_cost_R_from_sigma = 1.0 / torch.clamp(sig.reshape(-1) ** 2, min=eps)
        elif sig.ndim == 2:
            if sig.shape != (self.nu, self.nu):
                raise ValueError("noise_sigma cov must be (nu,nu)")
            self.ctrl_noise_mode = "cov"
            jitter = 1e-6 * torch.eye(self.nu, device=self.device, dtype=self.dtype)
            cov = sig + jitter
            self.noise_L = torch.linalg.cholesky(cov)
            self.noise_std = None
            # For generic running_cost that expects per-control weights,
            # use diagonal of Sigma^{-1}.
            inv_cov = torch.linalg.inv(cov)
            self.ctrl_cost_R_from_sigma = torch.diagonal(inv_cov)
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

    def _inject_sigma_inverse_control_weight(self):
        # Force control cost weight to come from Sigma^{-1} (requested behavior).
        self.cost_kwargs["R"] = self.ctrl_cost_R_from_sigma

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
        self._inject_sigma_inverse_control_weight()

        if return_samples:
            ns = int(min(max(1, int(n_show)), self.M))
            g = torch.Generator(device=self.device)
            g.manual_seed(int(show_seed))
            sample_idx = torch.randperm(self.M, generator=g, device=self.device)[:ns]
        else:
            ns = 0
            sample_idx = None

        for it in range(self.I):
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
                    print(f"[WARN] weight collapse it={it}, w_sum={float(w_sum.item())}")
                continue
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
      self.cost_kwargs["O_mean"]  (T,K,2)
      self.cost_kwargs["radii"]   (K,)
    """

    def __init__(
        self,
        *args,
        cvar_alpha: float = 0.9,
        cvar_N: int = 64,
        obs_pos_sigma: Union[Tuple[float, float], np.ndarray, Tensor] = (0.25, 0.25),
        obs_noise_mode: str = "static",  # "static" or "per_step"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.cvar_alpha = float(cvar_alpha)
        self.cvar_N = int(cvar_N)

        # IMPORTANT: use a different name than control noise mode
        self.obs_noise_mode = str(obs_noise_mode)
        if self.obs_noise_mode not in ("static", "per_step"):
            raise ValueError("obs_noise_mode must be 'static' or 'per_step'")

        self.obs_pos_sigma = _as_torch(obs_pos_sigma, self.device, self.dtype).reshape(2)

    def _sample_obstacles(
        self,
        O_mean: Tensor,                          # (T,K,2)
        obs_noise_std: Optional[Tensor] = None,  # optional standard normal noise
    ) -> Tensor:
        """
        Returns O_samp: (T,K,N,2)

        obs_noise_mode:
          - static: one noise per (K,N) shared over time
          - per_step: different noise per (T,K,N)
        """
        T, K, _ = O_mean.shape
        N = self.cvar_N
        if K == 0:
            return torch.zeros((T, 0, N, 2), device=self.device, dtype=self.dtype)

        sig = self.obs_pos_sigma.reshape(1, 1, 2)

        if self.obs_noise_mode == "static":
            if obs_noise_std is None:
                noise_kn = torch.randn((K, N, 2), device=self.device, dtype=self.dtype)
            else:
                noise_kn = obs_noise_std.to(device=self.device, dtype=self.dtype)
                if noise_kn.shape != (K, N, 2):
                    raise ValueError(f"obs_noise_std must be (K,N,2) got {tuple(noise_kn.shape)}")
            noise_kn = noise_kn * sig
            return O_mean[:, :, None, :] + noise_kn[None, :, :, :]
        else:
            if obs_noise_std is None:
                noise_tkn = torch.randn((T, K, N, 2), device=self.device, dtype=self.dtype)
            else:
                noise_tkn = obs_noise_std.to(device=self.device, dtype=self.dtype)
                if noise_tkn.shape != (T, K, N, 2):
                    raise ValueError(f"obs_noise_std must be (T,K,N,2) got {tuple(noise_tkn.shape)}")
            noise_tkn = noise_tkn * sig.reshape(1, 1, 1, 2)
            return O_mean[:, :, None, :] + noise_tkn

    def _cvar_from_sorted(self, L_sorted: Tensor) -> Tensor:
        """
        Matches your CuPy formula.
        L_sorted shape: (..., N)
        returns: (...) CVaR
        """
        alpha = self.cvar_alpha
        N = self.cvar_N
        k = int(np.ceil(alpha * N))
        k_idx = k - 1
        tail_sum = torch.sum(L_sorted[..., k_idx:], dim=-1)
        frac_term = (k - alpha * N) * L_sorted[..., k_idx]
        return (tail_sum + frac_term) / (N * (1.0 - alpha))

    def cvar_feasible_mask(
        self,
        X_hist_xy: Tensor,           # (T,M,2)
        O_mean: Optional[Tensor],    # (T,K,2)
        radii: Optional[Tensor],     # (K,)
        obs_noise_std: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
          feasible: (M,) bool
          cvar_max: (M,) float
        """
        T, M, _ = X_hist_xy.shape
        if (O_mean is None) or (radii is None) or (O_mean.numel() == 0) or (O_mean.shape[1] == 0):
            feasible = torch.ones((M,), device=self.device, dtype=torch.bool)
            cvar_max = torch.full((M,), -torch.inf, device=self.device, dtype=self.dtype)
            return feasible, cvar_max

        O_mean = O_mean.to(device=self.device, dtype=self.dtype)
        radii = radii.to(device=self.device, dtype=self.dtype).reshape(-1)
        K = int(O_mean.shape[1])

        O_samp = self._sample_obstacles(O_mean, obs_noise_std=obs_noise_std)  # (T,K,N,2)

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
        self._inject_sigma_inverse_control_weight()

        if obs_noise_std is not None:
            obs_noise_std = _as_torch(obs_noise_std, self.device, self.dtype)

        # read obstacle data from cost_kwargs (same place you update it in your loop)
        O_mean = self.cost_kwargs.get("O_mean", None)
        radii = self.cost_kwargs.get("radii", None)

        if return_samples:
            ns = int(min(max(1, int(n_show)), self.M))
            g = torch.Generator(device=self.device)
            g.manual_seed(int(show_seed))
            sample_idx = torch.randperm(self.M, generator=g, device=self.device)[:ns]
        else:
            ns = 0
            sample_idx = None

        debug = {}

        for it in range(self.I):
            eps = self._sample_eps()
            J = torch.zeros((self.M,), device=self.device, dtype=self.dtype)

            X = x0_t.unsqueeze(0).repeat(self.M, 1)
            X_hist_xy = torch.zeros((self.T, self.M, 2), device=self.device, dtype=self.dtype)

            if return_samples:
                nx = int(x0_t.numel())
                Xsamp = torch.zeros((self.T + 1, ns, nx), device=self.device, dtype=self.dtype)
                Xsamp[0] = X.index_select(0, sample_idx)
            else:
                Xsamp = None

            for t in range(self.T):
                U_t = torch.clamp(U[t].unsqueeze(0) + eps[:, t, :], self.u_min, self.u_max)
                X = self.dynamics(X, U_t, self.dt, **self.dyn_kwargs)
                X_hist_xy[t] = X[:, :2]

                c = self.running_cost(X, U_t, t, **self.cost_kwargs)
                J = J + c

                if return_samples:
                    Xsamp[t + 1] = X.index_select(0, sample_idx)

            J = J + self.terminal_cost(X, self.T, **self.cost_kwargs)
            J = torch.nan_to_num(J, nan=torch.tensor(float("inf"), device=self.device, dtype=self.dtype))

            feasible, cvar_max = self.cvar_feasible_mask(X_hist_xy, O_mean, radii, obs_noise_std=obs_noise_std)

            if not bool(torch.any(feasible).item()):
                best = int(torch.argmin(cvar_max).item())
                feasible = torch.zeros_like(feasible)
                feasible[best] = True

            Jf = J[feasible]
            rho = torch.min(Jf)

            w = torch.zeros_like(J)
            w[feasible] = torch.exp(-(Jf - rho) / self.lam)

            w_sum = torch.sum(w)
            w_sum_val = float(w_sum.item())
            feas_count = int(torch.sum(feasible).item())

            debug = {
                "it": it,
                "feasible_count": feas_count,
                "w_sum": w_sum_val,
                "cvar_min": float(torch.min(cvar_max).item()),
                "cvar_med": float(torch.median(cvar_max).item()),
                "cvar_max": float(torch.max(cvar_max).item()),
            }

            if (not np.isfinite(w_sum_val)) or (w_sum_val < self.weight_floor):
                if self.verbose:
                    print(f"[WARN] weight collapse it={it}: {debug}")
                continue

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
        X_hist_xy: Tensor,           # (T,M,2)
        O_mean: Optional[Tensor],    # (T,K,2)
        radii: Optional[Tensor],     # (K,)
        obs_noise_std: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Same as RA_MPPI.cvar_feasible_mask, but:
          - uses dx/dy distance calc (to avoid huge (..,2) diff tensor)
          - applies DR correction on cvar_per_obs
        Returns:
          feasible: (M,) bool
          cvar_max: (M,) float
        """
        T, M, _ = X_hist_xy.shape

        if (O_mean is None) or (radii is None) or (O_mean.numel() == 0) or (O_mean.shape[1] == 0):
            feasible = torch.ones((M,), device=self.device, dtype=torch.bool)
            cvar_max = torch.full((M,), -torch.inf, device=self.device, dtype=self.dtype)
            return feasible, cvar_max

        O_mean = O_mean.to(device=self.device, dtype=self.dtype)
        radii = radii.to(device=self.device, dtype=self.dtype).reshape(-1)
        K = int(O_mean.shape[1])
        N = int(self.cvar_N)

        O_samp = self._sample_obstacles(O_mean, obs_noise_std=obs_noise_std)  # (T,K,N,2)

        # dist without allocating huge diff tensor:
        # X_hist_xy: (T,M,2), O_samp: (T,K,N,2)
        dx = X_hist_xy[:, :, None, None, 0] - O_samp[:, None, :, :, 0]  # (T,M,K,N)
        dy = X_hist_xy[:, :, None, None, 1] - O_samp[:, None, :, :, 1]  # (T,M,K,N)
        dist = torch.sqrt(dx * dx + dy * dy)                            # (T,M,K,N)

        # g = R - dist
        Rk = radii.reshape(1, 1, K, 1)
        g = Rk - dist                                                  # (T,M,K,N)

        # L_worst = max_t g
        L_worst = torch.amax(g, dim=0)                                 # (M,K,N)

        # sort over samples
        Ls, _ = torch.sort(L_worst, dim=-1)                            # (M,K,N)
        cvar_per_obs = self._cvar_from_sorted(Ls)                      # (M,K)

        # DR correction (exactly your CuPy)
        if self.dr_eps_cvar > 0.0:
            cvar_per_obs = cvar_per_obs + (self.dr_eps_cvar * self._cvar_lip)

        cvar_max = torch.amax(cvar_per_obs, dim=1)                     # (M,)
        feasible = (cvar_max <= 0.0)
        return feasible, cvar_max
