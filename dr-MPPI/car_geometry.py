#!/usr/bin/env python3

import math

import numpy as np
import torch

Tensor = torch.Tensor


def _rect_corners_np(center_xy: np.ndarray, phi: float, half_length: float, half_width: float) -> np.ndarray:
    c = math.cos(float(phi))
    s = math.sin(float(phi))
    u = np.array([c, s], dtype=float)
    v = np.array([-s, c], dtype=float)
    center_xy = np.asarray(center_xy, dtype=float)
    return np.stack(
        [
            center_xy + half_length * u + half_width * v,
            center_xy + half_length * u - half_width * v,
            center_xy - half_length * u - half_width * v,
            center_xy - half_length * u + half_width * v,
        ],
        axis=0,
    )


def _proj_overlap_np(corners_a: np.ndarray, corners_b: np.ndarray, axis: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=float)
    vals_a = corners_a @ axis
    vals_b = corners_b @ axis
    return float(min(vals_a.max(), vals_b.max()) - max(vals_a.min(), vals_b.min()))


def _point_segment_dist_sq_np(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    point = np.asarray(point, dtype=float)
    seg_a = np.asarray(seg_a, dtype=float)
    seg_b = np.asarray(seg_b, dtype=float)
    edge = seg_b - seg_a
    denom = float(np.dot(edge, edge))
    if denom <= 1e-12:
        diff = point - seg_a
        return float(np.dot(diff, diff))
    t = float(np.dot(point - seg_a, edge) / denom)
    t = min(1.0, max(0.0, t))
    closest = seg_a + t * edge
    diff = point - closest
    return float(np.dot(diff, diff))


def rectangle_signed_distance_np(
    ego_xy: np.ndarray,
    ego_phi: float,
    ego_half_length: float,
    ego_half_width: float,
    obs_xy: np.ndarray,
    obs_phi: float,
    obs_half_length: float,
    obs_half_width: float,
) -> float:
    corners_a = _rect_corners_np(ego_xy, ego_phi, ego_half_length, ego_half_width)
    corners_b = _rect_corners_np(obs_xy, obs_phi, obs_half_length, obs_half_width)

    axes = []
    for phi in (ego_phi, obs_phi):
        c = math.cos(float(phi))
        s = math.sin(float(phi))
        axes.append(np.array([c, s], dtype=float))
        axes.append(np.array([-s, c], dtype=float))

    overlaps = np.array([_proj_overlap_np(corners_a, corners_b, axis) for axis in axes], dtype=float)
    if np.all(overlaps >= 0.0):
        return float(-np.min(overlaps))

    best_sq = float("inf")
    next_a = np.roll(corners_a, -1, axis=0)
    next_b = np.roll(corners_b, -1, axis=0)
    for point in corners_a:
        for seg_a, seg_b in zip(corners_b, next_b):
            best_sq = min(best_sq, _point_segment_dist_sq_np(point, seg_a, seg_b))
    for point in corners_b:
        for seg_a, seg_b in zip(corners_a, next_a):
            best_sq = min(best_sq, _point_segment_dist_sq_np(point, seg_a, seg_b))
    return float(math.sqrt(max(best_sq, 0.0)))


def min_signed_distance_to_obstacles_np(
    ego_xy: np.ndarray,
    ego_phi: float,
    obs_xy: np.ndarray,
    obs_phi: np.ndarray,
    ego_half_length: float,
    ego_half_width: float,
    obs_half_length: float,
    obs_half_width: float,
) -> float:
    obs_xy = np.asarray(obs_xy, dtype=float)
    obs_phi = np.asarray(obs_phi, dtype=float).reshape(-1)
    if obs_xy.shape[0] == 0:
        return np.nan

    best = float("inf")
    for idx in range(obs_xy.shape[0]):
        best = min(
            best,
            rectangle_signed_distance_np(
                ego_xy=ego_xy,
                ego_phi=ego_phi,
                ego_half_length=ego_half_length,
                ego_half_width=ego_half_width,
                obs_xy=obs_xy[idx],
                obs_phi=float(obs_phi[idx]),
                obs_half_length=obs_half_length,
                obs_half_width=obs_half_width,
            ),
        )
    return float(best)


def _rect_basis_torch(phi: Tensor) -> tuple[Tensor, Tensor]:
    c = torch.cos(phi)
    s = torch.sin(phi)
    u = torch.stack((c, s), dim=-1)
    v = torch.stack((-s, c), dim=-1)
    return u, v


def _rect_corners_torch(center_xy: Tensor, phi: Tensor, half_length: float | Tensor, half_width: float | Tensor) -> Tensor:
    hl = torch.as_tensor(half_length, device=center_xy.device, dtype=center_xy.dtype)
    hw = torch.as_tensor(half_width, device=center_xy.device, dtype=center_xy.dtype)
    u, v = _rect_basis_torch(phi)
    return torch.stack(
        [
            center_xy + hl * u + hw * v,
            center_xy + hl * u - hw * v,
            center_xy - hl * u - hw * v,
            center_xy - hl * u + hw * v,
        ],
        dim=-2,
    )


def _proj_overlap_torch(corners_a: Tensor, corners_b: Tensor, axis: Tensor) -> Tensor:
    vals_a = torch.sum(corners_a * axis.unsqueeze(-2), dim=-1)
    vals_b = torch.sum(corners_b * axis.unsqueeze(-2), dim=-1)
    return torch.minimum(vals_a.max(dim=-1).values, vals_b.max(dim=-1).values) - torch.maximum(
        vals_a.min(dim=-1).values, vals_b.min(dim=-1).values
    )


def _point_segment_min_dist_sq_torch(points: Tensor, seg_start: Tensor, seg_end: Tensor, eps: float = 1e-9) -> Tensor:
    edge = seg_end - seg_start
    edge_norm_sq = torch.sum(edge * edge, dim=-1).clamp_min(eps)
    rel = points.unsqueeze(-2) - seg_start.unsqueeze(-3)
    t = torch.sum(rel * edge.unsqueeze(-3), dim=-1) / edge_norm_sq.unsqueeze(-2)
    t = torch.clamp(t, 0.0, 1.0)
    closest = seg_start.unsqueeze(-3) + t.unsqueeze(-1) * edge.unsqueeze(-3)
    dist_sq = torch.sum((points.unsqueeze(-2) - closest) ** 2, dim=-1)
    return dist_sq.reshape(*dist_sq.shape[:-2], -1).min(dim=-1).values


def rectangle_signed_distance_torch(
    ego_xy: Tensor,
    ego_phi: Tensor,
    ego_half_length: float | Tensor,
    ego_half_width: float | Tensor,
    obs_xy: Tensor,
    obs_phi: Tensor,
    obs_half_length: float | Tensor,
    obs_half_width: float | Tensor,
    eps: float = 1e-9,
) -> Tensor:
    corners_a = _rect_corners_torch(ego_xy, ego_phi, ego_half_length, ego_half_width)
    corners_b = _rect_corners_torch(obs_xy, obs_phi, obs_half_length, obs_half_width)
    u_a, v_a = _rect_basis_torch(ego_phi)
    u_b, v_b = _rect_basis_torch(obs_phi)

    overlaps = torch.stack(
        [
            _proj_overlap_torch(corners_a, corners_b, u_a),
            _proj_overlap_torch(corners_a, corners_b, v_a),
            _proj_overlap_torch(corners_a, corners_b, u_b),
            _proj_overlap_torch(corners_a, corners_b, v_b),
        ],
        dim=-1,
    )
    separated = torch.any(overlaps < 0.0, dim=-1)
    penetration = overlaps.min(dim=-1).values

    next_a = torch.roll(corners_a, shifts=-1, dims=-2)
    next_b = torch.roll(corners_b, shifts=-1, dims=-2)
    a_to_b_sq = _point_segment_min_dist_sq_torch(corners_a, corners_b, next_b, eps=eps)
    b_to_a_sq = _point_segment_min_dist_sq_torch(corners_b, corners_a, next_a, eps=eps)
    sep_dist = torch.sqrt(torch.minimum(a_to_b_sq, b_to_a_sq).clamp_min(0.0))
    return torch.where(separated, sep_dist, -penetration)
