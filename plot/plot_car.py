#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Dict

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D


METHOD_TO_FILE: Dict[str, str] = {
    "mppi": "mppi_car_simulation.npz",
    "ramppi": "ramppi_car_simulation.npz",
    "drmppi": "drmppi_car_simulation.npz",
    "dramppi": "dramppi_car_simulation.npz",
}

METHOD_ORDER = ["mppi", "ramppi", "drmppi", "dramppi"]
LEGEND_ORDER = ["mppi", "dramppi", "ramppi", "drmppi"]
METHOD_LABELS: Dict[str, str] = {
    "mppi": "MPPI",
    "ramppi": "RA-MPPI",
    "drmppi": "DR-MPPI",
    "dramppi": "DRA-MPPI",
}
METHOD_COLORS: Dict[str, str] = {
    "mppi": "#095ed5",
    "ramppi": "#9509b8",
    "drmppi": "#26867b",
    "dramppi": "#ff7300",
}
GROUND_COLOR = "#ece9e2"
ROAD_COLOR = "#646464"
ROUNDABOUT_ISLAND_COLOR = "#8d8d8d"
TRANSPARENT_SPRITE = np.zeros((1, 1, 4), dtype=np.float32)


def _hex_to_rgb01(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def _tint_sprite(img: np.ndarray, color: str, strength: float = 0.72) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32).copy()
    tint = _hex_to_rgb01(color)

    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return arr

    rgb = arr[..., :3]
    rgb = (1.0 - strength) * rgb + strength * (rgb * tint[None, None, :])
    arr[..., :3] = np.clip(rgb, 0.0, 1.0)
    return arr


def set_img_pose(img_artist, x, y, phi, length_along_heading, width_lateral, ax):
    L = float(length_along_heading) * 2.0
    W = float(width_lateral) * 2.0
    img_artist.set_extent([-L / 2.0, L / 2.0, -W / 2.0, W / 2.0])
    tr = Affine2D().rotate(phi).translate(x, y) + ax.transData
    img_artist.set_transform(tr)


def _set_obstacle_label(label_artist, x: float, y: float, oid: int, obs_length: float, obs_width: float) -> None:
    y_offset = 0.65 * max(float(obs_length), float(obs_width)) + 0.04
    label_artist.set_position((float(x), float(y) + y_offset))
    label_artist.set_text(f"({int(oid)})")
    label_artist.set_visible(True)


def _load_npz(base_dir: str, method: str) -> dict:
    if method not in METHOD_TO_FILE:
        raise ValueError(f"Unknown method: {method}")
    p = os.path.join(base_dir, METHOD_TO_FILE[method])
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing saved data: {p}. Run {method}.py with --save first.")
    data = np.load(p, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _draw_road_markings(
    ax,
    y_bottom: float,
    y_divider: float,
    y_top: float,
    y_top_outer: float | None,
    y_divider_top_1: float | None,
    y_divider_top_2: float | None,
) -> None:
    ax.axhline(y_bottom, color="white", linewidth=3.0, zorder=2)
    ax.axhline(
        y_divider,
        color="white",
        linewidth=2.0,
        linestyle=(0, (12, 12)),
        alpha=0.9,
        zorder=2,
    )
    if y_top_outer is not None and y_divider_top_1 is not None and y_divider_top_2 is not None:
        ax.axhline(y_divider_top_1, color="white", linewidth=2.0, alpha=0.9, zorder=2)
        ax.axhline(y_divider_top_2, color="white", linewidth=2.0, linestyle=(0, (12, 12)), alpha=0.9, zorder=2)
        ax.axhline(y_top_outer, color="white", linewidth=3.0, zorder=2)
    else:
        ax.axhline(y_top, color="white", linewidth=3.0, zorder=2)


def _finite_minmax(arr: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.0
    return float(np.min(vals)), float(np.max(vals))


def _lane_width_from_data(data: dict) -> float:
    widths = []
    if "y_bottom" in data and "y_divider" in data:
        widths.append(abs(float(data["y_divider"]) - float(data["y_bottom"])))
    if "y_divider" in data and "y_top" in data:
        widths.append(abs(float(data["y_top"]) - float(data["y_divider"])))
    widths = [w for w in widths if np.isfinite(w) and w > 1e-6]
    return float(np.median(widths)) if widths else 0.8


def _figure_size_for_scenario(scenario: int) -> tuple[float, float]:
    if scenario in (2, 3):
        return (12.5, 8.5)
    return (14.0, 5.0)


def _roundabout_center(
    data: dict,
    world_xmin: float,
    world_xmax: float,
    reference_span: tuple[float, float],
) -> tuple[float, float]:
    lane_y = float(data["lane_y"])
    fallback_center_x = float(world_xmin) + 0.33 * max(float(reference_span[0]), float(world_xmax - world_xmin), 1.0)
    center_x = float(data["roundabout_center_x"]) if "roundabout_center_x" in data else fallback_center_x
    center_y = float(data["roundabout_center_y"]) if "roundabout_center_y" in data else lane_y
    if not np.isfinite(center_x):
        center_x = fallback_center_x
    if not np.isfinite(center_y):
        center_y = lane_y
    return center_x, center_y


def _scenario_camera_view(
    data: dict,
    scenario: int,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    reference_span: tuple[float, float],
    world_xmin: float,
    world_xmax: float,
) -> tuple[float, float, float, float]:
    x_left = float(x_limits[0])
    x_right = float(x_limits[1])
    y_low = float(y_limits[0])
    y_high = float(y_limits[1])

    x_center = 0.5 * (x_left + x_right)
    y_center = 0.5 * (y_low + y_high)
    x_span = float(reference_span[0])
    y_span = float(reference_span[1])

    if scenario == 2:
        lane_width = _lane_width_from_data(data)
        _roundabout_center_x, roundabout_center_y = _roundabout_center(data, world_xmin, world_xmax, reference_span)

        # Scenario 2 should read like a focused roundabout shot rather than the
        # wide highway camera used by the original straight-road replay.
        x_center = 0.38 * x_left + 0.62 * x_right
        y_center = roundabout_center_y
        x_span = max(12.0, 0.72 * float(reference_span[0]), 10.5 * lane_width)
        y_span = max(5.4, 1.55 * float(reference_span[1]), 6.4 * lane_width)
    elif scenario == 3:
        lane_width = _lane_width_from_data(data)
        roundabout_center_x, roundabout_center_y = _roundabout_center(data, world_xmin, world_xmax, reference_span)

        # Scenario 3 keeps the roundabout fixed on screen while cars move
        # through the scene.
        x_center = roundabout_center_x
        y_center = roundabout_center_y
        x_span = max(15.0, 0.85 * float(reference_span[0]), 12.5 * lane_width)
        y_span = max(6.0, 1.70 * float(reference_span[1]), 6.8 * lane_width)

    return x_center, y_center, x_span, y_span


def _draw_roundabout_environment(
    ax,
    center_x: float,
    center_y: float,
    lane_width: float,
    world_xmin: float,
    world_xmax: float,
    world_ymin: float,
    world_ymax: float,
    reference_span: tuple[float, float],
) -> None:
    road_half_width = max(1.05 * float(lane_width), 0.55)
    # Match the roundabout ring thickness to the straight-road width.
    ring_center_radius = max(2.0 * float(lane_width), 1.55)
    outer_radius = ring_center_radius + road_half_width
    island_radius = max(ring_center_radius - road_half_width, 0.45)
    ring_radius = 0.5 * (outer_radius + island_radius)
    x_pad = max(0.5 * float(reference_span[0]), 6.0 * float(lane_width), 2.0)
    y_pad = max(0.5 * float(reference_span[1]), 9.0 * float(lane_width), 4.0)
    left_x = float(world_xmin) - x_pad
    right_x = float(world_xmax) + x_pad
    bottom_y = min(float(world_ymin) - y_pad, center_y - 8.0 * float(lane_width))
    top_y = max(float(world_ymax) + y_pad, center_y + 8.0 * float(lane_width))
    join = 0.90 * outer_radius

    ax.set_facecolor(GROUND_COLOR)

    road_boxes = [
        (left_x, center_y - road_half_width, center_x - join - left_x, 2.0 * road_half_width),
        (center_x + join, center_y - road_half_width, right_x - (center_x + join), 2.0 * road_half_width),
        (center_x - road_half_width, center_y + join, 2.0 * road_half_width, top_y - (center_y + join)),
        (center_x - road_half_width, bottom_y, 2.0 * road_half_width, center_y - join - bottom_y),
    ]
    for x0, y0, w, h in road_boxes:
        if w > 0.0 and h > 0.0:
            ax.add_patch(Rectangle((x0, y0), w, h, facecolor=ROAD_COLOR, edgecolor="none", zorder=0))

    ax.add_patch(Circle((center_x, center_y), outer_radius, facecolor=ROAD_COLOR, edgecolor="none", zorder=0))
    ax.add_patch(
        Circle(
            (center_x, center_y),
            island_radius,
            facecolor=ROUNDABOUT_ISLAND_COLOR,
            edgecolor="#767676",
            linewidth=1.0,
            zorder=1,
        )
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    ax.plot(
        center_x + ring_radius * np.cos(theta),
        center_y + ring_radius * np.sin(theta),
        color="white",
        linewidth=1.6,
        linestyle=(0, (8, 8)),
        alpha=0.9,
        zorder=2,
    )
    ax.plot([left_x, center_x - outer_radius], [center_y, center_y], color="white", linewidth=1.8, linestyle=(0, (12, 12)), alpha=0.9, zorder=2)
    ax.plot([center_x + outer_radius, right_x], [center_y, center_y], color="white", linewidth=1.8, linestyle=(0, (12, 12)), alpha=0.9, zorder=2)
    ax.plot([center_x, center_x], [center_y + outer_radius, top_y], color="white", linewidth=1.8, linestyle=(0, (12, 12)), alpha=0.9, zorder=2)
    ax.plot([center_x, center_x], [bottom_y, center_y - outer_radius], color="white", linewidth=1.8, linestyle=(0, (12, 12)), alpha=0.9, zorder=2)


def _setup_scene(
    ax,
    data: dict,
    scenario: int,
    reference_span: tuple[float, float],
    world_xmin: float,
    world_xmax: float,
    world_ymin: float,
    world_ymax: float,
) -> tuple[list, list]:
    lane_y = float(data["lane_y"])
    y_bottom = float(data["y_bottom"])
    y_top = float(data["y_top"])
    y_divider = float(data["y_divider"])
    y_divider_top_1 = float(data["y_divider_top_1"]) if "y_divider_top_1" in data else None
    y_divider_top_2 = float(data["y_divider_top_2"]) if "y_divider_top_2" in data else None
    y_top_outer = float(data["y_top_outer"]) if "y_top_outer" in data else None

    if scenario == 1:
        ax.set_facecolor(ROAD_COLOR)
        ax.set_aspect("auto")
        _draw_road_markings(
            ax,
            y_bottom=y_bottom,
            y_divider=y_divider,
            y_top=y_top,
            y_top_outer=y_top_outer,
            y_divider_top_1=y_divider_top_1,
            y_divider_top_2=y_divider_top_2,
        )
        lane_center_line, = ax.plot(
            [world_xmin - 10.0, world_xmax + 10.0],
            [lane_y, lane_y],
            "--",
            linewidth=1.5,
            color="#abbac6",
            alpha=0.8,
            label="Lane center",
        )
        return [lane_center_line], ["Lane center"]

    if scenario in (2, 3):
        ax.set_aspect("equal", adjustable="box")
        lane_width = _lane_width_from_data(data)
        center_x, center_y = _roundabout_center(data, world_xmin, world_xmax, reference_span)
        _draw_roundabout_environment(
            ax,
            center_x=center_x,
            center_y=center_y,
            lane_width=lane_width,
            world_xmin=world_xmin,
            world_xmax=world_xmax,
            world_ymin=world_ymin,
            world_ymax=world_ymax,
            reference_span=reference_span,
        )
        return [], []

    raise ValueError(f"Unknown scenario: {scenario}")


def _print_summary(method: str, data: dict) -> None:
    solve_ms = np.asarray(data["solve_ms"], dtype=float)
    x_hist = np.asarray(data["X_hist"], dtype=float)
    print(
        f"[{method}] file={METHOD_TO_FILE[method]} steps={x_hist.shape[0]} "
        f"solve_ms(min/mean/max)=({solve_ms.min():.2f}/{solve_ms.mean():.2f}/{solve_ms.max():.2f})"
    )


def _view_span(data: dict) -> tuple[float, float]:
    xlim_hist = np.asarray(data["xlim_hist"], dtype=float)
    ylim_hist = np.asarray(data["ylim_hist"], dtype=float)
    x_span = xlim_hist[:, 1] - xlim_hist[:, 0]
    y_span = ylim_hist[:, 1] - ylim_hist[:, 0]
    return float(np.nanmedian(x_span)), float(np.nanmedian(y_span))


def _ensure_frames_root(plot_dir: str) -> str:
    frames_root = os.path.join(plot_dir, "frames")
    os.makedirs(frames_root, exist_ok=True)
    return frames_root


def _frame_run_dir(frames_root: str, mode: str, scenario: int) -> str:
    frame_dir = os.path.join(frames_root, mode, f"scenario{scenario}")
    os.makedirs(frame_dir, exist_ok=True)
    return frame_dir


def _save_frame(fig, frame_dir: str, frame_idx: int) -> None:
    frame_path = os.path.join(frame_dir, f"frame_{frame_idx:04d}.svg")
    fig.canvas.draw()
    fig.savefig(frame_path, format="svg")


def _reference_path_xy(data: dict, scenario: int) -> np.ndarray:
    """Return saved waypoint/reference path, or straight lane reference for scenario 1."""
    if "ref_path_xy" in data:
        ref_path = np.asarray(data["ref_path_xy"], dtype=float)
        if ref_path.ndim == 2 and ref_path.shape[1] >= 2 and ref_path.shape[0] > 1:
            valid = np.all(np.isfinite(ref_path[:, :2]), axis=1)
            ref_path = ref_path[valid, :2]
            if ref_path.shape[0] > 1:
                return ref_path

    if scenario == 1 and "X_path" in data and "lane_y" in data:
        x_path = np.asarray(data["X_path"], dtype=float)
        valid_x = x_path[:, 0][np.isfinite(x_path[:, 0])]
        if valid_x.size >= 2:
            xs = np.linspace(float(np.min(valid_x)), float(np.max(valid_x)), 200)
            ys = np.full_like(xs, float(data["lane_y"]))
            return np.column_stack([xs, ys])

    return np.zeros((0, 2), dtype=float)


def _ego_obs_thresholds(data: dict) -> tuple[float, float, float]:
    ego_length = float(data["ego_length"])
    ego_width = float(data["ego_width"])
    obs_length = float(data["obs_length"])
    obs_width = float(data["obs_width"])
    ego_radius = 0.5 * np.sqrt(ego_length**2 + ego_width**2)
    obs_radius = 0.5 * np.sqrt(obs_length**2 + obs_width**2)
    collision_threshold = float(ego_radius + obs_radius)
    return ego_radius, obs_radius, collision_threshold


def _infer_min_dist_hist(
    data: dict,
    obs_xy: np.ndarray,
    k_hist: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    if "min_dist_hist" in data:
        hist = np.asarray(data["min_dist_hist"], dtype=float).reshape(-1)
        out = np.full((n_steps,), np.nan, dtype=float)
        out[: min(n_steps, hist.shape[0])] = hist[: min(n_steps, hist.shape[0])]
        return out

    x_hist = np.asarray(data["X_hist"], dtype=float)
    out = np.full((n_steps,), np.nan, dtype=float)
    for i in range(min(n_steps, x_hist.shape[0])):
        kk = int(k_hist[i]) if i < k_hist.shape[0] else 0
        if kk <= 0:
            continue
        valid_obs = obs_xy[i, :kk, :]
        valid_mask = np.all(np.isfinite(valid_obs), axis=1)
        valid_obs = valid_obs[valid_mask]
        if valid_obs.shape[0] == 0 or not np.all(np.isfinite(x_hist[i, :2])):
            continue
        out[i] = float(np.min(np.linalg.norm(valid_obs - x_hist[i, :2][None, :], axis=1)))
    return out


def _infer_collision_hist(
    data: dict,
    obs_xy: np.ndarray,
    k_hist: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    if "collision_hist" in data:
        hist = np.asarray(data["collision_hist"], dtype=int).reshape(-1)
        out = np.zeros((n_steps,), dtype=int)
        out[: min(n_steps, hist.shape[0])] = hist[: min(n_steps, hist.shape[0])]
        return out

    dmin_hist = _infer_min_dist_hist(data, obs_xy, k_hist, n_steps)
    _ego_radius, _obs_radius, collision_threshold = _ego_obs_thresholds(data)
    out = np.zeros((n_steps,), dtype=int)
    valid = np.isfinite(dmin_hist)
    out[valid] = (dmin_hist[valid] < collision_threshold).astype(int)
    return out


def _replay_method(
    method: str,
    data: dict,
    root_dir: str,
    reference_span: tuple[float, float],
    scenario: int,
    frames_root: str | None,
) -> None:
    dt = float(data["dt"])
    lane_y = float(data["lane_y"])
    y_bottom = float(data["y_bottom"])
    y_top = float(data["y_top"])
    y_divider = float(data["y_divider"])
    y_divider_top_1 = float(data["y_divider_top_1"]) if "y_divider_top_1" in data else None
    y_divider_top_2 = float(data["y_divider_top_2"]) if "y_divider_top_2" in data else None
    y_top_outer = float(data["y_top_outer"]) if "y_top_outer" in data else None
    ego_length = float(data["ego_length"])
    ego_width = float(data["ego_width"])
    obs_length = float(data["obs_length"])
    obs_width = float(data["obs_width"])

    sim_time = np.asarray(data["sim_time"], dtype=float)
    solve_ms = np.asarray(data["solve_ms"], dtype=float)
    if "ego_speed_hist" in data:
        ego_speed_hist = np.asarray(data["ego_speed_hist"], dtype=float)
    else:
        xy_step = np.diff(np.asarray(data["X_path"], dtype=float), axis=0)
        if dt > 0.0:
            ego_speed_hist = np.linalg.norm(xy_step, axis=1) / dt
        else:
            ego_speed_hist = np.full((sim_time.shape[0],), np.nan, dtype=float)
    if "v_des_hist" in data:
        v_des_hist = np.asarray(data["v_des_hist"], dtype=float)
    else:
        v_des_hist = None
    if "min_dist_hist" in data:
        min_dist_hist = np.asarray(data["min_dist_hist"], dtype=float)
    else:
        min_dist_hist = _infer_min_dist_hist(data, obs_xy=np.asarray(data["obs_xy"], dtype=float), k_hist=np.asarray(data["K_hist"], dtype=int), n_steps=np.asarray(data["X_hist"], dtype=float).shape[0])
    if "safety_hist" in data:
        safety_hist = np.asarray(data["safety_hist"], dtype=int)
    else:
        safety_hist = None
    if "collision_hist" in data:
        collision_hist = np.asarray(data["collision_hist"], dtype=int)
    else:
        collision_hist = _infer_collision_hist(data, obs_xy=np.asarray(data["obs_xy"], dtype=float), k_hist=np.asarray(data["K_hist"], dtype=int), n_steps=np.asarray(data["X_hist"], dtype=float).shape[0])
    x_hist = np.asarray(data["X_hist"], dtype=float)
    x_path = np.asarray(data["X_path"], dtype=float)
    ref_path_xy = _reference_path_xy(data, scenario)
    pred_nominal_xy = np.asarray(data["pred_nominal_xy"], dtype=float)
    pred_samples_xy = np.asarray(data["pred_samples_xy"], dtype=float)
    obs_xy = np.asarray(data["obs_xy"], dtype=float)
    obs_phi = np.asarray(data["obs_phi"], dtype=float)
    obs_ids = np.asarray(data["obs_ids"], dtype=int)
    k_hist = np.asarray(data["K_hist"], dtype=int)
    xlim_hist = np.asarray(data["xlim_hist"], dtype=float)
    ylim_hist = np.asarray(data["ylim_hist"], dtype=float)
    reference_x_span, reference_y_span = reference_span

    n_steps = x_hist.shape[0]
    n_show = pred_samples_xy.shape[2]
    max_obs_draw = obs_xy.shape[1]

    ego_img_path = os.path.join(root_dir, "Data", "car_ego.png")
    if not os.path.exists(ego_img_path):
        raise FileNotFoundError(f"Missing ego sprite: {ego_img_path}")
    car_ego_img = mpimg.imread(ego_img_path)

    obs_sprite_paths = [os.path.join(root_dir, "Data", f"car_obs{i}.png") for i in range(1, 15)]
    obs_sprite_imgs = []
    for p in obs_sprite_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing obstacle sprite: {p}")
        obs_sprite_imgs.append(mpimg.imread(p))

    frame_dir = _frame_run_dir(frames_root, method, scenario) if frames_root is not None else None

    num_sprites = len(obs_sprite_imgs)
    id_to_sprite = {}
    next_sprite = 0

    world_xmin, world_xmax = _finite_minmax(x_path[:, 0])
    world_ymin, world_ymax = _finite_minmax(x_path[:, 1])

    plt.ion()
    fig, ax = plt.subplots(figsize=_figure_size_for_scenario(scenario))
    fig.subplots_adjust(right=0.80, top=0.84)

    ax.set_xlabel("x [scaled m]")
    ax.set_ylabel("y [scaled m]")
    extra_handles, extra_labels = _setup_scene(
        ax,
        data=data,
        scenario=scenario,
        reference_span=reference_span,
        world_xmin=world_xmin,
        world_xmax=world_xmax,
        world_ymin=world_ymin,
        world_ymax=world_ymax,
    )
    if scenario == 3:
        x_center, y_center, x_span, y_span = _scenario_camera_view(
            data=data,
            scenario=scenario,
            x_limits=(world_xmin, world_xmax),
            y_limits=(world_ymin, world_ymax),
            reference_span=reference_span,
            world_xmin=world_xmin,
            world_xmax=world_xmax,
        )
        ax.set_xlim(x_center - 0.5 * x_span, x_center + 0.5 * x_span)
        ax.set_ylim(y_center - 0.5 * y_span, y_center + 0.5 * y_span)

    sample_lines = []
    for _ in range(n_show):
        ln, = ax.plot([], [], lw=1.1, color="#9ee2e9", alpha=0.12, zorder=1)
        sample_lines.append(ln)

    path_line, = ax.plot([], [], lw=2.6, color="#67bde2", label="Ego path")
    pred_line, = ax.plot([], [], lw=2.2, color="#36ff0e", alpha=0.95, label="MPPI prediction", zorder=3)
    # Reference / waypoint plotting intentionally disabled.

    ego_img_artist = ax.imshow(car_ego_img, extent=[-0.5, 0.5, -0.5, 0.5], zorder=6)

    obs_imgs = []
    obs_labels = []
    for _ in range(max_obs_draw):
        im = ax.imshow(TRANSPARENT_SPRITE, extent=[-0.5, 0.5, -0.5, 0.5], zorder=4, visible=False)
        obs_imgs.append(im)
        txt = ax.text(
            0.0,
            0.0,
            "",
            color="white",
            fontsize=8,
            ha="center",
            va="bottom",
            visible=False,
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=(0.0, 0.0, 0.0, 0.42), edgecolor="none"),
        )
        obs_labels.append(txt)

    legend_handles = [path_line, pred_line]
    legend_labels = ["Ego path", "MPPI prediction"]
    legend_handles += extra_handles
    legend_labels += extra_labels
    ax.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
        handlelength=2.6,
    )

    for i in range(n_steps):
        path_line.set_data(x_path[: i + 2, 0], x_path[: i + 2, 1])

        nom = pred_nominal_xy[i]
        valid_nom = np.isfinite(nom[:, 0]) & np.isfinite(nom[:, 1])
        pred_line.set_data(nom[valid_nom, 0], nom[valid_nom, 1])

        for j in range(n_show):
            sxy = pred_samples_xy[i, :, j, :]
            valid = np.isfinite(sxy[:, 0]) & np.isfinite(sxy[:, 1])
            if np.any(valid):
                sample_lines[j].set_data(sxy[valid, 0], sxy[valid, 1])
            else:
                sample_lines[j].set_data([], [])

        set_img_pose(ego_img_artist, x_hist[i, 0], x_hist[i, 1], x_hist[i, 2], ego_length, ego_width, ax)

        kk = int(k_hist[i])
        for j in range(max_obs_draw):
            if j < kk and np.all(np.isfinite(obs_xy[i, j])) and np.isfinite(obs_phi[i, j]):
                oid = int(obs_ids[i, j])
                if oid not in id_to_sprite:
                    id_to_sprite[oid] = next_sprite
                    next_sprite = (next_sprite + 1) % num_sprites

                sprite_idx = id_to_sprite[oid]
                obs_imgs[j].set_data(obs_sprite_imgs[sprite_idx])
                obs_imgs[j].set_visible(True)
                set_img_pose(
                    obs_imgs[j],
                    obs_xy[i, j, 0],
                    obs_xy[i, j, 1],
                    obs_phi[i, j],
                    obs_length,
                    obs_width,
                    ax,
                )
                # _set_obstacle_label(obs_labels[j], obs_xy[i, j, 0], obs_xy[i, j, 1], oid, obs_length, obs_width)
            else:
                obs_imgs[j].set_data(TRANSPARENT_SPRITE)
                obs_imgs[j].set_visible(False)
                obs_labels[j].set_visible(False)

        if scenario != 3 and np.all(np.isfinite(xlim_hist[i])) and np.all(np.isfinite(ylim_hist[i])):
            x_center, y_center, x_span, y_span = _scenario_camera_view(
                data=data,
                scenario=scenario,
                x_limits=(float(xlim_hist[i, 0]), float(xlim_hist[i, 1])),
                y_limits=(float(ylim_hist[i, 0]), float(ylim_hist[i, 1])),
                reference_span=reference_span,
                world_xmin=world_xmin,
                world_xmax=world_xmax,
            )
            ax.set_xlim(x_center - 0.5 * x_span, x_center + 0.5 * x_span)
            ax.set_ylim(y_center - 0.5 * y_span, y_center + 0.5 * y_span)

        title_main = (
            f"{method.upper()} | t={sim_time[i]:.2f}s | "
            f"v={ego_speed_hist[i]:.2f} | K={kk} | solve={solve_ms[i]:.2f}ms"
        )
        if v_des_hist is not None:
            title_main = (
                f"{method.upper()} | t={sim_time[i]:.2f}s | "
                f"v={ego_speed_hist[i]:.2f} | v_des={v_des_hist[i]:.2f} | "
                f"K={kk} | solve={solve_ms[i]:.2f}ms"
            )

        status_parts = []
        if min_dist_hist is not None and i < len(min_dist_hist) and np.isfinite(min_dist_hist[i]):
            status_parts.append(f"dmin={min_dist_hist[i]:.3f}")
        if safety_hist is not None and i < len(safety_hist):
            status_parts.append(f"safety={int(safety_hist[i])}")
        if collision_hist is not None and i < len(collision_hist):
            status_parts.append(f"collided={int(collision_hist[i])}")

        # if status_parts:
        #     ax.set_title(f"{title_main}\n" + " | ".join(status_parts))
        # else:
        #     ax.set_title(title_main)
        if frame_dir is not None:
            _save_frame(fig, frame_dir, i)
        plt.pause(max(0.005, dt))

    plt.ioff()
    if frame_dir is not None:
        print(f"[{method}] saved {n_steps} frame(s) to {frame_dir}")
    plt.show()


def _replay_compare(
    all_data: Dict[str, dict],
    root_dir: str,
    reference_span: tuple[float, float],
    scenario: int,
    frames_root: str | None,
) -> None:
    base_method = METHOD_ORDER[0]
    base_data = all_data[base_method]

    dt = float(base_data["dt"])
    lane_y = float(base_data["lane_y"])
    y_bottom = float(base_data["y_bottom"])
    y_top = float(base_data["y_top"])
    y_divider = float(base_data["y_divider"])
    y_divider_top_1 = float(base_data["y_divider_top_1"]) if "y_divider_top_1" in base_data else None
    y_divider_top_2 = float(base_data["y_divider_top_2"]) if "y_divider_top_2" in base_data else None
    y_top_outer = float(base_data["y_top_outer"]) if "y_top_outer" in base_data else None
    sim_times = {method: np.asarray(all_data[method]["sim_time"], dtype=float) for method in METHOD_ORDER}
    solve_ms_hist = {method: np.asarray(all_data[method]["solve_ms"], dtype=float) for method in METHOD_ORDER}
    x_hists = {method: np.asarray(all_data[method]["X_hist"], dtype=float) for method in METHOD_ORDER}
    x_paths = {method: np.asarray(all_data[method]["X_path"], dtype=float) for method in METHOD_ORDER}
    ref_path_xy = _reference_path_xy(base_data, scenario)
    xlim_hists = {method: np.asarray(all_data[method]["xlim_hist"], dtype=float) for method in METHOD_ORDER}
    ylim_hists = {method: np.asarray(all_data[method]["ylim_hist"], dtype=float) for method in METHOD_ORDER}
    reference_x_span, reference_y_span = reference_span

    n_steps = min(x_hists[method].shape[0] for method in METHOD_ORDER)
    max_path_len = min(x_paths[method].shape[0] for method in METHOD_ORDER)
    sim_time = sim_times[base_method][:n_steps]

    obs_xy = np.asarray(base_data["obs_xy"], dtype=float)[:n_steps]
    obs_phi = np.asarray(base_data["obs_phi"], dtype=float)[:n_steps]
    obs_ids = np.asarray(base_data["obs_ids"], dtype=int)[:n_steps]
    k_hist = np.asarray(base_data["K_hist"], dtype=int)[:n_steps]
    max_obs_draw = obs_xy.shape[1]
    collision_hists = {
        method: _infer_collision_hist(all_data[method], obs_xy, k_hist, n_steps)
        for method in METHOD_ORDER
    }
    collision_ever_hists = {
        method: np.maximum.accumulate(collision_hists[method]).astype(int)
        for method in METHOD_ORDER
    }

    obs_length = float(base_data["obs_length"])
    obs_width = float(base_data["obs_width"])

    obs_sprite_paths = [os.path.join(root_dir, "Data", f"car_obs{i}.png") for i in range(1, 15)]
    obs_sprite_imgs = []
    for p in obs_sprite_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing obstacle sprite: {p}")
        obs_sprite_imgs.append(mpimg.imread(p))

    frame_dir = _frame_run_dir(frames_root, "compare", scenario) if frames_root is not None else None

    num_sprites = len(obs_sprite_imgs)
    id_to_sprite = {}
    next_sprite = 0

    world_xmin = min(_finite_minmax(x_paths[method][:max_path_len, 0])[0] for method in METHOD_ORDER)
    world_xmax = max(_finite_minmax(x_paths[method][:max_path_len, 0])[1] for method in METHOD_ORDER)
    world_ymin = min(_finite_minmax(x_paths[method][:max_path_len, 1])[0] for method in METHOD_ORDER)
    world_ymax = max(_finite_minmax(x_paths[method][:max_path_len, 1])[1] for method in METHOD_ORDER)

    plt.ion()
    fig, ax = plt.subplots(figsize=_figure_size_for_scenario(scenario))
    fig.subplots_adjust(right=0.80, top=0.84)

    ax.set_xlabel("x [scaled m]")
    ax.set_ylabel("y [scaled m]")
    extra_handles, extra_labels = _setup_scene(
        ax,
        data=base_data,
        scenario=scenario,
        reference_span=reference_span,
        world_xmin=world_xmin,
        world_xmax=world_xmax,
        world_ymin=world_ymin,
        world_ymax=world_ymax,
    )
    if scenario == 3:
        x_center, y_center, x_span, y_span = _scenario_camera_view(
            data=base_data,
            scenario=scenario,
            x_limits=(world_xmin, world_xmax),
            y_limits=(world_ymin, world_ymax),
            reference_span=reference_span,
            world_xmin=world_xmin,
            world_xmax=world_xmax,
        )
        ax.set_xlim(x_center - 0.5 * x_span, x_center + 0.5 * x_span)
        ax.set_ylim(y_center - 0.5 * y_span, y_center + 0.5 * y_span)

    # Reference / waypoint plotting intentionally disabled.

    path_lines = {}
    ego_markers = {}
    for method in METHOD_ORDER:
        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        line, = ax.plot([], [], lw=2.8, color=color, label=label, zorder=3)
        marker, = ax.plot(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=8.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=6 + 0.1 * METHOD_ORDER.index(method),
        )
        path_lines[method] = line
        ego_markers[method] = marker

    obs_imgs = []
    obs_labels = []
    for _ in range(max_obs_draw):
        im = ax.imshow(TRANSPARENT_SPRITE, extent=[-0.5, 0.5, -0.5, 0.5], zorder=4, visible=False)
        obs_imgs.append(im)
        txt = ax.text(
            0.0,
            0.0,
            "",
            color="white",
            fontsize=8,
            ha="center",
            va="bottom",
            visible=False,
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=(0.0, 0.0, 0.0, 0.42), edgecolor="none"),
        )
        obs_labels.append(txt)

    compare_legend_handles = [path_lines[method] for method in LEGEND_ORDER]
    compare_legend_labels = [METHOD_LABELS[method] for method in LEGEND_ORDER]
    compare_legend_handles += extra_handles
    compare_legend_labels += extra_labels

    ax.legend(
        compare_legend_handles,
        compare_legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
        handlelength=2.6,
    )

    for i in range(n_steps):
        for method in METHOD_ORDER:
            path_end = min(i + 2, x_paths[method].shape[0])
            path_lines[method].set_data(x_paths[method][:path_end, 0], x_paths[method][:path_end, 1])
            ego_markers[method].set_data([x_hists[method][i, 0]], [x_hists[method][i, 1]])
            if int(collision_ever_hists[method][i]) != 0:
                ego_markers[method].set_markeredgecolor("#ff2d2d")
                ego_markers[method].set_markeredgewidth(2.2)
            else:
                ego_markers[method].set_markeredgecolor("white")
                ego_markers[method].set_markeredgewidth(1.0)

        kk = int(k_hist[i])
        for j in range(max_obs_draw):
            if j < kk and np.all(np.isfinite(obs_xy[i, j])) and np.isfinite(obs_phi[i, j]):
                oid = int(obs_ids[i, j])
                if oid not in id_to_sprite:
                    id_to_sprite[oid] = next_sprite
                    next_sprite = (next_sprite + 1) % num_sprites

                sprite_idx = id_to_sprite[oid]
                obs_imgs[j].set_data(obs_sprite_imgs[sprite_idx])
                obs_imgs[j].set_visible(True)
                set_img_pose(
                    obs_imgs[j],
                    obs_xy[i, j, 0],
                    obs_xy[i, j, 1],
                    obs_phi[i, j],
                    obs_length,
                    obs_width,
                    ax,
                )
                # _set_obstacle_label(obs_labels[j], obs_xy[i, j, 0], obs_xy[i, j, 1], oid, obs_length, obs_width)
            else:
                obs_imgs[j].set_data(TRANSPARENT_SPRITE)
                obs_imgs[j].set_visible(False)
                obs_labels[j].set_visible(False)

        x_lows = []
        x_highs = []
        y_lows = []
        y_highs = []
        for method in METHOD_ORDER:
            if i < xlim_hists[method].shape[0] and np.all(np.isfinite(xlim_hists[method][i])):
                x_lows.append(float(xlim_hists[method][i, 0]))
                x_highs.append(float(xlim_hists[method][i, 1]))
            if i < ylim_hists[method].shape[0] and np.all(np.isfinite(ylim_hists[method][i])):
                y_lows.append(float(ylim_hists[method][i, 0]))
                y_highs.append(float(ylim_hists[method][i, 1]))

        if scenario != 3 and x_lows and x_highs and y_lows and y_highs:
            x_center, y_center, x_span, y_span = _scenario_camera_view(
                data=base_data,
                scenario=scenario,
                x_limits=(min(x_lows), max(x_highs)),
                y_limits=(min(y_lows), max(y_highs)),
                reference_span=reference_span,
                world_xmin=world_xmin,
                world_xmax=world_xmax,
            )
            ax.set_xlim(x_center - 0.5 * x_span, x_center + 0.5 * x_span)
            ax.set_ylim(y_center - 0.5 * y_span, y_center + 0.5 * y_span)

        collision_text = " | ".join(
            f"{METHOD_LABELS[method]} c={int(collision_hists[method][i])}" for method in METHOD_ORDER
        )
        #ax.set_title(f"Compare replay | t={sim_time[i]:.2f}s\n{collision_text}")
        if frame_dir is not None:
            _save_frame(fig, frame_dir, i)
        plt.pause(max(0.005, dt))

    plt.ioff()
    if frame_dir is not None:
        print(f"[compare] saved {n_steps} frame(s) to {frame_dir}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved car simulations.")
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Scenario 1 keeps the current straight-road view. Scenario 2 draws a moving-camera roundabout view. Scenario 3 keeps the roundabout fixed while cars move.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Optional directory containing saved *.npz replay files. Defaults to this plot directory.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save replay frames as SVG files under plot/frames/<mode>/scenario<id>/.",
    )
    parser.add_argument("mode", choices=[*METHOD_TO_FILE.keys(), "all", "compare"], help="Method to replay")
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(here, ".."))
    data_dir = args.data_dir if args.data_dir else here
    frames_root = _ensure_frames_root(here) if args.save else None

    if args.mode == "compare":
        all_data = {method: _load_npz(data_dir, method) for method in METHOD_ORDER}
        reference_span = _view_span(all_data["drmppi"])
        for method in METHOD_ORDER:
            _print_summary(method, all_data[method])
        _replay_compare(all_data, root_dir, reference_span, args.scenario, frames_root)
        return

    reference_span = _view_span(_load_npz(data_dir, "drmppi"))

    if args.mode == "all":
        methods = METHOD_ORDER
    else:
        methods = [args.mode]

    for method in methods:
        data = _load_npz(data_dir, method)
        _print_summary(method, data)
        _replay_method(method, data, root_dir, reference_span, args.scenario, frames_root)


if __name__ == "__main__":
    main()
