#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


POS_SCALE = 0.10
Y_OFFSET_RAW = 23.0
ROUNDABOUT_CENTER_X = 6.1
ROUNDABOUT_CENTER_Y = 1.4
STRAIGHT_LANE_Y = ROUNDABOUT_CENTER_Y - 0.42

TIME_START = 10.0
DT = 0.04
FRAME_START = 251
NUM_PATH_ACTORS = 5


def angle_wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def snap_time_to_grid(t_value: float) -> float:
    step_idx = int(np.round((float(t_value) - TIME_START) / DT))
    return float(TIME_START + step_idx * DT)


def dense_path_from_waypoints(waypoints_xy: np.ndarray, ds: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(waypoints_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        raise ValueError("waypoints_xy must have shape (N, 2) with N >= 2.")

    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = float(s[-1])
    if total_len <= 0.0:
        raise ValueError("Path length must be positive.")

    n_samples = max(int(math.ceil(total_len / max(float(ds), 1e-4))) + 1, pts.shape[0])
    sq = np.linspace(0.0, total_len, n_samples)
    x = np.interp(sq, s, pts[:, 0])
    y = np.interp(sq, s, pts[:, 1])
    return np.column_stack([x, y]), sq


def sample_xy_at_s(path_xy: np.ndarray, path_s: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    sq = np.clip(np.asarray(s_query, dtype=float), 0.0, float(path_s[-1]))
    x = np.interp(sq, path_s, path_xy[:, 0])
    y = np.interp(sq, path_s, path_xy[:, 1])
    return np.column_stack([x, y])


def kinematics_from_xy(xy_scaled: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_raw = xy_scaled[:, 0] / POS_SCALE
    y_raw = xy_scaled[:, 1] / POS_SCALE + Y_OFFSET_RAW

    vx_raw = np.gradient(x_raw, times)
    vy_raw = np.gradient(y_raw, times)
    phi = np.unwrap(np.arctan2(vy_raw, vx_raw))
    phi = np.array([angle_wrap(v) for v in phi], dtype=float)

    speed = np.hypot(vx_raw, vy_raw)
    accel = np.gradient(speed, times)
    phi_unwrapped = np.unwrap(phi)
    phi_dot = np.gradient(phi_unwrapped, times)
    return speed, phi, phi_dot, accel


def left_to_right_avoid_roundabout_waypoints(straight_y: float, radius: float) -> np.ndarray:
    cx = ROUNDABOUT_CENTER_X
    cy = ROUNDABOUT_CENTER_Y
    r = float(radius)

    approach = np.array(
        [
            [-2.4, straight_y],
            [-1.0, straight_y],
            [0.6, straight_y],
            [2.4, straight_y],
            [4.0, straight_y],
        ],
        dtype=float,
    )

    arc_angles_deg = np.array([194.0, 220.0, 248.0, 270.0, 292.0, 320.0, 346.0], dtype=float)
    arc = np.column_stack(
        [
            cx + r * np.cos(np.deg2rad(arc_angles_deg)),
            cy + r * np.sin(np.deg2rad(arc_angles_deg)),
        ]
    )

    exit_lane = np.array(
        [
            [8.25, straight_y],
            [9.8, straight_y],
            [11.4, straight_y],
            [13.0, straight_y],
            [14.2, straight_y],
        ],
        dtype=float,
    )
    return np.vstack([approach, arc, exit_lane])


def top_to_down_left_avoid_roundabout_waypoints(approach_x: float, radius: float, down_lane_x: float) -> np.ndarray:
    cx = ROUNDABOUT_CENTER_X
    cy = ROUNDABOUT_CENTER_Y
    r = float(radius)

    approach = np.array(
        [
            [approach_x, 6.6],
            [approach_x, 5.6],
            [approach_x, 4.6],
            [approach_x, 3.4],
            [cx, cy + r],
        ],
        dtype=float,
    )

    arc_angles_deg = np.array([90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0], dtype=float)
    arc = np.column_stack(
        [
            cx + r * np.cos(np.deg2rad(arc_angles_deg)),
            cy + r * np.sin(np.deg2rad(arc_angles_deg)),
        ]
    )

    exit_lane = np.array(
        [
            [down_lane_x, -0.7],
            [down_lane_x, -1.8],
            [down_lane_x, -2.9],
            [down_lane_x, -3.9],
        ],
        dtype=float,
    )
    return np.vstack([approach[:-1], arc, exit_lane])


def black_path_waypoints() -> np.ndarray:
    cx = ROUNDABOUT_CENTER_X
    cy = ROUNDABOUT_CENTER_Y
    lane_x = cx - 0.45
    radius = 1.65

    approach = np.array(
        [
            [lane_x, 6.6],
            [lane_x, 5.8],
            [lane_x, 5.0],
            [lane_x, 4.2],
            [lane_x, 3.45],
        ],
        dtype=float,
    )

    arc_angles_deg = np.array([108.0, 128.0, 150.0, 180.0, 210.0, 232.0, 252.0], dtype=float)
    arc = np.column_stack(
        [
            cx + radius * np.cos(np.deg2rad(arc_angles_deg)),
            cy + radius * np.sin(np.deg2rad(arc_angles_deg)),
        ]
    )

    exit_lane = np.array(
        [
            [lane_x, -0.2],
            [lane_x, -1.2],
            [lane_x, -2.4],
            [lane_x, -3.8],
        ],
        dtype=float,
    )
    return np.vstack([approach, arc, exit_lane])


def make_duration_profile(n: int, duration_min: float, duration_max: float) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.array([0.5 * (duration_min + duration_max)], dtype=float)
    return np.linspace(float(duration_min), float(duration_max), int(n), dtype=float)


def schedule_starts(durations: np.ndarray, path_len: float, base_start: float, min_gap_dist: float) -> np.ndarray:
    if durations.ndim != 1:
        raise ValueError("durations must be a 1D array.")
    if durations.size == 0:
        return np.zeros(0, dtype=float)

    starts = np.zeros(durations.shape[0], dtype=float)
    starts[0] = snap_time_to_grid(float(base_start))
    for i in range(1, durations.shape[0]):
        lead_speed = float(path_len) / max(float(durations[i - 1]), 1e-6)
        gap_time = max(0.68, float(min_gap_dist) / max(lead_speed, 0.40))
        start_i = snap_time_to_grid(starts[i - 1] + gap_time)
        if start_i <= starts[i - 1]:
            start_i = starts[i - 1] + DT
        starts[i] = start_i
    return starts


def build_rows_for_actor(
    obstacle_id: int,
    xy_scaled: np.ndarray,
    times: np.ndarray,
    frames: np.ndarray,
) -> list[dict[str, str]]:
    speed, phi, phi_dot, accel = kinematics_from_xy(xy_scaled, times)
    rows: list[dict[str, str]] = []
    for i in range(times.shape[0]):
        rows.append(
            {
                "id": str(int(obstacle_id)),
                "frame": str(int(frames[i])),
                "t": f"{times[i]:.2f}".rstrip("0").rstrip("."),
                "x": f"{xy_scaled[i, 0] / POS_SCALE:.2f}",
                "y": f"{xy_scaled[i, 1] / POS_SCALE + Y_OFFSET_RAW:.2f}",
                "v": f"{speed[i]:.12f}".rstrip("0").rstrip("."),
                "phi": f"{phi[i]:.12f}".rstrip("0").rstrip("."),
                "phi_dot": f"{phi_dot[i]:.12f}".rstrip("0").rstrip("."),
                "a": f"{accel[i]:.12f}".rstrip("0").rstrip("."),
            }
        )
    return rows


def build_stream_rows(
    actor_ids: list[int],
    path_xy: np.ndarray,
    path_s: np.ndarray,
    base_start: float,
    duration_min: float,
    duration_max: float,
    min_gap_dist: float,
) -> list[dict[str, str]]:
    durations = make_duration_profile(len(actor_ids), duration_min=duration_min, duration_max=duration_max)
    starts = schedule_starts(durations=durations, path_len=float(path_s[-1]), base_start=base_start, min_gap_dist=min_gap_dist)

    rows: list[dict[str, str]] = []
    for actor_id, duration_s, start_t in zip(actor_ids, durations, starts):
        n_samples = int(np.round(float(duration_s) / DT)) + 1
        times = start_t + np.arange(n_samples, dtype=float) * DT
        frames = FRAME_START + np.round((times - TIME_START) / DT).astype(int)
        s_query = np.linspace(0.0, float(path_s[-1]), n_samples, dtype=float)
        xy_scaled = sample_xy_at_s(path_xy, path_s, s_query)
        rows.extend(
            build_rows_for_actor(
                obstacle_id=int(actor_id),
                xy_scaled=xy_scaled,
                times=times,
                frames=frames,
            )
        )
    return rows


def build_dataset(source_path: Path) -> list[dict[str, str]]:
    source_df = pd.read_csv(source_path).sort_values(["id", "frame"]).reset_index(drop=True)
    summary = (
        source_df.groupby("id")
        .agg(
            t0=("t", "min"),
            x0=("x", "first"),
            x1=("x", "last"),
        )
        .reset_index()
    )

    actor_ids = (
        summary[summary["x1"] < summary["x0"]]
        .sort_values(["t0", "id"])["id"]
        .astype(int)
        .tolist()[:NUM_PATH_ACTORS]
    )
    path_xy, path_s = dense_path_from_waypoints(black_path_waypoints())

    rows: list[dict[str, str]] = []
    rows.extend(
        build_stream_rows(
            actor_ids=actor_ids,
            path_xy=path_xy,
            path_s=path_s,
            base_start=10.80,
            duration_min=5.8,
            duration_max=6.4,
            min_gap_dist=1.85,
        )
    )

    rows.sort(key=lambda row: (float(row["t"]), int(row["id"])))
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "frame", "t", "x", "y", "v", "phi", "phi_dot", "a"]
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_source = repo_root / "Data" / "obstacle_data.csv"
    default_output = repo_root / "Data" / "obstacle_data_3.csv"

    ap = argparse.ArgumentParser(description="Build obstacle_data_3.csv with staggered roundabout-avoid traffic.")
    ap.add_argument("--source", type=Path, default=default_source)
    ap.add_argument("--output", type=Path, default=default_output)
    args = ap.parse_args()

    rows = build_dataset(args.source)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
