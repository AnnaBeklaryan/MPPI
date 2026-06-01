#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


POS_SCALE = 0.10
Y_OFFSET_RAW = 23.0
ROUNDABOUT_CENTER_X = 6.1
ROUNDABOUT_CENTER_Y = 1.4
ROUNDABOUT_RADIUS = 1.16

T_START = 10.0
T_END = 25.0
DT = 0.04
FRAME_START = 251


def angle_wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def dense_path_from_waypoints(waypoints_xy: np.ndarray, ds: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(waypoints_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        raise ValueError("waypoints_xy must have shape (N,2) with N >= 2.")

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


def sample_motion_along_path(
    path_xy: np.ndarray,
    path_s: np.ndarray,
    times: np.ndarray,
    move_start: float,
    move_end: float,
) -> np.ndarray:
    total_len = float(path_s[-1])
    if total_len <= 0.0:
        raise ValueError("Path must be non-empty.")
    if not (move_end > move_start):
        raise ValueError("move_end must be greater than move_start.")

    out = np.zeros((times.shape[0], 2), dtype=float)
    for i, t_now in enumerate(times):
        if t_now <= move_start:
            s_now = 0.0
        elif t_now >= move_end:
            s_now = total_len
        else:
            alpha = (float(t_now) - float(move_start)) / float(move_end - move_start)
            s_now = alpha * total_len
        out[i, 0] = np.interp(s_now, path_s, path_xy[:, 0])
        out[i, 1] = np.interp(s_now, path_s, path_xy[:, 1])
    return out


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


def north_to_south_waypoints() -> np.ndarray:
    cx = ROUNDABOUT_CENTER_X
    cy = ROUNDABOUT_CENTER_Y
    r = ROUNDABOUT_RADIUS
    approach = np.array(
        [
            [cx, 6.6],
            [cx, 5.4],
            [cx, 4.2],
            [cx, cy + r],
        ],
        dtype=float,
    )
    arc_deg = np.array([90.0, 60.0, 30.0, 0.0, -30.0, -60.0, -90.0], dtype=float)
    arc = np.column_stack(
        [
            cx + r * np.cos(np.deg2rad(arc_deg)),
            cy + r * np.sin(np.deg2rad(arc_deg)),
        ]
    )
    exit_lane = np.array(
        [
            [cx, cy - r - 1.0],
            [cx, cy - r - 2.2],
            [cx, cy - r - 3.6],
            [cx, cy - r - 5.0],
        ],
        dtype=float,
    )
    return np.vstack([approach[:-1], arc, exit_lane])


def left_to_right_roundabout_waypoints() -> np.ndarray:
    cx = ROUNDABOUT_CENTER_X
    cy = ROUNDABOUT_CENTER_Y
    r = ROUNDABOUT_RADIUS
    approach = np.array(
        [
            [0.2, cy],
            [1.8, cy],
            [3.3, cy],
            [cx - r, cy],
        ],
        dtype=float,
    )
    arc_deg = np.array([180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 360.0], dtype=float)
    arc = np.column_stack(
        [
            cx + r * np.cos(np.deg2rad(arc_deg)),
            cy + r * np.sin(np.deg2rad(arc_deg)),
        ]
    )
    exit_lane = np.array(
        [
            [cx + r + 1.4, cy],
            [cx + r + 3.0, cy],
            [cx + r + 5.2, cy],
            [cx + r + 7.5, cy],
            [cx + r + 10.0, cy],
            [cx + r + 12.5, cy],
        ],
        dtype=float,
    )
    return np.vstack([approach[:-1], arc, exit_lane])


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


def build_dataset() -> list[dict[str, str]]:
    times = np.round(np.arange(T_START, T_END + 1e-9, DT), 6)
    frames = np.arange(FRAME_START, FRAME_START + times.shape[0], dtype=int)

    ns_path_xy, ns_path_s = dense_path_from_waypoints(north_to_south_waypoints(), ds=0.04)
    lr_path_xy, lr_path_s = dense_path_from_waypoints(left_to_right_roundabout_waypoints(), ds=0.04)

    north_to_south_xy = sample_motion_along_path(
        ns_path_xy,
        ns_path_s,
        times,
        move_start=10.2,
        move_end=T_END,
    )
    left_to_right_xy = sample_motion_along_path(
        lr_path_xy,
        lr_path_s,
        times,
        move_start=12.0,
        move_end=T_END,
    )

    rows = []
    rows.extend(build_rows_for_actor(60, north_to_south_xy, times, frames))
    rows.extend(build_rows_for_actor(61, left_to_right_xy, times, frames))
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
    default_output = Path(__file__).resolve().parents[1] / "Data" / "obstacle_data_2.csv"
    ap = argparse.ArgumentParser(description="Build a simple roundabout obstacle CSV for scenario 2.")
    ap.add_argument("--output", type=Path, default=default_output)
    args = ap.parse_args()

    rows = build_dataset()
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
