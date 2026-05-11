#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Waypoint:
    t: float
    x: float
    y: float
    z: float
    yaw: float


def angle_wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def parse_actor(world_path: Path, actor_name: str) -> tuple[float, list[Waypoint]]:
    root = ET.parse(world_path).getroot()
    actor = root.find(f".//actor[@name='{actor_name}']")
    if actor is None:
        raise ValueError(f"Actor '{actor_name}' not found in {world_path}")

    scale_text = actor.findtext("./skin/scale", default="1.0")
    scale = float(scale_text)

    waypoints: list[Waypoint] = []
    for waypoint in actor.findall("./script/trajectory/waypoint"):
        time_text = waypoint.findtext("time")
        pose_text = waypoint.findtext("pose")
        if time_text is None or pose_text is None:
            raise ValueError("Each actor waypoint must contain <time> and <pose>.")
        pose_vals = [float(v) for v in pose_text.split()]
        if len(pose_vals) != 6:
            raise ValueError(f"Expected 6 pose values, got {len(pose_vals)} in '{pose_text}'")
        x, y, z, _roll, _pitch, yaw = pose_vals
        waypoints.append(Waypoint(t=float(time_text), x=x, y=y, z=z, yaw=yaw))

    if len(waypoints) < 2:
        raise ValueError("Need at least two actor waypoints to build a CSV.")

    for prev, cur in zip(waypoints, waypoints[1:]):
        if cur.t <= prev.t:
            raise ValueError("Waypoint times must be strictly increasing.")

    return scale, waypoints


def evaluate_pose(waypoints: list[Waypoint], t_query: float) -> tuple[float, float, float, float]:
    cycle_duration = waypoints[-1].t
    if cycle_duration <= 0.0:
        raise ValueError("Actor cycle duration must be positive.")

    phase = math.fmod(t_query, cycle_duration)
    if phase < 0.0:
        phase += cycle_duration

    start_idx = 0
    for idx in range(len(waypoints) - 1):
        if waypoints[idx].t <= phase < waypoints[idx + 1].t:
            start_idx = idx
            break

    w0 = waypoints[start_idx]
    w1 = waypoints[start_idx + 1]
    seg_dt = w1.t - w0.t
    alpha = 0.0 if seg_dt <= 0.0 else (phase - w0.t) / seg_dt

    x = w0.x + alpha * (w1.x - w0.x)
    y = w0.y + alpha * (w1.y - w0.y)
    z = w0.z + alpha * (w1.z - w0.z)
    yaw_delta = angle_wrap(w1.yaw - w0.yaw)
    yaw = angle_wrap(w0.yaw + alpha * yaw_delta)
    return x, y, z, yaw


def build_rows(
    waypoints: list[Waypoint],
    dt: float,
    duration: float,
    obstacle_id: int,
    scale: float,
) -> list[dict[str, float | int]]:
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if duration <= 0.0:
        raise ValueError("duration must be positive.")

    num_steps = int(math.floor(duration / dt + 1e-9)) + 1
    times = [round(i * dt, 6) for i in range(num_steps)]
    poses = [evaluate_pose(waypoints, t) for t in times]

    speeds: list[float] = []
    yaw_rates: list[float] = []
    for idx in range(len(times)):
        if idx + 1 < len(times):
            x0, y0, _z0, yaw0 = poses[idx]
            x1, y1, _z1, yaw1 = poses[idx + 1]
            xdot = (x1 - x0) / dt
            ydot = (y1 - y0) / dt
            speeds.append(xdot * math.cos(yaw0) + ydot * math.sin(yaw0))
            yaw_rates.append(angle_wrap(yaw1 - yaw0) / dt)
        elif speeds:
            speeds.append(speeds[-1])
            yaw_rates.append(yaw_rates[-1])
        else:
            speeds.append(0.0)
            yaw_rates.append(0.0)

    # Keep acceleration at zero so abrupt waypoint transitions do not produce
    # unrealistically large spikes that would break the simple CSV horizon model.
    accels = [0.0 for _ in times]

    rows: list[dict[str, float | int]] = []
    for frame, (t, pose, v, phi_dot, a) in enumerate(zip(times, poses, speeds, yaw_rates, accels)):
        x, y, z, yaw = pose
        rows.append(
            {
                "id": obstacle_id,
                "frame": frame,
                "t": t,
                "x": round(x, 6),
                "y": round(y, 6),
                "v": round(v, 6),
                "phi": round(yaw, 6),
                "phi_dot": round(phi_dot, 6),
                "a": round(a, 6),
                "z": round(z, 6),
                "scale": round(scale, 6),
            }
        )
    return rows


def write_csv(rows: list[dict[str, float | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "frame", "t", "x", "y", "v", "phi", "phi_dot", "a", "z", "scale"]
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    default_world = repo_root / "gazebo_sim" / "world" / "cafe.world"
    default_output = Path(__file__).resolve().parents[1] / "Data" / "walking_actor_1_obstacle.csv"

    ap = argparse.ArgumentParser(
        description="Convert a Gazebo actor trajectory into a dense CSV for MPPI moving-obstacle inputs."
    )
    ap.add_argument("--world", type=Path, default=default_world)
    ap.add_argument("--actor-name", type=str, default="walking_actor_1")
    ap.add_argument("--output", type=Path, default=default_output)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--obstacle-id", type=int, default=1)
    args = ap.parse_args()

    scale, waypoints = parse_actor(args.world, args.actor_name)
    rows = build_rows(
        waypoints=waypoints,
        dt=float(args.dt),
        duration=float(args.duration),
        obstacle_id=int(args.obstacle_id),
        scale=scale,
    )
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
