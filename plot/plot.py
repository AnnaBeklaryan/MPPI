#!/usr/bin/env python3
"""Replay saved Crazyflie MPPI simulations using only Matplotlib.

Examples:
  python plot/plot.py mppi
  python plot/plot.py ramppi
  python plot/plot.py all
  python plot/plot.py compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PLOT_DIR = Path(__file__).resolve().parent
MPPI_DIR = PLOT_DIR.parent
if str(MPPI_DIR) not in sys.path:
    sys.path.insert(0, str(MPPI_DIR))

METHODS = {
    "mppi": ("mppi_crazyflie", "mppi_simulation.npz"),
    "ramppi": ("RA_mppi_crazyflie", "ramppi_simulation.npz"),
    "drmppi": ("DR_mppi_crazyflie", "drmppi_simulation.npz"),
    "dramppi": ("DRA_mppi_crazyflie", "dramppi_simulation.npz"),
}
LABELS = {
    "mppi": "MPPI",
    "ramppi": "RA-MPPI",
    "drmppi": "DR-MPPI",
    "dramppi": "DRA-MPPI",
}
DRONE_COLORS = {
    "mppi": "#5B9BFF",
    "ramppi": "#C56BE3",
    "drmppi": "#55B8AC",
    "dramppi": "#FFAA5C",
}
OBSTACLE_COLORS = ["#EB1D87", "#34980C", "#EC6613", "#FF8C00"]

CRAZYFLIE_ARM_LEN = 0.04
SCENE_PADDING = np.array([0.20, 0.20, 0.35])
COMPARE_PADDING = np.array([0.08, 0.08, 0.12])
SAVED_AXIS_PADDING = 1.0
AXIS_LABEL_FONTSIZE = 22
AXIS_TICK_FONTSIZE = 22
AXIS_LABEL_PAD = 30
AXIS_TICK_PAD = 9


def _saved_path(method: str) -> Path:
    return PLOT_DIR / METHODS[method][1]


def _load_data(method: str) -> dict[str, np.ndarray]:
    path = _saved_path(method)
    if not path.exists():
        raise FileNotFoundError(f"Missing saved file: {path}")
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def _scalar(data: dict[str, np.ndarray], key: str, default: float) -> float:
    values = np.asarray(data.get(key, default), dtype=float).reshape(-1)
    return float(values[0]) if values.size and np.isfinite(values[0]) else default


def _flags(data: dict[str, np.ndarray], steps: int) -> np.ndarray:
    source = np.asarray(data.get("collision_flags", []), dtype=bool).reshape(-1)
    result = np.zeros(steps, dtype=bool)
    result[: min(steps, source.size)] = source[:steps]
    return result


def _obstacles(data: dict[str, np.ndarray], steps: int) -> np.ndarray:
    values = np.asarray(data.get("obs_path", np.zeros((steps, 0, 3))), dtype=float)
    if values.ndim == 2:
        values = values[:, None, :]
    return values[:steps] if values.ndim == 3 else np.zeros((steps, 0, 3))


def _bounds(data: dict[str, np.ndarray], paths: list[np.ndarray], compare: bool) -> tuple[np.ndarray, np.ndarray]:
    if compare:
        points = list(paths)
        obs = _obstacles(data, min(len(path) for path in paths))
        ref = np.asarray(data.get("ref_curve", np.empty((0, 3))), dtype=float)
        if obs.size:
            points.append(obs.reshape(-1, 3))
        if ref.size:
            points.append(ref.reshape(-1, 3))
        visible = np.vstack(points)
        visible = visible[np.all(np.isfinite(visible), axis=1)]
        return visible.min(axis=0) - COMPARE_PADDING, visible.max(axis=0) + COMPARE_PADDING

    path = paths[0]
    mins = np.asarray(data.get("mins", path.min(axis=0) - 1.0), dtype=float).reshape(3)
    maxs = np.asarray(data.get("maxs", path.max(axis=0) + 1.0), dtype=float).reshape(3)
    span = np.maximum(maxs - mins, 1e-6)
    inset = np.minimum(np.maximum(0.0, SAVED_AXIS_PADDING - SCENE_PADDING), span * 0.25)
    return mins + inset, maxs - inset


def _style_axes(ax, mins: np.ndarray, maxs: np.ndarray) -> None:
    ax.set_xlim(*mins[[0]], *maxs[[0]])
    ax.set_ylim(*mins[[1]], *maxs[[1]])
    ax.set_zlim(*mins[[2]], *maxs[[2]])
    ax.set_xlabel("x [m]", fontsize=AXIS_LABEL_FONTSIZE, color="black", labelpad=AXIS_LABEL_PAD)
    ax.set_ylabel("y [m]", fontsize=AXIS_LABEL_FONTSIZE, color="black", labelpad=AXIS_LABEL_PAD)
    ax.set_zlabel("z [m]", fontsize=AXIS_LABEL_FONTSIZE, color="black", labelpad=AXIS_LABEL_PAD)
    ax.set_facecolor("white")
    ax.set_box_aspect(np.maximum(maxs - mins, 1e-6), zoom=0.94)
    ax.view_init(elev=25, azim=-111)
    pane = (0.957, 0.976, 0.988, 1.0)
    grid = (0.70, 0.76, 0.82, 0.75)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    compact_number = FuncFormatter(lambda value, _position: f"{0.0 if abs(value) < 1e-12 else value:g}")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane)
        axis.set_major_formatter(compact_number)
        axis._axinfo["grid"]["color"] = grid
        axis._axinfo["grid"]["linewidth"] = 0.7
    ax.tick_params(labelsize=AXIS_TICK_FONTSIZE, colors="black", pad=AXIS_TICK_PAD)
    ax.grid(True, which="major")


def _sphere(ax, center: np.ndarray, radius: float, color: str, alpha: float = 0.11) -> None:
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=False)


def _cross(ax, center: np.ndarray, color: str, rotation: np.ndarray | None = None) -> None:
    if rotation is None:
        d1 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        d2 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    else:
        xb, yb = rotation[:, 0], rotation[:, 1]
        d1 = (xb + yb) / (np.linalg.norm(xb + yb) + 1e-12)
        d2 = (xb - yb) / (np.linalg.norm(xb - yb) + 1e-12)
    for direction in (d1, d2):
        ends = np.vstack((center - CRAZYFLIE_ARM_LEN * direction, center + CRAZYFLIE_ARM_LEN * direction))
        ax.plot(*ends.T, color=color, linewidth=2.2)


def _rotation(data: dict[str, np.ndarray], frame: int) -> np.ndarray | None:
    if "X_hist" not in data or np.asarray(data["X_hist"]).shape[1] < 9:
        return None
    psi, phi, theta = np.asarray(data["X_hist"])[frame, 6:9]
    cp, sp, ct, st, cy, sy = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return rz @ ry @ rx


def _cube(ax, cube: np.ndarray) -> None:
    cx, cy, radius, z0, z1 = map(float, cube)
    x0, x1, y0, y1 = cx - radius, cx + radius, cy - radius, cy + radius
    vertices = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    faces = [[vertices[i] for i in face] for face in
             ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
              (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))]
    ax.add_collection3d(Poly3DCollection(faces, facecolor="#8D959D", edgecolor="black",
                                         linewidth=0.7, alpha=0.42))


def _static_scene(ax, data: dict[str, np.ndarray]) -> None:
    ref = np.asarray(data.get("ref_curve", np.empty((0, 3))), dtype=float)
    if ref.size:
        ax.plot(*ref.T, color="#2a6fdb", linewidth=1.5, linestyle=":", label="Reference")
    for cube in np.asarray(data.get("cylinders", np.empty((0, 5))), dtype=float):
        _cube(ax, cube)


def _draw_obstacles(ax, data: dict[str, np.ndarray], obs: np.ndarray, frame: int,
                    obstacle_radius: float) -> None:
    for j in range(obs.shape[1]):
        color = OBSTACLE_COLORS[j % len(OBSTACLE_COLORS)]
        ax.plot(*obs[:frame + 1, j].T, color=color, linewidth=1.3, linestyle=":",
                label=f"Obstacle {j + 1}")
        point = obs[frame, j]
        _cross(ax, point, color)
        _sphere(ax, point, obstacle_radius, color, 0.09)


def _draw_single(ax, method: str, data: dict[str, np.ndarray], frame: int,
                 mins: np.ndarray, maxs: np.ndarray) -> None:
    path = np.asarray(data["X_path"], dtype=float)
    steps = len(path)
    obs = _obstacles(data, steps)
    flags = _flags(data, steps)
    drone_radius = _scalar(data, "drone_radius", 0.3662153322325755)
    obstacle_radius = max(1e-6, _scalar(data, "moving_collision_radius",
                                       0.3045201563139591 + drone_radius) - drone_radius)
    color = "#557EDC"

    ax.cla()
    _style_axes(ax, mins, maxs)
    _static_scene(ax, data)
    _draw_obstacles(ax, data, obs, frame, obstacle_radius)

    ax.plot(*path[:frame + 1].T, color=color, linewidth=1.8, label="Drone path")
    point = path[frame]
    ax.scatter(*point, color=color, s=24, depthshade=False)
    _cross(ax, point, color, _rotation(data, frame))
    _sphere(ax, point, drone_radius, color)

    samples = np.asarray(data.get("pred_samples_xyz", []), dtype=float)
    if samples.ndim == 4 and frame < len(samples):
        for sample in np.moveaxis(samples[frame], 1, 0):
            valid = np.all(np.isfinite(sample), axis=1)
            if np.any(valid):
                ax.plot(*sample[valid].T, color="#C70039", linewidth=0.5, alpha=0.25)
    nominal = np.asarray(data.get("pred_nominal_xyz", []), dtype=float)
    if nominal.ndim == 3 and frame < len(nominal):
        valid = np.all(np.isfinite(nominal[frame]), axis=1)
        if np.any(valid):
            ax.plot(*nominal[frame, valid].T, color="#C70039", linewidth=1.8)

    collided = path[:frame + 1][flags[:frame + 1]]
    if collided.size:
        ax.scatter(*collided.T, color="#D00000", marker="x", s=34, linewidths=1.6,
                   depthshade=False)
    collision = np.flatnonzero(flags)
    suffix = f"   collision step={collision[0]}" if collision.size and frame >= collision[0] else ""
    times = np.asarray(data["sim_time"])
    solve = np.asarray(data["solve_ms"])
    ax.set_title(f"{method.upper()} replay | t={times[frame]:.2f}s | solve={solve[frame]:.1f}ms{suffix}")


def _draw_compare(ax, all_data: dict[str, dict[str, np.ndarray]], frame: int,
                  mins: np.ndarray, maxs: np.ndarray) -> None:
    methods = list(METHODS)
    base = all_data["mppi"]
    steps = min(len(all_data[m]["X_path"]) for m in methods)
    obs = _obstacles(base, steps)
    base_radius = _scalar(base, "drone_radius", 0.3662153322325755)
    obstacle_radius = max(1e-6, _scalar(base, "moving_collision_radius",
                                       0.3045201563139591 + base_radius) - base_radius)

    ax.cla()
    _style_axes(ax, mins, maxs)
    _static_scene(ax, base)
    _draw_obstacles(ax, base, obs, frame, obstacle_radius)

    collided_labels = []
    for method in methods:
        data = all_data[method]
        path = np.asarray(data["X_path"], dtype=float)[:steps]
        flags = _flags(data, steps)
        color = DRONE_COLORS[method]
        ax.plot(*path[:frame + 1].T, color=color, linewidth=1.8, label=LABELS[method])
        point = path[frame]
        ax.scatter(*point, color=color, s=22, depthshade=False)
        _sphere(ax, point, _scalar(data, "drone_radius", base_radius), color, 0.09)
        collision_path = path[:frame + 1].copy()
        collision_path[~flags[:frame + 1]] = np.nan
        ax.plot(*collision_path.T, color="#D00000", linewidth=2.2)
        starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
        ends = np.flatnonzero(flags & ~np.r_[flags[1:], False])
        centers = ((starts + ends) // 2)
        centers = centers[centers <= frame]
        if centers.size:
            ax.scatter(*path[centers].T, color="#D00000", marker="*", s=52, depthshade=False)
        collisions = np.flatnonzero(flags)
        if collisions.size and frame >= collisions[0]:
            collided_labels.append(LABELS[method])

    suffix = f"   collision: {', '.join(collided_labels)}" if collided_labels else ""
    time = np.asarray(base["sim_time"])[frame]


def _make_replay(title: str, steps: int, dt: float, draw_frame) -> tuple[plt.Figure, FuncAnimation]:
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    # 3D axis labels extend beyond the axes' reported bounds, especially the
    # rotated Y label, so leave a little more room than a 2D plot needs.
    fig.subplots_adjust(left=0.14, right=0.92, bottom=0.25, top=0.93)

    slider_ax = fig.add_axes((0.29, 0.045, 0.49, 0.025))
    slider = Slider(slider_ax, "Frame", 0, steps - 1, valinit=0, valstep=1)
    play_ax = fig.add_axes((0.03, 0.035, 0.07, 0.045))
    save_ax = fig.add_axes((0.105, 0.035, 0.08, 0.045))
    save_all_ax = fig.add_axes((0.19, 0.035, 0.085, 0.045))
    play = Button(play_ax, "Pause")
    save = Button(save_ax, "Save SVG")
    save_all = Button(save_all_ax, "Save All")
    state = {"playing": True, "updating": False}
    controls = (slider_ax, play_ax, save_ax, save_all_ax)

    def render(frame: int) -> None:
        frame = int(frame)
        draw_frame(ax, frame)
        if int(slider.val) != frame:
            state["updating"] = True
            slider.set_val(frame)
            state["updating"] = False
        fig.canvas.draw_idle()

    def tick(frame: int):
        if not state["playing"]:
            return ()
        render(frame)
        return ()

    animation = FuncAnimation(fig, tick, frames=range(steps), interval=max(1, int(dt * 1000)),
                              repeat=False, cache_frame_data=False)

    def seek(value: float) -> None:
        if not state["updating"]:
            render(int(value))

    def toggle(_event) -> None:
        state["playing"] = not state["playing"]
        play.label.set_text("Pause" if state["playing"] else "Play")
        (animation.event_source.start if state["playing"] else animation.event_source.stop)()

    def write_clean_svg(path: Path) -> None:
        """Save the scene only, without Matplotlib's interactive controls."""
        for control in controls:
            control.set_visible(False)
        try:
            fig.canvas.draw()
            fig.savefig(
                path,
                format="svg",
                facecolor="white",
            )
        finally:
            for control in controls:
                control.set_visible(True)

    def save_svg(_event) -> None:
        path = PLOT_DIR / f"{title}_frame_{int(slider.val):04d}.svg"
        write_clean_svg(path)
        print(f"[plot] saved {path}")

    def save_all_svgs(_event) -> None:
        output_dir = PLOT_DIR / f"{title}_frames"
        output_dir.mkdir(parents=True, exist_ok=True)
        current_frame = int(slider.val)
        was_playing = state["playing"]
        state["playing"] = False
        animation.event_source.stop()
        play.label.set_text("Play")
        try:
            for frame in range(steps):
                render(frame)
                write_clean_svg(output_dir / f"frame_{frame:04d}.svg")
                if frame == 0 or (frame + 1) % 50 == 0 or frame + 1 == steps:
                    print(f"[plot] saved {frame + 1}/{steps} frames", flush=True)
        finally:
            render(current_frame)
            state["playing"] = was_playing
            play.label.set_text("Pause" if was_playing else "Play")
            if was_playing:
                animation.event_source.start()
        print(f"[plot] saved all frames to {output_dir}")

    slider.on_changed(seek)
    play.on_clicked(toggle)
    save.on_clicked(save_svg)
    save_all.on_clicked(save_all_svgs)
    render(0)
    # Keep widgets and animation alive for as long as their figure exists.
    fig._replay_objects = (animation, slider, play, save, save_all)  # type: ignore[attr-defined]
    return fig, animation


def _print_summary(method: str, data: dict[str, np.ndarray]) -> None:
    solve = np.asarray(data["solve_ms"], dtype=float)
    collisions = np.count_nonzero(np.asarray(data.get("collision_flags", []), dtype=bool))
    print(
        f"[{method}] file={_saved_path(method)} steps={len(data['X_path'])} "
        f"solve_ms(min/mean/max)=({solve.min():.2f}/{solve.mean():.2f}/{solve.max():.2f}) "
        f"collision_steps={collisions}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved Crazyflie simulations with Matplotlib.")
    parser.add_argument("mode", choices=[*METHODS, "all", "compare"])
    args = parser.parse_args()

    if args.mode == "compare":
        all_data = {method: _load_data(method) for method in METHODS}
        for method, data in all_data.items():
            _print_summary(method, data)
        paths = [np.asarray(all_data[m]["X_path"], dtype=float) for m in METHODS]
        steps = min(map(len, paths))
        mins, maxs = _bounds(all_data["mppi"], [path[:steps] for path in paths], True)
        dt = _scalar(all_data["mppi"], "dt", 0.03)
        _make_replay("crazyflie_compare", steps, dt,
                     lambda ax, i: _draw_compare(ax, all_data, i, mins, maxs))
    else:
        selected = list(METHODS) if args.mode == "all" else [args.mode]
        for method in selected:
            data = _load_data(method)
            _print_summary(method, data)
            path = np.asarray(data["X_path"], dtype=float)
            mins, maxs = _bounds(data, [path], False)
            dt = _scalar(data, "dt", 0.03)
            _make_replay(method, len(path), dt,
                         lambda ax, i, m=method, d=data, lo=mins, hi=maxs:
                         _draw_single(ax, m, d, i, lo, hi))

    plt.show()


if __name__ == "__main__":
    main()
