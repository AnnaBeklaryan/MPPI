#!/usr/bin/env python3
"""Plotly replayer for Crazyflie MPPI variants.

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

import numpy as np
import plotly.graph_objects as go

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


def _saved_path(method: str) -> Path:
    return PLOT_DIR / METHODS[method][1]


def _load_data(method: str) -> dict[str, np.ndarray]:
    path = _saved_path(method)
    if not path.exists():
        raise FileNotFoundError(f"Missing saved file: {path}")
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def _append_empty_line_traces(fig: go.Figure, count: int, color: str, width: int, dash: str | None = None, name: str | None = None) -> list[int]:
    idx: list[int] = []
    line = dict(color=color, width=width)
    if dash is not None:
        line["dash"] = dash
    for i in range(count):
        fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=line,
                name=name if i == 0 else None,
                showlegend=bool(name and i == 0),
            )
        )
        idx.append(len(fig.data) - 1)
    return idx


def _add_cube_plotly(fig: go.Figure, c: np.ndarray) -> None:
    cx, cy, r, zmin, zmax = [float(v) for v in c]
    x0, x1 = cx - r, cx + r
    y0, y1 = cy - r, cy + r
    z0, z1 = zmin, zmax
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 0, 1, 2, 3, 4, 4, 5, 6, 1, 2]
    j = [1, 2, 4, 2, 3, 0, 5, 6, 6, 7, 5, 6]
    k = [2, 4, 5, 3, 0, 4, 6, 7, 7, 4, 6, 7]
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color="#8D959D",
            opacity=0.42,
            showlegend=False,
            hoverinfo="skip",
        )
    )

    edges = [
        ((x0, y0, z0), (x1, y0, z0)), ((x1, y0, z0), (x1, y1, z0)),
        ((x1, y1, z0), (x0, y1, z0)), ((x0, y1, z0), (x0, y0, z0)),
        ((x0, y0, z1), (x1, y0, z1)), ((x1, y0, z1), (x1, y1, z1)),
        ((x1, y1, z1), (x0, y1, z1)), ((x0, y1, z1), (x0, y0, z1)),
        ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)),
        ((x1, y1, z0), (x1, y1, z1)), ((x0, y1, z0), (x0, y1, z1)),
    ]
    for p0, p1 in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[p0[0], p1[0]],
                y=[p0[1], p1[1]],
                z=[p0[2], p1[2]],
                mode="lines",
                line=dict(color="black", width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    h = max(1e-6, z1 - z0)
    n_floors = max(3, int(h / 0.8))
    n_cols = 3
    floor_h = h / (n_floors + 1)
    win_h = 0.40 * floor_h
    win_wx = 0.16 * (x1 - x0)
    win_wy = 0.16 * (y1 - y0)
    gap_x = ((x1 - x0) - n_cols * win_wx) / (n_cols + 1)
    gap_y = ((y1 - y0) - n_cols * win_wy) / (n_cols + 1)
    z_start = z0 + 0.35 * floor_h
    eps = 0.001

    wx_x, wx_y, wx_z = [], [], []
    wy_x, wy_y, wy_z = [], [], []
    for fi in range(n_floors):
        zb = z_start + fi * floor_h
        zt = min(zb + win_h, z1 - 0.05 * floor_h)
        for cj in range(n_cols):
            xl = x0 + gap_x + cj * (win_wx + gap_x)
            xr = xl + win_wx
            wx_x += [xl, xr, xr, xl, xl, None]
            wx_y += [y1 + eps, y1 + eps, y1 + eps, y1 + eps, y1 + eps, None]
            wx_z += [zb, zb, zt, zt, zb, None]

            yl = y0 + gap_y + cj * (win_wy + gap_y)
            yr = yl + win_wy
            wy_x += [x1 + eps, x1 + eps, x1 + eps, x1 + eps, x1 + eps, None]
            wy_y += [yl, yr, yr, yl, yl, None]
            wy_z += [zb, zb, zt, zt, zb, None]

    fig.add_trace(
        go.Scatter3d(
            x=wx_x,
            y=wx_y,
            z=wx_z,
            mode="lines",
            line=dict(color="#2C3E50", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=wy_x,
            y=wy_y,
            z=wy_z,
            mode="lines",
            line=dict(color="#2C3E50", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _build_plotly_figure(method: str, data: dict[str, np.ndarray], with_predictions: bool = True) -> go.Figure:
    x_path = np.asarray(data["X_path"], dtype=float)
    obs_path = np.asarray(data["obs_path"], dtype=float)
    sim_time = np.asarray(data["sim_time"], dtype=float)
    solve_ms = np.asarray(data["solve_ms"], dtype=float)
    ref_curve = np.asarray(data.get("ref_curve", np.empty((0, 3))), dtype=float)
    cylinders = np.asarray(data.get("cylinders", np.empty((0, 5))), dtype=float)

    if x_path.ndim != 2 or x_path.shape[1] != 3:
        raise ValueError("X_path must be shaped (N,3).")

    steps = x_path.shape[0]
    dt = float(data.get("dt", np.median(np.diff(sim_time)) if sim_time.size > 1 else 0.03))
    mins = np.asarray(data.get("mins", np.min(x_path, axis=0) - 1.0), dtype=float)
    maxs = np.asarray(data.get("maxs", np.max(x_path, axis=0) + 1.0), dtype=float)

    if obs_path.ndim == 2:
        obs_path = obs_path[:, np.newaxis, :]
    if obs_path.ndim != 3:
        obs_path = np.zeros((steps, 0, 3), dtype=float)
    n_obs = obs_path.shape[1]

    pred_samples = None
    if with_predictions and "pred_samples_xyz" in data:
        pred_samples = np.asarray(data["pred_samples_xyz"], dtype=float)
        if pred_samples.ndim != 4:
            pred_samples = None
    pred_nominal = None
    if with_predictions and "pred_nominal_xyz" in data:
        pred_nominal = np.asarray(data["pred_nominal_xyz"], dtype=float)
        if pred_nominal.ndim != 3:
            pred_nominal = None

    fig = go.Figure()

    if ref_curve.size > 0:
        fig.add_trace(
            go.Scatter3d(
                x=ref_curve[:, 0],
                y=ref_curve[:, 1],
                z=ref_curve[:, 2],
                mode="lines",
                line=dict(color="#2a6fdb", width=4, dash="dot"),
                name="Reference",
            )
        )

    for c in cylinders:
        _add_cube_plotly(fig, c)

    dynamic_trace_idx: list[int] = []
    drone_path_idx = _append_empty_line_traces(fig, 1, color="#557EDC", width=4, name="Drone path")[0]
    dynamic_trace_idx.append(drone_path_idx)

    obs_colors = ["#EB1D87", "#219937", "#EC6613", "#FF8C00"]
    for j in range(n_obs):
        c = obs_colors[j % len(obs_colors)]
        idx = _append_empty_line_traces(fig, 1, color=c, width=4, dash="dot", name=f"Obstacle {j + 1}")[0]
        dynamic_trace_idx.append(idx)
    for j in range(n_obs):
        c = obs_colors[j % len(obs_colors)]
        idx1 = _append_empty_line_traces(fig, 1, color=c, width=5)[0]
        idx2 = _append_empty_line_traces(fig, 1, color=c, width=5)[0]
        dynamic_trace_idx.append(idx1)
        dynamic_trace_idx.append(idx2)

    pred_count = int(pred_samples.shape[2]) if pred_samples is not None else 0
    for j in range(pred_count):
        fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="#C70039", width=2),
                opacity=0.25,
                name="MPPI samples" if j == 0 else None,
                showlegend=(j == 0),
            )
        )
        dynamic_trace_idx.append(len(fig.data) - 1)

    if pred_nominal is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="#C70039", width=5),
                name="Nominal plan",
                showlegend=True,
            )
        )
        nominal_idx = len(fig.data) - 1
        dynamic_trace_idx.append(nominal_idx)
    else:
        nominal_idx = -1

    fig.add_trace(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            marker=dict(color="#557EDC", size=4),
            name="Drone",
            showlegend=False,
        )
    )
    drone_marker_idx = len(fig.data) - 1
    dynamic_trace_idx.append(drone_marker_idx)

    fig.add_trace(
        go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="#557EDC", width=5), showlegend=False)
    )
    drone_arm1_idx = len(fig.data) - 1
    dynamic_trace_idx.append(drone_arm1_idx)

    fig.add_trace(
        go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="#557EDC", width=5), showlegend=False)
    )
    drone_arm2_idx = len(fig.data) - 1
    dynamic_trace_idx.append(drone_arm2_idx)

    frames: list[go.Frame] = []
    obs_arm_len = 0.3
    obs_d1 = np.array([1.0, 1.0, 0.0], dtype=float)
    obs_d2 = np.array([1.0, -1.0, 0.0], dtype=float)
    obs_d1 /= (np.linalg.norm(obs_d1) + 1e-12)
    obs_d2 /= (np.linalg.norm(obs_d2) + 1e-12)
    arm_len = 0.35
    x_hist = np.asarray(data["X_hist"], dtype=float) if "X_hist" in data else None
    for i in range(steps):
        frame_data: list[go.Scatter3d] = []

        frame_data.append(go.Scatter3d(x=x_path[: i + 1, 0], y=x_path[: i + 1, 1], z=x_path[: i + 1, 2]))
        for j in range(n_obs):
            frame_data.append(go.Scatter3d(x=obs_path[: i + 1, j, 0], y=obs_path[: i + 1, j, 1], z=obs_path[: i + 1, j, 2]))
        for j in range(n_obs):
            po = obs_path[i, j, :]
            op1a = po - obs_arm_len * obs_d1
            op1b = po + obs_arm_len * obs_d1
            op2a = po - obs_arm_len * obs_d2
            op2b = po + obs_arm_len * obs_d2
            frame_data.append(go.Scatter3d(x=[op1a[0], op1b[0]], y=[op1a[1], op1b[1]], z=[op1a[2], op1b[2]]))
            frame_data.append(go.Scatter3d(x=[op2a[0], op2b[0]], y=[op2a[1], op2b[1]], z=[op2a[2], op2b[2]]))

        for j in range(pred_count):
            pred_xyz = pred_samples[i, :, j, :]
            valid = np.isfinite(pred_xyz[:, 0])
            if np.any(valid):
                frame_data.append(go.Scatter3d(x=pred_xyz[valid, 0], y=pred_xyz[valid, 1], z=pred_xyz[valid, 2]))
            else:
                frame_data.append(go.Scatter3d(x=[], y=[], z=[]))

        if nominal_idx >= 0:
            nom = pred_nominal[i]
            valid_nom = np.isfinite(nom[:, 0])
            if np.any(valid_nom):
                frame_data.append(go.Scatter3d(x=nom[valid_nom, 0], y=nom[valid_nom, 1], z=nom[valid_nom, 2]))
            else:
                frame_data.append(go.Scatter3d(x=[], y=[], z=[]))

        p = x_path[i]
        frame_data.append(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]]))
        if x_hist is not None:
            psi = float(x_hist[i, 6])
            phi = float(x_hist[i, 7])
            theta = float(x_hist[i, 8])
            cphi = np.cos(phi)
            sphi = np.sin(phi)
            cth = np.cos(theta)
            sth = np.sin(theta)
            cpsi = np.cos(psi)
            spsi = np.sin(psi)
            rz = np.array([[cpsi, -spsi, 0.0], [spsi, cpsi, 0.0], [0.0, 0.0, 1.0]])
            ry = np.array([[cth, 0.0, sth], [0.0, 1.0, 0.0], [-sth, 0.0, cth]])
            rx = np.array([[1.0, 0.0, 0.0], [0.0, cphi, -sphi], [0.0, sphi, cphi]])
            rm = rz @ ry @ rx
            xb = rm[:, 0]
            yb = rm[:, 1]
            d1 = (xb + yb) / (np.linalg.norm(xb + yb) + 1e-12)
            d2 = (xb - yb) / (np.linalg.norm(xb - yb) + 1e-12)
            p1a, p1b = p - arm_len * d1, p + arm_len * d1
            p2a, p2b = p - arm_len * d2, p + arm_len * d2
        else:
            p1a, p1b = p, p
            p2a, p2b = p, p
        frame_data.append(go.Scatter3d(x=[p1a[0], p1b[0]], y=[p1a[1], p1b[1]], z=[p1a[2], p1b[2]]))
        frame_data.append(go.Scatter3d(x=[p2a[0], p2b[0]], y=[p2a[1], p2b[1]], z=[p2a[2], p2b[2]]))

        frames.append(
            go.Frame(
                data=frame_data,
                traces=dynamic_trace_idx,
                name=str(i),
                layout=go.Layout(
                    title_text=(
                        f"{method.upper()} replay | t={sim_time[i]:.2f}s "
                        f"| solve={solve_ms[i]:.1f}ms"
                    )
                ),
            )
        )

    fig.frames = frames
    fig.update_layout(
        title=f"{method.upper()} replay from saved data",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        scene=dict(
            xaxis=dict(
                title="X",
                range=[float(mins[0]), float(maxs[0])],
                autorange=False,
                gridcolor="#bfc4cc",
                gridwidth=4,
                backgroundcolor="#f4f9fc",
                showbackground=True,
            ),
            yaxis=dict(
                title="Y",
                range=[float(mins[1]), float(maxs[1])],
                autorange=False,
                gridcolor="#bfc4cc",
                gridwidth=4,
                backgroundcolor="#f4f9fc",
                showbackground=True,
            ),
            zaxis=dict(
                title="Altitude",
                range=[float(mins[2]), float(maxs[2])],
                autorange=False,
                gridcolor="#bfc4cc",
                gridwidth=4,
                backgroundcolor="#f4f9fc",
                showbackground=True,
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.8),
            dragmode="turntable",
            bgcolor="#ffffff",
        ),
        width=1250,
        height=900,
        showlegend=False,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": int(1000 * dt), "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "steps": [
                    {
                        "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                        "label": str(i),
                        "method": "animate",
                    }
                    for i in range(steps)
                ],
            }
        ],
    )
    return fig


def _build_compare_figure(all_data: dict[str, dict[str, np.ndarray]]) -> go.Figure:
    methods = ["mppi", "ramppi", "drmppi", "dramppi"]
    base = all_data["mppi"]

    x_paths = {m: np.asarray(all_data[m]["X_path"], dtype=float) for m in methods}
    sim_times = {m: np.asarray(all_data[m]["sim_time"], dtype=float) for m in methods}

    steps = min(x_paths[m].shape[0] for m in methods)
    dt = float(base.get("dt", 0.03))
    sim_time = sim_times["mppi"][:steps]

    obs_path = np.asarray(base.get("obs_path", np.zeros((steps, 0, 3))), dtype=float)
    if obs_path.ndim == 2:
        obs_path = obs_path[:, np.newaxis, :]
    obs_path = obs_path[:steps]
    n_obs = obs_path.shape[1] if obs_path.ndim == 3 else 0

    ref_curve = np.asarray(base.get("ref_curve", np.empty((0, 3))), dtype=float)
    cylinders = np.asarray(base.get("cylinders", np.empty((0, 5))), dtype=float)
    mins = np.asarray(base.get("mins", np.min(np.vstack([x_paths[m][:steps] for m in methods]), axis=0) - 1.0), dtype=float)
    maxs = np.asarray(base.get("maxs", np.max(np.vstack([x_paths[m][:steps] for m in methods]), axis=0) + 1.0), dtype=float)

    fig = go.Figure()

    if ref_curve.size > 0:
        fig.add_trace(
            go.Scatter3d(
                x=ref_curve[:, 0], y=ref_curve[:, 1], z=ref_curve[:, 2],
                mode="lines",
                line=dict(color="#2a6fdb", width=4, dash="dot"),
                name="Reference",
            )
        )
    for c in cylinders:
        _add_cube_plotly(fig, c)

    dynamic_trace_idx: list[int] = []
    obs_colors = ["#EB1D87", "#34980C", "#EC6613", "#FF8C00"]
    for j in range(n_obs):
        c = obs_colors[j % len(obs_colors)]
        idx = _append_empty_line_traces(fig, 1, color=c, width=4, dash="dot", name=f"Obstacle {j + 1}")[0]
        dynamic_trace_idx.append(idx)
        idx_arm1 = _append_empty_line_traces(fig, 1, color=c, width=5)[0]
        idx_arm2 = _append_empty_line_traces(fig, 1, color=c, width=5)[0]
        dynamic_trace_idx.append(idx_arm1)
        dynamic_trace_idx.append(idx_arm2)

    drone_colors = {
        "mppi": "#095ed5",
        "ramppi": "#9509b8",
        "drmppi": "#26867b",
        "dramppi": "#FF7300",
    }
    drone_labels = {
        "mppi": "MPPI",
        "ramppi": "RA-MPPI",
        "drmppi": "DR-MPPI",
        "dramppi": "DRA-MPPI",
    }
    for m in methods:
        c = drone_colors[m]
        idx_path = _append_empty_line_traces(fig, 1, color=c, width=5, name=drone_labels[m])[0]
        dynamic_trace_idx.append(idx_path)
        fig.add_trace(
            go.Scatter3d(
                x=[], y=[], z=[],
                mode="markers",
                marker=dict(color=c, size=6),
                showlegend=False,
            )
        )
        dynamic_trace_idx.append(len(fig.data) - 1)

    obs_arm_len = 0.3
    obs_d1 = np.array([1.0, 1.0, 0.0], dtype=float)
    obs_d2 = np.array([1.0, -1.0, 0.0], dtype=float)
    obs_d1 /= (np.linalg.norm(obs_d1) + 1e-12)
    obs_d2 /= (np.linalg.norm(obs_d2) + 1e-12)

    frames: list[go.Frame] = []
    for i in range(steps):
        frame_data: list[go.Scatter3d] = []
        for j in range(n_obs):
            frame_data.append(go.Scatter3d(x=obs_path[: i + 1, j, 0], y=obs_path[: i + 1, j, 1], z=obs_path[: i + 1, j, 2]))
            po = obs_path[i, j, :]
            op1a = po - obs_arm_len * obs_d1
            op1b = po + obs_arm_len * obs_d1
            op2a = po - obs_arm_len * obs_d2
            op2b = po + obs_arm_len * obs_d2
            frame_data.append(go.Scatter3d(x=[op1a[0], op1b[0]], y=[op1a[1], op1b[1]], z=[op1a[2], op1b[2]]))
            frame_data.append(go.Scatter3d(x=[op2a[0], op2b[0]], y=[op2a[1], op2b[1]], z=[op2a[2], op2b[2]]))
        for m in methods:
            xp = x_paths[m]
            frame_data.append(go.Scatter3d(x=xp[: i + 1, 0], y=xp[: i + 1, 1], z=xp[: i + 1, 2]))
            p = xp[i]
            frame_data.append(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]]))

        frames.append(
            go.Frame(
                data=frame_data,
                traces=dynamic_trace_idx,
                name=str(i),
                layout=go.Layout(title_text=f"Combined replay | t={sim_time[i]:.2f}s"),
            )
        )

    fig.frames = frames
    fig.update_layout(
        title="Combined Replay: MPPI / RA / DR / DRA",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        scene=dict(
            xaxis=dict(title="X", range=[float(mins[0]), float(maxs[0])], autorange=False, gridcolor="#bfc4cc", gridwidth=4, backgroundcolor="#f4f9fc", showbackground=True),
            yaxis=dict(title="Y", range=[float(mins[1]), float(maxs[1])], autorange=False, gridcolor="#bfc4cc", gridwidth=4, backgroundcolor="#f4f9fc", showbackground=True),
            zaxis=dict(title="Altitude", range=[float(mins[2]), float(maxs[2])], autorange=False, gridcolor="#bfc4cc", gridwidth=4, backgroundcolor="#f4f9fc", showbackground=True),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.8),
            dragmode="turntable",
            bgcolor="#ffffff",
        ),
        width=1250,
        height=900,
        showlegend=True,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": int(1000 * dt), "redraw": True}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "steps": [
                    {"args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], "label": str(i), "method": "animate"}
                    for i in range(steps)
                ],
            }
        ],
    )
    return fig


def _print_summary(method: str, data: dict[str, np.ndarray]) -> None:
    solve_ms = np.asarray(data["solve_ms"], dtype=float)
    print(
        f"[{method}] file={_saved_path(method)} steps={data['X_path'].shape[0]} "
        f"solve_ms(min/mean/max)=({solve_ms.min():.2f}/{solve_ms.mean():.2f}/{solve_ms.max():.2f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved Crazyflie MPPI simulations with Plotly.")
    parser.add_argument("mode", choices=[*METHODS.keys(), "all", "compare"], help="Controller mode to run or replay.")
    args = parser.parse_args()

    if args.mode == "compare":
        all_data = {m: _load_data(m) for m in METHODS}
        for m in METHODS:
            _print_summary(m, all_data[m])
        fig = _build_compare_figure(all_data)
        fig.show(
            config={
                "scrollZoom": True,
                "displaylogo": False,
            }
        )
        return

    selected = list(METHODS.keys()) if args.mode == "all" else [args.mode]

    for method in selected:
        data = _load_data(method)
        _print_summary(method, data)
        fig = _build_plotly_figure(method, data, with_predictions=True)
        fig.show(
            config={
                "scrollZoom": True,
                "displaylogo": False,
            }
        )


if __name__ == "__main__":
    main()
