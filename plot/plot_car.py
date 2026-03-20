#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Dict

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D


METHOD_TO_FILE: Dict[str, str] = {
    "mppi": "mppi_car_simulation.npz",
    "ramppi": "ramppi_car_simulation.npz",
    "drmppi": "drmppi_car_simulation.npz",
    "dramppi": "dramppi_car_simulation.npz",
}


def set_img_pose(img_artist, x, y, phi, length_along_heading, width_lateral, ax):
    L = float(length_along_heading) * 2.0
    W = float(width_lateral) * 2.0
    img_artist.set_extent([-L / 2.0, L / 2.0, -W / 2.0, W / 2.0])
    tr = Affine2D().rotate(phi).translate(x, y) + ax.transData
    img_artist.set_transform(tr)


def _load_npz(base_dir: str, method: str) -> dict:
    if method not in METHOD_TO_FILE:
        raise ValueError(f"Unknown method: {method}")
    p = os.path.join(base_dir, METHOD_TO_FILE[method])
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing saved data: {p}. Run {method}.py with --save first.")
    data = np.load(p, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _replay_method(method: str, data: dict, root_dir: str) -> None:
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
    x_hist = np.asarray(data["X_hist"], dtype=float)
    x_path = np.asarray(data["X_path"], dtype=float)
    pred_nominal_xy = np.asarray(data["pred_nominal_xy"], dtype=float)
    pred_samples_xy = np.asarray(data["pred_samples_xy"], dtype=float)
    obs_xy = np.asarray(data["obs_xy"], dtype=float)
    obs_phi = np.asarray(data["obs_phi"], dtype=float)
    obs_ids = np.asarray(data["obs_ids"], dtype=int)
    k_hist = np.asarray(data["K_hist"], dtype=int)
    xlim_hist = np.asarray(data["xlim_hist"], dtype=float)
    ylim_hist = np.asarray(data["ylim_hist"], dtype=float)

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

    num_sprites = len(obs_sprite_imgs)
    id_to_sprite = {}
    next_sprite = 0

    plt.ion()
    fig, ax = plt.subplots(figsize=(14.0, 5.0))
    fig.subplots_adjust(right=0.80)

    ax.set_facecolor("#646464")
    ax.set_aspect("auto")
    ax.set_xlabel("x [scaled m]")
    ax.set_ylabel("y [scaled m]")

    ax.axhline(y_bottom, color="white", linewidth=3.0, zorder=2)
    ax.axhline(y_divider, color="white", linewidth=2.0, linestyle=(0, (12, 12)), alpha=0.9, zorder=2)
    if y_top_outer is not None and y_divider_top_1 is not None and y_divider_top_2 is not None:
        ax.axhline(y_divider_top_1, color="white", linewidth=2.0, linestyle=(0, (12, 12)), alpha=0.9, zorder=2)
        ax.axhline(y_divider_top_2, color="white", linewidth=2.0, linestyle=(0, (12, 12)), alpha=0.9, zorder=2)
        ax.axhline(y_top_outer, color="white", linewidth=3.0, zorder=2)
    else:
        ax.axhline(y_top, color="white", linewidth=3.0, zorder=2)

    ax.plot([x_path[0, 0] - 100, x_path[0, 0] + 500], [lane_y, lane_y],
            "--", linewidth=1.5, color="#abbac6", alpha=0.8, label="Lane center")

    sample_lines = []
    for _ in range(n_show):
        ln, = ax.plot([], [], lw=1.1, color="#9ee2e9", alpha=0.12, zorder=1)
        sample_lines.append(ln)

    path_line, = ax.plot([], [], lw=2.6, color="#67bde2", label="Ego path")
    pred_line, = ax.plot([], [], lw=2.2, color="#36ff0e", alpha=0.95, label="MPPI prediction", zorder=3)

    ego_img_artist = ax.imshow(car_ego_img, extent=[-0.5, 0.5, -0.5, 0.5], zorder=6)

    obs_imgs = []
    for _ in range(max_obs_draw):
        im = ax.imshow(obs_sprite_imgs[0], extent=[-0.5, 0.5, -0.5, 0.5], zorder=4, visible=False)
        obs_imgs.append(im)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)

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
            else:
                obs_imgs[j].set_visible(False)

        if np.all(np.isfinite(xlim_hist[i])) and np.all(np.isfinite(ylim_hist[i])):
            ax.set_xlim(float(xlim_hist[i, 0]), float(xlim_hist[i, 1]))
            ax.set_ylim(float(ylim_hist[i, 0]), float(ylim_hist[i, 1]))

        ax.set_title(f"{method.upper()} | t={sim_time[i]:.2f}s | K={kk} | solve={solve_ms[i]:.2f}ms")
        plt.pause(max(0.005, dt))

    plt.ioff()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved car simulations.")
    parser.add_argument("mode", choices=[*METHOD_TO_FILE.keys(), "all"], help="Method to replay")
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(here, ".."))

    if args.mode == "all":
        methods = ["mppi", "ramppi", "drmppi", "dramppi"]
    else:
        methods = [args.mode]

    for method in methods:
        data = _load_npz(here, method)
        _replay_method(method, data, root_dir)


if __name__ == "__main__":
    main()
