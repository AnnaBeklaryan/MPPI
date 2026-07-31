#!/usr/bin/env python3
"""Fly CrazySim cf_1 backward around the MPPI figure-eight.

This is a standalone moving-obstacle controller. It reads cf_1 ground-truth
odometry directly from Gazebo, forwards the pose to the simulated estimator,
and streams position setpoints through cflib. ROS is not required.
"""

import argparse
import math
import time

import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from mppi_crazyflie_gazebo import (
    GazeboTracker,
    MOCAP_FIGURE8,
    MocapPoly4DFigure8Reference,
    configure_estimator,
)


def run_obstacle(args):
    if args.cycles <= 0.0:
        raise ValueError("--cycles must be positive")
    center = np.array(
        [args.center_x, args.center_y, args.center_z], dtype=float
    )
    traj = MocapPoly4DFigure8Reference(MOCAP_FIGURE8, center=center)
    phase_time = float(args.phase) * traj.total_time
    initial_reference = traj.eval(phase_time)
    initial_position = initial_reference[0:3]

    tracker = GazeboTracker(
        args.odom_topic,
        ego_name="cf_1",
        obstacle_name="",
    )

    try:
        print(f"Waiting for cf_1 odometry on {args.odom_topic}...")
        tracker.wait_for_body("cf_1", args.gazebo_timeout)

        cflib.crtp.init_drivers()
        with SyncCrazyflie(
            args.uri, cf=Crazyflie(rw_cache="./cache")
        ) as scf:
            cf = scf.cf
            tracker.on_ego_pose = lambda pos, quat: cf.extpos.send_extpose(
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(quat.x),
                float(quat.y),
                float(quat.z),
                float(quat.w),
            )

            print("Waiting for the cf_1 estimator...")
            configure_estimator(scf)
            commander = cf.high_level_commander

            print(f"Taking cf_1 off to {center[2]:.2f} m...")
            commander.takeoff(float(center[2]), args.takeoff_duration)
            time.sleep(args.takeoff_duration + 0.5)

            print(
                "Moving cf_1 to its trajectory phase point at "
                f"({initial_position[0]:.3f}, "
                f"{initial_position[1]:.3f}, "
                f"{initial_position[2]:.3f})..."
            )
            commander.go_to(
                float(initial_position[0]),
                float(initial_position[1]),
                float(initial_position[2]),
                math.degrees(0.0),
                args.positioning_duration,
                relative=False,
            )
            time.sleep(args.positioning_duration + 0.2)

            print(
                "cf_1 reverse figure-eight started. "
                "Press Ctrl-C to land early."
            )
            start_time = time.monotonic()
            next_tick = start_time
            flight_duration = float(args.cycles) * traj.total_time
            try:
                while True:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= flight_duration:
                        print(
                            f"cf_1 completed {args.cycles:g} reverse "
                            "figure-eight cycle(s)."
                        )
                        break

                    # Subtracting elapsed time makes cf_1 traverse the same
                    # closed trajectory in the opposite direction.
                    obstacle_time = (
                        phase_time - elapsed
                    ) % traj.total_time
                    reference = traj.eval(obstacle_time)
                    cf.commander.send_position_setpoint(
                        float(reference[0]),
                        float(reference[1]),
                        float(reference[2]),
                        math.degrees(float(reference[8])),
                    )

                    next_tick += args.period
                    sleep_time = next_tick - time.monotonic()
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
                    else:
                        next_tick = time.monotonic()
            except KeyboardInterrupt:
                print("Early landing requested for cf_1.")
            finally:
                cf.commander.send_notify_setpoint_stop()
                time.sleep(0.1)
                commander.land(0.0, args.land_duration)
                time.sleep(args.land_duration + 0.5)
                commander.stop()
                tracker.on_ego_pose = None
    finally:
        tracker.close()
        tracker.join(timeout=2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fly CrazySim cf_1 as a reverse figure-eight obstacle."
    )
    parser.add_argument("--uri", default="udp://127.0.0.1:19851")
    parser.add_argument("--odom-topic", default="/cf_1/odom")
    parser.add_argument("--center-x", type=float, default=0.0)
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--center-z", type=float, default=1.0)
    parser.add_argument(
        "--phase",
        type=float,
        default=0.25,
        help="Initial phase as a fraction of one trajectory cycle.",
    )
    parser.add_argument(
        "--cycles",
        type=float,
        default=3.0,
        help="Number of reverse cycles before landing.",
    )
    parser.add_argument("--period", type=float, default=0.02)
    parser.add_argument("--gazebo-timeout", type=float, default=10.0)
    parser.add_argument("--takeoff-duration", type=float, default=2.0)
    parser.add_argument("--positioning-duration", type=float, default=2.0)
    parser.add_argument("--land-duration", type=float, default=2.0)
    run_obstacle(parser.parse_args())
