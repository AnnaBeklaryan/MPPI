#!/usr/bin/env python3
"""Fly the physical Crazyflie used as the moving Vicon obstacle.

Run this process for ``cf_2`` independently from ``DR_mppi_crazyflie_hl.py``.
The obstacle follows the same Poly4D figure-eight in reverse. The MPPI process
does not connect to this radio; it observes this drone only through Vicon.
"""

from __future__ import annotations

import argparse
import math
import time
from queue import Empty
from threading import Lock, Thread

import numpy as np

import cflib.crtp
import motioncapture
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper
from mocap_hl_commander_sim import figure8 as MOCAP_FIGURE8

# Edit this value directly: 1.0 = normal speed, 0.5 = half speed.
SPEED_SCALE = 0.5


def set_led_ring(cf, red, green, blue):
    """Set the LED Ring Deck to a solid RGB color."""
    cf.param.set_value("ring.effect", "7")
    cf.param.set_value("ring.solidRed", str(int(red)))
    cf.param.set_value("ring.solidGreen", str(int(green)))
    cf.param.set_value("ring.solidBlue", str(int(blue)))


class Poly4DFigure8:
    def __init__(self, trajectory, center):
        self.trajectory = tuple(
            tuple(float(value) for value in row) for row in trajectory
        )
        self.center = np.asarray(center, dtype=float).reshape(3)
        self.segment_durations = np.asarray(
            [row[0] for row in self.trajectory], dtype=float
        )
        self.segment_ends = np.cumsum(self.segment_durations)
        self.total_time = float(self.segment_ends[-1])

    @staticmethod
    def _value(coefficients, t):
        return sum(
            coefficient * t**power
            for power, coefficient in enumerate(coefficients)
        )

    def eval(self, t):
        loop_t = float(t) % self.total_time
        index = int(
            np.searchsorted(self.segment_ends, loop_t, side="right")
        )
        index = min(index, len(self.trajectory) - 1)
        start = 0.0 if index == 0 else self.segment_ends[index - 1]
        local_t = loop_t - float(start)
        row = self.trajectory[index]
        position = self.center + np.array(
            [
                self._value(row[1:9], local_t),
                self._value(row[9:17], local_t),
                self._value(row[17:25], local_t),
            ],
            dtype=float,
        )
        yaw = self._value(row[25:33], local_t)
        return position, float(yaw)


class ObstacleViconTracker(Thread):
    """Track one Vicon body and forward its pose to its Crazyflie."""

    def __init__(self, host, body_name):
        super().__init__(daemon=True)
        self.host = host
        self.body_name = body_name
        self.on_pose = None
        self.error = None
        self._running = True
        self._lock = Lock()
        self._position = None
        self._stamp = 0.0
        self.start()

    def close(self):
        self._running = False

    def snapshot(self):
        with self._lock:
            if self._position is None:
                return None
            return self._position.copy(), float(self._stamp)

    def wait_for_body(self, timeout):
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if self.error is not None:
                raise RuntimeError("Vicon thread failed") from self.error
            state = self.snapshot()
            if state is not None:
                return state
            time.sleep(0.02)
        raise TimeoutError(
            f"No Vicon data received for rigid body {self.body_name!r}"
        )

    def run(self):
        try:
            mc = motioncapture.connect(
                "vicon", {"hostname": self.host}
            )
            while self._running:
                mc.waitForNextFrame()
                obj = mc.rigidBodies.get(self.body_name)
                if obj is None:
                    continue
                position = np.asarray(obj.position, dtype=float).reshape(3)
                if not np.all(np.isfinite(position)):
                    continue
                stamp = time.monotonic()
                with self._lock:
                    self._position = position
                    self._stamp = stamp
                callback = self.on_pose
                if callback is not None:
                    callback(position, obj.rotation)
        except Exception as exc:
            self.error = exc
            self._running = False


def make_extpose_forwarder(cf, rate_hz=50.0):
    minimum_period = 1.0 / float(rate_hz)
    callback_lock = Lock()
    last_send_time = [0.0]

    def forward(position, quaternion):
        now = time.monotonic()
        with callback_lock:
            if now - last_send_time[0] < minimum_period:
                return
            last_send_time[0] = now
        cf.extpos.send_extpose(
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(quaternion.x),
            float(quaternion.y),
            float(quaternion.z),
            float(quaternion.w),
        )

    return forward


def wait_for_position_estimator(scf, timeout=20.0):
    log_config = LogConfig(name="Kalman Variance", period_in_ms=500)
    variables = ("kalman.varPX", "kalman.varPY", "kalman.varPZ")
    for variable in variables:
        log_config.add_variable(variable, "float")
    histories = {name: [1000.0] * 10 for name in variables}
    deadline = time.monotonic() + float(timeout)
    with SyncLogger(scf, log_config) as logger:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    "Crazyflie position estimator did not converge"
                )
            try:
                log_entry = logger._queue.get(
                    timeout=min(0.5, remaining)
                )
            except Empty:
                continue
            if log_entry == SyncLogger.DISCONNECT_EVENT:
                raise ConnectionError(
                    "Crazyflie disconnected during estimator setup"
                )
            _, data, _ = log_entry
            for name, history in histories.items():
                history.append(float(data[name]))
                history.pop(0)
            ranges = [
                max(history) - min(history)
                for history in histories.values()
            ]
            if all(value < 0.001 for value in ranges):
                print("Obstacle estimator converged.")
                return


def configure_estimator(scf):
    cf = scf.cf
    cf.param.set_value("locSrv.extQuatStdDev", 8.0e-3)
    cf.param.set_value("stabilizer.estimator", "2")
    cf.param.set_value("locSrv.extQuatStdDev", 0.06)
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    wait_for_position_estimator(scf)


def fly_obstacle(args, scf, tracker, trajectory):
    cf = scf.cf
    commander = cf.high_level_commander
    phase_time = (float(args.phase) % 1.0) * trajectory.total_time
    speed_scale = float(SPEED_SCALE)
    initial_position, initial_yaw = trajectory.eval(phase_time)
    airborne = False

    try:
        time.sleep(0.5)
        print(f"Waiting for {args.body!r} estimator...")
        configure_estimator(scf)

        print(f"Taking {args.body!r} off to {args.center_z:.2f} m...")
        commander.takeoff(float(args.center_z), float(args.takeoff_duration))
        airborne = True
        time.sleep(float(args.takeoff_duration) + 1.0)

        print(
            f"Moving {args.body!r} to figure-eight phase "
            f"{args.phase:.3f}..."
        )
        commander.go_to(
            float(initial_position[0]),
            float(initial_position[1]),
            float(initial_position[2]),
            float(initial_yaw),
            float(args.initial_go_to_duration),
            relative=False,
        )
        time.sleep(float(args.initial_go_to_duration) + 0.2)

        duration = (
            None if int(args.cycles) == 0
            else int(args.cycles) * trajectory.total_time / speed_scale
        )
        print(
            "Obstacle reverse figure-eight started. "
            "Press Ctrl-C to land."
        )
        start_time = time.monotonic()
        next_tick = start_time
        step = 0
        while True:
            now = time.monotonic()
            if tracker.error is not None:
                raise RuntimeError("Vicon thread stopped") from tracker.error
            state = tracker.snapshot()
            if (
                state is None
                or now - state[1] > float(args.vicon_timeout)
            ):
                raise RuntimeError(
                    f"Lost fresh Vicon data for {args.body!r}"
                )
            elapsed = now - start_time
            if duration is not None and elapsed >= duration:
                print("Obstacle trajectory complete.")
                break

            sample_time = (
                phase_time - speed_scale * elapsed
            ) % trajectory.total_time
            position, yaw = trajectory.eval(sample_time)
            cf.commander.send_position_setpoint(
                float(position[0]),
                float(position[1]),
                float(position[2]),
                math.degrees(yaw),
            )
            if step % 20 == 0:
                print(
                    f"[obstacle {step:05d}] "
                    f"target=({position[0]:.3f}, "
                    f"{position[1]:.3f}, {position[2]:.3f})"
                )
            step += 1
            next_tick += float(args.period)
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("Obstacle landing requested...")
    finally:
        try:
            cf.commander.send_notify_setpoint_stop()
        except Exception as exc:
            print(f"Could not release position setpoint mode: {exc}")
        time.sleep(0.1)
        if airborne:
            try:
                commander.land(0.0, float(args.land_duration))
                time.sleep(float(args.land_duration) + 0.5)
            except Exception as exc:
                print(f"Could not command obstacle landing: {exc}")
        try:
            commander.stop()
        except Exception:
            pass


def main(args):
    if float(args.period) <= 0.0:
        raise ValueError("--period must be positive")
    if int(args.cycles) < 0:
        raise ValueError("--cycles must be non-negative")
    if float(SPEED_SCALE) <= 0.0:
        raise ValueError("SPEED_SCALE must be positive")
    center = np.array(
        [args.center_x, args.center_y, args.center_z], dtype=float
    )
    trajectory = Poly4DFigure8(MOCAP_FIGURE8, center)
    tracker = ObstacleViconTracker(args.vicon_host, args.body)
    try:
        print(f"Waiting for Vicon rigid body {args.body!r}...")
        tracker.wait_for_body(args.vicon_timeout)
        cflib.crtp.init_drivers()
        uri = uri_helper.uri_from_env(default=args.uri)
        print(f"Connecting obstacle {args.body!r} on {uri}...")
        with SyncCrazyflie(
            uri, cf=Crazyflie(rw_cache="./cache")
        ) as scf:
            set_led_ring(scf.cf, 255, 0, 0)
            print("Obstacle LED ring: red")
            tracker.on_pose = make_extpose_forwarder(
                scf.cf, rate_hz=50.0
            )
            try:
                fly_obstacle(args, scf, tracker, trajectory)
            finally:
                tracker.on_pose = None
    finally:
        tracker.close()
        tracker.join(timeout=1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Fly cf_2 as an independent reverse figure-eight obstacle."
        )
    )
    parser.add_argument(
        "--uri", default="radio://0/80/2M/E7E7E7E702"
    )
    parser.add_argument("--vicon-host", default="169.254.91.105")
    parser.add_argument("--body", default="cf_2")
    parser.add_argument("--phase", type=float, default=0.25)
    parser.add_argument("--center-x", type=float, default=0.0)
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--center-z", type=float, default=0.5)
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of cycles; 0 flies until Ctrl-C.",
    )
    parser.add_argument("--period", type=float, default=0.02)
    parser.add_argument("--vicon-timeout", type=float, default=0.25)
    parser.add_argument("--takeoff-duration", type=float, default=3.0)
    parser.add_argument(
        "--initial-go-to-duration", type=float, default=3.0
    )
    parser.add_argument("--land-duration", type=float, default=2.0)
    main(parser.parse_args())
