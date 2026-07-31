#!/usr/bin/env python3
"""Print live Vicon data while remaining compatible with stdlib ``logging``.

This file has the exact name requested by the user. Since that name shadows
Python's standard-library logging package, its public API is loaded and
re-exported first so libraries such as Torch continue to work from this folder.
"""

import argparse
import importlib.util
import math
import os
import sysconfig
import time


_STDLIB_LOGGING_DIR = os.path.join(sysconfig.get_path("stdlib"), "logging")
_STDLIB_LOGGING_PATH = os.path.join(_STDLIB_LOGGING_DIR, "__init__.py")
_STDLIB_SPEC = importlib.util.spec_from_file_location(
    "_stdlib_logging",
    _STDLIB_LOGGING_PATH,
    submodule_search_locations=[_STDLIB_LOGGING_DIR],
)
_STDLIB_MODULE = importlib.util.module_from_spec(_STDLIB_SPEC)
_STDLIB_SPEC.loader.exec_module(_STDLIB_MODULE)
for _name, _value in vars(_STDLIB_MODULE).items():
    if _name not in {
        "__name__", "__loader__", "__package__", "__spec__", "__file__",
        "__cached__", "__builtins__",
    }:
        globals()[_name] = _value

# Allow imports such as ``logging.handlers`` to resolve to the stdlib package.
__path__ = [_STDLIB_LOGGING_DIR]


def quaternion_to_euler_degrees(quaternion):
    """Convert the motioncapture quaternion to roll, pitch, yaw in degrees."""
    qx = float(quaternion.x)
    qy = float(quaternion.y)
    qz = float(quaternion.z)
    qw = float(quaternion.w)

    sin_roll = 2.0 * (qw * qx + qy * qz)
    cos_roll = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sin_roll, cos_roll)

    sin_pitch = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    pitch = math.asin(sin_pitch)

    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(sin_yaw, cos_yaw)

    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def print_vicon_data(host, system_type, body_filter):
    import motioncapture

    print(f"Connecting to {system_type} at {host}...")
    mocap = motioncapture.connect(system_type, {"hostname": host})
    print("Connected. Waiting for rigid-body data; press Ctrl-C to stop.")

    frame_number = 0
    start_time = time.monotonic()
    try:
        while True:
            mocap.waitForNextFrame()
            frame_number += 1
            elapsed = time.monotonic() - start_time
            bodies_printed = 0

            for name, body in mocap.rigidBodies.items():
                if body_filter and name != body_filter:
                    continue

                position = body.position
                quaternion = body.rotation
                roll, pitch, yaw = quaternion_to_euler_degrees(quaternion)
                print(
                    f"t={elapsed:9.3f}s frame={frame_number:07d} body={name:<12} "
                    f"position=({position[0]: .4f}, {position[1]: .4f}, {position[2]: .4f}) m "
                    f"quaternion=({quaternion.x: .5f}, {quaternion.y: .5f}, "
                    f"{quaternion.z: .5f}, {quaternion.w: .5f}) "
                    f"rpy=({roll: .2f}, {pitch: .2f}, {yaw: .2f}) deg",
                    flush=True,
                )
                bodies_printed += 1

            if bodies_printed == 0 and frame_number % 100 == 0:
                wanted = repr(body_filter) if body_filter else "any rigid body"
                print(
                    f"t={elapsed:9.3f}s frame={frame_number:07d} "
                    f"waiting for {wanted}...",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nStopped Vicon logging.")


def main():
    parser = argparse.ArgumentParser(
        description="Print live Vicon rigid-body position and orientation."
    )
    parser.add_argument(
        "--host",
        default="169.254.91.105",
        help="Vicon host name or IP address.",
    )
    parser.add_argument(
        "--system",
        default="vicon",
        help="motioncapture system type (default: vicon).",
    )
    parser.add_argument(
        "--body",
        default=None,
        help="Only print this rigid body, for example cf_1 or cf_2.",
    )
    args = parser.parse_args()
    print_vicon_data(args.host, args.system, args.body)


if __name__ == "__main__":
    main()
