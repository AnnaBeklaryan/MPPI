#!/usr/bin/env python3
"""Create an MP4 video from one method's saved SVG frames.

Examples:
    python plot/frames/make_video.py mppi
    python plot/frames/make_video.py drmppi --fps 30
    python plot/frames/make_video.py dramppi --output plot/dramppi.mp4
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


METHODS = ("mppi", "drmppi", "ramppi", "dramppi")
FRAMES_ROOT = Path(__file__).resolve().parent


def natural_key(path: Path) -> list[object]:
    """Sort frame_2 before frame_10, including frames in subdirectories."""
    relative_name = str(path.relative_to(FRAMES_ROOT)).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", relative_name)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn the SVG frames for an MPPI method into an H.264 MP4."
    )
    parser.add_argument("method", choices=METHODS, help="frame directory to use")
    parser.add_argument("--fps", type=float, default=20.0, help="video frame rate (default: 20)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output path (default: plot/frames/<method>.mp4)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output video"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        print("error: --fps must be greater than zero", file=sys.stderr)
        return 2

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("error: FFmpeg is not installed or is not on PATH", file=sys.stderr)
        return 1

    frame_dir = FRAMES_ROOT / args.method
    frames = sorted(frame_dir.rglob("*.svg"), key=natural_key)
    if not frames:
        print(f"error: no SVG frames found in {frame_dir}", file=sys.stderr)
        return 1

    output = (args.output or (FRAMES_ROOT / f"{args.method}.mp4")).expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        print("error: the output filename must end in .mp4", file=sys.stderr)
        return 2
    if output.exists() and not args.overwrite:
        print(f"error: output already exists: {output} (use --overwrite)", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)

    # A numbered symlink sequence lets FFmpeg consume naturally sorted frames,
    # even when their original numbering has gaps or they are in subdirectories.
    with tempfile.TemporaryDirectory(prefix=f"{args.method}_frames_") as temp_name:
        temp_dir = Path(temp_name)
        for index, frame in enumerate(frames):
            (temp_dir / f"frame_{index:08d}.svg").symlink_to(frame.resolve())

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if args.overwrite else "-n",
            "-framerate",
            str(args.fps),
            "-i",
            str(temp_dir / "frame_%08d.svg"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
            "-c:v",
            "libx264",
            "-movflags",
            "+faststart",
            str(output),
        ]
        print(f"Encoding {len(frames)} SVG frames at {args.fps:g} fps...")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"error: FFmpeg failed with exit code {exc.returncode}", file=sys.stderr)
            return exc.returncode

    print(f"Created: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
