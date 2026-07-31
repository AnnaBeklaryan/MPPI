"""Shared evaluator for the Poly4D figure-eight used by physical flights."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np


def _load_coefficients():
    source_path = Path(__file__).resolve().parent / "dr-MPPI" / "mocap_hl_commander_sim.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in module.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "figure8" for target in targets):
                return tuple(tuple(float(value) for value in row) for row in ast.literal_eval(node.value))
    raise RuntimeError(f"Could not find figure8 Poly4D coefficients in {source_path}")


COEFFICIENTS = _load_coefficients()
SEGMENT_ENDS = np.cumsum(np.asarray([row[0] for row in COEFFICIENTS], dtype=float))
POLY_TIME = float(SEGMENT_ENDS[-1])
PHYSICAL_PEAK = np.array([0.99577055, 0.54361863], dtype=float)


def _value(coefficients, t):
    return sum(coefficient * t**power for power, coefficient in enumerate(coefficients))


def _derivative(coefficients, t):
    return sum(
        power * coefficient * t ** (power - 1)
        for power, coefficient in enumerate(coefficients)
        if power > 0
    )


def evaluate(phase, phase_rate, center, x_peak=0.5, y_peak=0.2):
    """Return Poly4D position/velocity with the requested simulation extents."""
    phase = float(phase) % (2.0 * math.pi)
    poly_t = phase * POLY_TIME / (2.0 * math.pi)
    index = min(
        int(np.searchsorted(SEGMENT_ENDS, poly_t, side="right")),
        len(COEFFICIENTS) - 1,
    )
    segment_start = 0.0 if index == 0 else float(SEGMENT_ENDS[index - 1])
    local_t = poly_t - segment_start
    row = COEFFICIENTS[index]
    scale = np.array(
        [float(x_peak) / PHYSICAL_PEAK[0], float(y_peak) / PHYSICAL_PEAK[1], 0.0],
        dtype=float,
    )
    offset = scale * np.array(
        [_value(row[1:9], local_t), _value(row[9:17], local_t), _value(row[17:25], local_t)],
        dtype=float,
    )
    derivative = scale * np.array(
        [
            _derivative(row[1:9], local_t),
            _derivative(row[9:17], local_t),
            _derivative(row[17:25], local_t),
        ],
        dtype=float,
    )
    velocity = derivative * POLY_TIME / (2.0 * math.pi) * float(phase_rate)
    return np.asarray(center, dtype=float) + offset, velocity
