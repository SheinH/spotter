"""Crosshair-to-target geometry. All angles in degrees."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .format import Run

OCTANTS = ["R", "UR", "U", "UL", "L", "DL", "D", "DR"]


@dataclass
class View:
    """Where target rows sit relative to the crosshair."""

    ex: np.ndarray      # target right of crosshair (+) / left (-)
    ey: np.ndarray      # target above crosshair (+) / below (-)
    ang: np.ndarray     # total angle between crosshair and target centre
    arad: np.ndarray    # angular radius of the target
    miss: np.ndarray    # angle to the nearest point of the target's vertical axis; on target when <= arad
    tyaw: np.ndarray    # absolute direction of the target from the camera
    tpitch: np.ndarray


def view(run: Run, rows: np.ndarray, frames: np.ndarray) -> View:
    """Target `rows` as seen from the camera at `frames` (same length)."""
    yaw, pitch = np.radians(run.yaw[frames]), np.radians(run.pitch[frames])
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    d = run.tgt_pos[rows] - run.cam[frames]
    dist = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
    d = d / dist[:, None]
    fwd = d[:, 0] * cp * cy + d[:, 1] * cp * sy + d[:, 2] * sp
    right = -d[:, 0] * sy + d[:, 1] * cy
    up = -d[:, 0] * sp * cy - d[:, 1] * sp * sy + d[:, 2] * cp
    ex = np.degrees(np.arctan2(right, fwd))
    ey = np.degrees(np.arctan2(up, np.hypot(fwd, right)))
    ang = np.degrees(np.arccos(np.clip(fwd, -1, 1)))
    arad = np.degrees(np.arcsin(np.clip(run.tgt_radius[rows] / dist, 0, 1)))
    # Tall bots are a vertical capsule: a sphere of the same radius swept up and down.
    spine = np.degrees(np.arctan(np.maximum(run.tgt_hh[rows] - run.tgt_radius[rows], 0) / dist))
    tall = spine > 1e-6
    miss = np.where(tall, np.hypot(ex, np.maximum(np.abs(ey) - spine, 0)), ang)
    return View(
        ex=ex, ey=ey, ang=ang, arad=arad, miss=miss,
        tyaw=np.degrees(np.arctan2(d[:, 1], d[:, 0])),
        tpitch=np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1))),
    )


def target_view(run: Run) -> View:
    return view(run, np.arange(len(run.tgt_frame)), run.tgt_frame)


def nearest(run: Run, v: View) -> np.ndarray:
    """Per frame, the target row closest to the crosshair, or -1 if no target."""
    out = np.full(len(run.t), -1, dtype=np.int64)
    if len(run.tgt_frame) == 0:
        return out
    order = np.lexsort((v.ang, run.tgt_frame))
    frames = run.tgt_frame[order]
    first = np.r_[True, frames[1:] != frames[:-1]]
    out[frames[first]] = order[first]
    return out


def octant(x: float, y: float) -> str:
    """Screen direction of a vector (x right, y up)."""
    return OCTANTS[int(np.round(np.degrees(np.arctan2(y, x)) / 45.0)) % 8]


def smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Zero-phase Hann smoothing over n samples, edges held."""
    if n < 3:
        return x
    w = np.hanning(n + 2)[1:-1]
    w /= w.sum()
    pad = n // 2
    return np.convolve(np.pad(x, pad, mode="edge"), w, mode="valid")[: len(x)]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of each True run."""
    m = np.r_[False, mask, False]
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))
