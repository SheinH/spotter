""".spot log reader/writer. Mirrors hook/src/format.h (version 2; version 1 logs still read)."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"SPOT"
VERSION = 2
HEADER = struct.Struct("<4sHHQq40s")
FRAME = struct.Struct("<Q6fIBBH")
TARGET = struct.Struct("<I6f")
TARGET_V1 = struct.Struct("<I5f")   # no half_height
NO_SHOTS = 0xFFFFFFFF
BTN_FIRE = 1

assert HEADER.size == 64 and FRAME.size == 40 and TARGET.size == 28


@dataclass
class Run:
    """One log. Per-frame arrays have length N; target arrays are flat with
    length M and `tgt_frame` (non-decreasing) says which frame each row belongs to."""

    t: np.ndarray            # seconds since log start
    yaw: np.ndarray          # degrees, positive = right
    pitch: np.ndarray        # degrees, positive = up
    fov: np.ndarray
    cam: np.ndarray          # (N, 3)
    shots: np.ndarray        # cumulative shot counter, -1 where unavailable
    buttons: np.ndarray
    tgt_frame: np.ndarray
    tgt_id: np.ndarray
    tgt_pos: np.ndarray      # (M, 3)
    tgt_radius: np.ndarray
    tgt_hh: np.ndarray       # half height; equals radius for a sphere
    tgt_health: np.ndarray
    start_wall_us: int = 0
    game: str = ""

    def wall(self, t: float) -> float:
        """Unix seconds for log time t."""
        return self.start_wall_us / 1e6 + t

    def slice(self, t0: float, t1: float) -> "Run":
        keep = (self.t >= t0) & (self.t <= t1)
        new_index = np.cumsum(keep) - 1
        rows = keep[self.tgt_frame]
        return Run(
            t=self.t[keep], yaw=self.yaw[keep], pitch=self.pitch[keep], fov=self.fov[keep],
            cam=self.cam[keep], shots=self.shots[keep], buttons=self.buttons[keep],
            tgt_frame=new_index[self.tgt_frame[rows]], tgt_id=self.tgt_id[rows],
            tgt_pos=self.tgt_pos[rows], tgt_radius=self.tgt_radius[rows], tgt_hh=self.tgt_hh[rows],
            tgt_health=self.tgt_health[rows],
            start_wall_us=self.start_wall_us, game=self.game,
        )


def read_spot(path: str | Path, v1_hh_ratio: float = 1.0) -> Run:
    """v1_hh_ratio: half-height as a multiple of radius, for version 1 logs that lack it."""
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"{path}: too short to be a .spot file")
    magic, version, header_size, _qpc, start_wall_us, game = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: not a .spot file")
    if version not in (1, VERSION):
        raise ValueError(f"{path}: version {version}, this analyzer reads up to version {VERSION}")
    target = TARGET_V1 if version == 1 else TARGET

    frames: list[tuple] = []
    targets: list[tuple] = []
    off, end = header_size, len(data)
    # A log cut off mid-record (game closed or crashed) is read up to the last whole frame.
    while off + FRAME.size <= end:
        f = FRAME.unpack_from(data, off)
        n = f[9]
        if off + FRAME.size + n * target.size > end:
            break
        off += FRAME.size
        i = len(frames)
        frames.append(f)
        for _ in range(n):
            rec = target.unpack_from(data, off)
            if version == 1:   # id, pos, radius, health -> insert half_height
                rec = (*rec[:5], rec[4] * v1_hh_ratio, rec[5])
            targets.append((i, *rec))
            off += target.size

    fa = np.array(frames, dtype=np.float64).reshape(-1, 11)
    ta = np.array(targets, dtype=np.float64).reshape(-1, 8)
    shots = fa[:, 7].astype(np.int64)
    shots[shots == NO_SHOTS] = -1
    return Run(
        t=fa[:, 0] / 1e6, yaw=fa[:, 1], pitch=fa[:, 2], fov=fa[:, 3], cam=fa[:, 4:7],
        shots=shots, buttons=fa[:, 8].astype(np.uint8),
        tgt_frame=ta[:, 0].astype(np.int64), tgt_id=ta[:, 1].astype(np.int64),
        tgt_pos=ta[:, 2:5], tgt_radius=ta[:, 5], tgt_hh=ta[:, 6], tgt_health=ta[:, 7],
        start_wall_us=start_wall_us, game=game.rstrip(b"\0").decode("utf-8", "replace"),
    )


def write_spot(path: str | Path, run: Run) -> None:
    out = bytearray(HEADER.pack(MAGIC, VERSION, HEADER.size, 10_000_000, run.start_wall_us,
                                run.game.encode()[:40]))
    bounds = np.searchsorted(run.tgt_frame, np.arange(len(run.t) + 1))
    for i in range(len(run.t)):
        a, b = bounds[i], bounds[i + 1]
        shots = NO_SHOTS if run.shots[i] < 0 else int(run.shots[i])
        out += FRAME.pack(int(round(run.t[i] * 1e6)), run.yaw[i], run.pitch[i], run.fov[i],
                          *run.cam[i], shots, int(run.buttons[i]), b - a, 0)
        for r in range(a, b):
            out += TARGET.pack(int(run.tgt_id[r]), *run.tgt_pos[r], run.tgt_radius[r],
                               run.tgt_hh[r], run.tgt_health[r])
    Path(path).write_bytes(out)
