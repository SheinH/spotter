"""Find the runs inside a .spot log from the log itself, then attach KovaaK's stats files.

The hook logs for as long as the game is open, across many runs and the menus between them.
A run is a stretch where targets exist; it ends when the targets go away or the weapon's shot
counter resets (a restart). Runs the game wrote a stats file for get its exact Challenge Start
to end window; runs without one (restarted early, free play) are still found and analyzed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import kovaaks
from .format import Run

MIN_RUN_S = 20.0     # shorter stretches without a stats file are restarts, not runs
MERGE_GAP_S = 1.5    # target-less gaps shorter than this, with no restart, stay inside one run


@dataclass
class Span:
    t0: float                          # log seconds
    t1: float
    stats: kovaaks.Stats | None

    def name(self, log: Run) -> str:
        if self.stats:
            return f"{self.stats.scenario} - {self.stats.end:%Y.%m.%d-%H.%M.%S}"
        return f"unsaved run - {datetime.fromtimestamp(log.wall(self.t1)):%Y.%m.%d-%H.%M.%S}"


def target_spans(log: Run) -> list[tuple[float, float]]:
    """(t0, t1) stretches with at least one target, split where the shot counter resets."""
    n = len(log.t)
    has = np.zeros(n, dtype=bool)
    has[log.tgt_frame] = True
    if (log.shots >= 0).all():
        reset = np.r_[False, np.diff(log.shots) < 0]
    else:
        reset = np.zeros(n, dtype=bool)
    starts = has & (~np.r_[False, has[:-1]] | reset)
    ends = has & (~np.r_[has[1:], False] | np.r_[reset[1:], False])
    out: list[list[float]] = []
    for a, b in zip(np.flatnonzero(starts), np.flatnonzero(ends)):
        # clicking scenarios leave a frame or two with no target between kills: same run
        if out and not reset[a] and log.t[a] - out[-1][1] < MERGE_GAP_S:
            out[-1][1] = float(log.t[b])
        else:
            out.append([float(log.t[a]), float(log.t[b])])
    return [(a, b) for a, b in out]


def find_runs(log: Run, stats_dir: Path, min_s: float = MIN_RUN_S) -> list[Span]:
    wall0 = log.start_wall_us / 1e6
    stats = kovaaks.find_stats(stats_dir, log.wall(log.t[0]), log.wall(log.t[-1]))
    timed = [(st.start.timestamp() - wall0, st.end.timestamp() - wall0, st) for st in stats]
    out = [Span(a, b, st) for a, b, st in timed]
    for a, b in target_spans(log):
        # what's left of the stretch once the saved runs inside it are cut out
        pieces = [(a, b)]
        for s0, s1, _ in timed:
            pieces = [p for x0, x1 in pieces for p in ((x0, min(x1, s0)), (max(x0, s1), x1)) if p[1] > p[0]]
        out += [Span(x0, x1, None) for x0, x1 in pieces if x1 - x0 >= min_s]
    return sorted(out, key=lambda s: s.t0)
