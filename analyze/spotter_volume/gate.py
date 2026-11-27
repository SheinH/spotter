"""When may the overlay be on screen? Never while you are in a scenario.

From the Spotter log: you count as "in a scenario" while any target exists or the camera or
shot counter moved recently. On this game build targets load a few seconds before a run
starts and clear about a second after it ends, and the results screen has no targets and a
frozen camera, so the overlay comes back there and in menus.

If the log goes quiet, the hook's own text log says why: "no player/world: paused" (a menu:
fine to show) or nothing fresh at all (hook not running: we cannot see runs, so stay hidden
whenever the game is in front).
"""
from __future__ import annotations

import time
from enum import Enum
from pathlib import Path

import numpy as np

from spotter_analyze.format import NO_SHOTS

TARGET_HOLD_S = 1.5      # targets gone this long (log time) before we call the scenario over
MOVE_HOLD_S = 1.0        # camera/shots still this long, too
MOVE_EPS_DEG = 0.04      # yaw+pitch change that counts as the hand moving
STALE_S = 1.5            # no new frames for this long (wall time): the log went quiet
HOOK_ALIVE_S = 15.0      # the hook's .log heartbeat is every 10 s


class State(Enum):
    AIMING = "aiming"        # in a scenario: hide
    IDLE = "idle"            # results screen or menu: may show
    UNKNOWN = "unknown"      # no hook: hide while the game is in front


class Gate:
    def __init__(self) -> None:
        self.latest_t = -1.0          # log seconds
        self.last_target_t = -1e9
        self.last_move_t = -1e9
        self.last_arrival = -1e9      # monotonic seconds when frames last arrived
        self._prev: tuple[float, float, int] | None = None

    def feed(self, frames: np.ndarray, now: float | None = None) -> None:
        """New frames from SpotTailer.poll (columns: t_us, yaw, pitch, ..., shots at 7, n_targets at 9)."""
        if not len(frames):
            return
        self.last_arrival = time.monotonic() if now is None else now
        for row in frames:
            t, yaw, pitch = row[0] / 1e6, float(row[1]), float(row[2])
            shots = -1 if int(row[7]) == NO_SHOTS else int(row[7])
            if row[9] > 0:
                self.last_target_t = t
            if self._prev is not None:
                py, pp, ps = self._prev
                moved = abs((yaw - py + 180.0) % 360.0 - 180.0) + abs(pitch - pp) >= MOVE_EPS_DEG
                if moved or (shots >= 0 and ps >= 0 and shots > ps):
                    self.last_move_t = t
            self._prev = (yaw, pitch, shots)
            self.latest_t = t

    def state(self, hook_paused: bool, hook_alive: bool, now: float | None = None) -> State:
        now = time.monotonic() if now is None else now
        if now - self.last_arrival > STALE_S:
            # no fresh frames: a menu if the hook says so, otherwise we are blind
            return State.IDLE if (hook_alive and hook_paused) else State.UNKNOWN
        if (self.latest_t - self.last_target_t < TARGET_HOLD_S
                or self.latest_t - self.last_move_t < MOVE_HOLD_S):
            return State.AIMING
        return State.IDLE


class HookLog:
    """Follows the hook's text log: alive = written to in the last 15 s, paused = its latest
    state line is 'no player/world: paused'. Reads only what was appended since last time."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.offset = 0
        self.paused = False

    def status(self, now_wall: float | None = None) -> tuple[bool, bool]:
        if self.path is None or not self.path.exists():
            return False, False
        now_wall = time.time() if now_wall is None else now_wall
        try:
            alive = now_wall - self.path.stat().st_mtime < HOOK_ALIVE_S
            with self.path.open("rb") as fh:
                fh.seek(self.offset)
                new = fh.read()
        except OSError:
            return False, self.paused
        cut = new.rfind(b"\n") + 1          # whole lines only
        self.offset += cut
        text = new[:cut].decode("utf-8", "replace")
        i_pause, i_run = text.rfind("no player/world: paused"), text.rfind("in a scenario: sampling")
        if i_pause >= 0 or i_run >= 0:
            self.paused = i_pause > i_run
        return alive, self.paused
