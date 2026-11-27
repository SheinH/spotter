"""Incremental .spot tail reader and live run-boundary state machine.

Unlike format.read_spot(), SpotTailer remembers its byte offset in an open/growing .spot file
and parses only newly flushed records, keeping O(window) slicing for 200+ MB sittings.

Real KovaaK's logs have a crucial quirk on the post-run Score Screen: targets can remain
alive in the arena behind the UI while camera yaw/pitch and shot count freeze completely.
LiveTracker therefore pairs active camera/shot motion + shot-counter resets with a Stats.csv
watcher so completed runs trigger immediately and early restarts reset silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path

import numpy as np

from . import kovaaks
from .format import (
    FRAME_COLS,
    HEADER,
    MAGIC,
    NO_SHOTS,
    TARGET,
    TARGET_V1,
    VERSION,
    Run,
    arrays_to_run,
    frame_struct,
)

MERGE_GAP_S = 1.5      # target-less gap tolerated inside one run (e.g. clicking respawn)
FREEZE_IDLE_S = 2.0    # frozen camera + frozen shots this long = paused / score screen / idle
STATS_WAIT_S = 2.0     # max wait after run stops before declaring it an unsaved early restart
MOVE_EPS_DEG = 0.04    # camera yaw+pitch delta that proves active player input


@dataclass
class RunStarted:
    t0: float


@dataclass
class RunAborted:
    t0: float
    t1: float
    reason: str


@dataclass
class RunCompleted:
    t0: float
    t1: float
    stats: kovaaks.Stats


LiveEvent = RunStarted | RunAborted | RunCompleted


class SpotTailer:
    """Append-only reader for a live .spot file."""

    def __init__(self, path: str | Path, v1_hh_ratio: float = 1.0) -> None:
        self.path = Path(path)
        self.v1_hh_ratio = v1_hh_ratio
        self.offset = 0
        self.version = VERSION
        self.start_wall_us = 0
        self.game = ""
        self._header_read = False

        # Chunked storage to avoid O(N) reallocations on every 250 ms poll
        self._frame_chunks: list[np.ndarray] = []
        self._chunk_end_frames: list[int] = []
        self._chunk_max_t: list[float] = []
        self._chunk_min_t: list[float] = []
        self._target_chunks: list[np.ndarray] = []
        self._n_frames = 0

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def wall0(self) -> float:
        return self.start_wall_us / 1e6

    @property
    def latest_t(self) -> float:
        return float(self._frame_chunks[-1][-1, 0] / 1e6) if self._frame_chunks else 0.0

    def poll(self, max_bytes: int | None = None) -> np.ndarray:
        """Read any newly flushed whole frames; returns the new frames array (K, FRAME_COLS)."""
        if not self.path.is_file():
            return np.empty((0, FRAME_COLS), dtype=np.float64)

        with self.path.open("rb") as fh:
            if not self._header_read:
                hdr = fh.read(HEADER.size)
                if len(hdr) < HEADER.size:
                    return np.empty((0, FRAME_COLS), dtype=np.float64)
                magic, version, header_size, _qpc, start_wall_us, game = HEADER.unpack_from(hdr, 0)
                if magic != MAGIC:
                    raise ValueError(f"{self.path}: not a .spot file")
                if version not in (1, 2, VERSION):
                    raise ValueError(f"{self.path}: unsupported version {version}")
                self.version = version
                self.start_wall_us = start_wall_us
                self.game = game.rstrip(b"\0").decode("utf-8", "replace")
                self.offset = header_size
                self._header_read = True

            fh.seek(self.offset)
            data = fh.read() if max_bytes is None else fh.read(max_bytes)

        if not data:
            return np.empty((0, FRAME_COLS), dtype=np.float64)

        target_struct = TARGET_V1 if self.version == 1 else TARGET
        tsize = target_struct.size
        frame = frame_struct(self.version)
        fsize = frame.size
        pos, end = 0, len(data)
        frames: list[tuple] = []
        targets: list[tuple] = []
        base_frame = self._n_frames

        while pos + fsize <= end:
            f = frame.unpack_from(data, pos)
            if self.version < 3:
                f = (*f, 0.0, 0.0)
            nt = f[9]
            rec_bytes = fsize + nt * tsize
            if pos + rec_bytes > end:
                break  # partial frame at EOF; wait for next flush
            f_idx = base_frame + len(frames)
            frames.append(f)
            t_off = pos + fsize
            for _ in range(nt):
                rec = target_struct.unpack_from(data, t_off)
                if self.version == 1:
                    rec = (*rec[:5], rec[4] * self.v1_hh_ratio, rec[5])
                targets.append((f_idx, *rec))
                t_off += tsize
            pos += rec_bytes

        if not frames:
            return np.empty((0, FRAME_COLS), dtype=np.float64)

        self.offset += pos
        fa = np.array(frames, dtype=np.float64).reshape(-1, FRAME_COLS)
        ta = np.array(targets, dtype=np.float64).reshape(-1, 8) if targets else np.empty((0, 8), dtype=np.float64)

        self._n_frames += len(fa)
        self._frame_chunks.append(fa)
        self._chunk_end_frames.append(self._n_frames)
        self._chunk_min_t.append(float(fa[0, 0] / 1e6))
        self._chunk_max_t.append(float(fa[-1, 0] / 1e6))
        self._target_chunks.append(ta)
        return fa

    def slice(self, t0: float, t1: float) -> Run:
        """Build a Run for [t0, t1] (log seconds) scanning only overlapping chunks."""
        if not self._frame_chunks:
            return _empty_run(self.start_wall_us, self.game)

        f_parts: list[np.ndarray] = []
        t_parts: list[np.ndarray] = []
        new_base = 0
        chunk_start_frame = 0

        for fa, ta, c_end, c_min, c_max in zip(
            self._frame_chunks, self._target_chunks, self._chunk_end_frames,
            self._chunk_min_t, self._chunk_max_t,
        ):
            if c_max < t0 or c_min > t1:
                chunk_start_frame = c_end
                continue
            ts = fa[:, 0] / 1e6
            keep = (ts >= t0) & (ts <= t1)
            if not keep.any():
                chunk_start_frame = c_end
                continue
            kept_fa = fa[keep]
            f_parts.append(kept_fa)
            if len(ta):
                local_f = ta[:, 0].astype(np.int64) - chunk_start_frame
                t_keep = keep[local_f]
                if t_keep.any():
                    remap = np.cumsum(keep) - 1 + new_base
                    sub_ta = ta[t_keep].copy()
                    sub_ta[:, 0] = remap[local_f[t_keep]]
                    t_parts.append(sub_ta)
            new_base += len(kept_fa)
            chunk_start_frame = c_end

        if not f_parts:
            return _empty_run(self.start_wall_us, self.game)

        fa_all = np.vstack(f_parts)
        ta_all = np.vstack(t_parts) if t_parts else np.empty((0, 8), dtype=np.float64)
        return _arrays_to_run(fa_all, ta_all, self.start_wall_us, self.game)

    def to_run(self) -> Run:
        if not self._frame_chunks:
            return _empty_run(self.start_wall_us, self.game)
        fa_all = np.vstack(self._frame_chunks)
        non_empty = [ta for ta in self._target_chunks if len(ta)]
        ta_all = np.vstack(non_empty) if non_empty else np.empty((0, 8), dtype=np.float64)
        return _arrays_to_run(fa_all, ta_all, self.start_wall_us, self.game)


def _empty_run(start_wall_us: int, game: str) -> Run:
    z = np.empty(0, dtype=np.float64)
    zi = np.empty(0, dtype=np.int64)
    return Run(
        t=z, yaw=z, pitch=z, fov=z, cam=np.empty((0, 3), dtype=np.float64),
        shots=zi, buttons=np.empty(0, dtype=np.uint8),
        tgt_frame=zi, tgt_id=zi, tgt_pos=np.empty((0, 3), dtype=np.float64),
        tgt_radius=z, tgt_hh=z, tgt_health=z,
        start_wall_us=start_wall_us, game=game,
    )


_arrays_to_run = arrays_to_run


class StatsWatcher:
    """Watches KovaaK's stats directory (often 1000s of files) in O(1) via dir mtime + os.scandir."""

    @staticmethod
    def _stamp_ts(name: str, fallback_mtime: float) -> float:
        m = kovaaks._NAME.match(name)
        if m:
            try:
                return datetime.strptime(m["stamp"], "%Y.%m.%d-%H.%M.%S").timestamp()
            except ValueError:
                pass
        return fallback_mtime

    def __init__(self, stats_dir: Path, since_wall_s: float) -> None:
        self.stats_dir = stats_dir
        self._seen: set[str] = set()
        self._baseline_mtime = since_wall_s
        self._last_dir_mtime_ns: int = -1
        self._candidates: list[tuple[float, str, Path]] = []  # (end_ts, name, path)

        if stats_dir.is_dir():
            try:
                self._last_dir_mtime_ns = stats_dir.stat().st_mtime_ns
                with os.scandir(stats_dir) as it:
                    for entry in it:
                        if not entry.name.endswith(" Stats.csv"):
                            continue
                        ts = self._stamp_ts(entry.name, entry.stat().st_mtime)
                        if ts <= since_wall_s:
                            self._seen.add(entry.name)
                        else:
                            self._candidates.append((ts, entry.name, Path(entry.path)))
                self._candidates.sort(key=lambda x: x[0])
            except OSError:
                pass

    def _refresh_dir(self) -> None:
        try:
            cur_ns = self.stats_dir.stat().st_mtime_ns
        except OSError:
            return
        if cur_ns == self._last_dir_mtime_ns:
            return
        self._last_dir_mtime_ns = cur_ns
        known = {name for _, name, _ in self._candidates} | self._seen
        try:
            with os.scandir(self.stats_dir) as it:
                for entry in it:
                    if not entry.name.endswith(" Stats.csv") or entry.name in known:
                        continue
                    ts = self._stamp_ts(entry.name, entry.stat().st_mtime)
                    if ts < self._baseline_mtime - 2.0:
                        self._seen.add(entry.name)
                    else:
                        self._candidates.append((ts, entry.name, Path(entry.path)))
            self._candidates.sort(key=lambda x: x[0])
        except OSError:
            pass

    def poll(self, wall0: float, latest_log_t: float) -> list[kovaaks.Stats]:
        if not self.stats_dir.is_dir():
            return []
        self._refresh_dir()
        if not self._candidates:
            return []

        latest_wall = wall0 + latest_log_t
        found: list[kovaaks.Stats] = []
        remaining: list[tuple[float, str, Path]] = []

        for mtime, name, path in self._candidates:
            # Files are sorted by mtime; stop once we're well ahead of the log's current wall clock
            if mtime > latest_wall + 120.0:
                remaining.append((mtime, name, path))
                continue
            if mtime < wall0 - 5.0:
                self._seen.add(name)
                continue
            try:
                st = kovaaks.parse_stats(path)
            except Exception:
                remaining.append((mtime, name, path))
                continue
            end_ts = st.end.timestamp()
            if end_ts <= wall0 + 1.0:
                self._seen.add(name)
            elif end_ts <= latest_wall + 2.5:
                self._seen.add(name)
                found.append(st)
            else:
                remaining.append((mtime, name, path))

        self._candidates = remaining
        return sorted(found, key=lambda s: s.end)


class LiveTracker:
    """Tracks active runs in real time from SpotTailer frames + StatsWatcher."""

    def __init__(self, tailer: SpotTailer, stats_dir: Path | None = None,
                 watch_existing_stats: bool = False) -> None:
        self.tailer = tailer
        self.stats_dir = stats_dir
        self._watcher: StatsWatcher | None = None
        self._watch_existing = watch_existing_stats

        self.in_run = False
        self.run_t0: float = 0.0
        self._last_target_t: float = 0.0
        self._last_active_t: float = 0.0
        self._prev_yaw: float | None = None
        self._prev_pitch: float | None = None
        self._prev_shots: int = -1
        self._active_streak: int = 0
        self._pending_stop: tuple[float, float, str, float] | None = None  # (t0, t1, reason, wall_deadline)
        self._completed_until_t: float = -1.0

    def step(self, now_monotonic: float, max_bytes: int | None = None) -> list[LiveEvent]:
        fa = self.tailer.poll(max_bytes=max_bytes)
        events: list[LiveEvent] = []

        if self.tailer._header_read and self._watcher is None and self.stats_dir is not None:
            since = 0.0 if self._watch_existing else self.tailer.wall0
            self._watcher = StatsWatcher(self.stats_dir, since_wall_s=since)

        # 1. Process new telemetry frames
        for row in fa:
            t = float(row[0] / 1e6)
            yaw, pitch = float(row[1]), float(row[2])
            raw_shots = int(row[7])
            shots = -1 if raw_shots == NO_SHOTS else raw_shots
            nt = int(row[9])

            moved = False
            if self._prev_yaw is not None and self._prev_pitch is not None:
                dy = abs((yaw - self._prev_yaw + 180.0) % 360.0 - 180.0)
                dp = abs(pitch - self._prev_pitch)
                moved = (dy + dp) >= MOVE_EPS_DEG
            shot_inc = shots >= 0 and self._prev_shots >= 0 and shots > self._prev_shots
            shot_reset = shots >= 0 and self._prev_shots > 0 and shots < self._prev_shots
            active_input = moved or shot_inc

            if not self.in_run:
                # Ignore buzzer wind-down & score-screen shot-counter zeroing right after a completed run
                if t <= self._completed_until_t:
                    self._active_streak = 0
                elif nt > 0 and active_input:
                    self._active_streak += 1
                elif nt == 0:
                    self._active_streak = 0

                # Require sustained active aiming/shooting (~6 frames) to start a real run
                if nt > 0 and t > self._completed_until_t and self._active_streak >= 6:
                    if self._pending_stop is not None:
                        p_t0, p_t1, p_reason, _ = self._pending_stop
                        events.append(RunAborted(p_t0, p_t1, p_reason))
                        self._pending_stop = None
                    self.in_run = True
                    self.run_t0 = t
                    self._last_target_t = t
                    self._last_active_t = t
                    self._active_streak = 0
                    events.append(RunStarted(t0=t))
            else:
                if shot_reset:
                    # Immediate restart (or score-screen reset): close current run, wait briefly in case Stats.csv lands
                    prev_t1 = max(self._last_target_t, self.run_t0)
                    self._pending_stop = (self.run_t0, prev_t1, "shot_reset", now_monotonic + STATS_WAIT_S)
                    self.in_run = False
                    if nt > 0 and moved:
                        # Instant mid-run restart where hand is already moving
                        p_t0, p_t1, p_reason, _ = self._pending_stop
                        events.append(RunAborted(p_t0, p_t1, p_reason))
                        self._pending_stop = None
                        self.in_run = True
                        self.run_t0 = t
                        self._last_target_t = t
                        self._last_active_t = t
                        events.append(RunStarted(t0=t))
                else:
                    if nt > 0:
                        self._last_target_t = t
                    if active_input:
                        self._last_active_t = t

                    gap_no_tgt = (t - self._last_target_t) >= MERGE_GAP_S if nt == 0 else False
                    gap_frozen = (t - self._last_active_t) >= FREEZE_IDLE_S
                    if gap_no_tgt or gap_frozen:
                        reason = "targets_gone" if gap_no_tgt else "frozen"
                        stop_t1 = self._last_target_t if gap_no_tgt else self._last_active_t
                        self._pending_stop = (self.run_t0, max(stop_t1, self.run_t0), reason,
                                              now_monotonic + STATS_WAIT_S)
                        self.in_run = False

            self._prev_yaw, self._prev_pitch, self._prev_shots = yaw, pitch, shots

        # 2. Poll Stats.csv for completed runs (authoritative Challenge Start -> End)
        if self._watcher is not None and self.tailer.n_frames > 0:
            for st in self._watcher.poll(self.tailer.wall0, self.tailer.latest_t):
                t0 = max(0.0, st.start.timestamp() - self.tailer.wall0)
                # KovaaK's filename stamp has 1-second resolution (truncates .000-.999s);
                # include up to +1.0s / last kill / current log head so the final kills aren't cut off
                # AND trailing buzzer frames don't immediately fire a phantom RunStarted!
                last_kill_ts = st.kill_times[-1].timestamp() if st.kill_times else st.end.timestamp()
                end_wall = max(st.end.timestamp() + 0.999, last_kill_ts + 0.1)
                t1 = min(max(end_wall - self.tailer.wall0, t0 + 0.5), self.tailer.latest_t + 1.0)
                self._completed_until_t = max(t1, self.tailer.latest_t) + 2.5
                self._active_streak = 0
                self._pending_stop = None
                if self.in_run and self.run_t0 <= t1 + 2.0:
                    self.in_run = False
                events.append(RunCompleted(t0=t0, t1=t1, stats=st))

        # 3. Expire pending unsaved stops once STATS_WAIT_S passes with no Stats.csv
        if self._pending_stop is not None and now_monotonic >= self._pending_stop[3]:
            p_t0, p_t1, p_reason, _ = self._pending_stop
            self._pending_stop = None
            events.append(RunAborted(p_t0, p_t1, p_reason))

        return events
