"""Aim time: every run KovaaK's saved, plus runs you restarted before the end.

Saved runs come from the stats folder (Challenge Start to the file's end time), cached by
filename so only new files are read. Restarted runs leave no stats file; the overlay sees them
in the Spotter log and records them in reports/volume_restarts.jsonl. The log does not say
which scenario a restart was, so it is attributed by context (see attribute()): restarts
cluster before the run you finally finish, so the next saved run within 10 minutes wins,
else the previous one. Attribution is worked out fresh each time, so a restart first credited
to the run before it moves to the next run once that one is saved.

A day rolls over at 05:00, so a session that runs past midnight counts for the evening.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from spotter_analyze import kovaaks

DAY_ROLLOVER_H = 5
MAX_RUN_S = 15 * 60          # longer than this means a broken Challenge Start; clamp
MIN_RESTART_S = 3.0          # shorter restarts are misclicks, not aiming
ATTRIBUTE_WINDOW = timedelta(minutes=10)   # a restart this close to a saved run belongs to it
CACHE_NAME = "volume_cache.json"
RESTARTS_NAME = "volume_restarts.jsonl"


def day_of(t: datetime) -> date:
    return (t - timedelta(hours=DAY_ROLLOVER_H)).date()


@dataclass(frozen=True)
class Run:
    scenario: str | None       # None = restarted run, scenario unknown
    start: datetime
    end: datetime

    @property
    def seconds(self) -> float:
        return min(max((self.end - self.start).total_seconds(), 0.0), MAX_RUN_S)


@dataclass(frozen=True)
class Summary:
    today_s: float
    target_s: float
    streak_days: int
    scenario: str | None       # the scenario played last
    scenario_today_s: float
    scenario_all_s: float
    all_s: float

    @property
    def done(self) -> bool:
        return self.today_s >= self.target_s


class Ledger:
    def __init__(self, stats_dir: Path, data_dir: Path) -> None:
        self.stats_dir = stats_dir
        self.data_dir = data_dir
        self.saved: list[Run] = []
        self.restarts: list[Run] = []
        self._restart_keys: set[str] = set()

    # --- loading -------------------------------------------------------------------------

    def load(self) -> "Ledger":
        """Read everything. The first call can take ~30 s on a big stats folder; run it off
        the UI thread."""
        self.saved = sorted(self._scan(), key=lambda r: r.end)
        path = self.data_dir / RESTARTS_NAME
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                    self.restarts.append(Run(None, datetime.fromisoformat(d["start"]),
                                             datetime.fromisoformat(d["end"])))
                    self._restart_keys.add(d["key"])
                except (ValueError, KeyError, TypeError):
                    continue
        return self

    def _scan(self) -> list[Run]:
        cache_path = self.data_dir / CACHE_NAME
        try:
            cache: dict[str, list] = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        out, dirty = [], False
        if self.stats_dir.is_dir():
            for p in self.stats_dir.glob("* Stats.csv"):
                row = cache.get(p.name)
                if row is None:
                    try:
                        st = kovaaks.parse_stats(p)
                        row = [st.scenario, st.start.isoformat(), st.end.isoformat()]
                    except (ValueError, IndexError, OSError):
                        row = []                        # remember unreadable files too
                    cache[p.name] = row
                    dirty = True
                if row:
                    out.append(Run(row[0], datetime.fromisoformat(row[1]), datetime.fromisoformat(row[2])))
        if dirty:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        return out

    # --- live updates --------------------------------------------------------------------

    def add_saved(self, st: kovaaks.Stats) -> Run | None:
        """A run KovaaK's just saved; None if we already had it."""
        run = Run(st.scenario, st.start, st.end)
        if run in self.saved[-200:]:
            return None
        self.saved.append(run)
        self.saved.sort(key=lambda r: r.end)
        return run

    def add_restart(self, key: str, start: datetime, end: datetime) -> Run | None:
        """Record a run that ended without a stats file. Ignored if it is too short, already
        recorded (same key), or overlaps a saved run (then it was not a restart after all)."""
        run = Run(None, start, end)
        if key in self._restart_keys or run.seconds < MIN_RESTART_S:
            return None
        slack = timedelta(seconds=1)
        if any(s.start - slack < end and start < s.end + slack for s in self.saved[-50:]):
            return None
        self._restart_keys.add(key)
        self.restarts.append(run)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with (self.data_dir / RESTARTS_NAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "start": start.isoformat(), "end": end.isoformat()}) + "\n")
        return run

    # --- numbers for the overlay ---------------------------------------------------------

    def summary(self, now: datetime, target_min: float) -> Summary:
        today = day_of(now)
        daily: dict[date, float] = {}
        for r in self.saved + self.restarts:
            d = day_of(r.start)
            daily[d] = daily.get(d, 0.0) + r.seconds
        target_s = target_min * 60
        scenario = self.saved[-1].scenario if self.saved else None
        mine = [r for r in self.saved if r.scenario == scenario]
        mine += [r for r, sc in zip(self.restarts, attribute(self.restarts, self.saved)) if sc == scenario]
        return Summary(
            today_s=daily.get(today, 0.0),
            target_s=target_s,
            streak_days=streak(daily, today, target_s),
            scenario=scenario,
            scenario_today_s=sum(r.seconds for r in mine if day_of(r.start) == today),
            scenario_all_s=sum(r.seconds for r in mine),
            all_s=sum(daily.values()),
        )


def attribute(restarts: list[Run], saved: list[Run]) -> list[str | None]:
    """Best guess at each restart's scenario: the next saved run if it starts within 10
    minutes of the restart ending, otherwise the previous saved run if it ended within 10
    minutes of the restart starting, otherwise unknown (None)."""
    by_start = sorted(saved, key=lambda r: r.start)
    starts = [r.start for r in by_start]
    by_end = sorted(saved, key=lambda r: r.end)
    ends = [r.end for r in by_end]
    out: list[str | None] = []
    for r in restarts:
        i = bisect.bisect_left(starts, r.end)
        if i < len(by_start) and by_start[i].start - r.end <= ATTRIBUTE_WINDOW:
            out.append(by_start[i].scenario)
            continue
        j = bisect.bisect_right(ends, r.start) - 1
        if j >= 0 and r.start - by_end[j].end <= ATTRIBUTE_WINDOW:
            out.append(by_end[j].scenario)
            continue
        out.append(None)
    return out


def streak(daily: dict[date, float], today: date, target_s: float) -> int:
    """Days in a row at or over target, ending today, or yesterday while today is still open."""
    day = today if daily.get(today, 0.0) >= target_s else today - timedelta(days=1)
    n = 0
    while daily.get(day, 0.0) >= target_s:
        n += 1
        day -= timedelta(days=1)
    return n
