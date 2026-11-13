"""Read the files KovaaK's writes itself: stats CSVs and .perf protobufs."""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path(r"C:\Program Files (x86)\Steam\steamapps\common\FPSAimTrainer\FPSAimTrainer")
_NAME = re.compile(r"^(?P<scenario>.+) - (?P<mode>[^-]+) - (?P<stamp>\d{4}\.\d\d\.\d\d-\d\d\.\d\d\.\d\d) Stats\.csv$")


@dataclass
class Stats:
    path: Path
    scenario: str
    end: datetime                      # local time the run finished (from the filename)
    start: datetime                    # local time of "Challenge Start"
    kill_times: list[datetime] = field(default_factory=list)
    kill_shots: list[int] = field(default_factory=list)
    shots: int = 0                     # weapon totals
    hits: int = 0
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def perf_path(self) -> Path:
        return self.path.parent.parent / "performances" / self.path.name.replace(
            " Stats.csv", " Performance.perf")


def _time_of_day(text: str, end: datetime) -> datetime:
    """KovaaK's logs times as local HH:MM:SS.mmm with no date; anchor to the run's end."""
    h, m, s = text.split(":")
    t = end.replace(hour=int(h), minute=int(m), second=0, microsecond=0) + timedelta(seconds=float(s))
    return t - timedelta(days=1) if t > end + timedelta(seconds=5) else t


def parse_stats(path: str | Path) -> Stats:
    path = Path(path)
    m = _NAME.match(path.name)
    if not m:
        raise ValueError(f"{path.name}: not a KovaaK's stats filename")
    end = datetime.strptime(m["stamp"], "%Y.%m.%d-%H.%M.%S")
    st = Stats(path=path, scenario=m["scenario"], end=end, start=end)

    blocks = path.read_text(encoding="utf-8", errors="replace").split("\n\n")
    for line in blocks[0].splitlines()[1:]:
        c = line.split(",")
        if len(c) >= 7 and c[0].isdigit():
            st.kill_times.append(_time_of_day(c[1], end))
            st.kill_shots.append(int(c[5]))
    for block in blocks[1:]:
        lines = block.strip().splitlines()
        if lines and lines[0].startswith("Weapon,Shots,Hits"):
            for line in lines[1:]:
                c = line.split(",")
                st.shots += int(float(c[1]))
                st.hits += int(float(c[2]))
            continue
        for line in lines:
            key, sep, value = line.partition(":,")
            if sep:
                st.fields[key.strip()] = value.strip()
    if "Challenge Start" in st.fields:
        st.start = _time_of_day(st.fields["Challenge Start"], end)
    return st


def find_stats(stats_dir: Path, wall_start: float, wall_end: float) -> list[Stats]:
    """Stats files for runs that overlap [wall_start, wall_end] (Unix seconds), oldest first.
    A run the log only caught part of (hook injected mid-run) still counts."""
    found = []
    if not stats_dir.is_dir():
        return found
    for p in stats_dir.glob("* Stats.csv"):
        mtime = p.stat().st_mtime
        if not (wall_start <= mtime <= wall_end + 120):
            continue
        try:
            st = parse_stats(p)
        except (ValueError, IndexError):
            continue
        if st.end.timestamp() > wall_start + 5 and st.start.timestamp() < wall_end - 5:
            found.append(st)
    return sorted(found, key=lambda s: s.end)


# --- .perf: protobuf, one header message then one small message per counter per second ---

_PERF_COUNTERS = {2: "shots", 3: "hits", 4: "misses", 5: "damage", 6: "damage_possible",
                  7: "score", 8: "kills"}


def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 127) << s
        s += 7
        if c < 128:
            return r, i


def _fields(b: bytes) -> list[tuple[int, int, object]]:
    i, out = 0, []
    while i < len(b):
        key, i = _varint(b, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            v, i = _varint(b, i)
        elif wire == 1:
            v = struct.unpack_from("<d", b, i)[0]
            i += 8
        elif wire == 5:
            v = struct.unpack_from("<f", b, i)[0]
            i += 4
        elif wire == 2:
            n, i = _varint(b, i)
            v = b[i:i + n]
            i += n
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        out.append((num, wire, v))
    return out


def parse_perf(path: str | Path) -> dict:
    """Returns {'scenario', 'duration', 't': seconds, '<counter>': per-interval values}."""
    top = _fields(Path(path).read_bytes())
    out: dict = {"scenario": "", "duration": 0.0}
    series: dict[float, dict[str, float]] = {}
    for num, _wire, v in top:
        if num == 1:
            for n2, _w2, v2 in _fields(v):
                if n2 == 1:
                    out["scenario"] = v2.decode("utf-8", "replace")
                elif n2 == 5:
                    out["duration"] = next((x for n3, w3, x in _fields(v2) if n3 == 1 and w3 == 5), 0.0)
        elif num == 2:
            sub = _fields(v)
            t = float(sub[0][2])
            for n2, _w2, v2 in sub[1:]:
                if n2 in _PERF_COUNTERS:
                    series.setdefault(t, {})[_PERF_COUNTERS[n2]] = float(_fields(v2)[0][2])
    times = sorted(series)
    out["t"] = np.array(times)
    for name in _PERF_COUNTERS.values():
        out[name] = np.array([series[t].get(name, 0.0) for t in times])
    return out
