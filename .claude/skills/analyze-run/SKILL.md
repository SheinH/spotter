---
name: analyze-run
description: Analyze a KovaaK's aim-training run from local files with Spotter. Use when the user asks to analyze, review or coach a run or session, asks what went wrong in a scenario, wants Spotter started or checked, or wants raw numbers pulled from Spotter .spot logs or KovaaK's stats/perf files.
---

# Analyze a KovaaK's run with Spotter

Spotter is this repo: a DLL injected into KovaaK's that logs crosshair and target telemetry
(`hook/`), a launcher that injects it (`launch/`), and a Python analyzer that turns a log into a
report (`analyze/`). Everything is local files; nothing is fetched from the network.

## Where things are

| What | Path |
| --- | --- |
| Launcher + hook | `build/bin/Release/spotter-launch.exe`, `spotter.dll` |
| Spotter logs | `runs/spotter-YYYYMMDD-HHMMSS.spot` (telemetry) and `.log` (hook's own text log) |
| Reports | `reports/<scenario> - <YYYY.MM.DD-HH.MM.SS>/report.md` + `error_scatter.png`, `velocity_profiles.png` |
| Session pages | `reports/<scenario> - session <YYYY.MM.DD>/session.md` |
| Per-run metrics history | `reports/history.jsonl` (one JSON record per saved run) |
| KovaaK's stats | `C:\Program Files (x86)\Steam\steamapps\common\FPSAimTrainer\FPSAimTrainer\stats\<scenario> - <mode> - <stamp> Stats.csv` |
| KovaaK's per-second counters | same folder's sibling `performances\<same name> Performance.perf` |

`runs/` and `reports/` are gitignored. Python is run with `uv` (never pip/venv directly).

## 1. Make sure Spotter is recording

One hook instance lives for as long as the game process does and writes one `.spot`/`.log` pair
named after the moment it was injected. It records every run played after injection; **a run
played before injection is not in any log** and can only be discussed from KovaaK's own files.

Check, in this order:

1. Is the game up? `tasklist /FI "IMAGENAME eq FPSAimTrainer-Win64-Shipping.exe"`
2. Is the hook alive? Read the tail of the newest `runs/spotter-*.log`. A healthy hook writes a
   `frames=... targets=... shots=...` heartbeat every 10 s, so the last line should be seconds
   old and `frames` should be climbing while a scenario is open.

Start it (safe to run any time; if the DLL is already loaded it says so and exits 0):

```bash
build/bin/Release/spotter-launch.exe --wait 120
```

- `--wait N` waits up to N seconds for the game to appear. If the game is not running, ask the
  user to start KovaaK's (or, if they asked you to, `Start-Process steam://rungameid/824270`).
- `--require-offline` refuses to inject unless Steam is in offline mode; use it when the user
  wants that guarantee.
- Exit codes: 0 loaded / already loaded, 1 game not running, DLL missing or injection failed,
  2 bad arguments, 3 refused by `--require-offline`.
- Launcher missing? Build it: `cmake -S . -B build` then `cmake --build build --config Release`.
  `spotter.dll` cannot be rebuilt while the game has it loaded.

Lines in the `.log` and what they mean:

| Line | Meaning |
| --- | --- |
| `in a scenario: sampling` | recording |
| `no player/world: paused` | in a menu or loading; nothing to record, not a fault |
| `target class ...` | a new kind of target was seen (radius, height, health) |
| `frames=N skipped_reads=M ...` | heartbeat; `targets=0` means no live targets right now |
| `this is not the game build the offsets were dumped from; not sampling` | KovaaK's updated. No `.spot` is written. `hook/src/offsets.h` must be re-dumped; tell the user, do not guess offsets |

After starting the hook, the user has to play (and finish) a run before there is anything new to
analyze. The `.spot` is flushed every second, so it can be read while the game is still open.
KovaaK's writes its Stats.csv only when a challenge runs to the end.

## 2. Analyze

Run from `analyze/`:

```bash
uv run python -m spotter_analyze
```

| Want | Command |
| --- | --- |
| The latest finished run (default) | `uv run python -m spotter_analyze` |
| Every run in the newest log | `... --all` |
| Every run + one page for the latest scenario's runs that day | `... --session` |
| A session for a named scenario from a date on | `... --session "voxTargetSwitch 2" --since 2026-11-14` |
| An older log | `... ../runs/spotter-20261112-135147.spot --all` |
| Raw traces of the N worst events appended | `... --worst 3` |
| Force the analysis type | `... --mode tracking\|clicking\|switching` |
| Pin the stats file by hand | `... --csv "<path to Stats.csv>"` |

It prints the path of each `report.md` (and `session.md`) it wrote. Read that file, and look at
the two PNGs next to it with the Read tool.

How it works, so results can be sanity-checked:

- A log covers a whole game sitting. Runs are found as stretches where targets exist, split when
  the shot counter resets. Stats files whose time window overlaps the log are attached by time
  and give the run its name, score and exact start/end. A stretch with no stats file (restarted
  early, free play) is reported as `unsaved run - <time>` and is skipped if under 20 s.
- Without `--all`/`--session`, the newest run *with a stats file* is chosen, else the newest.
- Mode is automatic: no continuous fire -> `clicking`; continuous fire with kills -> `switching`;
  continuous fire with no kills -> `tracking`.
- Only saved runs are added to `reports/history.jsonl` (replaced by name, so re-running is safe).
- `<name>: too little data, skipped` (exit code still 0, no report path printed) or
  `only N frames` for the newest log usually means the hook was injected after
  the run, or the user has not played since. Check the log's start time (in its filename)
  against the newest Stats.csv, say so, and fall back to the previous log or KovaaK's files.

Reading a report:

- Header line: score, sens (cm/360), DPI, FOV, fps. Always note settings changes between runs
  before comparing them.
- **Summary** is the plain-language version. **"Against your other runs" (noise layer)** is the
  judge of what matters: `usual`/`sd` come from the last ~20 runs, `verdict` says whether this
  run is outside that spread, `noise_2runs` is how far two single runs must differ to mean
  anything. Treat "within noise" as no change, however big the raw difference looks.
- **Cross-check** compares the log's kills/shots with the game's. A large mismatch means the log
  missed part of the run; say so and trust the affected numbers less.
- Switching/clicking: time budget (reaction / travel / off again / on target), splits by flick
  distance, direction, target choice and third of the run, then one row per kill. Units: `_r`
  columns are target radii (negative = short, |x| < 1 = on target), `_ms`, `deg`, `dps`.
- Tracking: on-target %, lead/lag in radii, reaction to reversals, off-target episodes, splits
  by target speed / position / time since reversal.
- Tension section: jitter is labelled "not yet validated" and hand activity is "context". Do not
  turn either into a tense/relaxed verdict.

When coaching from it: lead with the one or two findings that are outside noise or that the time
budget says cost the most, back each with its number, and separate what the data shows from
what you infer. Single-run splits with small `n` are hints, not findings; a session page or
`history.jsonl` across runs is the evidence.

## 3. Pull raw data yourself

For anything the report does not answer, write a short script to the scratchpad and run it with
the analyzer's environment (works from any directory):

```bash
uv run --project analyze python path/to/script.py
```

### Spotter logs

```python
from pathlib import Path
import numpy as np
from spotter_analyze import geometry, kovaaks, spans
from spotter_analyze.format import read_spot

REPO = Path(r"C:\Users\meow\Downloads\spotter")
STATS = kovaaks.DEFAULT_ROOT / "stats"

spot = max((REPO / "runs").glob("spotter-*.spot"), key=lambda p: p.stat().st_mtime)
log = read_spot(spot)                      # the whole sitting; big logs take a few seconds
runs = spans.find_runs(log, STATS)         # [Span(t0, t1, stats | None)], oldest first
for sp in runs:
    print(sp.name(log), sp.t0, sp.t1, bool(sp.stats))

sp = [s for s in runs if s.stats][-1]      # newest saved run
run = log.slice(sp.t0, sp.t1)              # same structure, just that run
```

`Run` fields (see `analyze/spotter_analyze/format.py`):

- Per frame (length N, ~335 Hz): `t` seconds since log start, `yaw`/`pitch` degrees (right / up
  positive), `fov`, `cam` (N,3) world position, `shots` cumulative shot counter (-1 = unknown;
  a drop means a restart), `buttons` (bit 1 = fire held).
- Per target per frame (flat, length M): `tgt_frame` (index into the frame arrays), `tgt_id`
  (stable while that target lives), `tgt_pos` (M,3), `tgt_radius`, `tgt_hh` (half height; equals
  radius for a sphere), `tgt_health`.
- `log.wall(t)` -> Unix seconds, for lining a log time up with KovaaK's timestamps.

Crosshair-to-target geometry, in degrees:

```python
v = geometry.target_view(run)              # one entry per target row
# v.ex / v.ey: target right / above the crosshair; v.ang: total angle to its centre
# v.arad: its angular radius; v.miss: angle to the nearest point of its axis (on target if <= arad)
near = geometry.nearest(run, v)            # per frame: row of the closest target, -1 if none
has = near >= 0
on_target = v.miss[near[has]] <= v.arad[near[has]]
```

The full per-event tables behind a report come from the analyzers directly:
`clicking.analyze(run, "clicking" | "switching")` and `tracking.analyze(run)` return dicts; print
the keys rather than assuming them. `tension.analyze(run, mode, mm_per_deg)` likewise.

### KovaaK's files

```python
st = kovaaks.parse_stats(path_to_stats_csv)   # or sp.stats from a Span
st.scenario, st.start, st.end                 # local datetimes
st.fields["Score"], st.fields["Horiz Sens"], st.fields["Sens Scale"], st.fields["DPI"], st.fields["FOV"]
st.shots, st.hits                             # weapon totals
st.kill_times, st.kill_shots                  # per kill: datetime, shots fired at it

perf = kovaaks.parse_perf(st.perf_path)       # per-second counters for the same run
perf["t"], perf["score"], perf["kills"], perf["shots"], perf["hits"], perf["misses"], perf["damage"]

from spotter_analyze import history
every = history.score_history(STATS, "voxTargetSwitch 2")   # every stats file of a scenario, oldest first
scores = [(s.end, float(s.fields["Score"]), s.fields.get("Horiz Sens")) for s in every]
```

The stats folder holds thousands of files going back years, including runs Spotter never saw, so
it is the source for long-term score trends and for runs played without the hook. A Stats.csv is
a per-kill table, a blank line, weapon totals, then `Key:,value` lines; `fields` holds all of the
latter as strings. Filenames carry the run's *end* time; `Challenge Start` is in the fields.

### Analyzed-run history

```python
hist = history.load(REPO / "reports" / "history.jsonl")
mine = [r for r in hist if r["scenario"] == "voxTargetSwitch 2"]
# r: name, scenario, end (ISO), mode, score, sens, sens_scale, dpi, fov, metrics{...}, bins{...}
```

Use this for trends in Spotter's own metrics (reaction, overflick, flick gain, on-target %)
across runs; it only contains runs that have been through the analyzer, so run `--all` on older
logs first if runs are missing. Compare like with like: same scenario, same `mode`, and call out
sens/FOV differences.
