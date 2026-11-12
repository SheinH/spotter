"""spotter_analyze: .spot log -> reports/<run>/report.md + two PNGs.

    uv run python -m spotter_analyze                 newest log in ../runs, newest run inside it
    uv run python -m spotter_analyze path/to.spot --mode clicking --worst 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import clicking, kovaaks, report, tracking
from .format import read_spot

REPO = Path(__file__).resolve().parents[2]


def pick_mode(run) -> str:
    if not clicking.is_continuous_fire(run):
        return "clicking"
    run = clicking.split_respawns(run)
    return "switching" if len(clicking.kills(run, clicking.shot_frames(run))) > 2 else "tracking"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="spotter_analyze", description=__doc__.splitlines()[0])
    ap.add_argument("spot", nargs="?", type=Path, help=".spot log (default: newest in runs/)")
    ap.add_argument("--mode", choices=["auto", "tracking", "clicking", "switching"], default="auto")
    ap.add_argument("--csv", type=Path, help="KovaaK's stats CSV for this run (default: matched by time)")
    ap.add_argument("--stats-dir", type=Path, default=kovaaks.DEFAULT_ROOT / "stats")
    ap.add_argument("--all", action="store_true", help="report every run in the log, not just the last")
    ap.add_argument("--worst", type=int, default=0, help="append raw traces of the N worst events")
    ap.add_argument("--hh-ratio", type=float, default=1.0,
                    help="old (version 1) logs only: target half-height as a multiple of its radius")
    ap.add_argument("--out", type=Path, default=REPO / "reports")
    args = ap.parse_args(argv)

    spot = args.spot
    if spot is None:
        logs = sorted((REPO / "runs").glob("*.spot"), key=lambda p: p.stat().st_mtime)
        if not logs:
            print(f"no .spot logs in {REPO / 'runs'}", file=sys.stderr)
            return 1
        spot = logs[-1]
    log = read_spot(spot, args.hh_ratio)
    if len(log.t) < 100:
        print(f"{spot}: only {len(log.t)} frames", file=sys.stderr)
        return 1

    if args.csv:
        matched = [kovaaks.parse_stats(args.csv)]
    else:
        matched = kovaaks.find_stats(args.stats_dir, log.wall(log.t[0]), log.wall(log.t[-1]))
        if not args.all:
            matched = matched[-1:]

    jobs = []
    for st in matched:
        t0 = st.start.timestamp() - log.start_wall_us / 1e6
        t1 = st.end.timestamp() - log.start_wall_us / 1e6
        jobs.append((f"{st.scenario} - {st.end:%Y.%m.%d-%H.%M.%S}", log.slice(t0, t1), st))
    if not jobs:
        jobs.append((spot.stem, log, None))

    for name, run, st in jobs:
        mode = pick_mode(run) if args.mode == "auto" else args.mode
        result = tracking.analyze(run) if mode == "tracking" else clicking.analyze(run, mode)
        path = report.write(args.out / name, name, result, st, run, args.worst)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
