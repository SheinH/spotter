"""spotter_analyze: .spot log -> reports/<run>/report.md + two PNGs.

    uv run python -m spotter_analyze                 newest log in ../runs, newest run inside it
    uv run python -m spotter_analyze --all           every run in the newest log
    uv run python -m spotter_analyze --session       every run, plus one page for the latest
                                                     scenario's runs that day
    uv run python -m spotter_analyze path/to.spot --mode clicking --worst 3
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

from . import clicking, history, kovaaks, report, session, spans, tension, tracking
from .format import read_spot

REPO = Path(__file__).resolve().parents[2]
warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
warnings.filterwarnings("ignore", "nperseg", UserWarning)          # very short runs
warnings.filterwarnings("ignore", "No artists with labels", UserWarning)


def pick_mode(run, stats: kovaaks.Stats | None = None) -> str:
    if not clicking.is_continuous_fire(run):
        return "clicking"
    if stats is not None:
        if not stats.kill_times and (stats.fields.get("Kills", "").strip() in ("0", "0.0")):
            return "tracking"          # e.g. invincible bots: the game counted no kills
    elif len(run.tgt_health) and run.tgt_health.min() == run.tgt_health.max():
        return "tracking"              # no stats file, and no target ever lost health
    run = clicking.split_respawns(run)
    return "switching" if len(clicking.kills(run, clicking.shot_frames(run))) > 2 else "tracking"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="spotter_analyze", description=__doc__.splitlines()[0])
    ap.add_argument("spot", nargs="?", type=Path, help=".spot log (default: newest in runs/)")
    ap.add_argument("--mode", choices=["auto", "tracking", "clicking", "switching"], default="auto")
    ap.add_argument("--csv", type=Path, help="KovaaK's stats CSV for this run (default: matched by time)")
    ap.add_argument("--stats-dir", type=Path, default=kovaaks.DEFAULT_ROOT / "stats")
    ap.add_argument("--all", action="store_true", help="report every run in the log, not just the last")
    ap.add_argument("--session", nargs="?", const="", metavar="SCENARIO",
                    help="report every run, then write one page for SCENARIO's runs (default: the "
                         "scenario of the last run) from --since on")
    ap.add_argument("--since", help="session start date YYYY-MM-DD (default: the day of the last run)")
    ap.add_argument("--min-run", type=float, default=spans.MIN_RUN_S,
                    help="seconds; shorter stretches without a stats file are skipped")
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

    wall0 = log.start_wall_us / 1e6
    if args.csv:
        st = kovaaks.parse_stats(args.csv)
        found = [spans.Span(st.start.timestamp() - wall0, st.end.timestamp() - wall0, st)]
    else:
        found = spans.find_runs(log, args.stats_dir, args.min_run)
        if not (args.all or args.session is not None):
            # newest run, preferring one the game finished and saved
            saved = [s for s in found if s.stats]
            found = (saved or found)[-1:]
    if not found:
        found = [spans.Span(float(log.t[0]), float(log.t[-1]), None)]
    if len(found) == 1 and not found[0].stats:
        name_of = lambda s: spot.stem
    else:
        name_of = lambda s: s.name(log)

    hist_path = args.out / "history.jsonl"
    last = None
    for sp in found:
        name, st = name_of(sp), sp.stats
        run = log.slice(sp.t0, sp.t1)
        if len(run.t) < 100 or len(run.tgt_frame) < 100:
            print(f"{name}: too little data, skipped", file=sys.stderr)
            continue
        mode = pick_mode(run, st) if args.mode == "auto" else args.mode
        try:
            result = tracking.analyze(run) if mode == "tracking" else clicking.analyze(run, mode)
        except ValueError as e:
            if mode == "tracking" or args.mode != "auto":
                print(f"{name}: {e}", file=sys.stderr)
                continue
            mode, result = "tracking", tracking.analyze(run)   # no kills after all
        tens = tension.analyze(run, mode, tension.mm_per_deg(st.fields if st else None))
        rec = history.make_record(name, mode, st, result, tens)
        noise, context = None, []
        if st:
            history.save(hist_path, rec)
            prior = [s for s in history.score_history(args.stats_dir, st.scenario) if s.end < st.end]
            noise = history.noise_rows(rec, history.load(hist_path),
                                       [history._num(s.fields["Score"]) for s in prior])
            context = history.score_context(st, args.stats_dir)
            last = rec
        path = report.write(args.out / name, name, result, st, run, args.worst, tens, noise, context)
        print(path)

    if args.session is not None:
        hist = history.load(hist_path)
        scenario = args.session or (last["scenario"] if last else None)
        if not scenario:
            print("no saved run to build a session from", file=sys.stderr)
            return 1
        mine = [r for r in hist if r["scenario"] == scenario]
        if not mine:
            print(f"no analyzed runs of {scenario!r} in {hist_path}", file=sys.stderr)
            return 1
        since = date.fromisoformat(args.since) if args.since else datetime.fromisoformat(mine[-1]["end"]).date()
        mine = [r for r in mine if datetime.fromisoformat(r["end"]).date() >= since]
        modes = {r["mode"] for r in mine}
        if len(modes) > 1:  # keep the mode most runs were analyzed in
            keep = max(modes, key=lambda m: sum(r["mode"] == m for r in mine))
            mine = [r for r in mine if r["mode"] == keep]
        out = args.out / f"{scenario} - session {since:%Y.%m.%d}"
        print(session.write(out, scenario, mine, args.stats_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
