"""Turn an analysis result into report.md plus two PNGs."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import plots
from .kovaaks import Stats


def cell(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if math.isnan(x):
        return "-"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def tsv(columns: list[str], rows: list[list]) -> str:
    return "\n".join(["\t".join(columns)] + ["\t".join(cell(c) for c in row) for row in rows])


def header(name: str, result: dict, stats: Stats | None, run) -> list[str]:
    lines = [f"# Spotter: {stats.scenario if stats else name} ({result['mode']})", ""]
    if stats:
        f = stats.fields
        lines.append(
            f"score {f.get('Score', '?')} | {stats.end:%Y-%m-%d %H:%M} | "
            f"sens {f.get('Horiz Sens', '?')} {f.get('Sens Scale', '')} @ {f.get('DPI', '?')} dpi | "
            f"FOV {f.get('FOV', '?')} ({f.get('FOVScale', '?')}) | {f.get('Resolution', '?')} | "
            f"avg fps {float(f.get('Avg FPS', 'nan')):.0f} | game {f.get('Game Version', '?')[:8]}")
        if int(f.get('Pause Count', '0') or 0):
            lines.append(f"paused {f['Pause Count']}x for {f.get('Pause Duration', '?')}s during the run; "
                         "the pause is inside the analyzed window")
    else:
        lines.append("no KovaaK's stats file for this stretch (restarted before the end, free play, or not "
                     "KovaaK's): run boundaries come from the log, scenario and settings unknown")
    if stats:
        want = (stats.end - stats.start).total_seconds()
        got = run.t[-1] - run.t[0]
        if want > 0 and got < 0.95 * want:
            lines.append(f"the log only covers {got:.0f}s of this {want:.0f}s run (hook started or stopped mid-run)")
    dts = np.diff(run.t)
    lines.append(f"log: {run.t[-1] - run.t[0]:.1f}s, {len(run.t)} frames, median {1 / np.median(dts):.0f} Hz")
    return lines


def crosscheck(result: dict, stats: Stats | None, wall0: float = 0.0) -> list[str]:
    """Compare what the log shows against what the game itself recorded."""
    if not stats:
        return []
    out = []
    if result["mode"] == "tracking":
        if stats.shots:
            out.append(f"on-target from log {result['on_pct']:.1f}% vs game hits/shots "
                       f"{100 * stats.hits / stats.shots:.1f}% (game only counts while firing)")
    else:
        out.append(f"kills: log {result['kills']} vs game {len(stats.kill_times) or stats.fields.get('Kills', '?')}; "
                   f"shots: log {result['shots']} vs game {stats.shots}")
        if stats.kill_times and len(result["kill_times"]):
            game = np.array([k.timestamp() for k in stats.kill_times])
            mine = result["kill_times"] + wall0
            delta = np.array([mine[np.argmin(np.abs(mine - g))] - g for g in game]) * 1000
            out.append(f"{int((np.abs(delta) < 15).sum())} of {len(game)} game kills have a log kill within 15 ms "
                       f"(median offset {np.median(delta):+.1f} ms)")
    return out


def write(out_dir: Path, name: str, result: dict, stats: Stats | None, run, worst: int = 0,
          tension: dict | None = None, noise: tuple | None = None, context: list[str] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = header(name, result, stats, run)
    lines += ["", "## Summary", ""] + [f"- {s}" for s in (context or []) + result["summary"]]
    if noise:
        columns, rows, n = noise
        lines += ["", "## Against your other runs of this scenario (noise layer)", ""]
        if n < 3:
            lines.append(f"Only {n} other analyzed run(s) of this scenario so far; the spread needs at least 3. "
                         "Until then treat every difference of a few points as noise.")
        lines.append("usual/sd = mean and spread of your last runs (score: from KovaaK's own files). "
                     "noise_2runs = how far two single runs must differ before it means anything. "
                     "\"within noise\" = this run is an ordinary draw from your usual spread; don't read into it.")
        if n < 3:
            rows = [r for r in rows if r[4] >= 3]
        if rows:
            lines += ["", "```tsv", tsv(columns, rows), "```"]
    checks = crosscheck(result, stats, run.start_wall_us / 1e6)
    if checks:
        lines += ["", "## Cross-check against the game's own files", ""] + [f"- {c}" for c in checks]
    for title, columns, rows in result["splits"]:
        lines += ["", f"## {title}", "", "```tsv", tsv(columns, rows), "```"]
    if tension and tension["summary"]:
        lines += ["", "## Tension (jitter and braking)", ""] + [f"- {s}" for s in tension["summary"]]
        for title, columns, rows in tension["splits"]:
            lines += ["", f"### {title}", "", "```tsv", tsv(columns, rows), "```"]
    lines += ["", "## Events", "", result["legend"], "", "```tsv", tsv(result["columns"], result["rows"]), "```"]
    if worst:
        lines += ["", f"## Worst {worst} events, raw traces", ""]
        for title, columns, rows in result["worst"](worst):
            lines += [f"### {title}", "", "```tsv", tsv(columns, rows), "```", ""]
    lines += ["", "Images: error_scatter.png, velocity_profiles.png"]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    plots.error_scatter(result, out_dir / "error_scatter.png")
    plots.velocity_profiles(result, out_dir / "velocity_profiles.png")
    return path
