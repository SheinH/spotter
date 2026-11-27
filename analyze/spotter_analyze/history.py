"""Run history and the noise layer.

Every analyzed run leaves one line in reports/history.jsonl: scenario, settings, score, the
run's headline metrics and its fixed-edge bins. A report then compares each metric against
the same scenario's other runs, so a 2-point change that is ordinary run-to-run spread is
labelled as noise instead of read as a change in skill.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import kovaaks

RECENT = 20          # compare against at most this many other runs, the most recent ones
MIN_RUNS = 3         # below this there is no spread to speak of

# label, whether higher is better (None = neither)
LABELS = {
    "score": ("score", True),
    "on_pct": ("on target %", True),
    "centred_pct": ("centred on target %", True),
    "far_behind_pct": ("far behind (>1 radius) %", False),
    "fast_on_pct": ("on target at 85+ deg/s %", True),
    "lead_r": ("lead (+) / lag (-), radii", None),
    "react_ms": ("reaction ms", False),
    "recov_ms": ("back on target after reversal ms", False),
    "off_per_min": ("off-target episodes per min", None),
    "off_med_ms": ("median off-target stretch ms", False),
    "off_p90_ms": ("long off-target stretches (90th pct) ms", False),
    "hold_jitter": ("on-target jitter (not yet validated)", None),
    "activity": ("hand activity 6-13 Hz (context)", None),
    "kills_per_min": ("kills per min", True),
    "ttk_ms": ("time per kill ms", False),
    "acq_ms": ("time to first on target ms", False),
    "confirm_ms": ("confirm before shot ms", None),
    "peak_dps": ("flick peak speed deg/s", None),
    "land_r": ("first flick lands, radii (- short)", None),
    "short_pct": ("first flick short %", False),
    "over_pct": ("overflick (past far edge) %", False),
    "slip_pct": ("kills with a slip off target %", False),
    "corr": ("corrections per kill", False),
    "miss_per_kill": ("misses per kill", False),
    "flick_gain": ("flick gain (reach / distance, 1 = exact)", None),
    "flick_spread_pct": ("flick scatter, % of distance", False),
    "off_after_pct": ("time off target after first contact %", False),
    "brake_ratio": ("flick braking: peak stop / peak push (1 = symmetric, >1 = hard stop)", None),
    "wobble_r": ("post-flick wobble, radii (first 150 ms; 0 = clean)", False),
    "flick_ms": ("flick time ms (hand moving until first movement ends)", False),
    "flick_on_pct": ("flick lands on the target %", True),
    "flick_err_pct": ("flick leaves, % of distance", False),
    "micro_ms": ("micro-adjust time ms (flick end until settled)", False),
    "micro_saved_s": ("micro slower than own best for same leftover, s", False),
    "flick_cost_s": ("micro time caused by the flick's leftover, s", False),
    "micro_corr": ("corrections per micro-adjust", False),
    "nearest_pct": ("went to the nearest target %", None),
    "extra_deg": ("extra flick distance vs nearest target, deg", None),
    "shift_on_pct": ("on target % with delay removed (ceiling)", True),
    "shift_gain_pts": ("on-target points lost to delay", False),
}


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def make_record(name: str, mode: str, stats: kovaaks.Stats | None, result: dict, tension: dict | None) -> dict:
    f = stats.fields if stats else {}
    metrics = dict(result.get("metrics", {}))
    if tension:
        for key in ("hold_jitter", "activity"):
            if _num(tension.get(key)) is not None:
                metrics[key] = tension[key]
    return {
        "name": name,
        "scenario": stats.scenario if stats else None,
        "end": stats.end.isoformat() if stats else None,
        "mode": mode,
        "score": _num(f.get("Score")),
        "sens": _num(f.get("Horiz Sens")), "sens_scale": f.get("Sens Scale"),
        # live sensitivity from v3 logs (sens randomizer); None for older logs
        "sens_live": _num((result.get("sens") or {}).get("cm360_median")),
        "sens_randomized": (result.get("sens") or {}).get("randomized"),
        "dpi": _num(f.get("DPI")), "fov": _num(f.get("FOV")),
        "metrics": {k: _num(v) for k, v in metrics.items()},
        "bins": result.get("bins", {}),
    }


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path: Path, rec: dict) -> None:
    """Add or replace (by name) one record. Runs without a stats file are not kept: they have
    no scenario name to compare against."""
    if not rec["scenario"]:
        return
    recs = [r for r in load(path) if r["name"] != rec["name"]] + [rec]
    recs.sort(key=lambda r: r["end"] or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def others(rec: dict, hist: list[dict]) -> list[dict]:
    """The same scenario's other runs in the same mode, most recent RECENT."""
    same = [r for r in hist if r["scenario"] == rec["scenario"] and r["mode"] == rec["mode"]
            and r["name"] != rec["name"]]
    return same[-RECENT:]


def spread(values: list[float]) -> tuple[float, float, int]:
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan"), len(v)
    return float(v.mean()), float(v.std(ddof=1)), len(v)


def verdict(x: float | None, mean: float, sd: float, n: int, better: bool | None) -> str:
    """Is one run's value outside the scatter of n others? A new single run differs from the
    mean of n by more than 2 SD*sqrt(1+1/n) about 5% of the time by chance alone."""
    if x is None or n < MIN_RUNS or not sd > 0:
        return "-"
    z = (x - mean) / (sd * math.sqrt(1 + 1 / n))
    if abs(z) < 2:
        return "within noise"
    word = "higher" if z > 0 else "lower"
    if better is not None:
        word += " (better)" if (z > 0) == better else " (worse)"
    return f"{word}, {z:+.1f} SD"


def noise_rows(rec: dict, hist: list[dict], score_prev: list[float] | None = None) -> tuple[list[str], list[list], int]:
    """Table comparing this run against the scenario's other runs in the history. Score can
    come from the game's own stats files instead (`score_prev`), which go back much further."""
    prev = others(rec, hist)
    rows = []
    for key, x in [("score", rec["score"])] + list(rec["metrics"].items()):
        if key not in LABELS:
            continue
        if key == "score" and score_prev:
            vals = score_prev[-RECENT:]
        else:
            vals = [r["score"] if key == "score" else r["metrics"].get(key) for r in prev]
        mean, sd, n = spread(vals)
        label, better = LABELS[key]
        rows.append([label, float("nan") if x is None else x, mean, sd, n, verdict(x, mean, sd, n, better),
                     2 * math.sqrt(2) * sd if n >= MIN_RUNS else float("nan")])
    return ["metric", "this_run", "usual", "sd", "runs", "verdict", "noise_2runs"], rows, len(prev)


def score_history(stats_dir: Path, scenario: str) -> list[kovaaks.Stats]:
    """Every stats file KovaaK's has for this scenario, oldest first."""
    out = []
    for p in stats_dir.glob(f"{_glob_escape(scenario)} - * Stats.csv"):
        try:
            st = kovaaks.parse_stats(p)
        except (ValueError, IndexError, OSError):
            continue
        if st.scenario == scenario and _num(st.fields.get("Score")) is not None:
            out.append(st)
    return sorted(out, key=lambda s: s.end)


def _glob_escape(s: str) -> str:
    return "".join(f"[{c}]" if c in "[]*?" else c for c in s)


def score_context(stats: kovaaks.Stats, stats_dir: Path) -> list[str]:
    """Where this score sits among every run of the scenario KovaaK's has a file for."""
    allruns = score_history(stats_dir, stats.scenario)
    me = _num(stats.fields.get("Score"))
    prior = [s for s in allruns if s.end < stats.end]
    if me is None or not prior:
        return []
    scores = np.array([_num(s.fields["Score"]) for s in prior])
    best = prior[int(np.argmax(scores))]
    recent = scores[-RECENT:]
    pct = 100 * float((scores < me).mean())
    out = [f"score {me:.0f}: beats {pct:.0f}% of your {len(scores)} earlier runs; previous best "
           f"{scores.max():.0f} ({best.end:%Y-%m-%d}); last {len(recent)} runs {np.median(recent):.0f} median, "
           f"{recent.std(ddof=1) if len(recent) > 1 else 0:.0f} SD"]
    return out
