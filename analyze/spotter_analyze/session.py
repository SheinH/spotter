"""One page for a set of runs of one scenario: per-run table, pooled numbers with their
run-to-run spread, the trend across the session, what moves with score, and settings."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as sps

from . import history, tracking
from .report import tsv

SHORT = {  # compact column names for the per-run table
    "on_pct": "on%", "centred_pct": "centred%", "far_behind_pct": "farbehind%", "fast_on_pct": "fast_on%",
    "lead_r": "lead_r", "react_ms": "react", "recov_ms": "recov", "off_per_min": "offs/min",
    "off_med_ms": "off_med", "off_p90_ms": "off_p90", "hold_jitter": "hold_jit", "activity": "activity",
    "kills_per_min": "kpm", "ttk_ms": "ttk", "acq_ms": "acq", "confirm_ms": "confirm", "peak_dps": "peak",
    "land_r": "land_r", "short_pct": "short%", "over_pct": "over%", "slip_pct": "slip%", "corr": "corr", "miss_per_kill": "miss/k",
    "flick_gain": "fgain", "flick_spread_pct": "fscatter%", "off_after_pct": "offafter%",
    "brake_ratio": "brake", "wobble_r": "wobble_r",
    "flick_ms": "flick", "flick_on_pct": "flick_on%", "flick_err_pct": "flick_err%", "micro_ms": "micro",
    "micro_saved_s": "micro_slack_s", "flick_cost_s": "flick_cost_s", "micro_corr": "micro_corr",
    "nearest_pct": "nearest%", "extra_deg": "extra_deg",
    "shift_on_pct": "shift_on%", "shift_gain_pts": "lag_pts",
}


def _vals(recs, key):
    return [r["score"] if key == "score" else r["metrics"].get(key) for r in recs]


def _arr(vals):
    return np.array([np.nan if v is None else v for v in vals], dtype=float)


def write(out_dir: Path, scenario: str, recs: list[dict], stats_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = sorted(recs, key=lambda r: r["end"])
    n = len(recs)
    t0, t1 = (datetime.fromisoformat(recs[i]["end"]) for i in (0, -1))
    keys = [k for k in history.LABELS if k != "score" and any(r["metrics"].get(k) is not None for r in recs)]
    lines = [f"# Spotter session: {scenario} ({recs[-1]['mode']})", "",
             f"{n} runs, {t0:%Y-%m-%d %H:%M} to {t1:%H:%M}" if t0.date() == t1.date()
             else f"{n} runs, {t0:%Y-%m-%d %H:%M} to {t1:%Y-%m-%d %H:%M}"]

    # --- the game's own score history for context ---
    allruns = history.score_history(stats_dir, scenario)
    before = [history._num(s.fields["Score"]) for s in allruns if s.end < t0]
    scores = _arr(_vals(recs, "score"))
    if before:
        b = np.array(before)
        lines.append(f"before this session: {len(b)} runs, best {b.max():.0f}, last {min(len(b), 20)} median "
                     f"{np.median(b[-20:]):.0f}; this session best {np.nanmax(scores):.0f} "
                     f"(beats {100 * (b < np.nanmax(scores)).mean():.0f}% of earlier runs), median {np.nanmedian(scores):.0f}")

    # --- per run ---
    lines += ["", "## Runs", "", "Columns: " + "; ".join(f"{SHORT[k]} = {history.LABELS[k][0]}" for k in keys), "",
              "```tsv", tsv(["#", "time", "sens", "fov", "score"] + [SHORT[k] for k in keys],
                            [[i + 1, f"{datetime.fromisoformat(r['end']):%H:%M}", r["sens"] or float("nan"),
                              r["fov"] or float("nan"), r["score"] if r["score"] is not None else float("nan")]
                             + [_nan(r["metrics"].get(k)) for k in keys] for i, r in enumerate(recs)]), "```"]

    # --- pooled with spread ---
    pooled = []
    for k in ["score"] + keys:
        v = _arr(_vals(recs, k))
        v = v[~np.isnan(v)]
        if len(v) >= 2:
            sd = float(v.std(ddof=1))
            pooled.append([history.LABELS[k][0], float(v.mean()), sd, float(v.min()), float(v.max()),
                           2 * math.sqrt(2) * sd, 2 * math.sqrt(2) * sd / math.sqrt(3)])
    lines += ["", "## Pooled across runs", "",
              "noise_1v1 = how far two single runs must differ before it means anything (2.8 SD); "
              "noise_3v3 = the same for averages of 3 runs each side.", "",
              "```tsv", tsv(["metric", "mean", "sd", "min", "max", "noise_1v1", "noise_3v3"], pooled), "```"]

    # --- trend: first half against second half ---
    if n >= 4:
        h = n // 2
        rows = []
        for k in ["score"] + keys:
            v = _arr(_vals(recs, k))
            a, b = v[:h][~np.isnan(v[:h])], v[h:][~np.isnan(v[h:])]
            if len(a) < 2 or len(b) < 2:
                continue
            sd = float(np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)))
            d = float(b.mean() - a.mean())
            need = 2 * sd * math.sqrt(1 / len(a) + 1 / len(b))
            label, better = history.LABELS[k]
            if sd > 0 and abs(d) > need:
                word = "real change" + ("" if better is None else (" (better)" if (d > 0) == better else " (worse)"))
            else:
                word = "within noise"
            rows.append([label, float(a.mean()), float(b.mean()), d, need, word])
        lines += ["", f"## Trend: first {h} runs against last {n - h}", "",
                  f"Testing {len(rows)} metrics at once, about one in 20 will read \"real change\" by chance; "
                  "trust one only if a neighbouring metric agrees or it repeats next session.", "",
                  "```tsv", tsv(["metric", "first", "last", "change", "needed", "verdict"], rows), "```"]

    # --- what moves with score ---
    if n >= 5 and np.isfinite(scores).sum() >= 5:
        rows = []
        for k in keys:
            v = _arr(_vals(recs, k))
            m = ~np.isnan(v) & ~np.isnan(scores)
            if m.sum() >= 5 and np.ptp(v[m]) > 0:
                r, p = sps.spearmanr(v[m], scores[m])
                rows.append([history.LABELS[k][0], float(r), float(p), int(m.sum())])
        rows.sort(key=lambda x: -abs(x[1]))
        lines += ["", "## What moves with score (rank correlation across runs)", "",
                  f"With {n} runs, only p below about 0.05 is worth reading; the rest is chance. "
                  "On a tracking scenario on-target % is nearly the score itself, so its r says nothing new.", "",
                  "```tsv", tsv(["metric", "r", "p", "runs"], rows), "```"]

    # --- settings ---
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        groups.setdefault((r["sens"], r["fov"]), []).append(r)
    rows = [[f"{s} {recs[0].get('sens_scale') or ''}".strip(), fv, len(g), float(np.nanmedian(_arr(_vals(g, "score")))),
             float(np.nanmean(_arr(_vals(g, "on_pct")))) if any(x.get("metrics", {}).get("on_pct") is not None for x in g)
             else float("nan")] for (s, fv), g in sorted(groups.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1] or 0))]
    lines += ["", "## By settings, this session", "", "Fewer than 3 runs at a setting says nothing about it.", "",
              "```tsv", tsv(["sens", "fov", "runs", "median_score", "on_pct"], rows), "```"]
    if allruns:
        by: dict[str, list[float]] = {}
        for s in allruns:
            by.setdefault(f"{s.fields.get('Horiz Sens', '?')} {s.fields.get('Sens Scale', '')}".strip(), []).append(
                history._num(s.fields["Score"]))
        rows = [[k, len(v), float(np.median(v)), float(np.max(v)), float(np.median(v[-20:]))]
                for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))[:8]]
        lines += ["", "### By sensitivity, every run KovaaK's has a file for", "",
                  "```tsv", tsv(["sens", "runs", "median", "best", "last20_median"], rows), "```"]

    # --- pooled bins ---
    for key in ("speed", "position", "since_rev"):
        many = [r["bins"][key] for r in recs if r.get("bins", {}).get(key)]
        if len(many) >= 2:
            title, head, rows = tracking.bin_split(key, tracking.pool_bins(many))
            # per-run spread of on_pct in each bin, so a band difference can be judged
            per = np.array([[100 * x[2] / x[1] if x[1] >= 0.25 else np.nan for x in rr] for rr in many])
            sds = {lab: float(np.nanstd(per[:, j], ddof=1)) if np.isfinite(per[:, j]).sum() >= 2 else float("nan")
                   for j, lab in enumerate(r[0] for r in many[0])}
            lines += ["", f"## {title}, pooled", "",
                      "```tsv", tsv(head + ["on_pct_sd"], [row + [sds.get(row[0], float("nan"))] for row in rows]), "```"]

    path = out_dir / "session.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _nan(x):
    return float("nan") if x is None else x
