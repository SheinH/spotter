"""Live sensitivity vs accuracy, for runs played with KovaaK's sens randomizer (v3 logs).

    uv run python -m spotter_analyze.sensplot                      # newest log, every saved run in it
    uv run python -m spotter_analyze.sensplot --scenario "Whisphere Viscose" --logs 3

Each run is cut into 1 s windows. A window's x is its median live cm/360, its y the share of
the window spent on target. Three panels:
  A  on target % vs live cm/360 (log axis), one dot per second, binned mean with 95% CI
  B  the same, against how fast the randomizer was moving the sens (adaptation cost)
  C  timeline of the newest run: live cm/360 with on-target % per second behind it
Target speed varies second to second and is a confound, so the binned table also gives
on-target % adjusted for it (residual from a fit on target speed, added back to the mean).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

from . import geometry, kovaaks, sens, spans
from .format import Run, read_spot
from .plots import plt

REPO = Path(__file__).resolve().parents[2]
WIN_S = 1.0
N_BINS = 6
BOOT = 1000


def windows(run: Run, fields: dict | None) -> dict | None:
    """Per 1 s window: t, live cm/360, on-target share, target speed, sens rate (ln/s)."""
    live = sens.cm360(run, fields)
    if live is None:
        return None
    v = geometry.target_view(run)
    near = geometry.nearest(run, v)
    has = near >= 0
    on = np.zeros(len(run.t), bool)
    on[has] = v.miss[near[has]] <= v.arad[near[has]]
    # target angular speed per frame (deg/s), from the nearest target's angle to the crosshair
    ex = np.full(len(run.t), np.nan)
    ey = np.full(len(run.t), np.nan)
    ex[has], ey[has] = v.ex[near[has]], v.ey[near[has]]
    yaw = np.unwrap(run.yaw, period=360)
    dt = np.gradient(run.t)
    tx, ty = np.gradient(ex + yaw) / dt, np.gradient(ey + run.pitch) / dt   # target direction in world angles
    tspeed = np.hypot(tx, ty)
    w = np.clip(np.diff(run.t, append=run.t[-1]), 0, 0.02)

    edges = np.arange(run.t[0], run.t[-1], WIN_S)
    out = {k: [] for k in ("t", "cm360", "on", "tspeed", "rate")}
    idx = np.searchsorted(run.t, edges)
    for a, b in zip(idx[:-1], idx[1:]):
        sel = slice(a, b)
        ww = w[sel] * has[sel]
        if ww.sum() < 0.5 * WIN_S:          # mostly no target: respawn, start, end
            continue
        out["t"].append(run.t[a] - run.t[0])
        out["cm360"].append(float(np.median(live[sel])))
        out["on"].append(float(np.sum(ww * on[sel]) / ww.sum()) * 100)
        out["tspeed"].append(float(np.nanmedian(tspeed[sel][has[sel]])))
        lt = np.log(live[sel])
        tt = run.t[sel]
        out["rate"].append(float(np.polyfit(tt - tt[0], lt, 1)[0]) if np.ptp(lt) > 0 else 0.0)
    return {k: np.array(x) for k, x in out.items()}


def binned(x: np.ndarray, y: np.ndarray, n_bins: int, rng) -> list[tuple]:
    """Quantile bins of x: (lo, hi, n, mean y, ci lo, ci hi)."""
    q = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    rows = []
    for i, (lo, hi) in enumerate(zip(q[:-1], q[1:])):
        sel = (x >= lo) & ((x < hi) if i < len(q) - 2 else (x <= hi))   # each window in one bin
        ys = y[sel]
        if len(ys) < 3:
            continue
        boots = rng.choice(ys, (BOOT, len(ys))).mean(1)
        rows.append((lo, hi, len(ys), ys.mean(), *np.percentile(boots, [2.5, 97.5])))
    return rows


def collect(logs: list[Path], scenario: str | None, stats_dir: Path):
    runs = []
    for p in logs:
        log = read_spot(p)
        if not np.isfinite(log.yaw_scale).any():
            print(f"{p.name}: no live sensitivity in this log. It was recorded by the old hook; the new "
                  "one only loads into a freshly started game. Close KovaaK's, start it again, run "
                  "build/bin/Release/spotter-launch.exe --wait 120, then play.", file=sys.stderr)
            continue
        for sp in spans.find_runs(log, stats_dir):
            if not sp.stats or (scenario and sp.stats.scenario != scenario):
                continue
            run = log.slice(sp.t0, sp.t1)
            wd = windows(run, sp.stats.fields)
            if wd is None or len(wd["t"]) < 10:
                continue
            runs.append((sp.stats, wd, sens.configured_cm360(sp.stats.fields)))
    return runs


def plot(runs: list, out: Path, title: str) -> list[str]:
    rng = np.random.default_rng(0)
    allw = {k: np.concatenate([r[1][k] for r in runs]) for k in runs[0][1]}
    x, y = allw["cm360"], allw["on"]
    # adjust for target speed: residual of a linear fit on target speed, added to the mean
    ok = np.isfinite(allw["tspeed"])
    fit = np.polyfit(allw["tspeed"][ok], y[ok], 1) if ok.sum() > 10 else np.array([0.0, y.mean()])
    yadj = np.where(ok, y - np.polyval(fit, np.nan_to_num(allw["tspeed"])) + y.mean(), y)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax, bx, cx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    cmap = plt.get_cmap("viridis")
    for i, (st, wd, base) in enumerate(runs):
        c = cmap(i / max(1, len(runs) - 1))
        ax.scatter(wd["cm360"], wd["on"], s=10, alpha=0.35, color=c, label=f"{st.end:%H:%M} ({st.fields.get('Score')})")
    rows = binned(x, y, N_BINS, rng)
    rows_adj = binned(x, yadj, N_BINS, rng)
    mids = [np.sqrt(r[0] * r[1]) for r in rows]
    ax.errorbar(mids, [r[3] for r in rows], yerr=[[r[3] - r[4] for r in rows], [r[5] - r[3] for r in rows]],
                color="k", lw=2, capsize=4, marker="o", label="binned mean, 95% CI")
    bases = {r[2] for r in runs if r[2]}
    for b in bases:
        ax.axvline(b, color="crimson", ls="--", lw=1)
        ax.text(b, 99, f" set {b:g}", color="crimson", va="top", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([t for t in (10, 15, 20, 25, 30, 35, 40, 50, 60, 80) if x.min() * 0.9 <= t <= x.max() * 1.1])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(NullFormatter())
    ax.set_xlabel("live sensitivity, cm/360 (log; left = faster)")
    ax.set_ylabel("on target, % of each second")
    ax.set_ylim(0, 100)
    ax.set_title("A. Accuracy vs live sensitivity")
    ax.legend(fontsize=7, loc="lower right")

    rate = allw["rate"] * 100   # % change in cm/360 per second
    bx.scatter(rate, y, s=10, alpha=0.3, c=np.log(x), cmap="coolwarm_r")
    rr = binned(rate, y, N_BINS, rng)
    bx.errorbar([(r[0] + r[1]) / 2 for r in rr], [r[3] for r in rr],
                yerr=[[r[3] - r[4] for r in rr], [r[5] - r[3] for r in rr]], color="k", lw=2, capsize=4, marker="o")
    bx.axvline(0, color="grey", lw=0.8)
    bx.set_xlabel("sens drift, % per second (- = getting faster, + = getting slower)")
    bx.set_ylabel("on target, %")
    bx.set_ylim(0, 100)
    bx.set_title("B. Does the change itself cost you? (dot colour: red = fast sens)")

    st, wd, base = runs[-1]
    cx.bar(wd["t"], wd["on"], width=WIN_S * 0.9, color="lightsteelblue", label="on target % (left)")
    cx.set_ylabel("on target, %")
    cx.set_ylim(0, 100)
    cx2 = cx.twinx()
    cx2.plot(wd["t"], wd["cm360"], color="crimson", lw=2, label="live cm/360 (right)")
    if base:
        cx2.axhline(base, color="crimson", ls="--", lw=1)
    cx2.set_ylabel("live cm/360")
    cx.set_xlabel("seconds into the run")
    cx.set_title(f"C. Newest run {st.end:%H:%M}, score {st.fields.get('Score')}")
    cx.legend(loc="upper left", fontsize=8)
    cx2.legend(loc="upper right", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)

    slope = np.polyfit(np.log(x), y, 1)[0] * np.log(1.25)
    lines = [f"# {title}", "", f"{len(runs)} runs, {len(x)} one-second windows", "",
             f"- trend: {slope:+.1f} on-target points per 25% slower sens (log fit; noisy, see CI)",
             f"- target speed explains part of the spread: {fit[0]:+.2f} points per deg/s; "
             "the adjusted column removes it", "",
             "```tsv", "cm360_lo\tcm360_hi\tseconds\ton_pct\tci95\ton_pct_adj\tci95_adj"]
    for r, a in zip(rows, rows_adj):
        lines.append(f"{r[0]:.1f}\t{r[1]:.1f}\t{r[2]}\t{r[3]:.1f}\t{r[4]:.1f}-{r[5]:.1f}\t{a[3]:.1f}\t{a[4]:.1f}-{a[5]:.1f}")
    lines += ["```", "", "```tsv", "drift_pct_s_lo\tdrift_pct_s_hi\tseconds\ton_pct\tci95"]
    for r in rr:
        lines.append(f"{r[0]:+.1f}\t{r[1]:+.1f}\t{r[2]}\t{r[3]:.1f}\t{r[4]:.1f}-{r[5]:.1f}")
    lines += ["```", "", f"![sens vs accuracy]({out.name})"]
    r1 = lambda v: round(float(v), 2)
    data = {
        "title": title, "base": sorted(bases), "trend_per_25pct": r1(slope),
        "runs": [{"label": f"{st.end:%H:%M}", "score": st.fields.get("Score"),
                  "points": [[r1(a), r1(b), r1(t), r1(d * 100)] for a, b, t, d in
                             zip(wd["cm360"], wd["on"], wd["t"], wd["rate"])]} for st, wd, _ in runs],
        "bins": [[r1(r[0]), r1(r[1]), int(r[2]), r1(r[3]), r1(r[4]), r1(r[5]), r1(a[3])] for r, a in zip(rows, rows_adj)],
        "drift_bins": [[r1(r[0]), r1(r[1]), int(r[2]), r1(r[3]), r1(r[4]), r1(r[5])] for r in rr],
    }
    out.with_suffix(".json").write_text(json.dumps(data), encoding="utf-8")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spot", nargs="*", type=Path, help=".spot logs (default: the newest --logs logs)")
    ap.add_argument("--logs", type=int, default=1, help="how many of the newest logs to scan")
    ap.add_argument("--scenario", help="only this scenario (default: the newest run's)")
    ap.add_argument("--stats-dir", type=Path, default=kovaaks.DEFAULT_ROOT / "stats")
    ap.add_argument("--out", type=Path, default=REPO / "reports")
    a = ap.parse_args(argv)
    logs = a.spot or sorted((REPO / "runs").glob("spotter-*.spot"), key=lambda p: p.stat().st_mtime)[-a.logs:]
    runs = collect(logs, a.scenario, a.stats_dir)
    if a.scenario is None and runs:
        scen = runs[-1][0].scenario
        runs = [r for r in runs if r[0].scenario == scen]
    if not runs:
        print("no saved runs with live sensitivity in those logs", file=sys.stderr)
        return 1
    scen = runs[-1][0].scenario
    d = a.out / f"{scen} - sens vs accuracy {datetime.now():%Y.%m.%d}"
    d.mkdir(parents=True, exist_ok=True)
    lines = plot(runs, d / "sens_vs_accuracy.png", f"{scen}: live sensitivity vs accuracy")
    (d / "sens.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(d / "sens.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
