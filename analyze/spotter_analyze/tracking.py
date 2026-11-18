"""Tracking analysis: one row per segment between target direction changes."""
from __future__ import annotations

import math

import numpy as np

from .format import Run
from .geometry import nearest, octant, runs, smooth, target_view

FS = 250.0          # analysis grid, Hz
MOVE = 3.0          # deg/s; below this the target counts as stationary
MAX_SEG = 1.5       # s; longer stretches without a reversal are split so the table keeps a time axis
REV_WIN = 0.15      # s; a reversal compares target velocity this long before and after

# Fixed bin edges (identical in every run so they pool across runs and sessions)
SPEED_EDGES = [0, 25, 45, 65, 85, 105, np.inf]                       # target deg/s
SPEED_LABELS = ["0-25", "25-45", "45-65", "65-85", "85-105", "105+"]
ZONE_EDGES = [-np.inf, -1.0, -0.3, 0.3, 1.0, np.inf]                 # lead_r, target radii
ZONE_LABELS = ["far behind", "behind", "centred", "ahead", "past"]
SINCE_EDGES = [0, 100, 200, 300, 500, 800, np.inf]                   # ms since direction change
SINCE_LABELS = ["0-100", "100-200", "200-300", "300-500", "500-800", "800+"]
BIN_TITLES = {
    "speed": ("By target speed (deg/s, fixed bands)", "tgt_dps"),
    "position": ("Where the crosshair sits along the target's travel (radii: far behind <-1, "
                 "behind -1..-0.3, centred +-0.3, ahead 0.3..1, past >1)", "zone"),
    "since_rev": ("By time since the target's last direction change (ms)", "since_ms"),
}

COLUMNS = ["i", "t_s", "dur_ms", "rev", "dir", "tgt_dps", "on_pct", "lead_r", "cross_r",
           "react_ms", "gain", "revlag_r", "recov_ms"]
LEGEND = (
    "rev=1 if the segment starts at a target direction change; dir=target's screen direction; "
    "tgt_dps=target angular speed; lead_r=crosshair ahead(+)/behind(-) of target centre along its "
    "motion, in target radii; cross_r=mean distance off the motion line, radii; react_ms=time until "
    "the crosshair is moving the new way; gain=crosshair speed / target speed after reacting; "
    "revlag_r=peak distance behind after the reversal, radii; recov_ms=reversal until back on target."
)


def _first_sustained(mask: np.ndarray, n: int) -> int | None:
    if len(mask) < n:
        return None
    hit = np.convolve(mask.astype(float), np.ones(n), mode="valid") >= n
    return int(np.argmax(hit)) if hit.any() else None


def _pct(mask: np.ndarray) -> float:
    return 100.0 * float(mask.mean()) if len(mask) else float("nan")


def analyze(run: Run) -> dict:
    v = target_view(run)
    ni = nearest(run, v)
    has = ni >= 0
    if has.sum() < 100:
        raise ValueError("log has too few frames with a target to analyze")
    rows, th = ni[has], run.t[has]

    tu = np.arange(run.t[0], run.t[-1], 1.0 / FS)
    n, dt = len(tu), 1.0 / FS
    idx = np.clip(np.searchsorted(run.t, tu), 0, len(run.t) - 1)
    ok = has[idx]
    tid = np.full(len(run.t), -1, dtype=np.int64)
    tid[has] = run.tgt_id[rows]
    tidu = tid[idx]

    def ip(a):
        return np.interp(tu, th, a)

    ex, ey, ang, arad = ip(v.ex[rows]), ip(v.ey[rows]), ip(v.miss[rows]), ip(v.arad[rows])
    tyaw, tpitch = ip(np.unwrap(v.tyaw[rows], period=360)), ip(v.tpitch[rows])
    yaw = np.interp(tu, run.t, np.unwrap(run.yaw, period=360))
    pitch = np.interp(tu, run.t, run.pitch)

    cvx = np.gradient(smooth(yaw, 5), dt) * np.cos(np.radians(pitch))
    cvy = np.gradient(smooth(pitch, 5), dt)
    tvx = np.gradient(smooth(tyaw, 5), dt) * np.cos(np.radians(tpitch))
    tvy = np.gradient(smooth(tpitch, 5), dt)
    tsp = np.hypot(tvx, tvy)

    # Velocities are meaningless across a respawn or a gap, so keep clear of block edges.
    change = np.r_[True, (tidu[1:] != tidu[:-1]) | (ok[1:] != ok[:-1])]
    pad = 15
    valid = ok & ~(np.convolve(change.astype(float), np.ones(2 * pad + 1), mode="same") > 0)
    moving = valid & (tsp > MOVE)
    on = ang <= arad

    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = tvx / tsp, tvy / tsp
        along_r = np.where(moving, (-ex * ux - ey * uy) / arad, np.nan)
        cross_r = np.where(moving, (ex * uy - ey * ux) / arad, np.nan)

    # --- reversals: velocity REV_WIN after a point points against velocity REV_WIN before it.
    # A wide symmetric window catches smooth turnarounds (targets that slow down and curve
    # through the turn) as well as instant ones; the turn itself is where the velocity
    # component along the old direction crosses zero.
    # Targets that curve up or down through a turn still reverse horizontally, which is what
    # the hand has to do, so a flip of the horizontal component alone also counts.
    k = int(REV_WIN * FS)
    cand = np.zeros(n, dtype=bool)
    horiz = np.zeros(n, dtype=bool)
    if n > 2 * k:
        pv, nv = slice(0, n - 2 * k), slice(2 * k, n)
        both = valid[pv] & valid[nv] & valid[k:n - k]
        full = (tvx[pv] * tvx[nv] + tvy[pv] * tvy[nv] < 0) & (tsp[pv] > 2 * MOVE) & (tsp[nv] > 2 * MOVE)
        hflip = (tvx[pv] * tvx[nv] < 0) & (np.abs(tvx[pv]) > 2 * MOVE) & (np.abs(tvx[nv]) > 2 * MOVE)
        cand[k:n - k] = both & (full | hflip)
        horiz[k:n - k] = hflip & ~full
    revs: list[int] = []
    rev_axis: dict[int, tuple[float, float]] = {}   # unit vector of the reversed motion, new way
    for a, b in runs(cand):
        c = (a + b) // 2
        ux0, uy0 = tvx[c - k], (0.0 if horiz[a:b].all() else tvy[c - k])
        proj = tvx[c - k:c + k + 1] * ux0 + tvy[c - k:c + k + 1] * uy0
        zc = np.flatnonzero((proj[:-1] > 0) & (proj[1:] <= 0))
        i = c - k + int(zc[np.argmin(np.abs(zc - k))]) + 1 if len(zc) else c
        if not revs or i - revs[-1] > 2 * k:
            revs.append(i)
            nrm = math.hypot(ux0, uy0)
            rev_axis[i] = (-ux0 / nrm, -uy0 / nrm)
    revset = set(revs)

    # --- segments ---
    table: list[dict] = []
    seg_bounds: list[tuple[int, int]] = []
    starts = np.flatnonzero(change).tolist() + [n]
    for a, b in zip(starts[:-1], starts[1:]):
        if not ok[a]:
            continue
        cuts = [a] + [r for r in revs if a < r < b] + [b]
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            pieces = max(1, math.ceil((c1 - c0) / (MAX_SEG * FS)))
            edges = np.linspace(c0, c1, pieces + 1).astype(int)
            for j in range(pieces):
                s0, s1 = int(edges[j]), int(edges[j + 1])
                if s1 - s0 < int(0.1 * FS):
                    continue
                axis = rev_axis.get(c0) if j == 0 else None
                table.append(_segment(s0, s1, axis, tu, tvx, tvy, tsp, cvx, cvy,
                                      ex, ey, arad, on, along_r, cross_r, moving))
                seg_bounds.append((s0, s1))
    for i, row in enumerate(table):
        row["i"] = i + 1

    # --- whole-run numbers ---
    lag_ms = _xcorr_lag(tvx, tvy, cvx, cvy, valid) if moving.sum() > 0.2 * max(valid.sum(), 1) else float("nan")
    # Tall bots: half-length of the capsule's spine as seen, so the shifted on-target test matches `on`.
    tdist = np.maximum(np.linalg.norm(run.tgt_pos[rows] - run.cam[run.tgt_frame[rows]], axis=1), 1e-6)
    spine = ip(np.degrees(np.arctan(np.maximum(run.tgt_hh[rows] - run.tgt_radius[rows], 0) / tdist)))
    shift = _shift_curve(yaw, pitch, ex, ey, arad, spine, valid, tidu)
    off = [(b - a) / FS * 1000 for a, b in runs(ok & ~on)]
    rev_rows = [r for r in table if r["rev"]]

    def med(key, src=None):
        vals = [r[key] for r in (table if src is None else src) if not math.isnan(r[key])]
        return float(np.median(vals)) if vals else float("nan")

    summary = [
        f"on target {_pct(on[ok]):.1f}% of {ok.sum() / FS:.1f}s with a target; "
        f"mean target radius {np.mean(arad[ok]):.2f} deg; median target speed {np.median(tsp[moving]) if moving.any() else 0:.0f} deg/s",
        f"crosshair sits {np.nanmean(along_r):+.2f} radii along the target's motion (- = behind) "
        f"and {np.nanmean(np.abs(cross_r)):.2f} radii off its line",
        f"mean offset from target centre: {np.mean(-ex[ok] / arad[ok]):+.2f} radii right, "
        f"{np.mean(-ey[ok] / arad[ok]):+.2f} radii up",
        f"velocity lag (cross-correlation) {lag_ms:.0f} ms; speed-match gain {med('gain'):.2f}",
        f"reversals {len(rev_rows)}: median reaction {med('react_ms', rev_rows):.0f} ms, "
        f"peak lag {med('revlag_r', rev_rows):.2f} radii, back on target after {med('recov_ms', rev_rows):.0f} ms",
        f"off-target episodes {len(off)}, median {np.median(off) if off else 0:.0f} ms, "
        f"longest {max(off) if off else 0:.0f} ms",
    ]
    if shift:
        summary.append(
            f"lag-shift ceiling: on target {shift['base']:.1f}% as played; replaying your crosshair "
            f"{shift['best_ms']:.0f} ms earlier gives {shift['best']:.1f}% (+{shift['best'] - shift['base']:.1f} points "
            f"from timing alone); the {100 - shift['best']:.1f}% still off is imprecision, not delay")

    # --- splits ---
    splits: list[tuple[str, list[str], list[list]]] = []
    thirds = np.array_split(np.flatnonzero(ok), 3)
    splits.append(("By third of the run", ["third", "on_pct", "lead_r", "cross_r"], [
        [i + 1, _pct(on[ix]), np.nanmean(along_r[ix]) if len(ix) else np.nan,
         np.nanmean(np.abs(cross_r[ix])) if len(ix) else np.nan] for i, ix in enumerate(thirds)]))
    mv = np.flatnonzero(moving)
    if len(mv):
        octs = (np.round(np.degrees(np.arctan2(tvy[mv], tvx[mv])) / 45.0).astype(int)) % 8
        from .geometry import OCTANTS
        rows_o = []
        for o in range(8):
            ix = mv[octs == o]
            if len(ix) >= 0.02 * len(mv):
                rows_o.append([OCTANTS[o], 100.0 * len(ix) / len(mv), _pct(on[ix]),
                               np.nanmean(along_r[ix]), np.nanmean(np.abs(cross_r[ix]))])
        splits.append(("By target direction", ["dir", "time_pct", "on_pct", "lead_r", "cross_r"], rows_o))

    # Fixed-edge bins, the same in every run, so runs and sessions can be pooled and compared.
    def binned(ix, code, labels):
        out = []
        for j, lab in enumerate(labels):
            s = ix[code == j]
            lr = along_r[s]
            lr = lr[~np.isnan(lr)]
            out.append([lab, len(s) / FS, float(on[s].sum()) / FS, float(lr.sum()), len(lr)])
        return out

    bins: dict[str, list] = {}
    if len(mv):
        bins["speed"] = binned(mv, np.digitize(tsp[mv], SPEED_EDGES[1:-1]), SPEED_LABELS)
        bins["position"] = binned(mv, np.digitize(along_r[mv], ZONE_EDGES[1:-1]), ZONE_LABELS)
    vi = np.flatnonzero(valid)
    if len(revs) and len(vi):
        ra = np.array(revs)
        last = np.searchsorted(ra, vi, side="right") - 1
        since = np.where(last >= 0, (vi - ra[np.maximum(last, 0)]) / FS * 1000, np.nan)
        # a respawn or gap since the last reversal resets the clock
        block = np.maximum.accumulate(np.where(change, np.arange(n), 0))
        since[(last < 0) | (ra[np.maximum(last, 0)] < block[vi])] = np.nan
        code = np.where(np.isnan(since), -1, np.digitize(np.nan_to_num(since), SINCE_EDGES[1:-1]))
        bins["since_rev"] = binned(vi, code, SINCE_LABELS)
    for key in ("speed", "position", "since_rev"):
        if key in bins:
            splits.append(bin_split(key, bins[key]))
    if shift:
        splits.append(("Lag-shift: on target if your crosshair had moved this much earlier (same path, "
                       "shifted in time)", ["earlier_ms", "on_pct", "gain_pts"],
                       [[ms, p, p - shift["base"]] for ms, p in zip(shift["ms"], shift["on"]) if ms % 20 == 0]))

    # --- reversal-aligned velocity profiles for the plot ---
    pre, post = int(0.2 * FS), int(0.5 * FS)
    prof_t = np.arange(-pre, post) / FS * 1000
    prof_c, prof_tg = [], []
    for r in revs:
        if r - pre < 0 or r + post > n or not valid[r - pre:r + post].all():
            continue
        mx, my = rev_axis[r]
        w = slice(r - pre, r + post)
        prof_c.append(cvx[w] * mx + cvy[w] * my)
        prof_tg.append(tvx[w] * mx + tvy[w] * my)

    def worst(count: int) -> list[tuple[str, list[str], list[list]]]:
        order = sorted((i for i, r in enumerate(table) if r["dur_ms"] >= 300),
                       key=lambda i: table[i]["on_pct"])[:count]
        out = []
        step = int(FS / 25)
        for i in order:
            a, b = seg_bounds[i]
            sl = slice(a, b, step)
            proj = cvx[sl] * np.nan_to_num(ux[sl]) + cvy[sl] * np.nan_to_num(uy[sl])
            out.append((f"segment {table[i]['i']} (on {table[i]['on_pct']:.0f}%)",
                        ["t_ms", "lead_r", "cross_r", "tgt_dps", "xhair_dps_along"],
                        [[(j * step) / FS * 1000, along_r[a + j * step], cross_r[a + j * step],
                          tsp[a + j * step], proj[j]] for j in range(len(proj))]))
        return out

    metrics = {
        "on_pct": _pct(on[ok]),
        "centred_pct": _share(bins.get("position"), "centred"),
        "far_behind_pct": _share(bins.get("position"), "far behind"),
        "fast_on_pct": _on_at(bins.get("speed"), SPEED_LABELS[4:]),
        "lead_r": float(np.nanmean(along_r)) if moving.any() else float("nan"),
        "react_ms": med("react_ms", rev_rows),
        "recov_ms": med("recov_ms", rev_rows),
        "off_per_min": len(off) / max(ok.sum() / FS / 60, 1e-6),
        "off_med_ms": float(np.median(off)) if off else 0.0,
        "off_p90_ms": float(np.percentile(off, 90)) if off else 0.0,
        "shift_on_pct": shift["best"] if shift else float("nan"),
        "shift_gain_pts": shift["best"] - shift["base"] if shift else float("nan"),
    }

    return {
        "mode": "tracking", "metrics": metrics, "bins": bins, "summary": summary, "columns": COLUMNS, "legend": LEGEND,
        "rows": [[r[c] for c in COLUMNS] for r in table], "splits": splits, "worst": worst,
        "on_pct": _pct(on[ok]), "lag_ms": lag_ms, "table": table,
        "plot": {"px": -ex[ok] / arad[ok], "py": -ey[ok] / arad[ok],
                 "prof_t": prof_t, "prof_c": np.array(prof_c), "prof_tg": np.array(prof_tg)},
    }


def _segment(a, b, axis, tu, tvx, tvy, tsp, cvx, cvy, ex, ey, arad, on, along_r, cross_r, moving) -> dict:
    """`axis` is set when the segment starts at a reversal: the unit vector of the motion
    that reversed, pointing the new way."""
    rev = axis is not None
    sl = slice(a, b)
    mv = moving[sl]
    tspd = float(tsp[sl].mean())
    # Direction of the new leg: just after a reversal the smoothed velocity is still ramping,
    # so average its first 150 ms; otherwise the whole segment.
    h = slice(a, min(b, a + int(0.15 * FS))) if rev else sl
    mx, my = float(tvx[h].mean()), float(tvy[h].mean())
    norm = math.hypot(mx, my)
    ux, uy = (mx / norm, my / norm) if norm > 1e-6 else (1.0, 0.0)
    cproj = cvx[sl] * ux + cvy[sl] * uy
    row = {
        "t_s": float(tu[a] - tu[0]), "dur_ms": (b - a) / FS * 1000, "rev": int(rev),
        "dir": octant(float(tvx[sl].mean()), float(tvy[sl].mean())) if tspd > MOVE else "-",
        "tgt_dps": tspd, "on_pct": _pct(on[sl]),
        "lead_r": float(np.nanmean(along_r[sl])) if mv.any() else float("nan"),
        "cross_r": float(np.nanmean(np.abs(cross_r[sl]))) if mv.any() else float("nan"),
        "react_ms": float("nan"), "gain": float("nan"),
        "revlag_r": float("nan"), "recov_ms": float("nan"),
    }
    if tspd <= MOVE:
        return row
    if not rev:
        row["gain"] = float(np.median(cproj)) / tspd
        return row
    # Reaction, measured along the axis that reversed: when the crosshair reaches a quarter of
    # the target's new-way speed, minus when the target itself got there (zero for an instant
    # reversal, longer for a target that eases through the turn). Negative = turned early.
    ax, ay = axis
    w = min(b - a, int(0.4 * FS))
    tr = tvx[sl] * ax + tvy[sl] * ay
    cr = cvx[sl] * ax + cvy[sl] * ay
    thr = 0.25 * float(np.max(tr[:w]))
    jt = _first_sustained(tr > thr, 5)
    jc = _first_sustained(cr > thr, 5)
    if jc is not None and thr > MOVE:
        row["react_ms"] = (jc - (jt or 0)) / FS * 1000
    j = _first_sustained(cproj > 0.25 * tspd, 5)
    if j is not None:
        row["gain"] = float(np.median(cproj[j:])) / tspd
    behind = (ex[sl] * ux + ey[sl] * uy)[:w] / arad[sl][:w]   # target ahead of crosshair, new direction
    pk = int(np.argmax(behind))
    row["revlag_r"] = float(behind[pk])
    back = _first_sustained(on[sl][pk:], 8)
    if on[sl][:w].all():
        row["recov_ms"] = 0.0          # never came off the target
    elif back is not None:
        row["recov_ms"] = (pk + back) / FS * 1000
    return row


SHIFT_MAX_MS = 200


def _shift_curve(yaw, pitch, ex, ey, arad, spine, valid, tidu) -> dict | None:
    """On-target % if the crosshair were where it actually was `s` ms later, for s = 0..SHIFT_MAX_MS.
    The best value is what fixing delay alone would buy; the rest of the gap to 100% is imprecision.
    Small-angle: the error moves by the crosshair's own yaw/pitch change over s. Every shift is scored
    on the same frames (valid now and SHIFT_MAX_MS later, same target), so the curve is comparable."""
    n = len(yaw)
    smax = int(round(SHIFT_MAX_MS / 1000 * FS))
    if n <= smax + 100:
        return None
    base = np.arange(n - smax)
    use = valid[base] & valid[base + smax] & (tidu[base] == tidu[base + smax])
    idx = base[use]
    if len(idx) < 0.5 * FS:
        return None
    cosp = np.cos(np.radians(pitch[idx]))
    step = max(1, int(round(0.004 * FS)))          # every 4 ms
    ms, on = [], []
    for s in range(0, smax + 1, step):
        dx = ex[idx] - (yaw[idx + s] - yaw[idx]) * cosp
        dy = ey[idx] - (pitch[idx + s] - pitch[idx])
        miss = np.hypot(dx, np.maximum(np.abs(dy) - spine[idx], 0))
        ms.append(s / FS * 1000)
        on.append(100.0 * float((miss <= arad[idx]).mean()))
    best = max(on)
    # A big target is hit over a range of shifts; take the middle of the top of the curve.
    best_ms = float(np.median([m for m, p in zip(ms, on) if p >= best - 0.25]))
    return {"ms": [round(m) for m in ms], "on": on, "base": on[0], "best": best, "best_ms": best_ms}


def _xcorr_lag(tvx, tvy, cvx, cvy, valid) -> float:
    """Delay (ms) at which crosshair velocity best matches target velocity."""
    n = len(tvx)
    lags = np.arange(-25, 126)
    score = np.full(len(lags), -np.inf)
    for i, lag in enumerate(lags):
        a = slice(0, n - lag) if lag >= 0 else slice(-lag, n)
        b = slice(lag, n) if lag >= 0 else slice(0, n + lag)
        m = valid[a] & valid[b]
        if m.sum() > 50:
            score[i] = float((tvx[a] * cvx[b] + tvy[a] * cvy[b])[m].mean())
    i = int(np.argmax(score))
    if not np.isfinite(score[i]):
        return float("nan")
    frac = 0.0
    if 0 < i < len(lags) - 1 and np.isfinite(score[i - 1]) and np.isfinite(score[i + 1]):
        denom = score[i - 1] - 2 * score[i] + score[i + 1]
        if denom < 0:
            frac = 0.5 * (score[i - 1] - score[i + 1]) / denom
    return float((lags[i] + frac) / FS * 1000)


def bin_split(key: str, rows: list) -> tuple[str, list[str], list[list]]:
    """Report table from bin rows [label, sec, on_sec, lead_sum, lead_n] (one run or pooled)."""
    title, head = BIN_TITLES[key]
    total = sum(r[1] for r in rows) or 1.0
    out = [[r[0], 100 * r[1] / total, 100 * r[2] / r[1] if r[1] else float("nan"),
            r[3] / r[4] if r[4] else float("nan")] for r in rows if r[1] >= 0.25]
    return title, [head, "time_pct", "on_pct", "lead_r"], out


def pool_bins(many: list[list]) -> list:
    """Sum bin rows from several runs (same labels, same order)."""
    out = [list(r) for r in many[0]]
    for rows in many[1:]:
        for o, r in zip(out, rows):
            for j in range(1, 5):
                o[j] += r[j]
    return out


def _share(rows, label) -> float:
    if not rows:
        return float("nan")
    total = sum(r[1] for r in rows)
    return 100 * sum(r[1] for r in rows if r[0] == label) / total if total else float("nan")


def _on_at(rows, labels) -> float:
    if not rows:
        return float("nan")
    sec = sum(r[1] for r in rows if r[0] in labels)
    return 100 * sum(r[2] for r in rows if r[0] in labels) / sec if sec >= 1.0 else float("nan")
