"""Clicking and target-switching analysis: one row per kill ("engagement")."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .format import BTN_FIRE, Run
from .geometry import OCTANTS, octant, smooth, target_view, view

COLUMNS = ["i", "t_s", "dist_deg", "dir", "react_ms", "flick_ms", "micro_ms", "peak_dps", "land_r", "off_r",
           "over_r", "past_ms", "corr", "acq_ms", "confirm_ms", "ttk_ms", "shots", "miss", "on_pct", "slips",
           "slip_ms", "brake", "wobble_r"]
# Braking: peak deceleration of the flick over its peak acceleration, from crosshair speed
# smoothed over BRAKE_SMOOTH samples (~22 ms at 400 Hz). A minimum-jerk flick is symmetric (1.0);
# clamping down to stop it brakes harder than it pushed (>1); releasing mid-flick and coasting
# in brakes softer (<1). The deceleration is looked for until BRAKE_TAIL_MS after the flick ends.
# Only flicks of BRAKE_MIN_DEG+ count: tiny nudges have no clear push and stop.
BRAKE_SMOOTH = 9
BRAKE_TAIL_MS = 60.0
BRAKE_MIN_DEG = 3.0
# Post-flick wobble: in the WOBBLE_MS after the flick ends, how much the crosshair moved
# against the target beyond the straight line to where it ended up (path length minus net
# change, in target radii, error smoothed over WOBBLE_SMOOTH samples). A clean correction or a
# clean landing reads ~0; bouncing, overshooting the correction or shaking reads high. Measured
# against the target, so following a moving one is not wobble.
WOBBLE_MS = 150.0
WOBBLE_SMOOTH = 7
SETTLE_MS = 50.0    # on target this long without a break = settled; later drift is tracking, not overflick
# Off target this long after settling = a slip. From 1025 kills of voxTargetSwitch 2 (Nov 2026): off
# stretches under 10 ms graze the edge by ~1% of a radius, 10-20 ms is nearly empty, and from 20 ms up
# they run continuously to 400+ ms and hold almost all off-target time.
SLIP_MS = 20.0
# Target choice is read this many samples after the previous kill, so the target that respawns
# in place of the dead one is among the options. The hand has not started moving yet.
CHOICE_LAG = 2
PATH_COLUMNS = ["opts", "pick", "extra_deg"]
PATH_LEGEND = (
    " opts=live targets to choose from when the previous one died; pick=which of them you went "
    "to by distance from the crosshair (1 = nearest); extra_deg=how much further that was than "
    "the nearest one."
)
LEGEND = (
    "One row per kill, timed from the previous kill (or the target's spawn if later). "
    "dist_deg=flick distance; dir=screen direction to the target; react_ms=start until the hand "
    "moves; flick_ms=hand starts moving until the first movement ends (speed down to a quarter "
    "of its peak); micro_ms=from there until settled on the target (on it for "
    f"{SETTLE_MS:.0f} ms, or still on it at the kill), 0 if the flick landed and stayed; "
    "peak_dps=peak crosshair speed; land_r=where the first movement ended along the flick "
    "line in target radii (- short, + past, |x|<1 on target); off_r=total distance from centre at "
    "that moment, radii; over_r=furthest past the target centre along the flick line, from peak "
    f"speed until settled on target ({SETTLE_MS:.0f} ms on it) or the kill, radii (>1 = beyond the "
    "far edge, 0 = never crossed the centre); past_ms=time beyond the far edge in that stretch; "
    "corr=corrective movements after it; acq_ms=start until first on target; "
    "confirm_ms=time on target before the killing shot; miss=missed shots with type "
    "(E fired before arriving, O past the target, D was on target then left); "
    "on_pct=share of time on target after first reaching it; "
    f"slips=times you left the target for {SLIP_MS:.0f}+ ms after settling on it and came back "
    "before it died; slip_ms=total time off the target in those slips; "
    "brake=peak deceleration of the flick over its peak acceleration (1 = symmetric, >1 = stopped "
    f"harder than you pushed, <1 = coasted in; flicks of {BRAKE_MIN_DEG:.0f}+ deg); wobble_r=back-and-forth "
    f"against the target in the {WOBBLE_MS:.0f} ms after the flick ended, beyond the straight line to "
    "where it ended up, radii (0 = clean)."
)


DIST_BINS = [("<10 deg", 0, 10), ("10-20 deg", 10, 20), ("20-35 deg", 20, 35), (">35 deg", 35, 1e9)]


def shot_frames(run: Run) -> np.ndarray:
    """Frame index of every shot. Uses the shot counter, or fire-button presses if there is none."""
    if len(run.shots) and (run.shots >= 0).all():
        d = np.diff(run.shots, prepend=run.shots[0])
        d[d < 0] = 0  # counter reset between runs
        return np.repeat(np.flatnonzero(d > 0), d[d > 0])
    fire = (run.buttons & BTN_FIRE) > 0
    return np.flatnonzero(fire & ~np.r_[False, fire[:-1]])


def is_continuous_fire(run: Run) -> bool:
    duration = max(run.t[-1] - run.t[0], 1e-6)
    if (run.shots >= 0).all():
        return len(shot_frames(run)) / duration > 15.0
    return float(((run.buttons & BTN_FIRE) > 0).mean()) > 0.5


def split_respawns(run: Run) -> Run:
    """Some scenarios respawn a target by teleporting the same actor, so it never vanishes.
    Give it a fresh id whenever it jumps further than any real movement could in one sample."""
    if len(run.tgt_id) == 0:
        return run
    order = np.lexsort((run.tgt_frame, run.tgt_id))
    ids, pos, rad = run.tgt_id[order], run.tgt_pos[order], run.tgt_radius[order]
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    same = ids[1:] == ids[:-1]
    # Median over every sample, still ones included: ~0 for static targets, the normal
    # per-sample travel for moving ones.
    typical = float(np.median(step[same])) if same.any() else 0.0
    jump = same & (step > np.maximum(2.0 * rad[1:], 25.0 * typical))
    if not jump.any():
        return run
    new_ids = np.empty_like(run.tgt_id)
    new_ids[order] = np.cumsum(np.r_[True, ~same | jump])
    return replace(run, tgt_id=new_ids)


def kills(run: Run, shots: np.ndarray) -> list[tuple[int, int, int]]:
    """(kill_frame, target_id, first_frame) for each target that vanished with a shot nearby.
    Targets that time out with no shot near are not kills."""
    n = len(run.t)
    out = []
    ids, inverse = np.unique(run.tgt_id, return_inverse=True)
    first = np.full(len(ids), n, dtype=np.int64)
    last = np.full(len(ids), -1, dtype=np.int64)
    np.minimum.at(first, inverse, run.tgt_frame)
    np.maximum.at(last, inverse, run.tgt_frame)
    continuous = is_continuous_fire(run)
    for tid, f0, f1 in zip(ids, first, last):
        if f1 >= n - 1:
            continue  # still alive when the log ends
        k = int(f1) + 1
        if continuous:
            near = bool(run.buttons[k] & BTN_FIRE) or bool(np.any((shots >= k - 3) & (shots <= k + 3)))
        else:
            near = bool(np.any((shots >= k - 3) & (shots <= k + 3)))
        if near:
            out.append((k, int(tid), int(f0)))
    return sorted(out)


def analyze(run: Run, mode: str = "clicking") -> dict:
    run = split_respawns(run)
    n = len(run.t)
    v = target_view(run)
    shots = shot_frames(run)
    kill_list = kills(run, shots)
    if not kill_list:
        raise ValueError("no kills found in log; is this a tracking run?")
    continuous = mode == "switching"

    yaw = np.unwrap(run.yaw, period=360)
    speed = smooth(np.hypot(np.gradient(yaw, run.t) * np.cos(np.radians(run.pitch)),
                            np.gradient(run.pitch, run.t)), 3)

    by_id = np.argsort(run.tgt_id, kind="stable")
    sorted_ids = run.tgt_id[by_id]
    frame_rows = np.searchsorted(run.tgt_frame, np.arange(n + 1))

    # The killing shot can register a sample either side of the target vanishing; pair each
    # kill with its nearest unused shot so that shot is never counted as a miss.
    killing = np.zeros(len(shots), dtype=bool)
    for k, _tid, _f0 in kill_list:
        near = np.flatnonzero(~killing & (np.abs(shots - k) <= 3))
        if len(near):
            killing[near[np.argmin(np.abs(shots[near] - k))]] = True
    miss_shots = shots[~killing]

    table: list[dict] = []
    traces: list[dict] = []
    prev_kill = 0
    for k, tid, f0 in kill_list:
        s = max(prev_kill, f0)
        prev_window = prev_kill
        prev_kill = k
        rows = by_id[np.searchsorted(sorted_ids, tid, "left"):np.searchsorted(sorted_ids, tid, "right")]
        rows = rows[run.tgt_frame[rows] >= s]
        if len(rows) < 3:
            continue
        fr = run.tgt_frame[rows]
        tt = (run.t[fr] - run.t[s]) * 1000.0
        ex, ey, ang, arad = v.ex[rows], v.ey[rows], v.ang[rows], v.arad[rows]
        sp = speed[fr]
        on = v.miss[rows] <= arad
        dist = float(ang[0])
        g = math.hypot(ex[0], ey[0])
        gx, gy = (ex[0] / g, ey[0] / g) if g > 1e-9 else (1.0, 0.0)
        rem = ex * gx + ey * gy            # distance still to go along the initial flick line

        row = {"t_s": float(run.t[s] - run.t[0]), "dist_deg": dist, "dir": octant(ex[0], ey[0]),
               "ttk_ms": float((run.t[k] - run.t[s]) * 1000.0)}
        nan = float("nan")
        row.update(react_ms=nan, flick_ms=nan, micro_ms=nan, peak_dps=nan, land_r=nan, off_r=nan, over_r=nan, past_ms=nan,
                   corr=0, acq_ms=nan, confirm_ms=nan, on_pct=nan, brake=nan, wobble_r=nan)

        if dist > arad[0]:
            p = int(np.argmax(sp))
            peak = float(sp[p])
            row["peak_dps"] = peak
            slow = np.flatnonzero(sp[:p] < max(0.1 * peak, 10.0))
            onset = int(slow[-1]) + 1 if len(slow) else 0
            row["react_ms"] = float(tt[onset])
            # The first movement ends when speed has dropped to a quarter of its peak.
            ended = np.flatnonzero(sp[p:] < 0.25 * peak)
            q = p + int(ended[0]) if len(ended) else len(sp) - 1
            row["land_r"] = float(-rem[q] / arad[q])
            row["off_r"] = float(ang[q] / arad[q])
            row["_arad_q"] = float(arad[q])
            # For flick gain: distance asked for, distance covered along the line, and sideways miss (deg)
            row["_D"], row["_M"] = g, float(g - rem[q])
            row["_lat"] = float(ex[q] * -gy + ey[q] * gx)
            row["over_r"], row["past_ms"] = _overflick(rem[p:] / arad[p:], on[p:], tt[p:])
            # with the trigger held the hand keeps moving after the flick; bumps are tracking, not corrections
            row["corr"] = float("nan") if continuous else _count_peaks(sp[q:], tt[q:], max(0.12 * peak, 8.0))
            row["_q"], row["_peak"] = q, peak
            if dist >= BRAKE_MIN_DEG:
                row["brake"] = _brake(sp, tt, onset, q)
            row["wobble_r"] = _wobble(rem / arad, (ex * -gy + ey * gx) / arad, tt, q)
        row.update(_choice(run, v, frame_rows, tid, min(s + CHOICE_LAG, int(fr[-1]))))
        row["_off"] = _off_episodes(on, v.miss[rows] / arad, tt)
        slips = [e for e in row["_off"] if e[0] >= SLIP_MS]
        row["slips"] = len(slips)
        row["slip_ms"] = float(sum(e[0] for e in slips))
        row["_wait_ms"] = float((run.t[s] - run.t[prev_window]) * 1000.0) if prev_window else 0.0
        row["_budget"] = _budget(row, on, tt)
        row.update(_phases(row, on, tt, sp))
        if on.any():
            first_on = int(np.argmax(on))
            row["acq_ms"] = float(tt[first_on])
            row["on_pct"] = 100.0 * float(on[first_on:].mean())
            # Walk back from the killing shot while still on target.
            j = len(on) - 1
            if not on[j] and j > 0 and on[j - 1]:
                j -= 1
            if on[j]:
                e = j
                while e > 0 and on[e - 1]:
                    e -= 1
                row["confirm_ms"] = float((run.t[k] - run.t[fr[e]]) * 1000.0)
            else:
                row["confirm_ms"] = 0.0

        mine = miss_shots[(miss_shots > prev_window) & (miss_shots <= k)]
        row["shots"] = len(mine) + 1
        types = ""
        if not continuous:
            for f in mine:                     # shots in this window that killed nothing
                i = int(np.searchsorted(fr, f))
                i = min(i, len(fr) - 1)
                if on[i]:
                    continue                   # multi-hit target: this one landed
                if on[:i].any():
                    types += "D"
                elif rem[i] > 0:
                    types += "E"
                else:
                    types += "O"
        row["miss"] = f"{len(types)}{types}" if types else "0"
        row["_miss_n"] = len(types)
        table.append(row)
        traces.append({"tt": tt, "rem_r": rem / arad, "lat_r": (ex * -gy + ey * gx) / arad,
                       "sp": sp, "onset_ms": row["react_ms"]})
    for i, row in enumerate(table):
        row["i"] = i + 1

    # --- every shot, as crosshair position relative to the nearest target, in radii ---
    scat = _shot_scatter(run, shots) if not continuous else None

    # --- summary ---
    def col(key, rows=None):
        vals = np.array([r[key] for r in (table if rows is None else rows)], dtype=float)
        return vals[~np.isnan(vals)]

    def med(key, rows=None):
        vals = col(key, rows)
        return float(np.median(vals)) if len(vals) else float("nan")

    land = col("land_r")
    misses = sum(r["_miss_n"] for r in table)
    all_types = "".join(r["miss"].lstrip("0123456789") for r in table)
    dur = float(run.t[-1] - run.t[0])
    summary = [
        f"{len(table)} kills in {dur:.1f}s ({len(table) / dur:.2f}/s); {len(shots)} shots"
        + ("" if continuous else f", {misses} misses "
           f"({all_types.count('E')} early, {all_types.count('O')} past, {all_types.count('D')} drifted off)"),
        f"median time per kill {med('ttk_ms'):.0f} ms = reaction {med('react_ms'):.0f} "
        f"+ travel to first on-target at {med('acq_ms'):.0f} ms, "
        + (f"then {med('confirm_ms'):.0f} ms on the target until it died" if continuous
           else f"confirm {med('confirm_ms'):.0f} ms before the shot"),
        f"flicks: median distance {med('dist_deg'):.1f} deg, peak speed {med('peak_dps'):.0f} deg/s"
        + ("" if continuous else f", {np.mean(col('corr')):.2f} corrections per kill"),
    ]
    if len(land):
        summary.append(
            f"first movement lands short on {100 * np.mean(land < -1):.0f}%, on target on "
            f"{100 * np.mean(np.abs(land) <= 1):.0f}%, past on {100 * np.mean(land > 1):.0f}% "
            f"(median {np.median(land):+.2f} radii)")
    def slip_pct(rows):
        return 100 * float(np.mean([r["slips"] > 0 for r in rows])) if rows else float("nan")

    over = col("over_r")
    if len(over):
        big = [r for r in table if r["over_r"] > 1.0]
        summary.append(
            f"overflick (went beyond the far edge before settling) on {100 * np.mean(over > 1):.0f}% of flicks"
            + (f"; on those, median {med('over_r', big):.2f} radii past centre for {med('past_ms', big):.0f} ms"
               if big else ""))
    if continuous:
        eps = [e for r in table for e in r["_off"] if e[0] >= SLIP_MS]
        line = (f"slipped off after settling on {slip_pct(table):.0f}% of kills "
                f"({len(eps)} slips, {sum(e[0] for e in eps) / 1000:.1f}s off target in total)")
        if eps:
            line += (f"; median slip {np.median([e[0] for e in eps]):.0f} ms, "
                     f"{np.median([e[1] for e in eps]):.2f} radii outside the edge")
        summary.append(line)

    brake, wob = col("brake"), col("wobble_r")
    if len(brake) >= 8:
        summary.append(
            f"stops: braking ratio {np.median(brake):.2f} (peak stopping force over peak pushing force, median of "
            f"{len(brake)} flicks; 1 = symmetric, above 1 = clamped to stop, below 1 = let go and coasted in); "
            f"stopped harder than you pushed on {100 * np.mean(brake > 1):.0f}% of flicks"
            + (f"; post-flick wobble median {np.median(wob):.2f} radii in the first {WOBBLE_MS:.0f} ms "
               f"(back-and-forth against the target beyond a straight correction; 0 = clean)" if len(wob) >= 8 else ""))

    fg = flick_gain(table)
    if fg:
        if fg["slope"] < 0.95:
            kind = "the shortfall grows with distance, so bigger flicks land further short (calibration)"
        elif fg["slope"] > 1.05:
            kind = "the overshoot grows with distance, so bigger flicks land further past (calibration)"
        elif fg["offset_deg"] <= -1.5:
            kind = "a fixed few degrees short whatever the distance"
        elif fg["offset_deg"] >= 1.5:
            kind = "a fixed few degrees long whatever the distance"
        else:
            kind = "about right on average; what's left is scatter"
        summary.append(
            f"flick gain {fg['gain']:.2f} (first movement covers {100 * fg['gain']:.0f}% of the distance, median "
            f"of {fg['n']}); fit reach = {fg['slope']:.2f} x distance {fg['offset_deg']:+.1f} deg: {kind}; "
            f"scatter +-{fg['spread_pct']:.0f}% of distance along the line, +-{fg['side_deg']:.2f} deg sideways (robust SD)")

    path = target_choice(table)
    if path:
        line = (f"target choice: went to the nearest of the live targets on {path['nearest_pct']:.0f}% of kills "
                f"(2nd nearest {path['second_pct']:.0f}%, further {path['further_pct']:.0f}%); your picks were "
                f"{path['picked_deg']:.1f} deg away on average against {path['nearest_deg']:.1f} for the nearest, "
                f"{path['extra_deg']:.1f} deg extra per kill")
        if path["ms_per_deg"] > 0:
            cost = path["extra_deg"] * path["ms_per_deg"] * path["n"] / 1000.0
            line += (f"; at this run's {path['ms_per_deg']:.1f} ms per degree that is about {cost:.1f}s "
                     "(a rough what-if: the nearest target is not always the best path)")
        summary.append(line)

    fm = flick_micro(table)
    if fm:
        summary += fm["summary"]

    b_cols, b_rows, b_share = time_budget(table, continuous)
    cuttable = [(lab, r[5]) for (key, lab), r in zip(BUDGET_PARTS, b_rows) if not math.isnan(r[5])]
    saved = sum(s for _l, s in cuttable)
    top = max(cuttable, key=lambda x: x[1])
    mean_ttk = float(np.mean([r["ttk_ms"] for r in table]))
    summary.append(
        f"time budget: reaction {b_share['react'][0]:.0f}%, travel {b_share['travel'][0]:.0f}%, off again after "
        f"first contact {b_share['settle'][0] + b_share['slip'][0]:.0f}% (slips {b_share['slip'][0]:.0f}%), on target "
        f"{b_share['on'][0]:.0f}%; cutting each part to your own best-quarter kill would save {saved:.1f}s, "
        f"about {1000 * saved / mean_ttk:.0f} kills (most from {top[0]}: {top[1]:.1f}s)")

    # --- splits ---
    head = ["n", "ttk_ms", "react_ms", "flick_ms", "micro_ms", "land_r", "gain", "over_pct", "past_ms", "slip_pct",
            "brake", "wobble_r", "corr", "confirm_ms", "misses"]

    def gain_med(rows):
        g = [r["_M"] / r["_D"] for r in rows if "_D" in r and r["_D"] >= 3.0]
        return float(np.median(g)) if g else float("nan")

    def over_pct(rows):
        v = col("over_r", rows)
        return 100 * float(np.mean(v > 1)) if len(v) else float("nan")

    def agg(rows):
        big = [r for r in rows if r["over_r"] > 1.0]
        return [len(rows), med("ttk_ms", rows), med("react_ms", rows), med("flick_ms", rows),
                med("micro_ms", rows), med("land_r", rows),
                gain_med(rows), over_pct(rows), med("past_ms", big) if big else float("nan"), slip_pct(rows),
                med("brake", rows), med("wobble_r", rows),
                float(np.mean(col("corr", rows))) if len(col("corr", rows)) else float("nan"),
                med("confirm_ms", rows), sum(r["_miss_n"] for r in rows)]

    splits = [("Time budget (where the kill time went; saved_s = if each kill's part were no longer than "
               "your own 25th percentile this run, a rough what-if)", b_cols, b_rows)]
    if fm:
        splits += fm["splits"]
    dists = np.array([r["dist_deg"] for r in table])
    if len(table) >= 9:
        # fixed edges, the same in every run, so runs can be compared and pooled
        buckets = [(lab, (dists >= lo) & (dists < hi)) for lab, lo, hi in DIST_BINS]
        buckets = [(lab, m) for lab, m in buckets if m.any()]
        splits.append(("By flick distance (fixed bands)", ["dist"] + head,
                       [[name] + agg([r for r, m in zip(table, mask) if m]) for name, mask in buckets]))
    else:
        buckets = [("all", np.ones(len(table), dtype=bool))]
    if path:
        groups = [("nearest", lambda r: r["pick"] == 1), ("2nd nearest", lambda r: r["pick"] == 2),
                  ("further", lambda r: r["pick"] >= 3)]
        splits.append(("By target choice (which of the live targets you went to)", ["pick"] + head,
                       [[name] + agg(sel) for name, want in groups
                        if (sel := [r for r in table if "_pick_deg" in r and want(r)])]))
    splits.append(("By direction", ["dir"] + head,
                   [[o] + agg([r for r in table if r["dir"] == o]) for o in OCTANTS
                    if any(r["dir"] == o for r in table)]))
    edges = np.linspace(0, len(table), 4).astype(int)
    splits.append(("By third of the run", ["third"] + head,
                   [[i + 1] + agg(table[edges[i]:edges[i + 1]]) for i in range(3)]))

    def worst(count: int):
        order = sorted(range(len(table)), key=lambda i: -table[i]["ttk_ms"])[:count]
        out = []
        for i in order:
            tr = traces[i]
            step = max(1, len(tr["tt"]) // 40)
            out.append((f"kill {table[i]['i']} ({table[i]['ttk_ms']:.0f} ms, {table[i]['dist_deg']:.0f} deg {table[i]['dir']})",
                        ["t_ms", "togo_r", "side_r", "dps"],
                        [[tr["tt"][j], tr["rem_r"][j], tr["lat_r"][j], tr["sp"][j]]
                         for j in range(0, len(tr["tt"]), step)]))
        return out

    metrics = {
        "kills_per_min": 60 * len(table) / dur,
        "ttk_ms": med("ttk_ms"), "react_ms": med("react_ms"), "acq_ms": med("acq_ms"),
        "peak_dps": med("peak_dps"), "over_pct": over_pct(table),
        "flick_gain": fg["gain"] if fg else float("nan"),
        "flick_spread_pct": fg["spread_pct"] if fg else float("nan"),
        "off_after_pct": b_share["settle"][0] + b_share["slip"][0],
        "brake_ratio": float(np.median(brake)) if len(brake) >= 8 else float("nan"),
        "wobble_r": float(np.median(wob)) if len(wob) >= 8 else float("nan"),
    }
    if fm:
        metrics.update(fm["metrics"])
    if path:
        metrics.update(nearest_pct=path["nearest_pct"], extra_deg=path["extra_deg"])
    if continuous:
        metrics["on_pct"] = med("on_pct")
        metrics["slip_pct"] = slip_pct(table)
    else:
        metrics.update(confirm_ms=med("confirm_ms"), land_r=med("land_r"),
                       short_pct=100 * float(np.mean(land < -1)) if len(land) else float("nan"),
                       corr=float(np.mean(col("corr"))) if len(col("corr")) else float("nan"),
                       miss_per_kill=misses / max(len(table), 1))

    columns = COLUMNS + PATH_COLUMNS if path else COLUMNS
    return {
        "mode": mode, "metrics": metrics, "bins": {}, "summary": summary, "columns": columns,
        "legend": LEGEND + PATH_LEGEND if path else LEGEND,
        "rows": [[r[c] for c in columns] for r in table], "splits": splits, "worst": worst,
        "table": table, "kills": len(table), "shots": len(shots), "misses": misses,
        "kill_times": np.array([run.t[k] for k, _tid, _f0 in kill_list]),
        "plot": {"scatter": scat, "traces": traces, "dists": dists,
                 "buckets": [(name, np.flatnonzero(mask)) for name, mask in buckets]},
    }


def _choice(run: Run, v, frame_rows: np.ndarray, tid: int, frame: int) -> dict:
    """Which of the targets alive at `frame` the player went on to kill: opts = how many there
    were, pick = its rank by angle from the crosshair (1 = nearest), extra_deg = its angle minus
    the nearest one's. pick and extra_deg are nan with fewer than two targets to choose from."""
    a, b = frame_rows[frame], frame_rows[frame + 1]
    ids, ang = run.tgt_id[a:b], v.ang[a:b]
    mine = ang[ids == tid]
    if len(ids) < 2 or len(mine) == 0:
        return {"opts": int(len(ids)), "pick": float("nan"), "extra_deg": float("nan")}
    return {"opts": int(len(ids)), "pick": int((ang < mine[0]).sum()) + 1,
            "extra_deg": float(mine[0] - ang.min()), "_pick_deg": float(mine[0])}


def target_choice(table: list[dict]) -> dict | None:
    """Run totals for target choice, or None when most kills had only one target to go to.
    ms_per_deg = least-squares slope of time per kill on the picked target's distance."""
    rows = [r for r in table if "_pick_deg" in r]
    if len(rows) < 8 or 2 * len(rows) < len(table):
        return None
    pick = np.array([r["pick"] for r in rows])
    deg = np.array([r["_pick_deg"] for r in rows])
    extra = np.array([r["extra_deg"] for r in rows])
    slope = float(np.polyfit(deg, [r["ttk_ms"] for r in rows], 1)[0]) if np.ptp(deg) > 1.0 else 0.0
    return {"n": len(rows), "nearest_pct": 100 * float(np.mean(pick == 1)),
            "second_pct": 100 * float(np.mean(pick == 2)), "further_pct": 100 * float(np.mean(pick >= 3)),
            "picked_deg": float(deg.mean()), "nearest_deg": float((deg - extra).mean()),
            "extra_deg": float(extra.mean()), "ms_per_deg": slope}


def _budget(row: dict, on: np.ndarray, tt: np.ndarray) -> dict:
    """Split one kill's time (ms) into parts that add up to ttk_ms: reaction, travel until first
    on the target, off it again afterwards (settling: bounces and overflicks; slips: leaving after
    settling), and on it."""
    ttk = row["ttk_ms"]
    react = row["react_ms"] if not math.isnan(row["react_ms"]) else 0.0
    if not on.any():
        react = min(react, ttk)
        return {"react": react, "travel": ttk - react, "settle": 0.0, "slip": 0.0, "on": 0.0}
    first = int(np.argmax(on))
    acq = float(tt[first])
    react = min(react, acq)
    dt = np.diff(tt, append=ttk)       # each sample lasts until the next; the last until the kill
    after = np.arange(len(on)) >= first
    on_ms = float(dt[after & on].sum())
    off_ms = float(dt[after & ~on].sum())
    slip = min(row["slip_ms"], off_ms)
    return {"react": react, "travel": acq - react, "settle": off_ms - slip, "slip": slip, "on": on_ms}


def _settle_start(on: np.ndarray, tt: np.ndarray) -> int | None:
    """Start of the first on-target stretch that lasts SETTLE_MS, or that is still going at the
    kill (a quick clicker shoots before SETTLE_MS is up)."""
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j == len(on) or tt[j] - tt[i] >= SETTLE_MS:
                return i
            i = j
        else:
            i += 1
    return None


def _phases(row: dict, on: np.ndarray, tt: np.ndarray, sp: np.ndarray) -> dict:
    """Split one kill into reaction + flick + micro + after settling (ms; adds up to ttk_ms).
    flick: hand starts moving until the first movement ends. micro: from there until settled on
    the target. after: settled until the kill (confirm / time under fire, slips included).
    Also how the micro went: corrective sub-movements, and times it touched the target and fell
    off again before settling."""
    ttk = row["ttk_ms"]
    s = _settle_start(on, tt)
    settle_t = float(tt[s]) if s is not None else ttk
    if "_q" not in row:                        # started on the target: nothing to flick
        return {"flick_ms": 0.0, "micro_ms": settle_t,
                "_phase": {"react": 0.0, "flick": 0.0, "micro": settle_t, "after": ttk - settle_t}}
    q = row["_q"]
    react = row["react_ms"]
    flick_end = float(tt[q])
    micro = max(settle_t - flick_end, 0.0)
    out = {"flick_ms": flick_end - react, "micro_ms": micro,
           "_phase": {"react": react, "flick": flick_end - react, "micro": micro, "after": ttk - flick_end - micro},
           "_micro_corr": 0, "_micro_touch": 0}
    if micro > 0:
        end = s if s is not None else len(on) - 1
        seg = on[q:end + 1]
        out["_micro_corr"] = _count_peaks(sp[q:end + 1], tt[q:end + 1], max(0.12 * row["_peak"], 8.0))
        out["_micro_touch"] = int(np.sum(seg[1:-1] & ~seg[2:]))   # on, then off again, before settling
    return out


def _pct_bar(x: np.ndarray, band: np.ndarray | None = None) -> np.ndarray:
    """Each kill's own-best-quarter bar: 25th percentile, per distance band when it has 8+ kills."""
    bar = np.full(len(x), np.percentile(x, 25))
    if band is not None:
        for b in np.unique(band):
            m = band == b
            if m.sum() >= 8:
                bar[m] = np.percentile(x[m], 25)
    return bar


LEFT_BINS = [("landed (<=1 r)", -1.0, 1.0), ("1-2 r", 1.0, 2.0), ("2-4 r", 2.0, 4.0), (">4 r", 4.0, 1e9)]


def flick_micro(table: list[dict]) -> dict | None:
    """Judge the flick and the micro-adjustment separately.

    Flick: time (a fixed part + ms per degree), and how close it got: landed on the target %,
    error left as % of the flick distance. Micro: time and corrections, split by how far the
    flick left you (left_r = distance from the target centre when the flick ended, radii).
    Blame: each micro is held to your own 25th percentile for the same leftover (LEFT_BINS).
    Up to that bar is what the flick's leftover costs even with a good micro (flick_cost_s);
    beyond it is the micro being slower than you can do it (micro_saved_s). flick_saved_s =
    flick time beyond your own 25th percentile in the same distance band.
    """
    rows = [r for r in table if "_q" in r]
    if len(rows) < 8:
        return None
    d = np.array([r["dist_deg"] for r in rows])
    fl = np.array([r["flick_ms"] for r in rows])
    mi = np.array([r["micro_ms"] for r in rows])
    left = np.array([r["off_r"] for r in rows])
    err = 100 * left * np.array([r["_arad_q"] for r in rows]) / np.maximum(d, 1e-6)
    landed = left <= 1.0
    corr = np.array([r["_micro_corr"] for r in rows], dtype=float)
    touch = np.array([r["_micro_touch"] for r in rows], dtype=float)
    had = mi > 0
    ph = {k: np.array([r["_phase"][k] for r in table]) for k in ("react", "flick", "micro", "after")}
    total = sum(float(v.sum()) for v in ph.values()) or 1.0
    mean_ttk = float(np.mean([r["ttk_ms"] for r in table]))
    nan = float("nan")

    def mean_or0(x, m):
        return float(x[m].mean()) if m.any() else 0.0

    def med_or_nan(x, m):
        return float(np.median(x[m])) if m.any() else nan

    ms_per_deg, fl0 = np.polyfit(d, fl, 1) if np.ptp(d) > 3.0 else (nan, float(np.median(fl)))
    band = np.array([next(i for i, (_l, lo, hi) in enumerate(DIST_BINS) if lo <= x < hi) for x in d])
    flick_saved = float(np.maximum(fl - _pct_bar(fl, band), 0).sum()) / 1000.0
    lband = np.array([next(i for i, (_l, lo, hi) in enumerate(LEFT_BINS) if lo < x <= hi) for x in left])
    mbar = _pct_bar(mi, lband)
    micro_saved = float(np.maximum(mi - mbar, 0).sum()) / 1000.0
    flick_cost = float(np.minimum(mi, mbar).sum()) / 1000.0

    share = {k: 100 * float(v.sum()) / total for k, v in ph.items()}
    summary = [
        f"switch phases: reaction {share['react']:.0f}%, flick {share['flick']:.0f}%, micro-adjust "
        f"{share['micro']:.0f}%, after settling {share['after']:.0f}% of kill time (median flick "
        f"{np.median(fl):.0f} ms, median micro {np.median(mi):.0f} ms)",
        f"flick verdict: lands on the target {100 * landed.mean():.0f}% of the time and leaves a median "
        f"{np.median(err):.0f}% of the distance ({np.median(left):.1f} radii) to fix"
        + ("" if math.isnan(ms_per_deg) else f"; takes {fl0:.0f} ms + {ms_per_deg:.1f} ms per degree")
        + f"; each flick no slower than your best quarter would save {flick_saved:.1f}s",
        "micro verdict: "
        + (f"{med_or_nan(mi, landed):.0f} ms after a flick that landed, " if landed.any() else "no flick landed, ")
        + (f"{med_or_nan(mi, ~landed):.0f} ms after one that missed" if (~landed).any() else "none missed")
        + f"; {mean_or0(corr, had):.1f} corrections and "
        f"{mean_or0(touch, had):.1f} touch-and-fall-offs per micro"
        + f"; of the run's {mi.sum() / 1000:.1f}s of micro, {flick_cost:.1f}s is what the flick left (your "
        f"best-quarter micro for that leftover would still take it) and {micro_saved:.1f}s is the micro "
        "being slower than your own best for the same leftover",
    ]
    big, small = sorted([("flick", flick_saved), ("micro", micro_saved)], key=lambda x: -x[1])
    summary.append(f"bigger opportunity: the {big[0]} ({big[1]:.1f}s back against {small[1]:.1f}s for the "
                   f"{small[0]}, about {1000 * (big[1] - small[1]) / mean_ttk:.0f} kills apart)")

    head = ["n", "react_ms", "flick_ms", "micro_ms", "after_ms", "landed_pct", "err_pct", "left_r",
            "micro_corr", "touches"]
    rp = np.array([r["_phase"]["react"] for r in rows])
    af = np.array([r["_phase"]["after"] for r in rows])

    def agg(m):
        return [int(m.sum()), float(np.median(rp[m])), float(np.median(fl[m])), float(np.median(mi[m])),
                float(np.median(af[m])), 100 * float(landed[m].mean()), float(np.median(err[m])),
                float(np.median(left[m])), mean_or0(corr, m & had), mean_or0(touch, m & had)]

    by_dist = [[lab] + agg(m) for lab, lo, hi in DIST_BINS if (m := (d >= lo) & (d < hi)).any()]
    by_dist.append(["all"] + agg(np.ones(len(rows), dtype=bool)))
    by_left = [[lab, int(m.sum()), float(np.median(mi[m])), float(np.percentile(mi[m], 25)),
                mean_or0(corr, m & had), mean_or0(touch, m & had), float(mi[m].sum()) / 1000.0]
               for lab, lo, hi in LEFT_BINS if (m := (left > lo) & (left <= hi)).any()]
    splits = [
        ("Flick vs micro by flick size (medians, ms; landed_pct = flick ended on the target; err_pct = "
         "what it left as % of the flick; left_r = same in target radii; micro_corr/touches = per micro)",
         ["dist"] + head, by_dist),
        ("Micro by what the flick left (radii from the centre when the flick ended: slow micro even "
         "after landed flicks is the micro itself, slow only after big misses is the flick; "
         "p25_ms = your best quarter for that leftover, the bar each micro is held to)",
         ["left", "n", "micro_ms", "p25_ms", "corr", "touches", "total_s"], by_left),
    ]
    metrics = {"flick_ms": float(np.median(fl)), "micro_ms": float(np.median(mi)),
               "flick_on_pct": 100 * float(landed.mean()), "flick_err_pct": float(np.median(err)),
               "micro_corr": mean_or0(corr, had), "flick_saved_s": flick_saved, "micro_saved_s": micro_saved,
               "flick_cost_s": flick_cost}
    return {"summary": summary, "splits": splits, "metrics": metrics}


BUDGET_PARTS = [("react", "reaction"), ("travel", "travel to first on target"),
                ("settle", "off again before settling (bounces, overflick)"),
                ("slip", "slips after settling"), ("on", "on target")]


def time_budget(table: list[dict], continuous: bool) -> tuple[list[str], list[list], dict]:
    """Where the run's kill time went. saved_s = time back if every kill's part were no longer
    than your own 25th percentile for it this run; kills_back = that over the mean time per kill.
    Reaction and travel grow with flick distance, so those are held to the 25th percentile of
    kills in the same distance band (when the band has 8+ kills). A rough what-if: it assumes the
    parts are independent and time saved turns into kills."""
    ttk = np.array([r["ttk_ms"] for r in table])
    total = float(ttk.sum()) or 1.0
    mean_ttk = float(ttk.mean()) if len(ttk) else float("nan")
    band = np.array([next(i for i, (_l, lo, hi) in enumerate(DIST_BINS) if lo <= r["dist_deg"] < hi)
                     if r["dist_deg"] >= 0 else 0 for r in table])
    rows, out = [], {}
    for key, label in BUDGET_PARTS:
        x = np.array([r["_budget"][key] for r in table])
        p25 = float(np.percentile(x, 25))
        bar = np.full(len(x), p25)
        if key in ("react", "travel"):
            for b in np.unique(band):
                m = band == b
                if m.sum() >= 8:
                    bar[m] = np.percentile(x[m], 25)
        saved = float(np.maximum(x - bar, 0).sum()) / 1000.0
        if key == "on" and continuous:
            label += " (set by target health)"
            saved = float("nan")       # can't be cut: the target needs that long under fire
        rows.append([label, x.sum() / 1000.0, 100 * x.sum() / total, float(np.median(x)), p25,
                     saved, saved * 1000.0 / mean_ttk])
        out[key] = (100 * x.sum() / total, saved)
    wait = sum(r["_wait_ms"] for r in table) / 1000.0
    if wait > 0.05:
        rows.append(["waiting for the next target to spawn", wait, float("nan"), float("nan"),
                     float("nan"), float("nan"), float("nan")])
    return ["part", "total_s", "pct", "median_ms", "p25_ms", "saved_s", "kills_back"], rows, out


def flick_gain(table: list[dict]) -> dict | None:
    """Reach M (how far the first movement went along the flick line) against distance D.
    gain = median M/D; slope/offset_deg = least-squares M = slope*D + offset; spread_pct =
    robust SD of M/D in %; side_deg = robust SD of the sideways miss."""
    pts = [(r["_D"], r["_M"], r["_lat"]) for r in table if "_D" in r and r["_D"] >= 3.0]
    if len(pts) < 8:
        return None
    D, M, L = (np.array(c) for c in zip(*pts))
    ratio = M / D
    slope, offset = np.polyfit(D, M, 1)
    mad = 1.4826 * float(np.median(np.abs(ratio - np.median(ratio))))
    return {"n": len(D), "gain": float(np.median(ratio)), "slope": float(slope), "offset_deg": float(offset),
            "spread_pct": 100 * mad, "side_deg": 1.4826 * float(np.median(np.abs(L - np.median(L))))}


def _settle_index(on: np.ndarray, tt: np.ndarray) -> int | None:
    """First frame at which the crosshair has been on the target for SETTLE_MS without a break."""
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if tt[j - 1] - tt[i] >= SETTLE_MS:
                return i + int(np.searchsorted(tt[i:j], tt[i] + SETTLE_MS))
            i = j
        else:
            i += 1
    return None


def _off_episodes(on: np.ndarray, miss_r: np.ndarray, tt: np.ndarray) -> list[tuple[float, float]]:
    """After settling on the target: every stretch off it that came back before the kill,
    as (duration ms, furthest outside the target edge in radii). Leaving for good at the end
    (the target died as you left) is not included."""
    s = _settle_index(on, tt)
    if s is None:
        return []
    out = []
    i = s
    while i < len(on):
        if on[i]:
            i += 1
            continue
        j = i
        while j < len(on) and not on[j]:
            j += 1
        if j == len(on):
            break
        # off from the frame it left to the frame it came back
        out.append((float(tt[j] - tt[i - 1]), float(miss_r[i:j].max() - 1.0)))
        i = j
    return out


def _overflick(rem_r: np.ndarray, on: np.ndarray, tt: np.ndarray) -> tuple[float, float]:
    """From peak flick speed onward: (furthest past the target centre along the flick line in
    radii, floored at 0; ms spent beyond the far edge). Stops once the crosshair has stayed on
    the target for SETTLE_MS, so passing straight through the target still counts."""
    s = _settle_index(on, tt)
    end = len(on) if s is None else s + 1
    past = -rem_r[:end]
    dt = np.diff(tt[:end], append=tt[end - 1] if end else 0.0)
    return float(max(past.max(), 0.0)) if end else 0.0, float(dt[past > 1.0].sum())


def _brake(sp: np.ndarray, tt: np.ndarray, onset: int, q: int) -> float:
    """Peak deceleration over peak acceleration of the flick from `onset` to the end of its
    first movement `q` (+ BRAKE_TAIL_MS). nan when either side is too short to tell."""
    s = smooth(sp, BRAKE_SMOOTH)
    p = int(np.argmax(s[:q + 1])) if q >= 0 else 0
    end = min(len(s) - 1, int(np.searchsorted(tt, tt[q] + BRAKE_TAIL_MS)))
    if p - onset < 3 or end - p < 3:
        return float("nan")
    acc = np.gradient(s, tt / 1000.0)
    push = float(acc[onset:p + 1].max())
    stop = float(-acc[p:end + 1].min())
    return stop / push if push > 0 else float("nan")


def _wobble(rem_r: np.ndarray, lat_r: np.ndarray, tt: np.ndarray, q: int) -> float:
    """Path of the error against the target in the WOBBLE_MS after the flick ended, minus the
    straight-line change over it, in radii. nan when the kill came before the window was up."""
    if tt[-1] < tt[q] + WOBBLE_MS:
        return float("nan")
    m = (tt >= tt[q]) & (tt <= tt[q] + WOBBLE_MS)
    x, y = smooth(rem_r, WOBBLE_SMOOTH)[m], smooth(lat_r, WOBBLE_SMOOTH)[m]
    if len(x) < 3:
        return float("nan")
    path = float(np.hypot(np.diff(x), np.diff(y)).sum())
    return path - float(np.hypot(x[-1] - x[0], y[-1] - y[0]))


def _count_peaks(sp: np.ndarray, tt: np.ndarray, threshold: float) -> int:
    """Separate speed bumps above threshold, at least 30 ms apart."""
    count, last_t = 0, -1e9
    for i in range(1, len(sp) - 1):
        if sp[i] > threshold and sp[i] >= sp[i - 1] and sp[i] > sp[i + 1] and tt[i] - last_t >= 30.0:
            count += 1
            last_t = tt[i]
    return count


def _shot_scatter(run: Run, shots: np.ndarray) -> dict:
    """A kill removes its target in the frame the shot registers, so each shot is compared
    against targets present in that frame and the one before, using the shot frame's camera."""
    bounds = np.searchsorted(run.tgt_frame, np.arange(len(run.t) + 1))
    xs, ys, hit = [], [], []
    for f in shots:
        rows = np.arange(bounds[max(f - 1, 0)], bounds[f + 1])
        if len(rows) == 0:
            continue
        vw = view(run, rows, np.full(len(rows), f))
        j = int(np.argmin(vw.miss - vw.arad))
        xs.append(-vw.ex[j] / vw.arad[j])
        ys.append(-vw.ey[j] / vw.arad[j])
        hit.append(vw.miss[j] <= vw.arad[j])
    return {"x": np.array(xs), "y": np.array(ys), "hit": np.array(hit, dtype=bool)}
