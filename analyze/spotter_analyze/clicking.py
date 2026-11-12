"""Clicking and target-switching analysis: one row per kill ("engagement")."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .format import BTN_FIRE, Run
from .geometry import OCTANTS, octant, smooth, target_view, view

COLUMNS = ["i", "t_s", "dist_deg", "dir", "react_ms", "peak_dps", "land_r", "off_r", "corr",
           "acq_ms", "confirm_ms", "ttk_ms", "shots", "miss", "on_pct"]
LEGEND = (
    "One row per kill, timed from the previous kill (or the target's spawn if later). "
    "dist_deg=flick distance; dir=screen direction to the target; react_ms=start until the hand "
    "moves; peak_dps=peak crosshair speed; land_r=where the first movement ended along the flick "
    "line in target radii (- short, + past, |x|<1 on target); off_r=total distance from centre at "
    "that moment, radii; corr=corrective movements after it; acq_ms=start until first on target; "
    "confirm_ms=time on target before the killing shot; miss=missed shots with type "
    "(E fired before arriving, O past the target, D was on target then left); "
    "on_pct=share of time on target after first reaching it."
)


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
        row.update(react_ms=nan, peak_dps=nan, land_r=nan, off_r=nan, corr=0, acq_ms=nan,
                   confirm_ms=nan, on_pct=nan)

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
            row["corr"] = _count_peaks(sp[q:], tt[q:], max(0.12 * peak, 8.0))
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
        f"+ travel to first on-target at {med('acq_ms'):.0f} ms, confirm {med('confirm_ms'):.0f} ms before the shot",
        f"flicks: median distance {med('dist_deg'):.1f} deg, peak speed {med('peak_dps'):.0f} deg/s, "
        f"{np.mean(col('corr')):.2f} corrections per kill",
    ]
    if len(land):
        summary.append(
            f"first movement lands short on {100 * np.mean(land < -1):.0f}%, on target on "
            f"{100 * np.mean(np.abs(land) <= 1):.0f}%, past on {100 * np.mean(land > 1):.0f}% "
            f"(median {np.median(land):+.2f} radii)")
    if continuous:
        summary.append(f"after first reaching a target, on it {med('on_pct'):.0f}% of the time until it dies")

    # --- splits ---
    head = ["n", "ttk_ms", "react_ms", "land_r", "corr", "confirm_ms", "misses"]

    def agg(rows):
        return [len(rows), med("ttk_ms", rows), med("react_ms", rows), med("land_r", rows),
                float(np.mean(col("corr", rows))) if rows else float("nan"),
                med("confirm_ms", rows), sum(r["_miss_n"] for r in rows)]

    splits = []
    dists = np.array([r["dist_deg"] for r in table])
    if len(table) >= 9:
        q = np.quantile(dists, [1 / 3, 2 / 3])
        buckets = [(f"<{q[0]:.0f} deg", dists < q[0]), (f"{q[0]:.0f}-{q[1]:.0f} deg", (dists >= q[0]) & (dists < q[1])),
                   (f">{q[1]:.0f} deg", dists >= q[1])]
        splits.append(("By flick distance", ["dist"] + head,
                       [[name] + agg([r for r, m in zip(table, mask) if m]) for name, mask in buckets]))
    else:
        buckets = [("all", np.ones(len(table), dtype=bool))]
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

    return {
        "mode": mode, "summary": summary, "columns": COLUMNS, "legend": LEGEND,
        "rows": [[r[c] for c in COLUMNS] for r in table], "splits": splits, "worst": worst,
        "table": table, "kills": len(table), "shots": len(shots), "misses": misses,
        "kill_times": np.array([run.t[k] for k, _tid, _f0 in kill_list]),
        "plot": {"scatter": scat, "traces": traces, "dists": dists,
                 "buckets": [(name, np.flatnonzero(mask)) for name, mask in buckets]},
    }


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
