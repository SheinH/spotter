"""Tension proxies from hand motion.

Tension is not measured directly; these are its visible side effects in the crosshair path:

- on-target jitter: wobble in the crosshair's *error against the target it is following*
  (target direction minus crosshair direction), so following a moving target perfectly reads
  zero at any speed. Only following time counts (after settling on the target until it dies;
  all target time in tracking), with slips kept in, and nothing within GUARD_MS of a flick or
  big correction. Following time is cut into pieces of 120-400 ms; in each piece a quadratic
  (the smooth part of following) is removed and what is left is smoothed to about 13 Hz and
  differentiated. Measuring each piece on its own matters: a band filter run over the whole
  log rings for 100+ ms after every correction, and in switching holds (~190 ms) that ringing
  was most of what it measured (4.3 deg/s with no tremor planted, in synth tests).
  Pieces are grouped by target speed and time since the target last reversed; the run's
  number is the equal-weight mean of the groups with enough data, so faster targets or more
  reversals don't make a run look shakier by themselves. A 90% range comes from resampling
  the pieces. Not yet validated against a known relaxed-vs-tense difference, so no
  better/worse verdict.
  Limits: switching gives only ~1.5-2.5 s of usable hold per minute. The sensor counts in
  ~8 um steps and pad friction holds the mouse still through small forces, so fine tremor
  mostly never reaches the log: a 20 um tremor reads ~0.7 mm/s, while real holds read 3-5,
  so what this sees is mostly small corrections and the mouse breaking free of friction.
- tremor peak: whether 6-13 Hz stands out of the spectrum as a hump. Only called a tremor
  when the hump peaks inside 7-12 Hz; a maximum at the band edge is the smooth fall-off of
  voluntary movement, not a shake.
- hand activity: 6-13 Hz of raw hand velocity over all active time (the old "jitter"). It
  mostly measures how busy and sharp the movement is, flicks included, so it is context only.
- flicks (clicking/switching only): braking shape and jitter right after the hand stops.
  Stopping a flick by clamping down makes the hand bounce back and shortens the
  deceleration; letting the flick coast in leaves a longer, gentler deceleration.

Speeds are in mm/s of mouse movement when the sensitivity is known, so they compare across
sensitivities, else in deg/s.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import clicking
from .format import Run
from .geometry import nearest, runs, smooth, target_view

FS = 500.0
BAND = (6.0, 13.0)       # Hz, jitter band
FLOOR = (15.0, 30.0)     # Hz, reference band above it
TREMOR = (7.0, 12.0)     # Hz; a hump peaking here is called a tremor
FLICK_MIN = 100.0        # deg/s; peak speed that counts as a flick
GUARD_MS = 30.0          # leave out following time this close to fast error movement (each piece is
                         # measured on its own, so this only has to clear a correction's tail)
FAST_DPS = 20.0          # error speed (smoothed) that counts as a flick or big correction
PIECE_MS = 400.0         # following time is measured in pieces of at most this long...
MIN_PIECE_MS = 120.0     # ...and at least this long
SMOOTH_N = 19            # samples at FS (38 ms Hann): keeps roughly what is below 13 Hz
BOOTSTRAP = 300
MIN_GROUP_S = 0.5        # a group needs this much following time to count in the run's number
MIN_SPECTRUM_S = 4.0     # unbroken following time needed to look for a tremor peak
TSPEED_EDGES = [0.0, 10.0, 30.0, np.inf]           # target deg/s
TSPEED_LABELS = ["<10", "10-30", "30+"]
SINCE_SPLIT_MS = 300.0   # "just reversed" vs "steady"
REV_WIN = 0.15           # s; a reversal compares target velocity this long before and after
REV_MIN_DPS = 6.0        # target speed both sides of a reversal


def mm_per_deg(fields: dict | None) -> float | None:
    """Mouse millimetres per degree of yaw, from the stats CSV, or None if unknown."""
    if not fields or fields.get("Sens Scale", "").strip().lower() != "cm/360":
        return None
    try:
        return float(fields["Horiz Sens"]) * 10.0 / 360.0
    except (KeyError, ValueError):
        return None


def _sos(lo: float, hi: float):
    return signal.butter(4, [lo, hi], btype="band", fs=FS, output="sos")


def _rms_env(x: np.ndarray, width_s: float) -> np.ndarray:
    n = max(1, int(width_s * FS))
    return np.sqrt(np.convolve(x ** 2, np.ones(n) / n, mode="same"))


def hand(run: Run, scale: float):
    """Uniform-grid hand velocity (vx, vy) in output units/s, plus the grid."""
    tu = np.arange(run.t[0], run.t[-1], 1.0 / FS)
    yaw = np.interp(tu, run.t, np.unwrap(run.yaw, period=360))
    pitch = np.interp(tu, run.t, run.pitch)
    return tu, np.gradient(yaw, 1 / FS) * scale, np.gradient(pitch, 1 / FS) * scale


def analyze(run: Run, mode: str, mmpd: float | None) -> dict:
    scale = mmpd or 1.0
    unit = "mm/s" if mmpd else "deg/s"
    tu, vx, vy = hand(run, scale)
    if len(tu) < 4 * FS:
        return {"summary": [], "splits": []}
    sos = _sos(*BAND)
    jx, jy = signal.sosfiltfilt(sos, vx), signal.sosfiltfilt(sos, vy)
    jit = _rms_env(np.hypot(jx, jy), 0.25)
    # Pauses and hands off the mouse would drag every median down; keep active time only.
    active = np.convolve(np.hypot(vx, vy), np.ones(int(FS)) / FS, mode="same") > (5.0 if mmpd else 6.0)

    ja = jit[active]
    thirds = [float(np.median(x)) for x in np.array_split(ja, 3)]
    hold = hold_jitter(run, mode)
    summary, splits = [], []
    if hold["seconds"] > 0:
        groups = hold["groups"]
        counted = [g for g in groups if g[2] >= MIN_GROUP_S]
        lo90, hi90 = hold["range90"]
        summary.append(
            f"on-target jitter (wobble in your error against the target while following it, flicks and "
            f"corrections left out): "
            + (f"{hold['value'] * scale:.2f} {unit} (90% range {lo90 * scale:.2f}-{hi90 * scale:.2f})"
               if counted else "too little following time in any group")
            + f", from {hold['seconds']:.1f}s of following time in {len(counted)} of {len(groups)} groups "
            f"(equal-weight mean of groups with {MIN_GROUP_S:.1f}s+). Mostly small corrections and the "
            f"mouse breaking free of pad friction, not tremor; no better/worse verdict until it is checked "
            f"against a relaxed-vs-tense test")
        if hold["thirds"]:
            t3 = [x * scale for x in hold["thirds"]]
            summary.append(f"on-target jitter by third of the run (raw, not grouped): "
                           f"{t3[0]:.2f} / {t3[1]:.2f} / {t3[2]:.2f} {unit}")
        splits.append((f"On-target jitter by target speed and time since its last reversal ({unit})",
                       ["target_dps", "since_reversal", "seconds", "jitter", "counted"],
                       [[s, r, sec, val * scale, "yes" if sec >= MIN_GROUP_S else "no"] for s, r, sec, val in groups]))
    else:
        summary.append("on-target jitter: no following time left after leaving out flicks and corrections")
    summary.append(_tremor_line(hold["spectrum"], hold["spectrum_s"]))
    summary.append(
        f"hand activity (6-13 Hz of raw hand velocity, flicks included; context, not tension) median "
        f"{np.median(ja):.1f} {unit}; by third {thirds[0]:.1f} / {thirds[1]:.1f} / {thirds[2]:.1f}")

    out = {"summary": summary, "splits": splits, "activity": float(np.median(ja)), "thirds": thirds,
           "hold_jitter": hold["value"] * scale if hold["value"] == hold["value"] else float("nan"),
           "hold_seconds": hold["seconds"],
           "hold_range90": tuple(x * scale for x in hold.get("range90", (float("nan"),) * 2)), "hold_thirds": [x * scale for x in hold["thirds"]],
           "tremor": hold["tremor"], "unit": unit}
    if mode in ("clicking", "switching"):
        fl = flicks(tu, vx / scale, vy / scale)
        out["flicks"] = fl
        if len(fl) >= 6:
            out["summary"] += _flick_summary(fl)
            out["splits"].append(_flick_split(fl))
    return out


def _following(run: Run, mode: str, v) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: the target row being followed (-1 = none) and whether this frame counts as
    following it. Switching/clicking: the target of each kill window, following from settling
    on it (SETTLE_MS on target) until the kill. Tracking: the nearest target, all the time."""
    n = len(run.t)
    if mode == "tracking":
        eng = nearest(run, v)
        return eng, eng >= 0
    eng = np.full(n, -1, dtype=np.int64)
    follow = np.zeros(n, dtype=bool)
    r = run
    prev = 0
    for k, tid, f0 in clicking.kills(r, clicking.shot_frames(r)):
        s, prev = max(prev, f0), k
        rows = np.flatnonzero((r.tgt_id == tid) & (r.tgt_frame >= s) & (r.tgt_frame < k))
        if len(rows) < 3:
            continue
        fr = r.tgt_frame[rows]
        eng[fr] = rows
        on = v.miss[rows] <= v.arad[rows]
        si = clicking._settle_index(on, (r.t[fr] - r.t[fr[0]]) * 1000.0)
        if si is not None:
            while si > 0 and on[si - 1]:          # the settled stretch counts from its start
                si -= 1
            follow[fr[si:]] = True
    return eng, follow


def _dilate(mask: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(mask.astype(float), np.ones(2 * n + 1), mode="same") > 0


def hold_jitter(run: Run, mode: str) -> dict:
    """On-target jitter in deg/s (see the module docstring)."""
    empty = {"value": float("nan"), "seconds": 0.0, "groups": [], "thirds": [],
             "spectrum": None, "spectrum_s": 0.0, "tremor": None}
    if len(run.tgt_frame) < 100:
        return empty
    run = clicking.split_respawns(run)        # a teleported respawn is a new target
    v = target_view(run)
    eng, follow = _following(run, mode, v)
    has = eng >= 0
    if has.sum() < 100:
        return empty
    tu = np.arange(run.t[0], run.t[-1], 1.0 / FS)
    n = len(tu)
    idx = np.clip(np.searchsorted(run.t, tu, side="right") - 1, 0, len(run.t) - 1)
    th, rows = run.t[has], eng[has]
    ex = np.interp(tu, th, v.ex[rows])
    ey = np.interp(tu, th, v.ey[rows])
    tyaw = np.interp(tu, th, np.unwrap(v.tyaw[rows], period=360))
    tpitch = np.interp(tu, th, v.tpitch[rows])
    tid = np.where(has, run.tgt_id[np.maximum(eng, 0)], -1)[idx]
    ok = has[idx]
    change = np.r_[True, tid[1:] != tid[:-1]]

    # error velocity: target direction minus crosshair direction, differentiated
    evx, evy = np.gradient(ex, 1 / FS), np.gradient(ey, 1 / FS)
    bad = ~ok | _dilate(change, 2)
    evx[bad], evy[bad] = 0.0, 0.0
    fast = np.hypot(smooth(evx, 9), smooth(evy, 9)) > FAST_DPS
    use = follow[idx] & ~bad & ~_dilate(fast, int(GUARD_MS / 1000 * FS))
    pieces = _pieces(use)
    if not pieces:
        return empty

    # target speed and time since its last reversal, within one target's block
    tvx = np.gradient(smooth(tyaw, 5), 1 / FS) * np.cos(np.radians(tpitch))
    tvy = np.gradient(smooth(tpitch, 5), 1 / FS)
    tsp = np.hypot(tvx, tvy)
    k = int(REV_WIN * FS)
    rev = np.zeros(n, dtype=bool)
    if n > 2 * k:
        a, b = slice(0, n - 2 * k), slice(2 * k, n)
        same = tid[a] == tid[b]
        rev[k:n - k] = same & (tvx[a] * tvx[b] + tvy[a] * tvy[b] < 0) & (tsp[a] > REV_MIN_DPS) & (tsp[b] > REV_MIN_DPS)
    marks = np.zeros(n, dtype=bool)
    for a0, b0 in runs(rev):
        marks[(a0 + b0) // 2] = True
    last = np.maximum.accumulate(np.where(marks | change, np.arange(n), 0))
    since_ms = (np.arange(n) - last) / FS * 1000.0
    just = marks[last] & (since_ms < SINCE_SPLIT_MS)     # last event was a reversal, recently

    # each piece: wobble above its own smooth trend, and the group it belongs to
    scode = np.digitize(tsp, TSPEED_EDGES[1:-1])
    labels = [(s, j) for s in range(len(TSPEED_LABELS)) for j in (True, False)]
    ss, cnt, grp = [], [], []
    for a, b in pieces:
        s2, c = _piece_wobble(ex[a:b], ey[a:b])
        mid = (a + b) // 2
        ss.append(s2); cnt.append(c); grp.append(labels.index((int(scode[mid]), bool(just[mid]))))
    ss, cnt, grp = np.array(ss), np.array(cnt), np.array(grp)

    def combine(sel: np.ndarray) -> float:
        vals = []
        for g in range(len(labels)):
            m = sel[grp[sel] == g] if len(sel) else sel
            if cnt[m].sum() / FS >= MIN_GROUP_S:
                vals.append(np.sqrt(ss[m].sum() / cnt[m].sum()))
        return float(np.mean(vals)) if vals else float("nan")

    allp = np.arange(len(pieces))
    value = combine(allp)
    # 90% range by resampling pieces: how much the number would move with other holds
    rng = np.random.default_rng(0)
    boots = [combine(rng.choice(allp, len(allp))) for _ in range(BOOTSTRAP)]
    boots = [x for x in boots if x == x]
    lo90, hi90 = (np.percentile(boots, [5, 95]) if boots else (float("nan"), float("nan")))

    groups = []
    for g, (si, jflag) in enumerate(labels):
        m = grp == g
        if m.any():
            groups.append([TSPEED_LABELS[si], f"0-{SINCE_SPLIT_MS:.0f} ms" if jflag else f"{SINCE_SPLIT_MS:.0f}+ ms or none",
                           float(cnt[m].sum() / FS), float(np.sqrt(ss[m].sum() / cnt[m].sum()))])
    thirds = [float(np.sqrt(ss[p].sum() / cnt[p].sum())) for p in np.array_split(allp, 3)] if len(allp) >= 3 else []

    # spectrum of the error velocity over unbroken following stretches, to look for a tremor peak
    seg = int(FS / 2)                                   # 0.5 s pieces: 2 Hz resolution
    long_runs = [(a0, b0) for a0, b0 in runs(use) if b0 - a0 >= seg]
    spec, spec_s, tremor = None, 0.0, None
    if long_runs:
        acc = None
        for a0, b0 in long_runs:
            for c in range(a0, b0 - seg + 1, seg // 2):
                f, px = signal.welch(evx[c:c + seg], FS, nperseg=seg, detrend="linear")
                _, py = signal.welch(evy[c:c + seg], FS, nperseg=seg, detrend="linear")
                acc = px + py if acc is None else acc + px + py
        spec_s = float(sum(b0 - a0 for a0, b0 in long_runs) / FS)
        spec = (f, acc)
        if spec_s >= MIN_SPECTRUM_S:
            tremor = _band_excess(f, acc)
    return {"value": value, "range90": (float(lo90), float(hi90)), "seconds": float(cnt.sum() / FS),
            "groups": groups, "thirds": thirds, "spectrum": spec, "spectrum_s": spec_s, "tremor": tremor}


def _pieces(use: np.ndarray) -> list[tuple[int, int]]:
    """Following stretches cut into pieces of at most PIECE_MS, each at least MIN_PIECE_MS, so
    every piece has the same trend removal whatever the scenario's hold length."""
    out = []
    lmax, lmin = int(PIECE_MS / 1000 * FS), int(MIN_PIECE_MS / 1000 * FS)
    for a, b in runs(use):
        k = max(1, int(np.ceil((b - a) / lmax)))
        edges = np.linspace(a, b, k + 1).astype(int)
        out += [(int(x), int(y)) for x, y in zip(edges[:-1], edges[1:]) if y - x >= lmin]
    return out


def _piece_wobble(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Sum of squared wobble velocity (deg/s) and sample count for one piece: the error minus a
    quadratic fitted to this piece (the smooth part of following), smoothed to about 13 Hz,
    differentiated, with the smoothing's unreliable ends dropped."""
    t = np.arange(len(x)) / FS
    k = SMOOTH_N // 2
    tot = 0.0
    for z in (x, y):
        r = z - np.polyval(np.polyfit(t, z, 2), t)
        dv = np.gradient(smooth(r, SMOOTH_N), 1 / FS)[k:-k]
        tot += float((dv ** 2).sum())
    return tot, len(x) - 2 * k


def _band_excess(f: np.ndarray, p: np.ndarray) -> dict:
    """How far 6-13 Hz sits above a straight line (log-log) from 3-5 Hz to 15-30 Hz, and where
    the band peaks above that line."""
    inband = (f >= BAND[0]) & (f <= BAND[1])
    ref = ((f >= 3.0) & (f < 5.0)) | ((f >= FLOOR[0]) & (f <= FLOOR[1]))
    good = f > 0
    lf, lp = np.log10(f[good]), np.log10(np.maximum(p[good], 1e-30))
    kk, cc = np.polyfit(lf[ref[good]], lp[ref[good]], 1)
    resid = lp - (kk * lf + cc)
    ib = inband[good]
    peak_hz = float(f[good][ib][np.argmax(resid[ib])])
    return {"excess_db": 10 * float(np.mean(resid[ib])), "peak_hz": peak_hz,
            "peak_db": 10 * float(resid[ib].max())}


def _tremor_line(spectrum, seconds: float) -> str:
    if spectrum is None or seconds < MIN_SPECTRUM_S:
        return (f"tremor peak: not tested ({seconds:.1f}s of unbroken 0.5 s+ following stretches; "
                f"needs {MIN_SPECTRUM_S:.0f}s)")
    t = _band_excess(*spectrum)
    if TREMOR[0] <= t["peak_hz"] <= TREMOR[1] and t["peak_db"] >= 1.0:
        return (f"tremor peak: yes, at {t['peak_hz']:.0f} Hz, {t['peak_db']:+.1f} dB above the smooth "
                f"fall-off of the error spectrum (from {seconds:.0f}s of following)")
    return (f"tremor peak: none (6-13 Hz peaks at {t['peak_hz']:.0f} Hz, {t['peak_db']:+.1f} dB; a tremor "
            f"would peak inside {TREMOR[0]:.0f}-{TREMOR[1]:.0f} Hz) from {seconds:.0f}s of following")


FLICK_COLS = ["peak", "dist", "accel_ms", "decel_ms", "rebound_pct", "t"]


def flicks(tu, vx, vy) -> np.ndarray:
    """One row per flick (columns FLICK_COLS), velocities in deg/s.

    rebound_pct: fastest motion back against the flick within 100 ms of stopping, as % of
    the flick's peak speed. A flick stopped by clamping down bounces; one that coasts in
    does not. Deliberate corrections come later (they need a reaction time) and mostly
    continue the flick's direction, so they don't count here.
    (Jitter right after a stop is not reported: the band filter smears the correction
    movement into it, and a planted post-stop shake did not show through.)"""
    lo = signal.butter(4, 25.0, fs=FS, output="sos")
    fx, fy = signal.sosfiltfilt(lo, vx), signal.sosfiltfilt(lo, vy)
    sp = np.hypot(fx, fy)
    peaks, _ = signal.find_peaks(sp, height=FLICK_MIN, distance=int(0.15 * FS))
    yaw = np.cumsum(vx) / FS
    pitch = np.cumsum(vy) / FS
    rows = []
    n = len(sp)

    for p in peaks:
        thr = 0.1 * sp[p]
        a = p
        while a > 0 and sp[a] > thr:
            a -= 1
        b = p
        while b < n - 1 and sp[b] > thr:
            b += 1
        if a == 0 or b + int(0.1 * FS) >= n:
            continue
        dx, dy = yaw[b] - yaw[a], pitch[b] - pitch[a]
        dist = float(np.hypot(dx, dy))
        if dist < 1e-6:
            continue
        along = (fx[b:b + int(0.1 * FS)] * dx + fy[b:b + int(0.1 * FS)] * dy) / dist
        rebound = 100 * max(0.0, -float(along.min())) / sp[p]
        rows.append([sp[p], dist, (p - a) / FS * 1000, (b - p) / FS * 1000, rebound, tu[a] - tu[0]])
    return np.array(rows).reshape(-1, len(FLICK_COLS))


def _terciles(fl: np.ndarray):
    q = np.quantile(fl[:, 0], [1 / 3, 2 / 3])
    return [("slow", fl[:, 0] < q[0]), ("mid", (fl[:, 0] >= q[0]) & (fl[:, 0] < q[1])), ("fast", fl[:, 0] >= q[1])]


def _med(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else float("nan")


def _flick_split(fl: np.ndarray):
    rows = []
    for name, m in _terciles(fl):
        r = fl[m]
        rows.append([name, int(m.sum()), _med(r[:, 0]), _med(r[:, 1]), _med(r[:, 2]), _med(r[:, 3]),
                     _med(r[:, 3] / r[:, 2]), _med(r[:, 4])])
    return ("Flicks by peak speed (braking and shake after stopping)",
            ["flicks", "n", "peak_dps", "dist_deg", "accel_ms", "decel_ms", "decel/accel",
             "rebound_pct"],
            rows)


def _flick_summary(fl: np.ndarray) -> list[str]:
    t = _terciles(fl)
    slow, fast = fl[t[0][1]], fl[t[2][1]]
    thirds = np.array_split(fl, 3)
    return [
        f"flicks {len(fl)}: deceleration lasts {_med(fl[:, 3] / fl[:, 2]):.2f}x the acceleration "
        f"(slow third {_med(slow[:, 3] / slow[:, 2]):.2f}x, fast third {_med(fast[:, 3] / fast[:, 2]):.2f}x; "
        f"below 1 = braking harder than the push)",
        f"rebound against the flick within 100 ms of stopping: median {_med(fl[:, 4]):.1f}% of peak speed "
        f"(slow flicks {_med(slow[:, 4]):.1f}%, fast {_med(fast[:, 4]):.1f}%; by third of run "
        + " / ".join(f"{_med(x[:, 4]):.1f}" for x in thirds) + ")",
    ]
