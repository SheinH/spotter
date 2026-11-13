"""Tension proxies from hand motion.

Tension is not measured directly; these are its visible side effects in the crosshair path:

- jitter: 6-13 Hz hand velocity, where physiological tremor lives. Co-contraction and fatigue
  raise it. It also grows with how fast the hand moves (motor noise scales with force), so it
  is reported per hand-speed band rather than as one number.
- tremor peak: whether that band stands out of the spectrum as a hump (a rhythmic
  oscillation) or just continues the smooth fall-off of voluntary movement.
- build-up: jitter by third of the run (tension budget running down).
- flicks (clicking/switching only): braking shape and jitter right after the hand stops.
  Stopping a flick by clamping down makes the hand bounce back and shortens the
  deceleration; letting the flick coast in leaves a longer, gentler deceleration.

Speeds are in mm/s of mouse movement when the sensitivity is known, so they compare across
sensitivities, else in deg/s.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .format import Run

FS = 500.0
BAND = (6.0, 13.0)       # Hz, jitter band
FLOOR = (15.0, 30.0)     # Hz, reference band above it
FLICK_MIN = 100.0        # deg/s; peak speed that counts as a flick


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
    lo = signal.butter(4, 4.0, fs=FS, output="sos")
    speed = np.hypot(signal.sosfiltfilt(lo, vx), signal.sosfiltfilt(lo, vy))
    # Pauses and hands off the mouse would drag every median down; keep active time only.
    active = np.convolve(np.hypot(vx, vy), np.ones(int(FS)) / FS, mode="same") > (5.0 if mmpd else 6.0)

    f, pxx = signal.welch(vx[active], FS, nperseg=int(2 * FS))
    _, pyy = signal.welch(vy[active], FS, nperseg=int(2 * FS))
    p = pxx + pyy
    inband = (f >= BAND[0]) & (f <= BAND[1])
    floor = (f >= FLOOR[0]) & (f <= FLOOR[1])
    shoulder = (f >= 3.0) & (f < 5.0)
    # Excess of the band over a straight line (in log-log) from 3-5 Hz to 15-30 Hz: a smooth
    # fall-off gives ~0 dB, a tremor hump gives a positive number.
    lf = np.log10(f[1:]); lp = np.log10(p[1:])
    xs = np.r_[lf[shoulder[1:]], lf[floor[1:]]]; ys = np.r_[lp[shoulder[1:]], lp[floor[1:]]]
    k, c = np.polyfit(xs, ys, 1)
    excess_db = 10 * float(np.mean(lp[inband[1:]] - (k * lf[inband[1:]] + c)))
    peak_hz = float(f[inband][np.argmax(p[inband])])

    ja = jit[active]
    thirds = [float(np.median(x)) for x in np.array_split(ja, 3)]
    summary = [
        f"jitter (6-13 Hz hand velocity) median {np.median(ja):.1f} {unit}, 90th pct {np.percentile(ja, 90):.1f}; "
        f"by third {thirds[0]:.1f} / {thirds[1]:.1f} / {thirds[2]:.1f} ({100 * (thirds[2] / thirds[0] - 1):+.0f}% first to last)",
        f"tremor hump: 6-13 Hz sits {excess_db:+.1f} dB above the smooth fall-off of the spectrum, "
        f"strongest at {peak_hz:.0f} Hz (0 dB = no hump; a few dB = a visible rhythmic shake)",
    ]

    # jitter against hand speed: the slope is expected; outliers are not
    splits = []
    sa = speed[active]
    edges = np.quantile(sa, [0, 0.25, 0.5, 0.75, 1.0])
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (sa >= a) & (sa <= b)
        if m.sum() > FS:
            rows.append([f"{a:.0f}-{b:.0f}", float(np.median(ja[m])), 100 * float(np.median(ja[m])) / max(float(np.median(sa[m])), 1e-6)])
    splits.append((f"Jitter by hand speed ({unit})", ["hand_speed", "jitter", "jitter_pct_of_speed"], rows))

    out = {"summary": summary, "splits": splits, "jitter": float(np.median(ja)), "thirds": thirds,
           "excess_db": excess_db, "peak_hz": peak_hz, "unit": unit}
    if mode in ("clicking", "switching"):
        fl = flicks(tu, vx / scale, vy / scale)
        out["flicks"] = fl
        if len(fl) >= 6:
            out["summary"] += _flick_summary(fl)
            out["splits"].append(_flick_split(fl))
    return out


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
