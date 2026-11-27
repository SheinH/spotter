"""Live sensitivity from v3 logs (KovaaK's sens randomizer changes it during a run).

The hook logs two engine values per frame, in the same units:
  user_sens  = AMetaPlayerController::UserYawSensitivity, the configured sensitivity
               (equals the stats CSV's "Sens Increment")
  yaw_scale  = APlayerController::InputYawScale, what the engine applies this frame
Degrees per mouse count = value * TURN_AXIS_SCALE. Distance per 360 is the reciprocal: a larger
value turns further per count, so it takes fewer cm to do a 360.
"""
from __future__ import annotations

import numpy as np

from .format import Run

TURN_AXIS_SCALE = 0.07   # deg per count per unit of sens; matches "Sens Increment" to the stats cm/360
CHANGE_TOL = 0.005       # relative; smaller wobbles are float noise, not a sens change


def configured_cm360(fields: dict | None) -> float | None:
    """cm/360 from the stats CSV, or None if unknown."""
    if not fields:
        return None
    try:
        if fields.get("Sens Scale", "").strip().lower() == "cm/360":
            return float(fields["Horiz Sens"])
        inc, dpi = float(fields["Sens Increment"]), float(fields["DPI"])
    except (KeyError, ValueError):
        return None
    return 360.0 / (inc * TURN_AXIS_SCALE) / dpi * 2.54 if inc > 0 and dpi > 0 else None


def multiplier(run: Run) -> np.ndarray:
    """Per frame: live cm/360 / configured cm/360 (NaN where not logged). >1 = slower sens."""
    return run.user_sens / run.yaw_scale


def cm360(run: Run, fields: dict | None) -> np.ndarray | None:
    """Per-frame live cm/360, or None when the log has no sens fields or the base is unknown.
    Frames with a missing reading are filled from the nearest logged frame."""
    m = multiplier(run)
    ok = np.isfinite(m)
    base = configured_cm360(fields)
    if base is None or not ok.any():
        return None
    if not ok.all():
        idx = np.arange(len(m))
        m = np.interp(idx, idx[ok], m[ok])
    return base * m


def analyze(run: Run, fields: dict | None) -> dict | None:
    """Summary of the live sensitivity over a run, or None for logs without it."""
    m = multiplier(run)
    ok = np.isfinite(m)
    if not ok.any():
        return None
    base = configured_cm360(fields)
    t, m = run.t[ok], m[ok]
    lo, hi, med = float(m.min()), float(m.max()), float(np.median(m))
    varies = hi / lo - 1 > CHANGE_TOL
    off_setting = abs(med - 1) > CHANGE_TOL
    # Count changes: steps between consecutive frames bigger than the tolerance, or for a
    # smooth drift, the share of the run spent away from the run's median.
    steps = int(np.sum(np.abs(np.diff(np.log(m))) > CHANGE_TOL))
    thirds = [float(np.median(x)) for x in np.array_split(m, 3)] if len(m) >= 3 else [med] * 3
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    out = {"randomized": bool(varies or off_setting), "varies": bool(varies), "mult_median": med,
           "mult_min": lo, "mult_max": hi, "steps": steps, "thirds": thirds, "base_cm360": base,
           "coverage": float(ok.mean()), "seconds": dur}
    unit = "cm/360" if base else "x setting"
    f = (lambda x: x * base) if base else (lambda x: x)
    out["cm360_median"] = f(med) if base else None
    if not out["randomized"]:
        line = (f"sensitivity steady at the configured {base:g} cm/360 all run" if base
                else "sensitivity steady at the configured value all run")
    else:
        line = ("sens randomizer ON: " if varies else "sensitivity differs from the configured value: ")
        line += (f"live sens {f(lo):.1f}-{f(hi):.1f} {unit}, median {f(med):.1f}"
                 + (f" (configured {base:g} cm/360)" if base else "")
                 + (f"; by third {f(thirds[0]):.1f} / {f(thirds[1]):.1f} / {f(thirds[2]):.1f}" if varies else "")
                 + (f"; {steps} step changes" if steps else ""))
    out["summary"] = [line]
    if out["coverage"] < 0.95:
        out["summary"].append(f"sensitivity readable on only {out['coverage']:.0%} of frames")
    return out


def attach(result: dict, run: Run, fields: dict | None) -> dict | None:
    """Add the live-sensitivity summary to an analyzer result (first summary lines + result["sens"])."""
    sv = analyze(run, fields)
    if sv:
        result["sens"] = sv
        result["summary"] = sv["summary"] + result["summary"]
    return sv
