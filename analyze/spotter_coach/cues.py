"""Mid-run rolling 12s cue evaluator, programmable InRunPlan reminders, and anti-nag gate.

Supports:
1. User-requested timed reminders ("hey remind me every 20s to relax my grip")
2. Deep-reasoner / voice-configured custom metric rules (e.g. watch extra_deg > 6.5)
3. Baseline history.jsonl 1.5-SD statistical drift rules with 10s warmup and cooldowns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from spotter_analyze import clicking, tracking
from spotter_analyze.__main__ import pick_mode
from spotter_analyze.format import Run

WARMUP_S = 10.0
WINDOW_S = 12.0
COOLDOWN_S = 18.0
MIN_WINDOW_KILLS = 6
MIN_TRACK_S = 6.0


@dataclass
class TimedReminder:
    cue_text: str
    every_s: float = 22.0
    next_at_s: float = 14.0


@dataclass
class MetricRule:
    metric: str
    direction: str  # "above" | "below"
    threshold: float
    cue_text: str


@dataclass
class InRunPlan:
    muted_in_run: bool = False
    focus_cue: str = ""
    pinned_metric: str | None = None
    coach_directive: str = ""
    reminders: list[TimedReminder] = field(default_factory=list)
    custom_rules: list[MetricRule] = field(default_factory=list)

    def reset_timers(self) -> None:
        for i, rem in enumerate(self.reminders):
            rem.next_at_s = max(8.0, min(rem.every_s, 14.0 + i * 6.0))


@dataclass
class MidRunCue:
    run_elapsed_s: float
    metric: str
    value: float
    baseline: float | None
    hint: str
    prompt: str


class CueEngine:
    """Stateful mid-run cue detector for a single active run."""

    def __init__(self, history_records: list[dict], scenario_hint: str | None = None) -> None:
        self.history_records = history_records
        self.scenario_hint = scenario_hint
        self.plan = InRunPlan()
        self.last_cue_elapsed_s: float = -999.0
        self.fired_cues: list[MidRunCue] = []
        self._fired_metrics_count: dict[str, int] = {}

    def reset_for_run(self, scenario_hint: str | None = None) -> None:
        if scenario_hint is not None:
            self.scenario_hint = scenario_hint
        self.plan.reset_timers()
        self.last_cue_elapsed_s = -999.0
        self.fired_cues.clear()
        self._fired_metrics_count.clear()

    def _baseline(self, mode: str, metric: str) -> tuple[float, float] | None:
        if not self.scenario_hint:
            return None
        pool = [
            r["metrics"].get(metric)
            for r in self.history_records
            if r.get("scenario") == self.scenario_hint and r.get("mode") == mode
        ]
        vals = [float(v) for v in pool[-20:] if v is not None and not math.isnan(float(v))]
        if len(vals) < 3:
            return None
        return float(np.mean(vals)), max(float(np.std(vals, ddof=1)), 1e-6)

    def evaluate(self, window_run: Run, run_elapsed_s: float) -> MidRunCue | None:
        if self.plan.muted_in_run:
            return None

        # 1. Check user/reasoner timed reminders first (respecting a minimum 10s gap from any cue)
        for rem in self.plan.reminders:
            if run_elapsed_s >= rem.next_at_s and (run_elapsed_s - self.last_cue_elapsed_s) >= 10.0:
                rem.next_at_s = run_elapsed_s + max(rem.every_s, 12.0)
                directive = f" Style note: {self.plan.coach_directive}" if self.plan.coach_directive else ""
                prompt = (
                    f"[IN_RUN_CUE] Player is {run_elapsed_s:.0f}s into an active run. "
                    f"Deliver this requested reminder in <= 7 calm words: '{rem.cue_text}'.{directive} "
                    f"Do NOT call any tools."
                )
                cue = MidRunCue(
                    run_elapsed_s=run_elapsed_s,
                    metric="reminder",
                    value=0.0,
                    baseline=None,
                    hint=rem.cue_text,
                    prompt=prompt,
                )
                self.last_cue_elapsed_s = run_elapsed_s
                self.fired_cues.append(cue)
                return cue

        if run_elapsed_s < WARMUP_S:
            return None
        if (run_elapsed_s - self.last_cue_elapsed_s) < COOLDOWN_S:
            return None
        if len(window_run.t) < 120 or len(window_run.tgt_frame) < 60:
            return None

        mode = pick_mode(window_run, None)
        candidates: list[tuple[float, str, float, float | None, str]] = []

        if mode == "tracking":
            if (window_run.t[-1] - window_run.t[0]) < MIN_TRACK_S:
                return None
            try:
                res = tracking.analyze(window_run)
            except Exception:
                return None
            m = res["metrics"]
            self._eval_custom_rules(candidates, m)
            self._check_high(candidates, mode, "far_behind_pct", m.get("far_behind_pct", float("nan")),
                             abs_hi=28.0, min_delta=6.0, hint="Stay ahead of the strafe, accelerate sooner")
            self._check_high(candidates, mode, "react_ms", m.get("react_ms", float("nan")),
                             abs_hi=225.0, min_delta=22.0, hint="React sharper on direction changes")
            self._check_low(candidates, mode, "on_pct", m.get("on_pct", float("nan")),
                            abs_lo=45.0, min_delta=7.0, hint="Smooth out your tracking speed")
        else:
            try:
                res = clicking.analyze(window_run, mode)
            except Exception:
                return None
            if res["kills"] < MIN_WINDOW_KILLS:
                return None
            m = res["metrics"]
            self._eval_custom_rules(candidates, m)

            fg = m.get("flick_gain", float("nan"))
            if not math.isnan(fg):
                base_fg = self._baseline(mode, "flick_gain")
                if fg < 0.87 or (base_fg and fg < base_fg[0] - max(1.5 * base_fg[1], 0.06)):
                    candidates.append(((0.92 - fg) / 0.05, "flick_gain", fg,
                                       base_fg[0] if base_fg else None, "Commit further on initial flicks"))
                elif fg > 1.12 or (base_fg and fg > base_fg[0] + max(1.5 * base_fg[1], 0.06)):
                    candidates.append(((fg - 1.06) / 0.05, "flick_gain", fg,
                                       base_fg[0] if base_fg else None, "Dial back flick distance slightly"))

            self._check_high(candidates, mode, "over_pct", m.get("over_pct", float("nan")),
                             abs_hi=36.0, min_delta=10.0, hint="Brake earlier before target edge")
            if mode == "switching":
                self._check_high(candidates, mode, "slip_pct", m.get("slip_pct", float("nan")),
                                 abs_hi=32.0, min_delta=10.0, hint="Hold steady on target through the kill")
            else:
                self._check_high(candidates, mode, "short_pct", m.get("short_pct", float("nan")),
                                 abs_hi=58.0, min_delta=15.0, hint="Throw initial flick all the way onto target")
                self._check_high(candidates, mode, "miss_per_kill", m.get("miss_per_kill", float("nan")),
                                 abs_hi=0.45, min_delta=0.15, hint="Confirm crosshair on target before clicking")
            self._check_high(candidates, mode, "extra_deg", m.get("extra_deg", float("nan")),
                             abs_hi=8.5, min_delta=2.5, hint="Pick closer targets after each kill")

        if not candidates:
            return None

        scored: list[tuple[float, str, float, float | None, str]] = []
        for sev, metric, val, base, hint in candidates:
            if self._fired_metrics_count.get(metric, 0) >= 2:
                continue
            boost = 1.45 if (self.plan.pinned_metric and metric == self.plan.pinned_metric) else 1.0
            scored.append((sev * boost, metric, val, base, hint))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        _, metric, val, base, hint = scored[0]
        base_txt = f" (usual {base:.2f})" if base is not None else ""
        directive = f" Player/Coach directive: {self.plan.coach_directive}." if self.plan.coach_directive else ""
        prompt = (
            f"[IN_RUN_CUE] Player is {run_elapsed_s:.0f}s into an active {mode} run. "
            f"Rolling 12s {metric}={val:.2f}{base_txt}.{directive} "
            f"Speak ONE calm, punchy <= 7-word adjustment right now based on: '{hint}'. "
            f"Do NOT call any tools and do NOT explain numbers."
        )
        cue = MidRunCue(
            run_elapsed_s=run_elapsed_s,
            metric=metric,
            value=val,
            baseline=base,
            hint=hint,
            prompt=prompt,
        )
        self.last_cue_elapsed_s = run_elapsed_s
        self._fired_metrics_count[metric] = self._fired_metrics_count.get(metric, 0) + 1
        self.fired_cues.append(cue)
        return cue

    def _eval_custom_rules(self, out: list, metrics: dict[str, Any]) -> None:
        for rule in self.plan.custom_rules:
            v = metrics.get(rule.metric)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            vf = float(v)
            if rule.direction == "above" and vf > rule.threshold:
                out.append((3.0, rule.metric, vf, rule.threshold, rule.cue_text))
            elif rule.direction == "below" and vf < rule.threshold:
                out.append((3.0, rule.metric, vf, rule.threshold, rule.cue_text))

    def _check_high(self, out: list, mode: str, metric: str, val: float,
                    abs_hi: float, min_delta: float, hint: str) -> None:
        if math.isnan(val):
            return
        base = self._baseline(mode, metric)
        if base is not None:
            mean, sd = base
            thresh = mean + max(1.5 * sd, min_delta)
            if val > thresh:
                out.append(((val - mean) / max(sd, 1.0), metric, val, mean, hint))
        elif val > abs_hi:
            out.append(((val - abs_hi) / max(min_delta, 1.0) + 1.0, metric, val, None, hint))

    def _check_low(self, out: list, mode: str, metric: str, val: float,
                   abs_lo: float, min_delta: float, hint: str) -> None:
        if math.isnan(val):
            return
        base = self._baseline(mode, metric)
        if base is not None:
            mean, sd = base
            thresh = mean - max(1.5 * sd, min_delta)
            if val < thresh:
                out.append(((mean - val) / max(sd, 1.0), metric, val, mean, hint))
        elif val < abs_lo:
            out.append(((abs_lo - val) / max(min_delta, 1.0) + 1.0, metric, val, None, hint))


def evaluate_cue_effect(fired_cues: list[MidRunCue], result: dict[str, Any]) -> list[str]:
    """Check 'By third of the run' splits to see if mid-run cues helped in the final third."""
    if not fired_cues:
        return []
    third_split = next((s for s in result.get("splits", []) if "third" in s[0].lower()), None)
    if not third_split:
        return [f"Mid-run cue at {c.run_elapsed_s:.0f}s: '{c.hint}'" for c in fired_cues[:3]]
    _title, cols, rows = third_split
    if len(rows) < 3:
        return [f"Mid-run cue at {c.run_elapsed_s:.0f}s: '{c.hint}'" for c in fired_cues[:3]]

    col_map = {
        "flick_gain": "gain",
        "short_pct": "land_r",
        "over_pct": "over_pct",
        "slip_pct": "slip_pct",
        "react_ms": "react_ms",
        "on_pct": "on_pct",
        "far_behind_pct": "far_behind_pct",
    }
    notes: list[str] = []
    seen: set[str] = set()
    for cue in fired_cues:
        key = f"{cue.metric}:{cue.hint}"
        if key in seen:
            continue
        seen.add(key)
        cname = col_map.get(cue.metric)
        if not cname or cname not in cols:
            notes.append(f"Mid-run reminder at {cue.run_elapsed_s:.0f}s: '{cue.hint}'")
            continue
        idx = cols.index(cname)
        first_v, last_v = rows[0][idx], rows[-1][idx]
        if isinstance(first_v, (int, float)) and isinstance(last_v, (int, float)):
            if not (math.isnan(first_v) or math.isnan(last_v)):
                notes.append(
                    f"Mid-run cue at {cue.run_elapsed_s:.0f}s ('{cue.hint}'): "
                    f"1st third {cname}={first_v:.2f} -> 3rd third {cname}={last_v:.2f}"
                )
    return notes
