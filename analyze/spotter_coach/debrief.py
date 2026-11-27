"""Completed-run debrief builder, noise-layer comparator, and tool data provider."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from spotter_analyze import clicking, history, kovaaks, report, sens, tension, tracking
from spotter_analyze.__main__ import pick_mode
from spotter_analyze.format import Run

from .cues import MidRunCue, evaluate_cue_effect


@dataclass
class DeltaPill:
    label: str
    value_text: str
    sub_text: str
    tone: str  # "good" | "bad" | "neutral"


@dataclass
class DebriefBundle:
    name: str
    scenario: str
    mode: str
    score: float | None
    usual_score: float | None
    score_sd: float | None
    score_verdict: str
    settings_note: str | None
    pills: list[DeltaPill]
    scatter_png: Path
    profiles_png: Path
    report_path: Path
    default_visual: str
    time_budget: tuple[list[str], list[list]] | None
    lag_shift: tuple[list[str], list[list]] | None
    splits: dict[str, tuple[str, list[str], list[list]]]
    trend_points: list[tuple[str, float]]
    cue_notes: list[str]
    record: dict[str, Any]
    llm_prompt: str
    raw_noise_rows: list[list] = field(default_factory=list)


def build_debrief(
    run: Run,
    stats: kovaaks.Stats,
    out_dir: Path,
    stats_dir: Path,
    fired_cues: list[MidRunCue] | None = None,
) -> DebriefBundle:
    """Run the full Spotter analysis for a completed run, save report & history, and package UI/LLM data."""
    name = f"{stats.scenario} - {stats.end:%Y.%m.%d-%H.%M.%S}"
    mode = pick_mode(run, stats)
    try:
        result = tracking.analyze(run) if mode == "tracking" else clicking.analyze(run, mode)
    except ValueError:
        mode, result = "tracking", tracking.analyze(run)

    sens.attach(result, run, stats.fields)
    tens = tension.analyze(run, mode, tension.mm_per_deg(stats.fields, run))
    rec = history.make_record(name, mode, stats, result, tens)

    hist_path = out_dir / "history.jsonl"
    existing_hist = history.load(hist_path)
    same_scen_prior = [
        r for r in existing_hist
        if r.get("scenario") == stats.scenario and r.get("mode") == mode and r.get("name") != name
    ]
    prev_rec = same_scen_prior[-1] if same_scen_prior else None

    history.save(hist_path, rec)
    all_hist = history.load(hist_path)

    prior_stats = [s for s in history.score_history(stats_dir, stats.scenario) if s.end < stats.end]
    prior_scores = [s_val for s in prior_stats if (s_val := history._num(s.fields.get("Score"))) is not None]
    noise = history.noise_rows(rec, all_hist, prior_scores)
    context = history.score_context(stats, stats_dir)

    run_dir = out_dir / name
    rep_path = report.write(run_dir, name, result, stats, run, 0, tens, noise, context)

    # Check if sensitivity or FOV changed since the previous analyzed run
    settings_note = None
    if prev_rec:
        changes = []
        if prev_rec.get("sens") != rec.get("sens") or prev_rec.get("sens_scale") != rec.get("sens_scale"):
            changes.append(f"sens {prev_rec.get('sens')} -> {rec.get('sens')} {rec.get('sens_scale') or ''}".strip())
        if prev_rec.get("fov") != rec.get("fov"):
            changes.append(f"FOV {prev_rec.get('fov')} -> {rec.get('fov')}")
        if changes:
            settings_note = "Settings changed vs previous run: " + ", ".join(changes)

    # Score summary & noise verdict
    score = rec.get("score")
    noise_cols, noise_rows, n_other = noise
    score_row = next((r for r in noise_rows if r[0] == "score"), None)
    usual_score = _to_float(score_row[2]) if score_row else None
    score_sd = _to_float(score_row[3]) if score_row else None
    score_verdict = str(score_row[5]) if score_row else "first runs"

    # Extract structured splits & visuals
    splits_map: dict[str, tuple[str, list[str], list[list]]] = {}
    time_budget_data: tuple[list[str], list[list]] | None = None
    lag_shift_data: tuple[list[str], list[list]] | None = None

    for title, cols, rows in result.get("splits", []):
        tl = title.lower()
        if tl.startswith("time budget"):
            time_budget_data = (cols, rows)
            splits_map["time_budget"] = (title, cols, rows)
        elif tl.startswith("lag-shift"):
            lag_shift_data = (cols, rows)
            splits_map["lag_shift"] = (title, cols, rows)
        elif tl.startswith("flick vs micro"):
            splits_map["flick_micro"] = (title, cols, rows)
        elif tl.startswith("micro by"):
            splits_map["micro"] = (title, cols, rows)
        elif "distance" in tl:
            splits_map["distance"] = (title, cols, rows)
        elif "target choice" in tl:
            splits_map["target_choice"] = (title, cols, rows)
        elif "direction" in tl:
            splits_map["direction"] = (title, cols, rows)
        elif "third" in tl:
            splits_map["thirds"] = (title, cols, rows)
        elif "speed" in tl:
            splits_map["speed"] = (title, cols, rows)
        elif "reversal" in tl or "since" in tl:
            splits_map["since_rev"] = (title, cols, rows)
        else:
            key = tl.split()[0].strip(":()")
            splits_map[key] = (title, cols, rows)

    pills = _build_pills(rec, prev_rec, noise_rows)
    cue_notes = evaluate_cue_effect(fired_cues or [], result)

    # Recent score trend (up to last 15 runs of this scenario)
    recent_stats = history.score_history(stats_dir, stats.scenario)[-15:]
    trend_points = [
        (s.end.strftime("%m/%d %H:%M"), float(v))
        for s in recent_stats
        if (v := history._num(s.fields.get("Score"))) is not None
    ]

    default_visual = "time_budget" if time_budget_data else ("lag_shift" if lag_shift_data else "error_scatter")

    llm_prompt = _format_llm_debrief_prompt(
        scenario=stats.scenario,
        mode=mode,
        score=score,
        usual_score=usual_score,
        score_sd=score_sd,
        score_verdict=score_verdict,
        settings_note=settings_note,
        summary_lines=(context or []) + result.get("summary", []),
        noise_cols=noise_cols,
        noise_rows=noise_rows,
        prev_rec=prev_rec,
        rec=rec,
        cue_notes=cue_notes,
        default_visual=default_visual,
        available_splits=list(splits_map.keys()),
    )

    return DebriefBundle(
        name=name,
        scenario=stats.scenario,
        mode=mode,
        score=score,
        usual_score=usual_score,
        score_sd=score_sd,
        score_verdict=score_verdict,
        settings_note=settings_note,
        pills=pills,
        scatter_png=run_dir / "error_scatter.png",
        profiles_png=run_dir / "velocity_profiles.png",
        report_path=rep_path,
        default_visual=default_visual,
        time_budget=time_budget_data,
        lag_shift=lag_shift_data,
        splits=splits_map,
        trend_points=trend_points,
        cue_notes=cue_notes,
        record=rec,
        llm_prompt=llm_prompt,
        raw_noise_rows=noise_rows,
    )


def _to_float(x: Any) -> float | None:
    if isinstance(x, (int, float, np.number)):
        f = float(x)
        return None if math.isnan(f) else f
    if isinstance(x, str):
        try:
            f = float(x)
            return None if math.isnan(f) else f
        except ValueError:
            return None
    return None


def _isnan(x: Any) -> bool:
    return _to_float(x) is None


def _fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _build_pills(rec: dict, prev_rec: dict | None, noise_rows: list[list]) -> list[DeltaPill]:
    pills: list[DeltaPill] = []
    # Map noise rows from history.py: [metric, this_run, usual, sd, runs, verdict, noise_2runs]
    scored_rows = [r for r in noise_rows if r[0] != "score" and _to_float(r[1]) is not None]

    def sort_key(r: list) -> tuple[int, float]:
        verdict_s = str(r[5])
        outside = 1 if ("SD" in verdict_s or "better" in verdict_s or "worse" in verdict_s) else 0
        val_f, usual_f, sd_f = _to_float(r[1]), _to_float(r[2]), _to_float(r[3])
        z = abs(val_f - usual_f) / sd_f if (val_f is not None and usual_f is not None and sd_f and sd_f > 0) else 0.0
        return (outside, z)

    for r in sorted(scored_rows, key=sort_key, reverse=True)[:4]:
        label_full, val, usual, _sd, _n, verdict, _n2 = r
        short = label_full.split("(")[0].strip()
        if len(short) > 22:
            short = short[:20] + "…"
        val_f = _to_float(val) or 0.0
        usual_f = _to_float(usual)
        v_str = _fmt(val_f)
        verdict_s = str(verdict)
        if "better" in verdict_s and usual_f is not None:
            tone = "good"
            sub = f"usual {_fmt(usual_f)} · REAL ▲"
        elif "worse" in verdict_s and usual_f is not None:
            tone = "bad"
            sub = f"usual {_fmt(usual_f)} · REAL ▼"
        elif "SD" in verdict_s and usual_f is not None:
            tone = "neutral"
            sub = f"usual {_fmt(usual_f)} · shift"
        elif usual_f is not None:
            tone = "neutral"
            sub = f"usual {_fmt(usual_f)} · noise"
        else:
            tone = "neutral"
            sub = "baseline building"
        pills.append(DeltaPill(label=short, value_text=v_str, sub_text=sub, tone=tone))

    if not pills:
        for k, v in list(rec.get("metrics", {}).items())[:3]:
            if v is not None and not _isnan(v):
                pills.append(DeltaPill(label=k, value_text=_fmt(float(v)), sub_text="first run", tone="neutral"))
    return pills


def _format_llm_debrief_prompt(
    scenario: str,
    mode: str,
    score: float | None,
    usual_score: float | None,
    score_sd: float | None,
    score_verdict: str,
    settings_note: str | None,
    summary_lines: list[str],
    noise_cols: list[str],
    noise_rows: list[list],
    prev_rec: dict | None,
    rec: dict,
    cue_notes: list[str],
    default_visual: str,
    available_splits: list[str],
) -> str:
    prev_deltas: dict[str, str] = {}
    if prev_rec:
        pm = prev_rec.get("metrics", {})
        for k, v in rec.get("metrics", {}).items():
            if v is not None and k in pm and pm[k] is not None:
                prev_deltas[k] = f"{pm[k]:.2f} -> {v:.2f} ({v - pm[k]:+.2f})"

    compact_noise = [
        {noise_cols[i]: (round(row[i], 2) if isinstance(row[i], float) and not math.isnan(row[i]) else row[i])
         for i in range(len(noise_cols))}
        for row in noise_rows
    ]

    payload = {
        "event": "POST_RUN_DEBRIEF",
        "scenario": scenario,
        "mode": mode,
        "score": score,
        "usual_score": round(usual_score, 1) if usual_score is not None else None,
        "score_sd": round(score_sd, 1) if score_sd is not None else None,
        "score_verdict": score_verdict,
        "settings_note": settings_note,
        "previous_run_score": prev_rec.get("score") if prev_rec else None,
        "deltas_vs_previous_run": prev_deltas,
        "mid_run_cue_outcomes": cue_notes,
        "summary_findings": summary_lines,
        "noise_comparison_vs_last_20": compact_noise,
        "currently_displayed_visual": default_visual,
        "available_splits_for_tools": available_splits,
    }
    return (
        "[POST_RUN_DEBRIEF] The player just finished a completed run and their mic is now open.\n"
        "Give a crisp, conversational 15-25 second spoken review:\n"
        "1. State what meaningfully changed vs usual/previous run — strictly treat 'within noise' metrics as "
        "normal variance, and lead with metrics outside noise or the biggest time-budget / lag-shift cost.\n"
        "2. If mid_run_cue_outcomes is non-empty, briefly mention whether the 3rd third improved.\n"
        "3. Call `show_visual` if switching to `error_scatter`, `velocity_profiles`, `time_budget`, `lag_shift`, "
        "`splits`, or `trend` helps illustrate your main point, and call `set_next_run_focus` to pin ONE clear "
        "goal for the next run.\n"
        "4. Stop after 2-4 sentences so the player can ask questions or jump into the next run.\n\n"
        f"DATA:\n{json.dumps(payload, ensure_ascii=False)}"
    )


class DebriefToolHandler:
    """Executes Gemini Live tool calls against the latest DebriefBundle and history.jsonl."""

    def __init__(self, out_dir: Path, stats_dir: Path) -> None:
        self.out_dir = out_dir
        self.stats_dir = stats_dir
        self.current: DebriefBundle | None = None

    def get_split_details(self, split_key: str) -> dict[str, Any]:
        if not self.current:
            return {"error": "No completed run loaded yet."}
        key = split_key.lower().strip()
        if key not in self.current.splits:
            # Fuzzy fallback
            match = next((k for k in self.current.splits if key in k or k in key), None)
            if not match:
                return {"available_splits": list(self.current.splits.keys())}
            key = match
        title, cols, rows = self.current.splits[key]
        formatted = [
            {cols[i]: (round(r[i], 2) if isinstance(r[i], float) and not math.isnan(r[i]) else r[i])
             for i in range(len(cols))}
            for r in rows
        ]
        return {"title": title, "split_key": key, "rows": formatted}

    def compare_with_history(self, target: str = "previous") -> dict[str, Any]:
        if not self.current:
            return {"error": "No completed run loaded yet."}
        hist = history.load(self.out_dir / "history.jsonl")
        mine = [
            r for r in hist
            if r.get("scenario") == self.current.scenario and r.get("mode") == self.current.mode
        ]
        cur = self.current.record
        prior = [r for r in mine if r.get("name") != cur.get("name")]
        if not prior:
            return {"note": "No prior analyzed runs of this scenario in history.jsonl."}

        t = target.lower().strip()
        if "pb" in t or "best" in t:
            scored = [r for r in prior if r.get("score") is not None]
            ref = max(scored, key=lambda r: float(r["score"])) if scored else prior[-1]
            label = "analyzed_pb"
        elif "today" in t or "session" in t:
            today = datetime.now().date()
            todays = [
                r for r in prior
                if r.get("end") and datetime.fromisoformat(r["end"]).date() == today
            ]
            pool = todays or prior[-5:]
            avg_m = {
                k: round(float(np.mean([p["metrics"][k] for p in pool if p["metrics"].get(k) is not None])), 2)
                for k in cur.get("metrics", {})
                if any(p["metrics"].get(k) is not None for p in pool)
            }
            scores = [float(p["score"]) for p in pool if p.get("score") is not None]
            return {
                "target": "session_average",
                "n_runs": len(pool),
                "avg_score": round(float(np.mean(scores)), 1) if scores else None,
                "current_score": cur.get("score"),
                "avg_metrics": avg_m,
                "current_metrics": cur.get("metrics"),
            }
        else:
            ref = prior[-1]
            label = "previous_run"

        return {
            "target": label,
            "ref_name": ref.get("name"),
            "ref_score": ref.get("score"),
            "current_score": cur.get("score"),
            "ref_metrics": ref.get("metrics"),
            "current_metrics": cur.get("metrics"),
        }
