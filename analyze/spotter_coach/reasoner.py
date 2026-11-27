"""Deep Reasoning Brain powered by Gemini 4 Neon (`gemini-4-neon`).

1. Post-run synthesis: analyzes full telemetry, split tables, tension, and 20-run noise verdicts
   to produce a high-IQ biomechanical diagnosis + calibrated in-run metric rule for the next run.
2. On-demand `ask_deep_coach` tool: answers deep mechanics, multi-run progression, or scenario
   meta questions (with Google Search grounding) for the live voice frontend.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from spotter_analyze import history

from .cues import MetricRule
from .debrief import DebriefBundle

DEFAULT_REASONER_MODEL = "gemini-4-neon"

DIAGNOSIS_SYSTEM = """You are Spotter's frontier aim-science & motor-control analyst (Gemini 4 Neon).
Analyze the player's completed KovaaK's telemetry and return STRICT JSON matching this schema:
{
  "spoken_brief": "2-3 crisp, high-IQ spoken sentences for the voice coach to deliver. Never read raw JSON keys. Distinguish real changes from 'within noise' variance. Explain the underlying motor-control cause (e.g., deceleration clamping causing post-settle slips, under-calibrated initial ballistic pulse, pure visual-motor delay vs path imprecision, or suboptimal target-choice routing) and state the #1 adjustment.",
  "best_visual": "One of: time_budget (simple kill-time bar graph), splits (simple distance/speed bar graph or thirds pacing line), lag_shift (simple delay ceiling line), trend (simple score trend line)",
  "split_key": "If best_visual is 'splits', one of: distance, target_choice, direction, thirds, speed, since_rev; else empty string",
  "next_run_focus": "Short <= 8-word imperative focus cue for next run",
  "watch_rule": {
    "metric": "One of: flick_gain, short_pct, over_pct, slip_pct, miss_per_kill, extra_deg, react_ms, far_behind_pct, on_pct (or empty string if none)",
    "direction": "above or below",
    "threshold": 0.0,
    "cue_text": "<= 6-word in-run callout if triggered"
  }
}
"""


class CarbonReasoner:
    """Calls `gemini-4-neon` asynchronously for post-run synthesis and deep Q&A."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = model or os.environ.get("SPOTTER_REASONER_MODEL", DEFAULT_REASONER_MODEL)
        self._client = (
            genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    api_version="v1beta",
                    timeout=40000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            if self.api_key else None
        )

    @property
    def available(self) -> bool:
        return self._client is not None

    async def diagnose_run(self, bundle: DebriefBundle) -> dict[str, Any] | None:
        """Synthesize post-run coaching insights and next-run rule with Gemini 4 Neon."""
        if not self._client:
            return None

        # Keep digest compact (~2 KB) so Gemini 4 Neon returns in ~4-6 seconds!
        splits_digest = {}
        for k, (title, cols, rows) in bundle.splits.items():
            splits_digest[k] = [
                {cols[i]: (round(r[i], 2) if isinstance(r[i], float) and r[i] == r[i] else r[i])
                 for i in range(len(cols))}
                for r in rows[:6]
            ]

        report_text = ""
        if bundle.report_path.is_file():
            raw = bundle.report_path.read_text(encoding="utf-8", errors="replace")
            report_text = raw.split("## Events")[0][:2400]

        prompt = (
            f"Scenario: {bundle.scenario} ({bundle.mode})\n"
            f"Score: {bundle.score} (Usual: {bundle.usual_score}, SD: {bundle.score_sd}, Verdict: {bundle.score_verdict})\n"
            f"Settings note: {bundle.settings_note or 'unchanged'}\n"
            f"Mid-run cue outcomes: {bundle.cue_notes}\n\n"
            f"REPORT SUMMARY:\n{report_text}\n\n"
            f"SPLITS:\n{json.dumps(splits_digest, ensure_ascii=False)}"
        )

        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=DIAGNOSIS_SYSTEM,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                ),
            )
            if not resp.text:
                return None
            return json.loads(resp.text)
        except Exception as exc:
            print(f"[spotter] Carbon diagnosis fallback ({exc})", flush=True)
            return None

    async def ask_deep(self, question: str, bundle: DebriefBundle | None, out_dir: Path) -> dict[str, Any]:
        """Answer complex biomechanical, training-plan, or web-grounded questions via Gemini 4 Neon."""
        if not self._client:
            return {"error": "No API key"}

        ctx_parts = [f"Player question: {question}"]
        if bundle:
            ctx_parts.append(f"Current scenario: {bundle.scenario} ({bundle.mode}), score={bundle.score}")
            if bundle.report_path.is_file():
                raw = bundle.report_path.read_text(encoding="utf-8", errors="replace")
                ctx_parts.append("Latest run report:\n" + raw.split("## Events")[0][:4000])
            hist = history.load(out_dir / "history.jsonl")
            same = [r for r in hist if r.get("scenario") == bundle.scenario][-12:]
            if same:
                ctx_parts.append("Recent history records:\n" + json.dumps(same, ensure_ascii=False)[:3000])

        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model,
                contents="\n\n".join(ctx_parts),
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are Spotter's Gemini 4 Neon aim-coaching strategist. "
                        "Give a concise, high-IQ, evidence-backed answer (3-5 sentences max) that the "
                        "voice coach can speak directly to the player. Use Google Search when asked about "
                        "Voltaic/Viscose/Revosect benchmarks, aim theory, pro settings, or scenario strategies."
                    ),
                    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MEDIUM),
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            return {"analysis": (resp.text or "").strip()}
        except Exception as exc:
            return {"error": str(exc)}


def parse_watch_rule(raw: Any) -> MetricRule | None:
    if not isinstance(raw, dict):
        return None
    metric = str(raw.get("metric") or "").strip()
    direction = str(raw.get("direction") or "").strip().lower()
    cue_text = str(raw.get("cue_text") or "").strip()
    thresh = raw.get("threshold")
    if not metric or direction not in ("above", "below") or not cue_text or thresh is None:
        return None
    try:
        return MetricRule(metric=metric, direction=direction, threshold=float(thresh), cue_text=cue_text)
    except (TypeError, ValueError):
        return None
