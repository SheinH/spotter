"""Async CoachController orchestrating telemetry, Carbon reasoner, Live voice, cues, and HUD."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import Any

from spotter_analyze import history
from spotter_analyze.stream import (
    LiveTracker,
    RunAborted,
    RunCompleted,
    RunStarted,
    SpotTailer,
)

from .audio import AudioEngine
from .cues import WINDOW_S, CueEngine, MetricRule, TimedReminder
from .debrief import DebriefBundle, DebriefToolHandler, build_debrief
from .live import GeminiLiveSession
from .overlay import CoachOverlay
from .reasoner import CarbonReasoner, parse_watch_rule

POLL_MS = 200
CUE_TICK_S = 2.0
IDLE_HIBERNATE_S = 90.0
DIAGNOSE_TIMEOUT_S = 30.0       # cap on the post-run reasoner call
SPEAK_START_TIMEOUT_S = 12.0    # Live voice must start within this after the debrief prompt


class CoachController:
    """State machine: IDLE -> IN_RUN -> DEBRIEF -> IDLE."""

    def __init__(
        self,
        runs_dir: Path,
        stats_dir: Path,
        out_dir: Path,
        overlay: CoachOverlay,
        loop: asyncio.AbstractEventLoop,
        spot_path: Path | None = None,
        api_key: str | None = None,
        model: str | None = None,
        reasoner_model: str | None = None,
        voice: str = "Sadachbia",
        enable_audio: bool = True,
        replay_path: Path | None = None,
        replay_speed: float = 10.0,
        debrief_latest: bool = False,
    ) -> None:
        self.runs_dir = runs_dir
        self.stats_dir = stats_dir
        self.out_dir = out_dir
        self.overlay = overlay
        self.loop = loop
        self._pinned_spot = spot_path
        self._replay_path = replay_path
        self._replay_speed = max(replay_speed, 0.25)
        self._debrief_latest_pending = debrief_latest

        self.state = "IDLE"
        self.muted = False
        self.last_scenario: str | None = None
        self._transcript_buf = ""
        self._user_transcript_buf = ""
        self._user_locked_focus = False

        self._current_spot: Path | None = None
        self._tailer: SpotTailer | None = None
        self._tracker: LiveTracker | None = None

        hist = history.load(out_dir / "history.jsonl")
        if hist:
            self.last_scenario = hist[-1].get("scenario")
        self._cues = CueEngine(hist, scenario_hint=self.last_scenario)
        self._tools = DebriefToolHandler(out_dir=out_dir, stats_dir=stats_dir)
        self.reasoner = CarbonReasoner(api_key=api_key, model=reasoner_model)

        self.audio = AudioEngine(
            loop=loop,
            on_mic_chunk=self._on_mic_chunk,
            enabled=enable_audio,
        )
        self.live = GeminiLiveSession(
            api_key=api_key,
            model=model,
            on_audio=self.audio.enqueue_output,
            on_interrupted=self._on_server_interrupted,
            on_transcript=self._on_transcript,
            on_tool_call=self._handle_tool_call,
            on_status=self._on_live_status,
            voice_name=voice,
        )

        self._last_cue_check_mono = 0.0
        self._last_activity_mono = time.monotonic()
        self._running = False
        self._main_task: asyncio.Task | None = None

        self.overlay.end_convo_requested.connect(lambda: self.loop.create_task(self.end_debrief()))
        self.overlay.mute_toggle_requested.connect(self.toggle_mute)
        self.overlay.mic_device_changed.connect(self.audio.set_input_device)

    @property
    def pinned_metric(self) -> str | None:
        return self._cues.plan.pinned_metric

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.overlay.populate_mics(self.audio.list_input_devices())
        self.audio.start_output()
        if not self.live.available:
            self.overlay.set_status("Spotter (visual-only; set GEMINI_API_KEY)", ok=False)
        else:
            self.overlay.set_status("Spotter Ready · Home=Hide", ok=True)
        self._main_task = self.loop.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._main_task is not None:
            self._main_task.cancel()
            try:
                await self._main_task
            except (asyncio.CancelledError, Exception):
                pass
            self._main_task = None
        self.audio.close()
        await self.live.disconnect()

    def _discover_spot(self) -> Path | None:
        if self._replay_path is not None:
            return self._replay_path
        if self._pinned_spot is not None:
            return self._pinned_spot if self._pinned_spot.is_file() else None
        if not self.runs_dir.is_dir():
            return None
        candidates = [p for p in self.runs_dir.glob("spotter-*.spot") if p.is_file()]
        if not candidates:
            candidates = [p for p in self.runs_dir.glob("*.spot") if p.is_file()]
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    def _ensure_tracker(self) -> bool:
        target = self._discover_spot()
        if target is None:
            return False
        if target != self._current_spot or self._tailer is None or self._tracker is None:
            self._current_spot = target
            self._tailer = SpotTailer(target)
            is_replay = self._replay_path is not None
            if not is_replay and self._pinned_spot is None:
                self._tailer.poll()
                self._tracker = LiveTracker(self._tailer, stats_dir=self.stats_dir, watch_existing_stats=False)
                from spotter_analyze.stream import StatsWatcher
                lookback = 600.0 if self._debrief_latest_pending else 0.0
                self._debrief_latest_pending = False
                since = max(self._tailer.wall0, time.time() - lookback)
                self._tracker._watcher = StatsWatcher(self.stats_dir, since_wall_s=since)
            else:
                self._tracker = LiveTracker(self._tailer, stats_dir=self.stats_dir,
                                            watch_existing_stats=is_replay)
        return True

    async def _run_loop(self) -> None:
        tick_s = POLL_MS / 1000.0
        replay_chunk = int(22_000 * tick_s * self._replay_speed) if self._replay_path else None

        while self._running:
            now = time.monotonic()

            if self._replay_path is not None and self.state == "DEBRIEF":
                self.overlay.set_voice_state(self.muted, self.audio.is_speaking, self.audio.mic_level)
                await asyncio.sleep(tick_s)
                continue

            if self._ensure_tracker() and self._tracker is not None and self._tailer is not None:
                events = self._tracker.step(now_monotonic=now, max_bytes=replay_chunk)
                for ev in events:
                    if isinstance(ev, RunStarted):
                        await self._on_run_started(ev)
                    elif isinstance(ev, RunAborted):
                        await self._on_run_aborted(ev)
                    elif isinstance(ev, RunCompleted):
                        await self._on_run_completed(ev)

                if self.state == "IN_RUN" and self._tracker.in_run:
                    if (now - self._last_cue_check_mono) >= CUE_TICK_S:
                        self._last_cue_check_mono = now
                        await self._check_mid_run_cue()

            if self.state == "DEBRIEF":
                self.overlay.set_voice_state(self.muted, self.audio.is_speaking, self.audio.mic_level)
            elif self.state == "IDLE" and (now - self._last_activity_mono) >= IDLE_HIBERNATE_S:
                await self.live.disconnect()
                self._last_activity_mono = now

            await asyncio.sleep(tick_s)

    # --- State Event Handlers ---

    async def _on_run_started(self, ev: RunStarted) -> None:
        print(f"[spotter] Run started at t={ev.t0:.1f}s (overlay hidden, mic OFF)", flush=True)
        self._last_activity_mono = time.monotonic()
        if self.state == "DEBRIEF":
            await self._cut_conversation_audio_and_mic()

        self.state = "IN_RUN"
        self.audio.set_mic_active(False)
        self.overlay.enter_in_run()

        hist = history.load(self.out_dir / "history.jsonl")
        self._cues.history_records = hist
        self._cues.reset_for_run(scenario_hint=self.last_scenario)
        self._last_cue_check_mono = time.monotonic()

        if self.live.available and not self._cues.plan.muted_in_run:
            self.loop.create_task(self.live.ensure_connected())

    async def _on_run_aborted(self, _ev: RunAborted) -> None:
        self._last_activity_mono = time.monotonic()
        self.audio.set_mic_active(False)
        self.live.suppress_current_output()
        self.audio.clear_playback()
        if self._tracker and self._tracker.in_run:
            self.state = "IN_RUN"
            self.overlay.enter_in_run()
        else:
            self.state = "IDLE"
            self.overlay.show_idle()

    async def _on_run_completed(self, ev: RunCompleted) -> None:
        self._last_activity_mono = time.monotonic()
        if self._tailer is None:
            return

        self.live.suppress_current_output()
        self.audio.clear_playback()
        self.last_scenario = ev.stats.scenario

        run_slice = self._tailer.slice(ev.t0, ev.t1)
        if len(run_slice.t) < 60 or len(run_slice.tgt_frame) < 40:
            self.state = "IDLE"
            self.overlay.set_status(f"{ev.stats.scenario}: log too short", ok=False)
            self.overlay.show_idle()
            return

        fired = list(self._cues.fired_cues)
        bundle: DebriefBundle = await asyncio.to_thread(
            build_debrief,
            run_slice,
            ev.stats,
            self.out_dir,
            self.stats_dir,
            fired,
        )
        self._tools.current = bundle
        self._transcript_buf = ""
        self._user_transcript_buf = ""

        # Show card immediately while Gemini 4 Neon synthesizes the deep diagnosis
        print(f"[spotter] Completed: {bundle.scenario} | score={bundle.score} ({bundle.score_verdict})", flush=True)
        self.state = "DEBRIEF"
        self.overlay.show_debrief(bundle)
        self.overlay.set_caption("Analyzing with Gemini 4 Neon…", is_user=False)
        self.audio.set_mic_active(not self.muted)
        self.overlay.set_voice_state(self.muted, False, 0.0)

        if self.live.available:
            self.loop.create_task(self._run_carbon_and_speak(bundle))

    def _debrief_is_current(self, bundle: DebriefBundle) -> bool:
        return self.state == "DEBRIEF" and self._tools.current is bundle

    async def _run_carbon_and_speak(self, bundle: DebriefBundle) -> None:
        # This runs as a fire-and-forget task: an uncaught error here used to vanish silently and
        # leave the HUD on "Analyzing…" forever. Log it and never leave the caption stuck.
        try:
            await self._diagnose_and_speak(bundle)
        except Exception as exc:
            print(f"[spotter] Debrief voice failed: {type(exc).__name__}: {exc}", flush=True)
            if self._debrief_is_current(bundle):
                self.overlay.set_caption("Voice coach unavailable — see the card above.", is_user=False)

    async def _diagnose_and_speak(self, bundle: DebriefBundle) -> None:
        # Warm Live WebSocket in parallel with Gemini 4 Neon's synthesis.
        live_task = self.loop.create_task(self.live.ensure_connected())
        try:
            diag = await asyncio.wait_for(self.reasoner.diagnose_run(bundle), DIAGNOSE_TIMEOUT_S)
        except asyncio.TimeoutError:
            print(f"[spotter] Diagnosis timed out after {DIAGNOSE_TIMEOUT_S:.0f}s; using raw debrief", flush=True)
            diag = None
        if not isinstance(diag, dict):
            diag = None
        await live_task

        if not self._debrief_is_current(bundle):
            return

        if diag:
            best_vis = str(diag.get("best_visual") or "").strip()
            split_key = str(diag.get("split_key") or "").strip()
            if best_vis:
                self.overlay.select_visual(best_vis, split_key)

            # Apply Carbon's recommended next-run focus & metric rule unless the user explicitly locked their own
            focus = str(diag.get("next_run_focus") or "").strip()
            rule = parse_watch_rule(diag.get("watch_rule"))
            if not self._user_locked_focus:
                if focus:
                    self._cues.plan.focus_cue = focus
                if rule:
                    self._cues.plan.pinned_metric = rule.metric
                    self._cues.plan.custom_rules = [rule]
                self._refresh_focus_badge()

            prompt = (
                f"[POST_RUN_DEBRIEF] Scenario: {bundle.scenario} ({bundle.mode}), Score: {bundle.score}.\n"
                f"Gemini 4 Neon has ALREADY analyzed the telemetry and updated the HUD chart & focus goal.\n"
                f"Do NOT call any tools right now — just speak this coaching diagnosis naturally and "
                f"conversationally in 2-3 crisp sentences, then stop so the player can reply:\n\n"
                f"{diag.get('spoken_brief', '')}"
            )
        else:
            prompt = bundle.llm_prompt

        self.overlay.set_caption("Coach is speaking…", is_user=False)
        # Watchdog: a half-open or server-closed Live socket can swallow the prompt with no error.
        # If no voice/transcript arrives, reconnect and resend once; then fall back to on-screen text.
        for attempt in (1, 2):
            sent_at = time.monotonic()
            await asyncio.wait_for(self.live.send_prompt(prompt), SPEAK_START_TIMEOUT_S)
            while time.monotonic() - sent_at < SPEAK_START_TIMEOUT_S:
                if self.live.last_output_mono >= sent_at or not self._debrief_is_current(bundle):
                    return
                await asyncio.sleep(0.2)
            print(f"[spotter] No voice {SPEAK_START_TIMEOUT_S:.0f}s after debrief prompt "
                  f"(attempt {attempt}); reconnecting Live session", flush=True)
            await self.live.disconnect()

        if self._debrief_is_current(bundle):
            brief = str((diag or {}).get("spoken_brief") or "").strip()
            self.overlay.set_caption(brief or "Voice coach didn't respond — see the card above.", is_user=False)
            self._on_live_status("Live error: voice didn't respond")

    def _refresh_focus_badge(self) -> None:
        plan = self._cues.plan
        parts: list[str] = []
        if plan.muted_in_run:
            parts.append("In-run cues muted")
        else:
            for r in plan.reminders:
                parts.append(f"{r.cue_text} ({r.every_s:.0f}s)")
            if plan.focus_cue and not plan.reminders:
                parts.append(plan.focus_cue)
        self.overlay.set_next_focus(" · ".join(parts))

    async def _check_mid_run_cue(self) -> None:
        if self._tailer is None or self._tracker is None or not self._tracker.in_run:
            return
        now_t = self._tailer.latest_t
        elapsed = now_t - self._tracker.run_t0
        win_t0 = max(self._tracker.run_t0, now_t - WINDOW_S)
        window_run = self._tailer.slice(win_t0, now_t)
        cue = await asyncio.to_thread(self._cues.evaluate, window_run, elapsed)
        if cue is not None and self.live.available:
            print(f"[spotter] Mid-run cue ({elapsed:.0f}s): {cue.metric} -> {cue.hint}", flush=True)
            self._last_activity_mono = time.monotonic()
            self.loop.create_task(self.live.send_prompt(cue.prompt))

    # --- User & AI Controls ---

    async def end_debrief(self) -> None:
        await self._cut_conversation_audio_and_mic()
        if self._tracker and self._tracker.in_run:
            self.state = "IN_RUN"
            self.overlay.enter_in_run()
        else:
            self.state = "IDLE"
            self.overlay.show_idle()

    async def _cut_conversation_audio_and_mic(self) -> None:
        self.live.suppress_current_output()
        self.audio.clear_playback()
        was_mic = self.audio.mic_active
        self.audio.set_mic_active(False)
        if was_mic:
            await self.live.end_mic_stream()

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        if self.state == "DEBRIEF":
            self.audio.set_mic_active(not self.muted)
            if self.muted:
                self.loop.create_task(self.live.end_mic_stream())
            self.overlay.set_voice_state(self.muted, self.audio.is_speaking, self.audio.mic_level)

    def _on_mic_chunk(self, pcm16k: bytes) -> None:
        if self.state == "DEBRIEF" and not self.muted:
            self._last_activity_mono = time.monotonic()
            self.loop.create_task(self.live.send_mic_pcm(pcm16k))

    def _on_server_interrupted(self) -> None:
        self.audio.clear_playback()
        self._transcript_buf = ""

    def _on_transcript(self, text: str, is_user: bool) -> None:
        if self.state != "DEBRIEF":
            return
        if is_user:
            self._transcript_buf = ""
            self._user_transcript_buf = (self._user_transcript_buf + text)[-160:]
            self.overlay.set_caption(self._user_transcript_buf, is_user=True)
        else:
            self._user_transcript_buf = ""
            self._transcript_buf = (self._transcript_buf + text)[-200:]
            self.overlay.set_caption(self._transcript_buf, is_user=False)

    def _on_live_status(self, status_text: str) -> None:
        ok = "error" not in status_text.lower() and "no gemini" not in status_text.lower()
        self.overlay.set_status(f"Spotter · {status_text}", ok=ok)

    async def _handle_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self._last_activity_mono = time.monotonic()
        if name == "show_visual":
            view = str(args.get("view", "time_budget"))
            split_key = str(args.get("split_key", ""))
            chosen = self.overlay.select_visual(view, split_key)
            return {"shown": chosen, "split_key": split_key}

        if name == "configure_in_run_cues":
            plan = self._cues.plan
            if args.get("clear_reminders"):
                plan.reminders.clear()
                self._user_locked_focus = False
            if "mute_in_run" in args and args["mute_in_run"] is not None:
                plan.muted_in_run = bool(args["mute_in_run"])
                self._user_locked_focus = bool(args["mute_in_run"])
            rem_txt = str(args.get("reminder_text") or "").strip()
            if rem_txt:
                every_s = float(args.get("reminder_every_s") or 20.0)
                plan.reminders.append(TimedReminder(cue_text=rem_txt, every_s=max(every_s, 12.0)))
                plan.muted_in_run = False
                self._user_locked_focus = True
            focus = str(args.get("focus_cue") or "").strip()
            if focus:
                plan.focus_cue = focus
                self._user_locked_focus = True
            directive = str(args.get("coach_directive") or "").strip()
            if directive:
                plan.coach_directive = directive
            w_metric = str(args.get("watch_metric") or "").strip()
            if w_metric:
                plan.pinned_metric = w_metric
                w_dir = str(args.get("watch_direction") or "").strip().lower()
                w_thresh = args.get("watch_threshold")
                w_cue = str(args.get("watch_cue") or focus or rem_txt or "Adjust aim").strip()
                if w_dir in ("above", "below") and w_thresh is not None:
                    plan.custom_rules = [
                        MetricRule(metric=w_metric, direction=w_dir, threshold=float(w_thresh), cue_text=w_cue)
                    ]
                    self._user_locked_focus = True
            self._refresh_focus_badge()
            return {
                "muted_in_run": plan.muted_in_run,
                "reminders": [f"{r.cue_text} every {r.every_s:.0f}s" for r in plan.reminders],
                "focus_cue": plan.focus_cue,
                "pinned_metric": plan.pinned_metric,
            }

        if name == "set_next_run_focus":
            cue = str(args.get("cue", "")).strip()
            metric = str(args.get("metric", "")).strip() or None
            self._cues.plan.focus_cue = cue
            if metric:
                self._cues.plan.pinned_metric = metric
            self._refresh_focus_badge()
            return {"pinned_cue": cue, "pinned_metric": metric}

        if name == "ask_deep_coach":
            q = str(args.get("question", "")).strip()
            return await self.reasoner.ask_deep(q, self._tools.current, self.out_dir)

        if name == "get_split_details":
            return self._tools.get_split_details(str(args.get("split_key", "thirds")))

        if name == "compare_with_history":
            return self._tools.compare_with_history(str(args.get("target", "previous")))

        if name == "end_debrief":
            self.loop.create_task(self.end_debrief())
            return {"ended": True}

        return {"error": f"Unknown tool {name}"}
