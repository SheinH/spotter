"""Gemini Live API (`gemini-3.8-live-extended-thinking` + Google Search) voice client."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import inspect
import os
import time
from typing import Any

from google import genai
from google.genai import types

SYSTEM_PROMPT = """You are Spotter, a high-IQ, calm, biomechanics-grounded KovaaK's aim coach.
Your analytical brain is powered by Gemini 4 Neon, and you also have live Google Search and direct HUD/in-run controls.

OPERATING MODES:
1. [IN_RUN_CUE]: The player is mid-run with their microphone OFF.
   - Deliver ONE calm, imperative voice cue in 7 words or fewer.
   - Never cite raw decimals, never lecture, and do NOT call tools during [IN_RUN_CUE].

2. [POST_RUN_DEBRIEF]: The player just completed a run and is on the score screen with an open mic.
   - If `carbon_diagnosis` is provided in the prompt, deliver its `spoken_brief` naturally in 2-4 crisp sentences (~15-20s) and stop so the player can talk.
   - Strictly treat 'within noise' metrics as normal variance.
   - Whenever the player asks you to remind them of something during runs (e.g. "remind me to relax my grip", "remind me every 15s to look at the target", "stay quiet mid-run", "warn me if I undershoot"), call `configure_in_run_cues` immediately and confirm briefly!
   - For deep biomechanical, training-plan, or cross-run diagnostic questions, call `ask_deep_coach` to consult Gemini 4 Neon, or use Google Search for Voltaic/Viscose benchmarks and scenario strategies.
   - Use `show_visual` to switch charts (`error_scatter`, `velocity_profiles`, `time_budget`, `lag_shift`, `splits`, `trend`).
   - Never speak filler narration before calling a tool (do NOT say "Let me check" or "I am reviewing...") — call the tool silently first, then speak once after the tool result arrives.
   - If the player says thanks / got it / end convo / wrap up, call `end_debrief` immediately.
"""

CONNECT_TIMEOUT_S = 15.0

PREFERRED_MODELS = (
    "gemini-3.8-live-extended-thinking",
    "gemini-3.8-live",
    "gemini-2.5-flash-native-audio-latest",
)

UI_SIDE_EFFECT_TOOLS = {
    "show_visual",
    "set_next_run_focus",
    "configure_in_run_cues",
    "end_debrief",
}


def _tool_declarations() -> list[types.Tool]:
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="show_visual",
                    description="Switch the visual card displayed on the player's overlay HUD.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "view": types.Schema(
                                type=types.Type.STRING,
                                description="One of: time_budget (simple bar graph), splits (simple bar/line split), lag_shift (simple delay line), trend (simple score line)",
                            ),
                            "split_key": types.Schema(
                                type=types.Type.STRING,
                                description="When view='splits', which split table to show (distance, target_choice, direction, thirds, speed, since_rev).",
                            ),
                        },
                        required=["view"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="configure_in_run_cues",
                    description="Set or update mid-run voice reminders, custom metric alerts, focus goal, or mid-run mute for upcoming runs.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "reminder_text": types.Schema(
                                type=types.Type.STRING,
                                description="Timed spoken reminder during runs (<= 7 words), e.g. 'Relax grip and breathe'",
                            ),
                            "reminder_every_s": types.Schema(
                                type=types.Type.NUMBER,
                                description="How often in seconds to speak reminder_text during a run (default 20).",
                            ),
                            "clear_reminders": types.Schema(
                                type=types.Type.BOOLEAN,
                                description="Set true to clear previous timed reminders.",
                            ),
                            "mute_in_run": types.Schema(
                                type=types.Type.BOOLEAN,
                                description="Set true if player wants complete silence mid-run; false to re-enable mid-run cues.",
                            ),
                            "focus_cue": types.Schema(
                                type=types.Type.STRING,
                                description="Short pinned goal shown on the HUD footer.",
                            ),
                            "watch_metric": types.Schema(
                                type=types.Type.STRING,
                                description="Optional metric to watch mid-run (flick_gain, short_pct, over_pct, slip_pct, miss_per_kill, extra_deg, react_ms, far_behind_pct, on_pct).",
                            ),
                            "watch_direction": types.Schema(
                                type=types.Type.STRING,
                                description="'above' or 'below' when setting a custom watch_threshold.",
                            ),
                            "watch_threshold": types.Schema(
                                type=types.Type.NUMBER,
                                description="Numeric threshold for watch_metric.",
                            ),
                            "watch_cue": types.Schema(
                                type=types.Type.STRING,
                                description="<= 6-word callout when watch_metric crosses watch_threshold.",
                            ),
                            "coach_directive": types.Schema(
                                type=types.Type.STRING,
                                description="Extra style or focus instruction for in-run cues.",
                            ),
                        },
                    ),
                ),
                types.FunctionDeclaration(
                    name="set_next_run_focus",
                    description="Pin a concise focus goal on the overlay for the player's next run.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "cue": types.Schema(type=types.Type.STRING),
                            "metric": types.Schema(type=types.Type.STRING),
                        },
                        required=["cue"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="ask_deep_coach",
                    description="Consult Gemini 4 Neon (frontier deep-reasoning brain + Google Search) for complex biomechanics, multi-run diagnosis, training routines, or benchmark questions.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "question": types.Schema(
                                type=types.Type.STRING,
                                description="The analytical or coaching question to evaluate deeply.",
                            ),
                        },
                        required=["question"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="get_split_details",
                    description="Fetch exact numbers from one of the run's split tables (distance, target_choice, flick_micro, micro, direction, thirds, speed, since_rev, time_budget, lag_shift).",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={"split_key": types.Schema(type=types.Type.STRING)},
                        required=["split_key"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="compare_with_history",
                    description="Compare this run against previous, pb, or session average.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={"target": types.Schema(type=types.Type.STRING)},
                        required=["target"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="end_debrief",
                    description="End the post-run debrief conversation, mute mic, and collapse the overlay card.",
                    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
                ),
            ]
        ),
        types.Tool(google_search=types.GoogleSearch()),
    ]


ToolCallback = Callable[[str, dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


class GeminiLiveSession:
    """Manages an async Gemini Live WebSocket with extended thinking, Google Search, and tools."""

    def __init__(
        self,
        api_key: str | None,
        model: str | None,
        on_audio: Callable[[bytes], None],
        on_interrupted: Callable[[], None],
        on_transcript: Callable[[str, bool], None],
        on_tool_call: ToolCallback,
        on_status: Callable[[str], None],
        voice_name: str = "Sadachbia",
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.requested_model = model or os.environ.get("SPOTTER_GEMINI_MODEL")
        self.voice_name = os.environ.get("SPOTTER_VOICE", voice_name)
        self._resolved_model: str | None = None

        self._on_audio = on_audio
        self._on_interrupted = on_interrupted
        self._on_transcript = on_transcript
        self._on_tool_call = on_tool_call
        self._on_status = on_status

        self._client = (
            genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})
            if self.api_key else None
        )
        self._session: Any = None
        self._conn_cm: Any = None
        self._recv_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._tools_in_flight = 0
        self._resume_handle: str | None = None
        self._accept_audio_out = True
        self._audio_bytes_this_turn = 0
        # monotonic time of the last model audio/transcript; lets callers detect a silent (dead) session
        self.last_output_mono = 0.0

    @property
    def available(self) -> bool:
        return self._client is not None

    async def _pick_model(self) -> str:
        if self.requested_model:
            return self.requested_model
        if self._resolved_model:
            return self._resolved_model
        if not self._client:
            return PREFERRED_MODELS[0]
        try:
            live_models: list[str] = []
            pager = await self._client.aio.models.list()
            async for m in pager:
                actions = getattr(m, "supported_actions", None) or []
                name = (getattr(m, "name", "") or "").removeprefix("models/")
                if "bidiGenerateContent" in actions and name:
                    live_models.append(name)
            for pref in PREFERRED_MODELS:
                for cand in live_models:
                    if cand == pref:
                        self._resolved_model = cand
                        return cand
            for cand in live_models:
                if "live" in cand or "native-audio" in cand:
                    self._resolved_model = cand
                    return cand
        except Exception:
            pass
        self._resolved_model = PREFERRED_MODELS[0]
        return self._resolved_model

    async def ensure_connected(self) -> bool:
        if not self._client:
            self._on_status("No GEMINI_API_KEY (visual-only mode)")
            return False
        async with self._lock:
            if self._session is not None:
                return True
            model = await self._pick_model()
            thinking_cfg = (
                types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
                if "thinking" in model else None
            )
            cfg = types.LiveConnectConfig(
                response_modalities=[types.Modality.AUDIO],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice_name)
                    )
                ),
                thinking_config=thinking_cfg,
                system_instruction=SYSTEM_PROMPT,
                tools=_tool_declarations(),
                output_audio_transcription=types.AudioTranscriptionConfig(),
                input_audio_transcription=types.AudioTranscriptionConfig(),
                context_window_compression=types.ContextWindowCompressionConfig(
                    trigger_tokens=14000,
                    sliding_window=types.SlidingWindow(target_tokens=8000),
                ),
                session_resumption=types.SessionResumptionConfig(
                    handle=self._resume_handle,
                ),
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(
                        disabled=False,
                        silence_duration_ms=550,
                    ),
                ),
            )
            try:
                self._conn_cm = self._client.aio.live.connect(model=model, config=cfg)
                self._session = await asyncio.wait_for(self._conn_cm.__aenter__(), CONNECT_TIMEOUT_S)
                self._recv_task = asyncio.create_task(self._receive_loop())
                short_m = "3.8-Thinking + Neon" if "thinking" in model else f"{model} + Neon"
                self._on_status(f"Ready ({short_m})")
                return True
            except Exception as exc:
                self._conn_cm = None
                self._session = None
                if isinstance(exc, asyncio.TimeoutError):
                    exc = f"connect timed out after {CONNECT_TIMEOUT_S:.0f}s"
                print(f"[spotter] Live connect failed: {exc}", flush=True)
                self._on_status(f"Live error: {exc}")
                return False

    async def send_prompt(self, text: str) -> None:
        self._accept_audio_out = True
        self._audio_bytes_this_turn = 0
        if not await self.ensure_connected():
            return
        try:
            async with self._send_lock:
                await self._session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part.from_text(text=text)]),
                    turn_complete=True,
                )
        except Exception:
            await self.disconnect()
            if await self.ensure_connected():
                async with self._send_lock:
                    await self._session.send_client_content(
                        turns=types.Content(role="user", parts=[types.Part.from_text(text=text)]),
                        turn_complete=True,
                    )

    async def send_mic_pcm(self, pcm16k: bytes) -> None:
        if self._session is None:
            if not await self.ensure_connected():
                return
        if not pcm16k:
            return
        self._accept_audio_out = True
        # While a blocking tool call is running/thinking, send silence so desk/breathing noise
        # never triggers server-side tool_call_cancellation!
        payload = b"\x00" * len(pcm16k) if self._tools_in_flight > 0 else pcm16k
        try:
            async with self._send_lock:
                if self._session is not None:
                    await self._session.send_realtime_input(
                        audio=types.Blob(data=payload, mime_type="audio/pcm;rate=16000")
                    )
        except Exception:
            pass

    async def end_mic_stream(self) -> None:
        if self._session is None:
            return
        try:
            async with self._send_lock:
                if self._session is not None:
                    await self._session.send_realtime_input(audio_stream_end=True)
        except Exception:
            pass

    def suppress_current_output(self) -> None:
        self._accept_audio_out = False

    async def _receive_loop(self) -> None:
        session = self._session
        if session is None:
            return
        try:
            while self._session is session:
                async for msg in session.receive():
                    upd = getattr(msg, "session_resumption_update", None)
                    if upd and getattr(upd, "new_handle", None):
                        self._resume_handle = upd.new_handle

                    sc = msg.server_content
                    if sc is not None:
                        if sc.interrupted:
                            self._on_interrupted()
                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if part.inline_data and part.inline_data.data and self._accept_audio_out:
                                    self.last_output_mono = time.monotonic()
                                    self._audio_bytes_this_turn += len(part.inline_data.data)
                                    self._on_audio(part.inline_data.data)
                        if sc.output_transcription and getattr(sc.output_transcription, "text", None):
                            if self._accept_audio_out:
                                self.last_output_mono = time.monotonic()
                                self._on_transcript(sc.output_transcription.text, False)
                        if sc.input_transcription and getattr(sc.input_transcription, "text", None):
                            self._audio_bytes_this_turn = 0
                            self._on_transcript(sc.input_transcription.text, True)

                    tc = msg.tool_call
                    if tc and tc.function_calls:
                        # Run tools off the receive loop so WebSocket reads/pings never stall
                        # during multi-second calls like ask_deep_coach!
                        asyncio.create_task(self._run_and_reply_tools(session, list(tc.function_calls)))
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._on_status(f"Session closed ({exc})")
        finally:
            if self._session is session:
                self._session = None

    async def _run_and_reply_tools(self, session: Any, calls: list[Any]) -> None:
        self._tools_in_flight += 1
        try:
            responses: list[types.FunctionResponse] = []
            for fc in calls:
                args = dict(fc.args or {})
                try:
                    maybe_coro = self._on_tool_call(fc.name, args)
                    result = await maybe_coro if inspect.isawaitable(maybe_coro) else maybe_coro
                except Exception as exc:
                    result = {"error": str(exc)}
                responses.append(
                    types.FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response={"result": result},
                    )
                )
            if self._session is session and not any(fc.name == "end_debrief" for fc in calls):
                async with self._send_lock:
                    if self._session is session:
                        await session.send_tool_response(function_responses=responses)
        except Exception as exc:
            print(f"[spotter] Tool response error: {exc}", flush=True)
        finally:
            self._tools_in_flight = max(0, self._tools_in_flight - 1)

    async def disconnect(self) -> None:
        async with self._lock:
            task, self._recv_task = self._recv_task, None
            cm, self._conn_cm = self._conn_cm, None
            self._session = None
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            if cm is not None:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    pass
