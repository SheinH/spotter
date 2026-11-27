"""Low-latency audio I/O with hard in-run mic off, mic selector/meter, and <25ms barge-in flush."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
import threading
import time

import numpy as np

IN_RATE = 16000          # Gemini Live expects 16 kHz 16-bit mono PCM
OUT_RATE = 24000         # Gemini Live outputs 24 kHz 16-bit mono PCM
BLOCK_MS = 20            # 20 ms hardware blocks -> instant interruption
ECHO_TAIL_S = 0.45       # cover Windows audio pipeline latency + room reverb after coach stops
BARGE_IN_RMS = 9500.0    # loud close-mic speech required to barge in over active speaker playback


class AudioEngine:
    """Manages 24 kHz PCM playback ring buffer and strictly gated 16 kHz mic capture."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_mic_chunk: Callable[[bytes], None],
        enabled: bool = True,
        input_device: int | None = None,
    ) -> None:
        self._loop = loop
        self._on_mic_chunk = on_mic_chunk
        self._enabled = enabled
        self.input_device: int | None = input_device

        self._lock = threading.Lock()
        self._out_buf = bytearray()
        self._playing = False
        self._last_play_mono: float = 0.0
        self._barge_streak: int = 0
        self.mic_level: float = 0.0  # 0.0 .. 1.0 live input meter

        self._sd = None
        self._out_stream = None
        self._in_stream = None
        self.mic_active = False

        if enabled:
            try:
                import sounddevice as sd
                self._sd = sd
            except Exception:
                self._sd = None

    def list_input_devices(self) -> list[tuple[int | None, str]]:
        """Return deduplicated MME input devices for the UI mic selector."""
        if not self._sd:
            return [(None, "No audio driver")]
        out: list[tuple[int | None, str]] = []
        try:
            def_in = self._sd.default.device[0]
            devs = self._sd.query_devices()
            def_name = devs[def_in]["name"] if (def_in is not None and 0 <= def_in < len(devs)) else "Default"
            out.append((None, f"Default ({def_name[:26]})"))

            seen_names: set[str] = set()
            for i, d in enumerate(devs):
                if d["max_input_channels"] <= 0:
                    continue
                api = self._sd.query_hostapis(d["hostapi"])["name"]
                if api not in ("MME", "Windows DirectSound"):
                    continue
                name = str(d["name"]).strip()
                if "Sound Mapper" in name or "Primary Sound" in name or name in seen_names:
                    continue
                seen_names.add(name)
                out.append((i, name[:34]))
        except Exception:
            if not out:
                out.append((None, "Default Mic"))
        return out

    def set_input_device(self, device_idx: int | None) -> None:
        if self.input_device == device_idx:
            return
        was_active = self.mic_active
        if was_active:
            self.set_mic_active(False)
        self.input_device = device_idx
        if was_active:
            self.set_mic_active(True)

    def start_output(self) -> None:
        if not self._sd or self._out_stream is not None:
            return
        blocksize = int(OUT_RATE * BLOCK_MS / 1000)

        def _out_cb(outdata, frames, _time, _status):
            want = frames * 2
            with self._lock:
                avail = len(self._out_buf)
                if avail >= want:
                    outdata[:] = self._out_buf[:want]
                    del self._out_buf[:want]
                    self._playing = True
                    self._last_play_mono = time.monotonic()
                elif avail > 0:
                    outdata[:avail] = self._out_buf
                    outdata[avail:] = b"\x00" * (want - avail)
                    self._out_buf.clear()
                    self._playing = True
                    self._last_play_mono = time.monotonic()
                else:
                    outdata[:] = b"\x00" * want
                    self._playing = False

        try:
            self._out_stream = self._sd.RawOutputStream(
                samplerate=OUT_RATE,
                blocksize=blocksize,
                channels=1,
                dtype="int16",
                callback=_out_cb,
            )
            self._out_stream.start()
        except Exception:
            self._out_stream = None

    def enqueue_output(self, pcm24k: bytes) -> None:
        if not pcm24k or not self._out_stream:
            return
        with self._lock:
            self._out_buf.extend(pcm24k)

    def clear_playback(self) -> None:
        """Flush buffered audio immediately (<20 ms cutoff on barge-in, F8, or run start)."""
        with self._lock:
            self._out_buf.clear()
            self._playing = False

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._playing or len(self._out_buf) > 0

    def set_mic_active(self, active: bool) -> None:
        """Hard gate: mic hardware stream is ONLY open during unmuted DEBRIEF."""
        if not self._sd:
            self.mic_active = False
            self.mic_level = 0.0
            return
        if active and not self.mic_active:
            blocksize = int(IN_RATE * 40 / 1000)  # 40 ms chunks to WebSocket

            def _in_cb(indata, _frames, _time, _status):
                chunk = bytes(indata)
                arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(arr * arr))) if len(arr) else 0.0

                # Echo gate: block speaker bleed into desk mics while coach is speaking (+450 ms tail),
                # unless the user speaks loudly right into the mic for >= 3 frames (~120 ms) to barge in.
                in_echo = self.is_speaking or (time.monotonic() - self._last_play_mono) < ECHO_TAIL_S
                if in_echo:
                    if rms >= BARGE_IN_RMS:
                        self._barge_streak += 1
                        if self._barge_streak >= 3:
                            self.clear_playback()
                            self._last_play_mono = 0.0
                            self.mic_level = min(1.0, rms / 3500.0)
                            self._loop.call_soon_threadsafe(self._on_mic_chunk, chunk)
                            return
                    else:
                        self._barge_streak = 0
                    self.mic_level = 0.0
                    self._loop.call_soon_threadsafe(self._on_mic_chunk, b"\x00" * len(chunk))
                    return

                self._barge_streak = 0
                self.mic_level = min(1.0, rms / 3500.0)
                self._loop.call_soon_threadsafe(self._on_mic_chunk, chunk)

            try:
                self._in_stream = self._sd.RawInputStream(
                    device=self.input_device,
                    samplerate=IN_RATE,
                    blocksize=blocksize,
                    channels=1,
                    dtype="int16",
                    callback=_in_cb,
                )
                self._in_stream.start()
                self.mic_active = True
            except Exception:
                self._in_stream = None
                self.mic_active = False
                self.mic_level = 0.0
        elif not active and self.mic_active:
            self.mic_active = False
            self.mic_level = 0.0
            if self._in_stream is not None:
                try:
                    self._in_stream.stop()
                    self._in_stream.close()
                except Exception:
                    pass
                self._in_stream = None

    def close(self) -> None:
        self.set_mic_active(False)
        self.clear_playback()
        if self._out_stream is not None:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None
