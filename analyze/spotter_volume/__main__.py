"""Aim-volume overlay: shows how long you aimed today, and stays off screen while you aim.

    analyze\\.venv\\Scripts\\pythonw.exe -m spotter_volume       (spotter-volume.bat does this)

Settings: spotter-volume.json at the repo root (daily target, font, corner, hotkey).
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QAbstractNativeEventFilter, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from spotter_analyze import kovaaks
from spotter_analyze.format import FRAME_COLS
from spotter_analyze.stream import LiveTracker, RunAborted, RunCompleted, SpotTailer

from .config import PATH as CONFIG_PATH, REPO, Config
from .gate import Gate, HookLog, State
from .ledger import Ledger
from .overlay import DARK, VolumeCard, fmt_clock

GAME_EXE = "fpsaimtrainer-win64-shipping.exe"
TICK_MS = 100              # how often we look at the log and the foreground window
CATCHUP_BYTES = 4_000_000  # per tick while reading an existing log, so the UI stays live
QUIT_AFTER_HOOK_GONE_S = 60
PEEK_S = 6.0               # "show now" lasts this long, or until you move the mouse
HOTKEYS = {"Home": 0x24, "End": 0x23, "Insert": 0x2D, "Pause": 0x13, "ScrollLock": 0x91,
           **{f"F{i}": 0x6F + i for i in range(1, 13)}}
HOTKEY_ID = 0x5EE

log = logging.getLogger("spotter_volume")


class _Recording:
    """SpotTailer that remembers the frames of its last poll, so the gate sees exactly what
    the run tracker saw."""

    def __init__(self, inner: SpotTailer) -> None:
        self.inner = inner
        self.last = np.empty((0, FRAME_COLS))

    def poll(self, max_bytes: int | None = None) -> np.ndarray:
        self.last = self.inner.poll(max_bytes=max_bytes)
        return self.last

    def __getattr__(self, name):
        return getattr(self.inner, name)


def game_in_front() -> bool:
    """Is KovaaK's the foreground window?"""
    if sys.platform != "win32":
        return True
    u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
    hwnd = u32.GetForegroundWindow()
    if not hwnd:
        return False
    pid = wintypes.DWORD()
    u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    h = k32.OpenProcess(0x1000, False, pid.value)          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        buf = ctypes.create_unicode_buffer(520)
        n = wintypes.DWORD(len(buf))
        if not k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
            return False
        return os.path.basename(buf.value).lower() == GAME_EXE
    finally:
        k32.CloseHandle(h)


class _HotkeyFilter(QAbstractNativeEventFilter):
    def __init__(self, callback) -> None:
        super().__init__()
        self.callback = callback

    def nativeEventFilter(self, event_type, message):
        if event_type == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == 0x0312 and msg.wParam == HOTKEY_ID:      # WM_HOTKEY
                self.callback()
                return True, 0
        return False, 0


class App:
    def __init__(self, qapp: QApplication, cfg: Config, runs_dir: Path, stats_dir: Path,
                 data_dir: Path) -> None:
        self.qapp, self.cfg = qapp, cfg
        self.runs_dir, self.stats_dir, self.data_dir = runs_dir, stats_dir, data_dir
        self.card = VolumeCard(cfg)
        self.ledger = Ledger(stats_dir, data_dir)
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._loading: Future | None = self._pool.submit(self.ledger.load)
        self.ready = False

        self.spot: Path | None = None
        self.tracker: LiveTracker | None = None
        self.rec: _Recording | None = None
        self.gate = Gate()
        self.hook = HookLog(None)
        self.caught_up = False
        self.dismissed = False          # hidden by hand; back after your next run
        self.peek_until = 0.0
        self._peek_move_t = -1e9
        self._hook_seen_alive = 0.0
        self._last_spot_check = 0.0

        self._tray = self._make_tray()
        self._register_hotkey()
        self.timer = QTimer(interval=TICK_MS)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    # --- setup ---------------------------------------------------------------------------

    def _make_tray(self) -> QSystemTrayIcon:
        pm = QPixmap(64, 64)
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(DARK.blue)
        p.setPen(QColor(0, 0, 0, 0))
        p.drawEllipse(8, 8, 48, 48)
        p.end()
        tray = QSystemTrayIcon(QIcon(pm))
        menu = QMenu()
        a_toggle = QAction("Show now / hide", menu)
        a_toggle.triggered.connect(self.toggle)
        a_cfg = QAction("Settings…", menu)
        a_cfg.triggered.connect(lambda: os.startfile(CONFIG_PATH))
        a_quit = QAction("Quit", menu)
        a_quit.triggered.connect(self.qapp.quit)
        menu.addActions([a_toggle, a_cfg, a_quit])
        tray.setContextMenu(menu)
        tray.setToolTip("Spotter volume")
        tray.activated.connect(lambda reason: self.toggle()
                               if reason == QSystemTrayIcon.ActivationReason.Trigger else None)
        tray.show()
        self._tray_menu = menu
        return tray

    def _register_hotkey(self) -> None:
        vk = HOTKEYS.get(self.cfg.hotkey or "")
        if not vk or sys.platform != "win32":
            return
        if not ctypes.windll.user32.RegisterHotKey(None, HOTKEY_ID, 0x4000, vk):   # MOD_NOREPEAT
            log.warning("hotkey %s is taken (is the coach running?); use the tray icon", self.cfg.hotkey)
            return
        self._filter = _HotkeyFilter(self.toggle)
        self.qapp.installNativeEventFilter(self._filter)

    def toggle(self) -> None:
        """Home key / tray click. Card up: hide it until after your next run. Card down: show
        it now for a few seconds, anywhere, but it still vanishes the moment you move the mouse
        or shoot, so a peek can never sit on screen while you aim."""
        if self.card.isVisible():
            self.dismissed, self.peek_until = True, 0.0
        else:
            self.dismissed = False
            self.peek_until = time.monotonic() + PEEK_S
            self._peek_move_t = self.gate.last_move_t
        self.tick()

    def _follow_newest_log(self) -> None:
        """Attach to the newest .spot (a new one appears each time the hook is injected)."""
        now = time.monotonic()
        if now - self._last_spot_check < 3.0 and self.spot is not None:
            return
        self._last_spot_check = now
        logs = sorted(self.runs_dir.glob("spotter-*.spot"), key=lambda p: p.stat().st_mtime)
        if not logs or logs[-1] == self.spot:
            return
        self.spot = logs[-1]
        self.rec = _Recording(SpotTailer(self.spot))
        self.tracker = LiveTracker(self.rec, self.stats_dir)
        self.gate = Gate()
        self.hook = HookLog(self.spot.with_suffix(".log"))
        self.caught_up = False
        log.info("following %s", self.spot.name)

    # --- the loop ------------------------------------------------------------------------

    def tick(self) -> None:
        try:
            self._tick()
        except Exception:                        # never let one bad tick kill the overlay
            log.exception("tick failed")
            self.card.conceal()

    def _tick(self) -> None:
        if not self.ready:
            if self._loading is None or not self._loading.done():
                self._apply(State.UNKNOWN)
                return
            self._loading.result()               # surface a load error in the log
            self.ready = True
            self._refresh()

        self._follow_newest_log()
        gained = 0.0
        if self.tracker is not None and self.rec is not None:
            # while catching up on an existing log, read it in pieces so the UI never stalls
            size = self.spot.stat().st_size if self.spot and self.spot.exists() else 0
            for ev in self.tracker.step(time.monotonic(), max_bytes=CATCHUP_BYTES):
                gained += self._on_event(ev)
            self.gate.feed(self.rec.last)
            self.caught_up = self.rec.inner.offset >= size - 65_536
        if gained:
            self._refresh(gained if self.caught_up else 0.0)

        alive, paused = self.hook.status()
        if alive:
            self._hook_seen_alive = time.monotonic()
        elif self._hook_seen_alive and time.monotonic() - self._hook_seen_alive > QUIT_AFTER_HOOK_GONE_S:
            log.info("hook gone (game closed?); quitting")
            self.qapp.quit()
            return
        state = self.gate.state(paused, alive) if self.caught_up else State.UNKNOWN
        self._apply(state)

    def _on_event(self, ev) -> float:
        wall0 = self.rec.inner.wall0 if self.rec else 0.0
        if isinstance(ev, RunCompleted):
            run = self.ledger.add_saved(ev.stats)
            return run.seconds if run else 0.0
        if isinstance(ev, RunAborted):
            run = self.ledger.add_restart(f"{self.spot.stem}:{ev.t0:.1f}",
                                          datetime.fromtimestamp(wall0 + ev.t0),
                                          datetime.fromtimestamp(wall0 + ev.t1))
            return run.seconds if run else 0.0
        return 0.0

    def _refresh(self, gained: float = 0.0) -> None:
        s = self.ledger.summary(datetime.now(), self.cfg.daily_target_min)
        self.card.set_summary(s, gained)
        self._tray.setToolTip(f"Aim today {fmt_clock(s.today_s)} / {fmt_clock(s.target_s)}")

    def _apply(self, state: State) -> None:
        if state is State.AIMING:
            self.dismissed = False
        if self.gate.last_move_t != self._peek_move_t:
            self.peek_until = 0.0                # you moved: the peek is over
        peeking = time.monotonic() < self.peek_until
        show = peeking or (state is State.IDLE and not self.dismissed and game_in_front())
        if show:
            self.card.present()
        else:
            self.card.conceal()


def main(argv: list[str] | None = None) -> int:
    data_dir = REPO / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=data_dir / "volume.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    # exact fractional scaling (125%, 150%) keeps text crisp instead of rounding the whole UI
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    qapp = QApplication(sys.argv[:1])
    qapp.setQuitOnLastWindowClosed(False)
    cfg = Config.load()
    app = App(qapp, cfg, REPO / "runs", kovaaks.DEFAULT_ROOT / "stats", data_dir)
    log.info("started; font %s, target %.0f min", app.card.fonts.family, cfg.daily_target_min)
    return qapp.exec()


if __name__ == "__main__":
    raise SystemExit(main())
