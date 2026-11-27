"""Live pixels around the card, for the glass, with "nothing changed" detection.

DXGI Desktop Duplication (dxcam) first: it reads the composed desktop, game included, and
says for free when the desktop has not changed at all. GDI (Qt's grabWindow) as a fallback.
Either way the region around the card is compared with the last one, and grab() returns None
when it is the same, so an unchanged screen costs a capture and a compare, never a render.

The card itself must be left out of capture (SetWindowDisplayAffinity WDA_EXCLUDEFROMCAPTURE),
or it would refract its own previous frame.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

import numpy as np


class Backdrop:
    """grab(rect) -> (h, w, 3) uint8 of the desktop rect (physical px), or None if it is the
    same as last time. The rect is clamped to the monitor; .offset says how far the requested
    top-left sits inside what was returned (non-zero only at the monitor's edge)."""

    def __init__(self) -> None:
        self.source: str | None = None       # "dxgi" | "gdi"
        self.offset = (0, 0)
        self._cam = None
        self._cam_origin = (0, 0)
        self._cam_mon = None
        self._prev: np.ndarray | None = None
        self._prev_rect: tuple | None = None
        self.captures = 0
        self.unchanged = 0

    def reset(self) -> None:
        """Forget the last frame (the next grab always counts as new)."""
        self._prev, self._prev_rect = None, None

    def grab(self, rect: tuple[int, int, int, int], hwnd: int | None = None) -> np.ndarray | None:
        frame = None
        if self.source in (None, "dxgi"):
            try:
                frame, fresh = self._grab_dxgi(rect, hwnd)
                self.source = "dxgi"
                if not fresh:
                    self.unchanged += 1
                    return None
            except Exception:
                self._cam, self.source = None, "gdi"
        if frame is None and self.source == "gdi":
            frame = self._grab_gdi(rect)
        if frame is None:
            return None
        return self._if_changed(frame, rect)

    def _if_changed(self, frame: np.ndarray, rect: tuple) -> np.ndarray | None:
        self.captures += 1
        if rect == self._prev_rect and self._prev is not None and np.array_equal(frame, self._prev):
            self.unchanged += 1
            return None
        self._prev, self._prev_rect = frame, rect
        return frame

    # --- DXGI ----------------------------------------------------------------------------

    def _grab_dxgi(self, rect, hwnd) -> tuple[np.ndarray | None, bool]:
        mon = _monitor_of(hwnd, rect)
        if self._cam is None or mon != self._cam_mon:
            self._open(mon)
        ox, oy = self._cam_origin
        w, h = self._cam.width, self._cam.height
        l, t, r, b = rect
        cl, ct = max(l - ox, 0), max(t - oy, 0)
        cr, cb = min(r - ox, w), min(b - oy, h)
        if cr <= cl or cb <= ct:
            raise RuntimeError("card is not on the captured monitor")
        self.offset = (cl - (l - ox), ct - (t - oy))
        frame = self._cam.grab(region=(cl, ct, cr, cb), new_frame_only=True)
        if frame is None:                    # DXGI: the desktop has not been redrawn since
            if self._prev is not None and self._prev_rect == rect:
                return None, False
            frame = self._cam.grab(region=(cl, ct, cr, cb), new_frame_only=False)
            if frame is None:
                return None, False
        return np.asarray(frame), True

    def _open(self, mon) -> None:
        import dxcam
        factory = getattr(dxcam, "__factory", None) or dxcam.DXFactory()   # its one instance
        pick = (0, 0)
        for d, outs in enumerate(factory.outputs):
            for o, out in enumerate(outs):
                if mon is not None and int(out.hmonitor or 0) == int(mon):
                    pick = (d, o)
        if self._cam is not None:
            try:
                self._cam.release()
            except Exception:
                pass
        self._cam = dxcam.create(device_idx=pick[0], output_idx=pick[1],
                                 processor_backend="numpy", output_color="RGB")
        c = factory.outputs[pick[0]][pick[1]].desc.DesktopCoordinates
        self._cam_origin = (c.left, c.top)
        self._cam_mon = mon
        self.reset()

    # --- GDI -----------------------------------------------------------------------------

    def _grab_gdi(self, rect) -> np.ndarray | None:
        from PySide6.QtGui import QGuiApplication, QImage
        l, t, r, b = rect
        screen = QGuiApplication.screenAt(_logical_point(l, t)) or QGuiApplication.primaryScreen()
        if screen is None:
            return None
        dpr = screen.devicePixelRatio()
        g = screen.geometry()
        # grabWindow(0) takes logical coordinates relative to the screen
        x = (l / dpr) - g.left()
        y = (t / dpr) - g.top()
        pm = screen.grabWindow(0, int(x), int(y), int(round((r - l) / dpr)), int(round((b - t) / dpr)))
        img = pm.toImage().convertToFormat(QImage.Format.Format_RGB888)
        if img.isNull():
            return None
        w, h, bpl = img.width(), img.height(), img.bytesPerLine()
        buf = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes()).reshape(h, bpl)
        self.offset = (0, 0)
        return buf[:, :w * 3].reshape(h, w, 3).copy()

    def release(self) -> None:
        if self._cam is not None:
            try:
                self._cam.release()
            except Exception:
                pass
        self._cam = None
        self.reset()


def _logical_point(x: int, y: int):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QGuiApplication
    s = QGuiApplication.primaryScreen()
    dpr = s.devicePixelRatio() if s else 1.0
    return QPoint(int(x / dpr), int(y / dpr))


def _monitor_of(hwnd: int | None, rect) -> int | None:
    if sys.platform != "win32":
        return None
    u = ctypes.windll.user32
    MONITOR_DEFAULTTONEAREST = 2
    if hwnd:
        u.MonitorFromWindow.restype = wintypes.HMONITOR
        return u.MonitorFromWindow(wintypes.HWND(hwnd), MONITOR_DEFAULTTONEAREST)
    u.MonitorFromPoint.restype = wintypes.HMONITOR
    pt = wintypes.POINT((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
    return u.MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST)


def windows_prefs() -> dict[str, bool]:
    """The Windows equivalents of Apple's Reduce Transparency / Reduce Motion / Increase
    Contrast: Settings > Personalization > Colors > Transparency effects, Settings > Ease of
    Access > Show animations, and High contrast."""
    prefs = {"transparency": True, "animations": True, "high_contrast": False}
    if sys.platform != "win32":
        return prefs
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            prefs["transparency"] = bool(winreg.QueryValueEx(k, "EnableTransparency")[0])
    except OSError:
        pass
    try:
        u = ctypes.windll.user32
        on = wintypes.BOOL()
        SPI_GETCLIENTAREAANIMATION = 0x1042
        if u.SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(on), 0):
            prefs["animations"] = bool(on.value)

        class HIGHCONTRAST(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwFlags", wintypes.DWORD),
                        ("lpszDefaultScheme", wintypes.LPWSTR)]
        hc = HIGHCONTRAST(ctypes.sizeof(HIGHCONTRAST), 0, None)
        SPI_GETHIGHCONTRAST, HCF_HIGHCONTRASTON = 0x0042, 0x1
        if u.SystemParametersInfoW(SPI_GETHIGHCONTRAST, hc.cbSize, ctypes.byref(hc), 0):
            prefs["high_contrast"] = bool(hc.dwFlags & HCF_HIGHCONTRASTON)
    except (AttributeError, OSError):
        pass
    return prefs
