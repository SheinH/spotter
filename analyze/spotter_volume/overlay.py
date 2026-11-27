"""The card: today's aim time against the target, this scenario, all-time hours, streak.

Painted by hand in Apple's style (system colors, SF Pro Text with Apple's tracking, tabular
figures, continuous corners), click-through and never focused. Two materials:

- solid: opaque systemGray6, what Apple shows with Reduce Transparency on.
- glass: live Liquid Glass rendered on the GPU (gpu_glass.py) from what is around the card
  (capture.py), re-rendered only when that changes. Styled after the Liquid Glass HIG: it
  turns light or dark with the backdrop, keeps color for the few things that matter, puts no
  glass on glass, and materializes in instead of fading. Falls back to solid when Windows has
  transparency effects off or high contrast on, or when the GPU or capture is unavailable.

Nothing is captured or rendered while the card is hidden. No scores, on purpose: this is
about volume.
"""
from __future__ import annotations

import ctypes
import logging
import math
import sys
from dataclasses import dataclass

import numpy as np

from PySide6.QtCore import (QEasingCurve, QPointF, QRectF, Qt, QTimer, QVariantAnimation,
                            Signal)
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF, QGuiApplication, QImage,
                           QPainter, QPainterPath, QPen)
from PySide6.QtWidgets import QWidget

from . import capture, glass
from .config import Config
from .ledger import Summary

log = logging.getLogger("spotter_volume")


@dataclass(frozen=True)
class Palette:
    """Apple's semantic system colors for one appearance."""
    label: QColor
    label2: QColor       # secondaryLabel
    label3: QColor       # tertiaryLabel
    fill: QColor         # systemFill
    separator: QColor
    blue: QColor
    green: QColor
    orange: QColor


DARK = Palette(QColor(255, 255, 255), QColor(235, 235, 245, 153), QColor(235, 235, 245, 77),
               QColor(120, 120, 128, 92), QColor(84, 84, 88, 150),
               QColor(10, 132, 255), QColor(48, 209, 88), QColor(255, 159, 10))
LIGHT = Palette(QColor(0, 0, 0), QColor(60, 60, 67, 153), QColor(60, 60, 67, 77),
                QColor(120, 120, 128, 51), QColor(60, 60, 67, 74),
                QColor(0, 122, 255), QColor(52, 199, 89), QColor(255, 149, 0))
GREEN = DARK.green
MATERIAL = QColor(28, 28, 30)            # systemGray6, opaque (Apple's Reduce Transparency look)
RIM = QColor(255, 255, 255, 26)
RIM_HIGH_CONTRAST = QColor(255, 255, 255, 140)
GLASS_RIM = QColor(255, 255, 255, 46)

WIDTH = 296          # card, logical px
PAD_X, PAD_Y = 18, 15
RADIUS = 22          # continuous corners read smaller than circular ones, so a bit larger
SHADOW = 48          # room around the card for its (wide, soft) shadow
GAIN_HOLD_MS = 7000  # how long "+1:02" stays before fading
GLASS_FPS = 30       # how often the glass looks for a changed backdrop while the card is up
VICINITY = 40        # logical px around the card the glass can sample (refraction + frost)
MATERIALIZE_MS = 320
APPEARANCE_MS = 200

# SF Pro tracking (px at these sizes), from Apple's typography tables; other fonts get none
SF_TRACKING = {11: 0.07, 12: 0.0, 13: -0.08, 15: -0.24, 17: -0.43, 34: -0.6}


def squircle(rect: QRectF, radius: float) -> QPainterPath:
    """Apple's continuous-corner rounded rectangle (what iOS draws for cards, icons and
    sheets): the curvature eases in along each edge instead of starting abruptly like a plain
    circular corner. Control points are the published reverse-engineering of UIBezierPath's
    continuous corners; each corner reaches 1.528665 x radius along both edges."""
    k = 1.52866483
    r = min(radius, min(rect.width(), rect.height()) / 2 / k)
    # one corner, as (back along the incoming edge, forward along the outgoing edge), in radii
    seg = [((1.08849299, 0.0), (0.86840701, 0.0), (0.63149399, 0.07491100)),
           ((0.37282392, 0.16905899), (0.16905899, 0.37282392), (0.07491100, 0.63149399)),
           ((0.0, 0.86840701), (0.0, 1.08849299), (0.0, k))]
    corners = [  # corner point, incoming edge direction, outgoing edge direction (clockwise)
        (rect.topRight(), (1, 0), (0, 1)), (rect.bottomRight(), (0, 1), (-1, 0)),
        (rect.bottomLeft(), (-1, 0), (0, -1)), (rect.topLeft(), (0, -1), (1, 0))]
    path = QPainterPath(QPointF(rect.left() + k * r, rect.top()))
    for c, d1, d2 in corners:
        pt = lambda a, b: QPointF(c.x() + (-a * d1[0] + b * d2[0]) * r, c.y() + (-a * d1[1] + b * d2[1]) * r)
        path.lineTo(pt(k, 0.0))
        for c1, c2, end in seg:
            path.cubicTo(pt(*c1), pt(*c2), pt(*end))
    path.closeSubpath()
    return path


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    return QColor(*(round(x + (y - x) * t) for x, y in zip(a.getRgb(), b.getRgb())))


def mix_palette(t: float) -> Palette:
    """0 = dark, 1 = light, anything between while the card crossfades."""
    if t <= 0:
        return DARK
    if t >= 1:
        return LIGHT
    return Palette(*(_mix(getattr(DARK, f), getattr(LIGHT, f), t) for f in Palette.__dataclass_fields__))


def fmt_clock(seconds: float) -> str:
    s = int(round(max(seconds, 0)))
    h, m, s = s // 3600, s // 60 % 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_hours(seconds: float) -> str:
    h = seconds / 3600
    if h < 10:
        return f"{h:.1f} h"
    return f"{h:,.0f} h" if h >= 100 else f"{h:.1f} h"


class Fonts:
    """One place that decides the typeface. Set "font" in spotter-volume.json to swap it."""

    def __init__(self, cfg: Config) -> None:
        fams = set(QFontDatabase.families())
        self.family = cfg.font if cfg.font in fams else cfg.fallback_font
        self.is_sf = self.family.startswith("SF Pro")

    def get(self, px: int, weight: QFont.Weight = QFont.Weight.Normal, numbers: bool = False) -> QFont:
        f = QFont(self.family)
        f.setPixelSize(px)
        f.setWeight(weight)
        # no hinting + antialiasing = the smooth, Mac-like rendering; full hinting looks pinched
        f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        if self.is_sf:
            f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, SF_TRACKING.get(px, 0.0))
        if numbers:
            try:   # tabular figures, so 1:11 and 8:08 take the same width and nothing jiggles
                f.setFeature(QFont.Tag("tnum"), 1)
            except (AttributeError, TypeError):
                pass
        return f


class VolumeCard(QWidget):
    shown_changed = Signal(bool)

    def __init__(self, cfg: Config) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool
                         | Qt.WindowType.WindowTransparentForInput
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.cfg = cfg
        self.fonts = Fonts(cfg)
        self.summary: Summary | None = None
        self.loading = True
        self._shown_today = 0.0          # what the bar currently shows (animates toward today_s)
        self._gain_s = 0.0
        self._gain_alpha = 0.0
        self._glass: QImage | None = None      # rendered glass background, card-sized
        self._shadow: QImage | None = None
        self._maps: glass.GlassMaps | None = None
        self._maps_key: tuple | None = None
        self.glass_ok: bool | None = None       # None = not tried yet; False = fell back to solid
        self._gpu = None                         # gpu_glass.GpuGlass, made on first use
        self._gpu_key: tuple | None = None
        self._backdrop = capture.Backdrop()
        self._appearance = glass.Appearance()
        self._luma = 0.0
        self._light = 0.0                        # 0 dark .. 1 light, animated
        self._amount = 1.0                       # materialize progress, 0..1
        self.prefs = capture.windows_prefs()
        self.composes = 0

        self._bar_anim = QVariantAnimation(self, duration=750,
                                           easingCurve=QEasingCurve(QEasingCurve.Type.OutCubic))
        self._bar_anim.valueChanged.connect(self._on_bar)
        self._gain_fade = QVariantAnimation(self, startValue=1.0, endValue=0.0, duration=900)
        self._gain_fade.valueChanged.connect(self._on_gain_alpha)
        self._gain_timer = QTimer(self, singleShot=True, interval=GAIN_HOLD_MS)
        self._gain_timer.timeout.connect(self._gain_fade.start)
        self._fade_in = QVariantAnimation(self, startValue=0.0, endValue=1.0, duration=220,
                                          easingCurve=QEasingCurve(QEasingCurve.Type.OutCubic))
        self._fade_in.valueChanged.connect(lambda v: self.setWindowOpacity(float(v)))
        self._materialize = QVariantAnimation(self, startValue=0.0, endValue=1.0, duration=MATERIALIZE_MS,
                                              easingCurve=QEasingCurve(QEasingCurve.Type.OutCubic))
        self._materialize.valueChanged.connect(self._on_amount)
        self._light_anim = QVariantAnimation(self, duration=APPEARANCE_MS,
                                             easingCurve=QEasingCurve(QEasingCurve.Type.InOutQuad))
        self._light_anim.valueChanged.connect(self._on_light)

        self.resize(WIDTH + 2 * SHADOW, self._card_height() + 2 * SHADOW)
        self._live = QTimer(self, interval=1000 // GLASS_FPS)
        self._live.timeout.connect(self._tick_glass)
        if cfg.style == "glass":
            _exclude_from_capture(self)         # the glass must never see the card itself

    # --- data ----------------------------------------------------------------------------

    def set_summary(self, s: Summary, gain_s: float = 0.0) -> None:
        """New numbers. With a gain (a run just ended) the bar grows into place and "+m:ss"
        shows for a few seconds, the next time the card is on screen."""
        self.loading = False
        first = self.summary is None
        self.summary = s
        if first:
            self._shown_today = s.today_s
        if gain_s > 0:
            self._gain_s, self._gain_alpha = gain_s, 1.0
            self._gain_fade.stop()
            self._gain_timer.stop()
        if self.isVisible():
            self._animate_in()
        self.update()

    # --- show / hide ---------------------------------------------------------------------

    @property
    def glass_wanted(self) -> bool:
        return (self.cfg.style == "glass" and self.glass_ok is not False
                and self.prefs["transparency"] and not self.prefs["high_contrast"])

    def present(self) -> None:
        if self.isVisible():
            return
        self.prefs = capture.windows_prefs()
        self._place()
        self._ensure_maps()
        animate = self.prefs["animations"]
        if self.glass_wanted:
            self._backdrop.reset()
            self._amount = 0.0 if animate else 1.0
            self._tick_glass()                  # first frame before the card is on screen
        if self.glass_wanted:
            self.setWindowOpacity(1.0)          # glass materializes instead of fading
            self.show()
            if animate:
                self._materialize.start()
            self._live.start()
        else:
            self._amount = 1.0
            self.setWindowOpacity(0.0 if animate else 1.0)
            self.show()
            if animate:
                self._fade_in.start()
        _keep_on_top(self)
        self._animate_in()
        self.shown_changed.emit(True)

    def conceal(self) -> None:
        """Instant: this runs when a scenario starts, so no fade-out, and from here on nothing
        is captured or rendered until the card comes back."""
        if not self.isVisible():
            return
        self._fade_in.stop()
        self._materialize.stop()
        self._live.stop()
        self.hide()
        self.shown_changed.emit(False)

    def _animate_in(self) -> None:
        if self.summary is None:
            return
        if abs(self._shown_today - self.summary.today_s) > 0.5:
            self._bar_anim.stop()
            self._bar_anim.setStartValue(float(self._shown_today))
            self._bar_anim.setEndValue(float(self.summary.today_s))
            self._bar_anim.start()
        if self._gain_alpha > 0 and not self._gain_timer.isActive() and self._gain_fade.state() != QVariantAnimation.State.Running:
            self._gain_timer.start()

    def _on_bar(self, v) -> None:
        self._shown_today = float(v)
        self.update()

    def _on_gain_alpha(self, v) -> None:
        self._gain_alpha = float(v)
        self.update()

    def _on_amount(self, v) -> None:
        self._amount = float(v)
        self._compose()

    def _on_light(self, v) -> None:
        self._light = float(v)
        self._compose()

    def _place(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if not screen:
            return
        g = screen.geometry()
        m = self.cfg.margin_px - SHADOW
        x = g.left() + m if "left" in self.cfg.corner else g.right() - self.width() - m + 1
        y = g.bottom() - self.height() - m + 1 if "bottom" in self.cfg.corner else g.top() + m
        self.move(x, y)

    # --- glass and shadow ----------------------------------------------------------------

    def _card_rect(self) -> QRectF:
        return QRectF(SHADOW, SHADOW, WIDTH, self._card_height())

    def _dpr(self) -> float:
        screen = QGuiApplication.primaryScreen()
        return float(screen.devicePixelRatio()) if screen else 1.0

    def _ensure_maps(self) -> None:
        """Glass maps and the shadow depend only on size, scale and material: build once."""
        dpr = self._dpr()
        glassy = self.glass_wanted
        key = (self.width(), self.height(), self._card_height(), dpr, glassy)
        if key == self._maps_key:
            return
        card = self._card_rect()
        cw, ch = round(card.width() * dpr), round(card.height() * dpr)
        self._maps = glass.build_maps(_shape_alpha(QRectF(0, 0, card.width(), card.height()), cw, ch, dpr), dpr)
        full = _shape_alpha(card, round(self.width() * dpr), round(self.height() * dpr), dpr)
        # Liquid Glass floats on a lighter shadow than an opaque card does
        sh = glass.shadow(full, dpr, opacity=0.16 if glassy else 0.30)
        rgba = np.zeros(sh.shape + (4,), np.uint8)
        rgba[..., 3] = (sh * 255 + 0.5).astype(np.uint8)        # black, premultiplied
        self._shadow = _qimage(rgba, dpr)
        self._maps_key = key

    def _fall_back(self, why: str) -> None:
        if self.glass_ok is not False:
            log.warning("glass off, using the solid card: %s", why)
        self.glass_ok = False
        self._glass = None
        self._live.stop()
        self._materialize.stop()
        self._amount = 1.0
        self.setWindowOpacity(1.0)
        self._ensure_maps()                     # the solid card's (darker) shadow
        self.update()

    def _capture_rect(self) -> tuple[tuple[int, int, int, int], int]:
        """Desktop rect (physical px) of the card plus everything the glass samples around
        it, and that margin in device px."""
        dpr = self._dpr()
        h, w = self._maps.alpha.shape
        m = int(math.ceil(VICINITY * dpr))
        left = round((self.x() + SHADOW) * dpr)
        top = round((self.y() + SHADOW) * dpr)
        return (left - m, top - m, left + w + m, top + h + m), m

    def _tick_glass(self) -> None:
        """Look at what is around the card; render only if it changed."""
        if not self.glass_wanted:
            self._live.stop()
            return
        try:
            self._ensure_maps()
            rect, m = self._capture_rect()
            frame = self._backdrop.grab(rect, int(self.winId()))
            if frame is None:
                return                          # nothing new around the card: no work
            if self.glass_ok is None:
                log.info("glass on: %s capture, GPU render", self._backdrop.source)
            ox, oy = self._backdrop.offset
            self.set_backdrop(frame, (m - ox, m - oy))
        except Exception as e:                  # the solid card always works
            self._fall_back(f"{type(e).__name__}: {e}")

    def set_backdrop(self, rgb: np.ndarray | None, offset: tuple[int, int] = (0, 0)) -> None:
        """Pixels around the card, (h, w, 3) uint8 at device resolution, with the card's
        top-left at offset; None or a blank (black) capture = solid card."""
        self._ensure_maps()
        if rgb is None or self._maps is None or glass.looks_blank(rgb):
            if rgb is not None:
                self._fall_back("the screen capture came back blank")
            return
        h, w = self._maps.alpha.shape
        ox, oy = offset
        under = rgb[max(oy, 0):oy + h, max(ox, 0):ox + w]
        self._luma = glass.backdrop_luma(under if under.size else rgb)
        light = 1.0 if self._appearance.update(self._luma) else 0.0
        self._renderer().set_backdrop(rgb, offset)
        self.glass_ok = True
        if light != self._light_target():
            if self.prefs["animations"] and self.isVisible():
                self._light_anim.stop()
                self._light_anim.setStartValue(self._light)
                self._light_anim.setEndValue(light)
                self._light_anim.start()
            else:
                self._light = light
        self._compose()

    def _light_target(self) -> float:
        if self._light_anim.state() == QVariantAnimation.State.Running:
            return float(self._light_anim.endValue())
        return self._light

    def _renderer(self):
        from .gpu_glass import GpuGlass
        if self._gpu is None or self._gpu_key != self._maps_key:
            if self._gpu is not None:
                self._gpu.release()
            self._gpu = GpuGlass(self._maps, self._dpr())
            self._gpu_key = self._maps_key
        return self._gpu

    def _compose(self) -> None:
        """Composite pass only: the frost is cached on the GPU, so animation frames and
        appearance crossfades never re-blur."""
        if self._gpu is None or not self._gpu.ready or not self.glass_wanted:
            self.update()
            return
        p = self._gpu.p
        tint = tuple((a + (b - a) * self._light) / 255.0 for a, b in zip(p.tint_dark, p.tint_light))
        alpha = glass.legibility_tint(self._luma, self._light, p)
        self._glass = _qimage(self._gpu.compose(self._amount, tint, alpha), self._dpr())
        self.composes += 1
        self.update()

    # --- painting ------------------------------------------------------------------------

    def _layout(self) -> dict[str, float]:
        """Vertical positions from the font's own metrics, relative to the card's top, so a
        taller font (Segoe UI) gets a taller card instead of crowding the bottom edge."""
        f = self.fonts
        m = lambda font: QFontMetricsF(font)
        head = m(f.get(13, QFont.Weight.DemiBold))
        big = m(f.get(34, QFont.Weight.DemiBold, numbers=True))
        body = m(f.get(13, QFont.Weight.Medium))
        foot = m(f.get(12, QFont.Weight.Normal, numbers=True))
        L = {"head": PAD_Y + head.ascent()}
        L["big"] = L["head"] + 6 + big.ascent()
        L["bar"] = L["big"] + 11
        L["sep"] = L["bar"] + 20
        L["body"] = L["sep"] + 9 + body.ascent()
        L["foot"] = L["body"] + body.descent() + 3 + foot.ascent()
        L["h"] = L["foot"] + foot.descent() + PAD_Y + 1
        return L

    def _card_height(self) -> int:
        return int(math.ceil(self._layout()["h"]))

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        L = self._layout()
        card = QRectF(SHADOW, SHADOW, WIDTH, math.ceil(L['h']))
        self._ensure_maps()

        glassy = self._glass is not None and self.glass_wanted
        amount = self._amount if glassy else 1.0
        pal = mix_palette(self._light) if glassy else DARK
        if glassy and amount < 1.0:             # materialize: grow in from 96% around the centre
            k = 0.96 + 0.04 * amount
            c = card.center()
            p.translate(c)
            p.scale(k, k)
            p.translate(-c)

        # real blurred drop shadow (wide, soft, shifted down), then glass or solid material
        if self._shadow is not None:
            p.setOpacity(amount)
            p.drawImage(QPointF(0, 0), self._shadow)
            p.setOpacity(1.0)
        path = squircle(card, RADIUS)
        if glassy:
            p.drawImage(card.topLeft(), self._glass)
            rim = QColor(GLASS_RIM)
            rim.setAlphaF(GLASS_RIM.alphaF() * amount)
            p.setPen(QPen(rim, 1.0))
        else:
            p.fillPath(path, MATERIAL)
            p.setPen(QPen(RIM_HIGH_CONTRAST if self.prefs["high_contrast"] else RIM, 1.0))
        p.drawPath(path)
        p.setOpacity(min(max((amount - 0.2) / 0.8, 0.0), 1.0))   # content follows the glass in

        x0, x1 = card.left() + PAD_X, card.right() - PAD_X
        f = self.fonts
        s = self.summary

        # row 1: "Today" and the streak
        top = card.top()
        head = f.get(13, QFont.Weight.DemiBold)
        y_base = top + L['head']
        self._text(p, head, pal.label2, x0, y_base, "Today")
        if s and s.streak_days:
            self._text(p, f.get(13, QFont.Weight.DemiBold, numbers=True), pal.orange, x1, y_base,
                       f"{s.streak_days}-day streak", right=True)

        # row 2: big time, "/ target" (or "goal met"), and the "+m:ss" pill
        big = f.get(34, QFont.Weight.DemiBold, numbers=True)
        fm_big = QFontMetricsF(big)
        y_base = top + L['big']
        today = self._shown_today if s else 0.0
        big_txt = fmt_clock(today) if s else "–:––"
        self._text(p, big, pal.label if s else pal.label3, x0, y_base, big_txt)
        sub = f.get(15, QFont.Weight.Normal, numbers=True)
        sx = x0 + fm_big.horizontalAdvance(big_txt) + 7
        if s and s.done:
            extra = s.today_s - s.target_s
            sub_txt = "goal met" + (f"  +{fmt_clock(extra)}" if extra >= 30 else "")
            self._text(p, f.get(15, QFont.Weight.DemiBold, numbers=True), pal.green, sx, y_base, sub_txt)
        else:
            self._text(p, sub, pal.label2, sx, y_base,
                       f"/ {fmt_clock(s.target_s)}" if s else f"/ {fmt_clock(self.cfg.daily_target_min * 60)}")
        if self._gain_alpha > 0.01 and self._gain_s > 0:
            self._pill(p, x1, y_base - fm_big.ascent() * 0.5, f"+{fmt_clock(self._gain_s)}",
                       self._gain_alpha, pal.green)

        # row 3: progress capsule
        bar_y = top + L['bar']
        track = QRectF(x0, bar_y, x1 - x0, 6)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(pal.fill)
        p.drawRoundedRect(track, 3, 3)
        if s:
            frac = min(today / s.target_s, 1.0) if s.target_s else 0.0
            if frac > 0:
                fill = QRectF(track.left(), track.top(), max(track.width() * frac, 6), track.height())
                p.setBrush(pal.green if today >= s.target_s else pal.blue)
                p.drawRoundedRect(fill, 3, 3)

        # hairline on the solid card; on glass the two groups are separated by space alone
        if not glassy:
            sep_y = top + L['sep']
            p.setPen(QPen(pal.separator, 0.5))
            p.drawLine(QPointF(x0, sep_y), QPointF(x1, sep_y))

        # row 4: this scenario, today
        body = f.get(13, QFont.Weight.Medium)
        body_num = f.get(13, QFont.Weight.Normal, numbers=True)
        y_base = top + L['body']
        if s and s.scenario:
            right = f"{fmt_clock(s.scenario_today_s)} today"
            rw = QFontMetricsF(body_num).horizontalAdvance(right)
            name = QFontMetricsF(body).elidedText(s.scenario, Qt.TextElideMode.ElideRight, x1 - x0 - rw - 12)
            self._text(p, body, pal.label, x0, y_base, name)
            self._text(p, body_num, pal.label2, x1, y_base, right, right=True)
        else:
            self._text(p, body, pal.label2, x0, y_base,
                       "Counting your history…" if self.loading else "No runs yet")

        # row 5: all-time hours
        foot = f.get(12, QFont.Weight.Normal, numbers=True)
        y_base = top + L['foot']
        if s:
            if s.scenario:
                self._text(p, foot, pal.label3, x0, y_base, f"{fmt_hours(s.scenario_all_s)} on this scenario")
            self._text(p, foot, pal.label3, x1, y_base, f"{fmt_hours(s.all_s)} all-time", right=True)
        p.end()

    def _text(self, p: QPainter, font: QFont, color: QColor, x: float, baseline: float,
              text: str, right: bool = False) -> None:
        p.setFont(font)
        p.setPen(color)
        if right:
            x -= QFontMetricsF(font).horizontalAdvance(text)
        p.drawText(QPointF(x, baseline), text)

    def _pill(self, p: QPainter, right_x: float, mid_y: float, text: str, alpha: float,
              green: QColor = GREEN) -> None:
        """A flat tinted capsule, never glass: Liquid Glass is not stacked on Liquid Glass."""
        font = self.fonts.get(13, QFont.Weight.DemiBold, numbers=True)
        fm = QFontMetricsF(font)
        w, h = fm.horizontalAdvance(text) + 16, 22
        r = QRectF(right_x - w, mid_y - h / 2, w, h)
        bg = QColor(green)
        bg.setAlphaF(0.18 * alpha)
        fg = QColor(green)
        fg.setAlphaF(alpha)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(r, h / 2, h / 2)
        p.setFont(font)
        p.setPen(fg)
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)


def _shape_alpha(rect: QRectF, w: int, h: int, dpr: float) -> np.ndarray:
    """The card's squircle rasterized (antialiased) on a w x h device-pixel canvas, 0..1."""
    img = QImage(w, h, QImage.Format.Format_Grayscale8)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(dpr, dpr)
    p.fillPath(squircle(rect, RADIUS), QColor(255, 255, 255))
    p.end()
    buf = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes()).reshape(h, img.bytesPerLine())
    return buf[:, :w].astype(np.float32) / 255.0


def _qimage(rgba: np.ndarray, dpr: float) -> QImage:
    """(h, w, 4) premultiplied RGBA uint8 -> QImage (copied, so the array can go away)."""
    h, w = rgba.shape[:2]
    data = np.ascontiguousarray(rgba)
    img = QImage(data.data, w, h, w * 4, QImage.Format.Format_RGBA8888_Premultiplied).copy()
    img.setDevicePixelRatio(dpr)
    return img


def _exclude_from_capture(w: QWidget) -> None:
    """Windows 10 2004+: leave this window out of screen capture, so the live glass never
    captures the card itself. Side effect: in glass mode the card is also missing from your own
    screenshots and recordings (the solid card is not excluded)."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetWindowDisplayAffinity(int(w.winId()), 0x11)   # WDA_EXCLUDEFROMCAPTURE
    except (AttributeError, OSError):
        pass


def _keep_on_top(w: QWidget) -> None:
    """Re-assert topmost without activating; some games push other topmost windows down."""
    if sys.platform != "win32":
        return
    HWND_TOPMOST, SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = -1, 0x1, 0x2, 0x10
    try:
        ctypes.windll.user32.SetWindowPos(int(w.winId()), HWND_TOPMOST, 0, 0, 0, 0,
                                          SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)
    except (AttributeError, OSError):
        pass
