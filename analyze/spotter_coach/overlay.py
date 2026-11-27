"""PySide6 frameless overlay HUD with a single minimal bar/line chart and Home/F8/F9 hotkeys."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import math
import sys

from PySide6.QtCharts import (
    QAbstractBarSeries,
    QAreaSeries,
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QLineSeries,
    QScatterSeries,
    QValueAxis,
)
from PySide6.QtCore import QMargins, QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .debrief import DebriefBundle, DeltaPill

# Win32 constants for click-through IDLE pill, DWM rounded corners, and global Home / F8 / F9 hotkeys
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000
VK_HOME = 0x24
VK_F8 = 0x77
VK_F9 = 0x78
HOTKEY_END_CONVO = 101
HOTKEY_TOGGLE_MUTE = 102
HOTKEY_TOGGLE_UI = 103

BG_DARK = "#111318"
BG_PILL = "#20242e"
BORDER = "#2c3240"
TEXT_MAIN = "#f1f3f8"
TEXT_DIM = "#949bb0"
ACCENT = "#4da3ff"
GOOD = "#2bd48f"
BAD = "#ff6b57"
WARN = "#f5b942"
UI_FONT = '"Segoe UI"'

STYLESHEET = f"""
QWidget#CoachRoot {{
    background-color: {BG_DARK};
}}
QFrame#RootCard {{
    background-color: {BG_DARK};
    border: none;
}}
QLabel {{
    color: {TEXT_MAIN};
    font-family: {UI_FONT};
    background: transparent;
    border: none;
}}
QPushButton {{
    background-color: {BG_PILL};
    color: {TEXT_MAIN};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 3px 9px;
    font-family: {UI_FONT};
    font-size: 11px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: #2a303d;
    border-color: #414a5e;
}}
QPushButton#EndBtn {{
    background-color: #361d1d;
    color: #ff8c7a;
    border: 1px solid #6e3430;
}}
QPushButton#EndBtn:hover {{
    background-color: #4a2525;
}}
QPushButton#MiniBtn {{
    padding: 2px 7px;
    font-size: 10px;
    font-weight: 600;
    color: {TEXT_DIM};
}}
QPushButton#MiniBtn[active="true"] {{
    background-color: #1b2d45;
    color: {ACCENT};
    border-color: #2f5b8f;
}}
QComboBox {{
    background-color: {BG_PILL};
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 2px 6px;
    font-family: {UI_FONT};
    font-size: 10px;
}}
QComboBox QAbstractItemView {{
    background-color: #181b22;
    color: {TEXT_MAIN};
    selection-background-color: #2c3547;
}}
"""


class SimpleChartWidget(QWidget):
    """Size-aware native C++ QtCharts widget (QHorizontalBarSeries & QAreaSeries/QLineSeries)."""

    def __init__(self) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 0)
        lay.setSpacing(2)

        self._title_lbl = QLabel("")
        self._title_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; font-weight: 600;")
        lay.addWidget(self._title_lbl)

        self._chart = QChart()
        self._chart.setBackgroundBrush(QBrush(QColor(BG_DARK)))
        self._chart.setBackgroundPen(QPen(Qt.PenStyle.NoPen))
        self._chart.setBackgroundRoundness(0)
        self._chart.setPlotAreaBackgroundVisible(False)
        self._chart.setMargins(QMargins(0, 0, 6, 0))
        if self._chart.layout() is not None:
            self._chart.layout().setContentsMargins(0, 0, 0, 0)
        self._chart.legend().hide()

        self._view = QChartView(self._chart)
        self._view.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
        )
        self._view.setFrameShape(QFrame.Shape.NoFrame)
        self._view.setStyleSheet(f"background-color: {BG_DARK}; border: none;")
        self._view.setMinimumHeight(148)
        self._view.setMaximumHeight(165)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay.addWidget(self._view)

    def _new_chart(self) -> QChart:
        c = QChart()
        c.setBackgroundBrush(QBrush(QColor(BG_DARK)))
        c.setBackgroundPen(QPen(Qt.PenStyle.NoPen))
        c.setBackgroundRoundness(0)
        c.setPlotAreaBackgroundVisible(False)
        c.setMargins(QMargins(2, 4, 10, 2))
        if c.layout() is not None:
            c.layout().setContentsMargins(0, 0, 0, 0)
        c.legend().hide()
        old = self._chart
        self._chart = c
        self._view.setChart(c)
        if old is not None:
            old.deleteLater()
        return c

    def set_bars(self, title: str, bars: list[tuple[str, float, str, str]], unit_suffix: str = "%") -> None:
        self._title_lbl.setText(title)
        chart = self._new_chart()
        if not bars:
            return

        items = bars[:4]
        series = QBarSeries(chart)
        bset = QBarSet("", series)
        bset.setColor(QColor(ACCENT))
        bset.setBorderColor(QColor(BG_DARK))
        bset.setLabelColor(QColor(TEXT_MAIN))
        bset.setLabelFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))

        raw_vals: list[float] = []
        cats: list[str] = []
        for label, frac, val_str, _col in items:
            cleaned = val_str.rstrip("%msx ")
            try:
                num = float(cleaned)
            except ValueError:
                num = frac * 100.0
            raw_vals.append(num)
            cats.append(label)
            bset.append(round(num, 1))

        series.setBarWidth(0.48)
        series.append(bset)
        series.setLabelsVisible(True)
        series.setLabelsPosition(QAbstractBarSeries.LabelsPosition.LabelsOutsideEnd)
        if items and items[0][2].endswith("%"):
            series.setLabelsFormat("@value%")
        elif items and items[0][2].endswith("ms"):
            series.setLabelsFormat("@value ms")
        else:
            series.setLabelsFormat("@value")
        chart.addSeries(series)

        ax_x = QBarCategoryAxis(chart)
        ax_x.append(cats)
        ax_x.setLabelsColor(QColor(TEXT_DIM))
        ax_x.setLabelsFont(QFont("Segoe UI", 8))
        ax_x.setGridLineVisible(False)
        ax_x.setLinePenColor(QColor("#242a36"))
        chart.addAxis(ax_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(ax_x)

        vmax = max(raw_vals) if raw_vals else 100.0
        ax_y = QValueAxis(chart)
        ax_y.setRange(0.0, max(vmax * 1.28, 1.0))
        ax_y.setTickCount(3)
        ax_y.setLabelsVisible(False)
        ax_y.setGridLineColor(QColor("#1c212c"))
        ax_y.setLineVisible(False)
        chart.addAxis(ax_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(ax_y)

    def set_line(self, title: str, pts: list[tuple[str, float]],
                 ref_val: float | None = None, unit: str = "") -> None:
        chart = self._new_chart()
        if len(pts) < 2:
            self._title_lbl.setText(title)
            return

        ys = [p[1] for p in pts]
        now_s = f"{ys[-1]:.1f}{unit}" if abs(ys[-1]) < 100 else f"{ys[-1]:.0f}{unit}"
        self._title_lbl.setText(f"{title}  ·  Latest {now_s}")

        all_v = ys + ([ref_val] if ref_val is not None else [])
        ymin, ymax = min(all_v), max(all_v)
        pad = max((ymax - ymin) * 0.22, 1.0)
        ylo, yhi = ymin - pad, ymax + pad
        n = len(ys)

        if ref_val is not None:
            ref_s = QLineSeries(chart)
            ref_pen = QPen(QColor("#384257"), 1.1, Qt.PenStyle.DashLine)
            ref_s.setPen(ref_pen)
            ref_s.append(0.0, ref_val)
            ref_s.append(float(n - 1), ref_val)
            chart.addSeries(ref_s)
        else:
            ref_s = None

        area = QAreaSeries(chart)
        upper = QLineSeries(area)
        lower = QLineSeries(area)
        for i, val in enumerate(ys):
            upper.append(float(i), val)
            lower.append(float(i), ylo)
        area.setUpperSeries(upper)
        area.setLowerSeries(lower)

        grad = QLinearGradient(QPointF(0, 0), QPointF(0, 1))
        grad.setCoordinateMode(QLinearGradient.CoordinateMode.ObjectBoundingMode)
        grad.setColorAt(0.0, QColor(77, 163, 255, 68))
        grad.setColorAt(1.0, QColor(77, 163, 255, 0))
        area.setBrush(QBrush(grad))
        area.setPen(QPen(Qt.PenStyle.NoPen))
        chart.addSeries(area)

        line = QLineSeries(chart)
        line.setPen(QPen(QColor(ACCENT), 2.0))
        for i, val in enumerate(ys):
            line.append(float(i), val)
        chart.addSeries(line)

        dots = QScatterSeries(chart)
        dots.setMarkerSize(7.0)
        dots.setColor(QColor(GOOD))
        dots.setBorderColor(QColor(BG_DARK))
        dots.append(float(n - 1), ys[-1])
        chart.addSeries(dots)

        ax_x = QValueAxis(chart)
        ax_x.setRange(-0.15, float(n - 1) + 0.25)
        ax_x.setLabelsVisible(False)
        ax_x.setGridLineVisible(False)
        ax_x.setLinePenColor(QColor("#242a36"))
        chart.addAxis(ax_x, Qt.AlignmentFlag.AlignBottom)

        ax_y = QValueAxis(chart)
        ax_y.setRange(ylo, yhi)
        ax_y.setTickCount(3)
        ax_y.setLabelFormat(f"%.0f{unit}")
        ax_y.setLabelsColor(QColor(TEXT_DIM))
        ax_y.setLabelsFont(QFont("Segoe UI", 8))
        ax_y.setGridLineColor(QColor("#1c212c"))
        ax_y.setLineVisible(False)
        chart.addAxis(ax_y, Qt.AlignmentFlag.AlignLeft)

        for s in (ref_s, area, line, dots):
            if s is not None:
                s.attachAxis(ax_x)
                s.attachAxis(ax_y)


class CoachOverlay(QWidget):
    """Compact 3-state overlay: IDLE pill, 100% hidden IN_RUN, clean 1-chart DEBRIEF card."""

    end_convo_requested = Signal()
    mute_toggle_requested = Signal()
    mic_device_changed = Signal(object)
    quit_requested = Signal()

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setObjectName("CoachRoot")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setStyleSheet(STYLESHEET)

        self._hotkeys_registered = False
        self._user_hidden = False
        self._mode_state = "IDLE"
        self._bundle: DebriefBundle | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._card = QFrame()
        self._card.setObjectName("RootCard")
        outer.addWidget(self._card)

        self._card_lay = QVBoxLayout(self._card)
        self._card_lay.setContentsMargins(12, 9, 12, 9)
        self._card_lay.setSpacing(0)

        self._idle_page = self._build_idle_page()
        self._debrief_page = self._build_debrief_page()
        self._card_lay.addWidget(self._idle_page)
        self._card_lay.addWidget(self._debrief_page)

        self.show_idle()

    def _build_idle_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(3)

        row = QHBoxLayout()
        row.setSpacing(7)
        self._idle_dot = QLabel("●")
        self._idle_dot.setStyleSheet(f"color: {GOOD}; font-size: 11px;")
        row.addWidget(self._idle_dot)

        self._idle_status = QLabel("Spotter Ready")
        self._idle_status.setStyleSheet("font-size: 11px; font-weight: 600;")
        row.addWidget(self._idle_status, 1)
        lay.addLayout(row)

        self._idle_focus = QLabel("")
        self._idle_focus.setWordWrap(True)
        self._idle_focus.setStyleSheet(f"color: {ACCENT}; font-size: 11px;")
        self._idle_focus.hide()
        lay.addWidget(self._idle_focus)
        return w

    def _build_debrief_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(7)

        # 1. Header: Scenario + Score + Verdict
        hdr = QHBoxLayout()
        self._scen_lbl = QLabel("Scenario")
        self._scen_lbl.setStyleSheet("font-size: 13px; font-weight: 700;")
        hdr.addWidget(self._scen_lbl, 1)

        self._score_lbl = QLabel("")
        self._score_lbl.setStyleSheet("font-size: 12px; font-weight: 600;")
        hdr.addWidget(self._score_lbl)
        lay.addLayout(hdr)

        # 2. Voice bar: status + Mute (F9) + End Convo (F8)
        vbar = QHBoxLayout()
        vbar.setSpacing(5)
        self._voice_lbl = QLabel("🎙 Listening")
        self._voice_lbl.setStyleSheet(f"color: {GOOD}; font-size: 11px; font-weight: 600;")
        vbar.addWidget(self._voice_lbl, 1)

        self._mute_btn = QPushButton("Mute [F9]")
        self._mute_btn.clicked.connect(self.mute_toggle_requested.emit)
        vbar.addWidget(self._mute_btn)

        self._end_btn = QPushButton("✕ End [F8]")
        self._end_btn.setObjectName("EndBtn")
        self._end_btn.clicked.connect(self.end_convo_requested.emit)
        vbar.addWidget(self._end_btn)
        lay.addLayout(vbar)

        # 2b. Compact Mic + view toggle row
        mrow = QHBoxLayout()
        mrow.setSpacing(4)
        self._mic_combo = QComboBox()
        self._mic_combo.currentIndexChanged.connect(self._on_mic_selected)
        mrow.addWidget(self._mic_combo, 1)

        self._btn_bars = QPushButton("Bars")
        self._btn_bars.setObjectName("MiniBtn")
        self._btn_bars.clicked.connect(lambda: self.select_visual("time_budget"))
        mrow.addWidget(self._btn_bars)

        self._btn_line = QPushButton("Trend")
        self._btn_line.setObjectName("MiniBtn")
        self._btn_line.clicked.connect(lambda: self.select_visual("trend"))
        mrow.addWidget(self._btn_line)

        quit_btn = QPushButton("Quit")
        quit_btn.setObjectName("MiniBtn")
        quit_btn.clicked.connect(self.quit_requested.emit)
        mrow.addWidget(quit_btn)
        lay.addLayout(mrow)

        # 3. Settings change warning (if any)
        self._settings_lbl = QLabel("")
        self._settings_lbl.setStyleSheet(f"color: {WARN}; font-size: 10px; font-weight: 600;")
        self._settings_lbl.hide()
        lay.addWidget(self._settings_lbl)

        # 4. Headline borderless stats
        self._pills_row = QHBoxLayout()
        self._pills_row.setSpacing(6)
        lay.addLayout(self._pills_row)

        # 5. Live caption
        self._caption_lbl = QLabel("")
        self._caption_lbl.setWordWrap(True)
        self._caption_lbl.setMaximumHeight(34)
        self._caption_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; font-style: italic;")
        self._caption_lbl.hide()
        lay.addWidget(self._caption_lbl)

        # 6. ONE clean, glanceable chart
        self._chart = SimpleChartWidget()
        lay.addWidget(self._chart)

        # 7. Pinned Next Run Focus footer
        self._debrief_focus = QLabel("🎯 —")
        self._debrief_focus.setWordWrap(True)
        self._debrief_focus.setStyleSheet(
            f"background-color: {BG_PILL}; color: {ACCENT}; border-radius: 6px; "
            "padding: 4px 8px; font-size: 11px; font-weight: 600;"
        )
        lay.addWidget(self._debrief_focus)
        return w

    # --- Sizing & State transitions ---

    def _fit_to_active_page(self, width: int) -> None:
        self._card.setFixedWidth(width)
        self.setFixedWidth(width)
        self._card.layout().activate()
        self.layout().activate()
        h = self._card.sizeHint().height()
        self.setFixedSize(width, max(h, 32))
        self._reposition_top_right()

    def toggle_user_visibility(self) -> None:
        self._user_hidden = not self._user_hidden
        if self._user_hidden or self._mode_state == "IN_RUN":
            self.hide()
        elif self._mode_state == "DEBRIEF":
            self.show()
            self._apply_win32_style(click_through=False)
        else:
            self.show()
            self._apply_win32_style(click_through=True)

    def show_idle(self) -> None:
        self._mode_state = "IDLE"
        self._debrief_page.hide()
        self._idle_page.show()
        self._fit_to_active_page(285)
        if not self._user_hidden:
            self.show()
            self._apply_win32_style(click_through=True)
        else:
            self._apply_win32_style(click_through=True)
            self.hide()

    def enter_in_run(self) -> None:
        self._mode_state = "IN_RUN"
        self.hide()

    def show_debrief(self, bundle: DebriefBundle) -> None:
        self._mode_state = "DEBRIEF"
        self._bundle = bundle
        self._scen_lbl.setText(f"● {bundle.scenario}")

        if bundle.score is not None:
            v = bundle.score_verdict
            if "better" in v:
                badge, col = "▲ REAL", GOOD
            elif "worse" in v:
                badge, col = "▼ REAL", BAD
            else:
                badge, col = "noise", TEXT_DIM
            usual_s = f" ({bundle.usual_score:.0f})" if bundle.usual_score is not None else ""
            self._score_lbl.setText(f"{bundle.score:.1f}{usual_s} · {badge}")
            self._score_lbl.setStyleSheet(f"color: {col}; font-size: 12px; font-weight: 700;")
        else:
            self._score_lbl.setText("")

        if bundle.settings_note:
            self._settings_lbl.setText(f"⚠ {bundle.settings_note}")
            self._settings_lbl.show()
        else:
            self._settings_lbl.hide()

        self._populate_pills(bundle.pills)
        self._caption_lbl.clear()
        self._caption_lbl.hide()

        self._idle_page.hide()
        self._debrief_page.show()
        self.select_visual(bundle.default_visual)
        self._fit_to_active_page(390)
        if not self._user_hidden:
            self.show()
        self._apply_win32_style(click_through=False)

    def _populate_pills(self, pills: list[DeltaPill]) -> None:
        while self._pills_row.count():
            it = self._pills_row.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        tone_color = {"good": GOOD, "bad": BAD, "neutral": TEXT_DIM}
        for pill in pills[:4]:
            col_w = QWidget()
            bl = QVBoxLayout(col_w)
            bl.setContentsMargins(2, 1, 2, 1)
            bl.setSpacing(0)
            t1 = QLabel(pill.label)
            t1.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
            t2 = QLabel(pill.value_text)
            t2.setStyleSheet("font-size: 15px; font-weight: 700;")
            t3 = QLabel(pill.sub_text)
            t3.setStyleSheet(f"color: {tone_color.get(pill.tone, TEXT_DIM)}; font-size: 9px; font-weight: 600;")
            bl.addWidget(t1)
            bl.addWidget(t2)
            bl.addWidget(t3)
            col_w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self._pills_row.addWidget(col_w)

    def select_visual(self, view: str, split_key: str = "") -> str:
        """Render ONE minimal bar or line graph onto SimpleChartWidget."""
        b = self._bundle
        v = view.lower().strip()
        sk = split_key.lower().strip()

        if b is None:
            return v

        # 1. Line graph requests: score trend, pacing by thirds, or lag-shift
        if "trend" in v or "hist" in v or "score" in v:
            pts = b.trend_points[-12:]
            self._chart.set_line("RECENT SCORE TREND", pts, ref_val=b.usual_score)
            self._mark_mini_btn(is_bars=False)
            return "trend"

        if "lag" in v or ("shift" in v and b.lag_shift):
            if b.lag_shift:
                _cols, lrows = b.lag_shift
                pts = [(f"-{float(r[0]):.0f}ms", float(r[1])) for r in lrows]
                self._chart.set_line("ON-TARGET % IF EARLIER (DELAY CEILING)", pts, unit="%")
                self._mark_mini_btn(is_bars=False)
                return "lag_shift"

        if sk == "thirds" or "third" in v:
            sp = b.splits.get("thirds")
            if sp:
                _t, cols, rows = sp
                metric_col = "ttk_ms" if "ttk_ms" in cols else ("on_pct" if "on_pct" in cols else cols[1])
                idx = cols.index(metric_col)
                unit = "ms" if "ms" in metric_col else ("%" if "pct" in metric_col else "")
                pts = [(str(r[0]), float(r[idx])) for r in rows if not math.isnan(float(r[idx]))]
                self._chart.set_line(f"PACING ACROSS RUN ({metric_col.upper()})", pts, unit=unit)
                self._mark_mini_btn(is_bars=False)
                return "splits"

        # 2. Bar graph requests: distance / speed / target choice splits, or default time breakdown
        if "split" in v or sk in ("distance", "target_choice", "speed"):
            key = sk if sk in b.splits else ("distance" if "distance" in b.splits else next(iter(b.splits), ""))
            if key and key in b.splits and key not in ("time_budget", "lag_shift"):
                _t, cols, rows = b.splits[key]
                val_col = "gain" if "gain" in cols else ("on_pct" if "on_pct" in cols else "ttk_ms")
                if val_col not in cols:
                    val_col = cols[min(2, len(cols) - 1)]
                idx = cols.index(val_col)
                vals = [float(r[idx]) for r in rows if isinstance(r[idx], (int, float)) and not math.isnan(float(r[idx]))]
                vmax = max(vals) if vals else 1.0
                bars = []
                for r in rows[:4]:
                    rv = float(r[idx]) if isinstance(r[idx], (int, float)) and not math.isnan(float(r[idx])) else 0.0
                    v_txt = f"{rv:.0f}ms" if "ms" in val_col else (f"{rv:.0f}%" if "pct" in val_col else f"{rv:.2f}")
                    bars.append((str(r[0])[:15], rv / max(vmax, 1e-6), v_txt, ACCENT))
                self._chart.set_bars(f"{key.upper().replace('_', ' ')} ({val_col})", bars)
                self._mark_mini_btn(is_bars=True)
                return "splits"

        # Default simple bar graph: Where Kill Time Went (clicking/switching) or Speed On-Target % (tracking)
        if b.time_budget:
            _cols, trows = b.time_budget
            palette = [ACCENT, "#2bd48f", WARN, BAD, "#7c859e"]
            short_names = {
                "reaction": "React",
                "travel": "Travel",
                "off again": "Settle",
                "slips": "Slips",
                "on target": "On Tgt",
            }
            bars = []
            for i, r in enumerate(trows):
                if isinstance(r[2], float) and math.isnan(r[2]):
                    continue
                raw_name = str(r[0]).lower()
                lbl = next((sn for k, sn in short_names.items() if k in raw_name), str(r[0])[:14])
                pct = max(0.0, float(r[2]))
                bars.append((lbl, pct / 100.0, f"{pct:.0f}%", palette[i % len(palette)]))
            # Combine tiny overshoot + slips if >4 bars so it stays <= 4 clean bars
            self._chart.set_bars("WHERE KILL TIME WENT", bars[:4])
            self._mark_mini_btn(is_bars=True)
            return "time_budget"

        if "speed" in b.splits:
            _t, cols, rows = b.splits["speed"]
            idx = cols.index("on_pct") if "on_pct" in cols else 2
            bars = [
                (f"Speed {r[0]}", float(r[idx]) / 100.0, f"{float(r[idx]):.0f}%", ACCENT)
                for r in rows[:4]
                if isinstance(r[idx], (int, float)) and not math.isnan(float(r[idx]))
            ]
            self._chart.set_bars("ON-TARGET % BY BOT SPEED", bars)
            self._mark_mini_btn(is_bars=True)
            return "splits"

        if b.lag_shift:
            _cols, lrows = b.lag_shift
            pts = [(f"-{float(r[0]):.0f}ms", float(r[1])) for r in lrows]
            self._chart.set_line("ON-TARGET % IF EARLIER (DELAY CEILING)", pts, unit="%")
            self._mark_mini_btn(is_bars=False)
            return "lag_shift"

        self._chart.set_line("RECENT SCORE TREND", b.trend_points[-12:], ref_val=b.usual_score)
        self._mark_mini_btn(is_bars=False)
        return "trend"

    def _mark_mini_btn(self, is_bars: bool) -> None:
        self._btn_bars.setProperty("active", "true" if is_bars else "false")
        self._btn_line.setProperty("active", "false" if is_bars else "true")
        for b in (self._btn_bars, self._btn_line):
            b.style().unpolish(b)
            b.style().polish(b)

    # --- Status & controls updates ---

    def set_status(self, text: str, ok: bool = True) -> None:
        self._idle_status.setText(text)
        self._idle_dot.setStyleSheet(f"color: {GOOD if ok else WARN}; font-size: 11px;")

    def populate_mics(self, devices: list[tuple[int | None, str]]) -> None:
        self._mic_combo.blockSignals(True)
        self._mic_combo.clear()
        for dev_id, label in devices:
            self._mic_combo.addItem(label, dev_id)
        self._mic_combo.blockSignals(False)

    def _on_mic_selected(self, idx: int) -> None:
        if idx >= 0:
            self.mic_device_changed.emit(self._mic_combo.itemData(idx))

    def set_voice_state(self, muted: bool, speaking: bool, mic_level: float = 0.0) -> None:
        bars = ("▁▁▁▁", "▃▁▁▁", "▃▅▁▁", "▃▅▆▁", "▃▅▆▇")
        meter = bars[min(4, int(mic_level * 5.0))]
        if muted:
            self._voice_lbl.setText("🔇 Muted (F9)")
            self._voice_lbl.setStyleSheet(f"color: {WARN}; font-size: 11px; font-weight: 600;")
            self._mute_btn.setText("Unmute [F9]")
        elif speaking:
            self._voice_lbl.setText(f"🔊 Speaking · {meter}")
            self._voice_lbl.setStyleSheet(f"color: {ACCENT}; font-size: 11px; font-weight: 600;")
            self._mute_btn.setText("Mute [F9]")
        else:
            self._voice_lbl.setText(f"🎙 Listening · {meter}")
            self._voice_lbl.setStyleSheet(f"color: {GOOD}; font-size: 11px; font-weight: 600;")
            self._mute_btn.setText("Mute [F9]")

    def set_caption(self, text: str, is_user: bool = False) -> None:
        clean = text.strip()
        if clean:
            prefix = "You: " if is_user else "Coach: "
            self._caption_lbl.setText(f"{prefix}“{clean[-130:]}”")
            self._caption_lbl.show()
            if self._mode_state == "DEBRIEF":
                self._fit_to_active_page(390)

    def set_next_focus(self, cue: str) -> None:
        clean = cue.strip()
        if clean:
            self._debrief_focus.setText(f"🎯 {clean}")
            self._idle_focus.setText(f"🎯 {clean}")
            self._idle_focus.show()
        else:
            self._idle_focus.hide()
        if self._mode_state == "IDLE":
            self._fit_to_active_page(285)
        elif self._mode_state == "DEBRIEF":
            self._fit_to_active_page(390)

    # --- Positioning, Win11 DWM chrome, click-through IDLE & global hotkeys ---

    def _reposition_top_right(self) -> None:
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(geo.right() - self.width() - 16, geo.top() + 16)

    def _apply_win32_style(self, click_through: bool) -> None:
        if sys.platform != "win32":
            return
        try:
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            ex = get_long(hwnd, GWL_EXSTYLE) | WS_EX_TOOLWINDOW
            if click_through:
                ex |= WS_EX_LAYERED | WS_EX_TRANSPARENT
                set_long(hwnd, GWL_EXSTYLE, ex)
                user32.SetLayeredWindowAttributes(hwnd, 0, 255, 0x02)
            else:
                ex &= ~(WS_EX_TRANSPARENT | WS_EX_LAYERED)
                set_long(hwnd, GWL_EXSTYLE, ex)

            dwmapi = ctypes.windll.dwmapi
            corner_pref = ctypes.c_int(2)
            dwmapi.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(corner_pref), ctypes.sizeof(corner_pref))
            border_colorref = ctypes.c_uint32(0x0040322C)
            dwmapi.DwmSetWindowAttribute(hwnd, 34, ctypes.byref(border_colorref), ctypes.sizeof(border_colorref))

            if not self._hotkeys_registered:
                user32.RegisterHotKey(hwnd, HOTKEY_END_CONVO, MOD_NOREPEAT, VK_F8)
                user32.RegisterHotKey(hwnd, HOTKEY_TOGGLE_MUTE, MOD_NOREPEAT, VK_F9)
                user32.RegisterHotKey(hwnd, HOTKEY_TOGGLE_UI, MOD_NOREPEAT, VK_HOME)
                self._hotkeys_registered = True
        except Exception:
            pass

    def nativeEvent(self, eventType, message):
        if sys.platform == "win32" and eventType == b"windows_generic_MSG":
            try:
                msg = wintypes.MSG.from_address(int(message))
                if msg.message == WM_HOTKEY:
                    if msg.wParam == HOTKEY_END_CONVO:
                        self.end_convo_requested.emit()
                        return True, 0
                    if msg.wParam == HOTKEY_TOGGLE_MUTE:
                        self.mute_toggle_requested.emit()
                        return True, 0
                    if msg.wParam == HOTKEY_TOGGLE_UI:
                        self.toggle_user_visibility()
                        return True, 0
            except Exception:
                pass
        return super().nativeEvent(eventType, message)

    def closeEvent(self, event) -> None:
        if sys.platform == "win32" and self._hotkeys_registered:
            try:
                hwnd = int(self.winId())
                ctypes.windll.user32.UnregisterHotKey(hwnd, HOTKEY_END_CONVO)
                ctypes.windll.user32.UnregisterHotKey(hwnd, HOTKEY_TOGGLE_MUTE)
                ctypes.windll.user32.UnregisterHotKey(hwnd, HOTKEY_TOGGLE_UI)
            except Exception:
                pass
            self._hotkeys_registered = False
        super().closeEvent(event)
