"""Aim-volume overlay: time math, when it may show, settings, and drawing."""
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from spotter_analyze.format import FRAME_COLS, NO_SHOTS
from spotter_volume import gate as gate_mod
from spotter_volume.config import Config
from spotter_volume.gate import Gate, HookLog, State
from spotter_volume.ledger import Ledger, day_of, streak


def _stats(d: Path, scenario: str, start: datetime, seconds: float) -> Path:
    end = start + timedelta(seconds=seconds)
    p = d / f"{scenario} - Challenge - {end:%Y.%m.%d-%H.%M.%S} Stats.csv"
    p.write_text("Kill #,Timestamp,Bot,Weapon,TTK,Shots,Hits\n\n"
                 f"Score:,99\nChallenge Start:,{start:%H:%M:%S}.000\n", encoding="utf-8")
    return p


# --- ledger ----------------------------------------------------------------------------

def test_today_scenario_and_all_time(tmp_path):
    sd = tmp_path / "stats"
    sd.mkdir()
    _stats(sd, "ww4t Voltaic", datetime(2026, 11, 20, 21, 0), 60)            # an older day
    _stats(sd, "ww4t Voltaic", datetime(2026, 11, 26, 22, 0), 60)
    _stats(sd, "voxTargetSwitch 2", datetime(2026, 11, 27, 0, 30), 60)       # after midnight, same evening
    _stats(sd, "voxTargetSwitch 2", datetime(2026, 11, 27, 1, 0), 60)
    led = Ledger(sd, tmp_path / "data").load()
    s = led.summary(datetime(2026, 11, 27, 2, 0), target_min=30)
    assert s.today_s == 180 and s.target_s == 1800 and not s.done
    assert s.scenario == "voxTargetSwitch 2"
    assert (s.scenario_today_s, s.scenario_all_s, s.all_s) == (120, 120, 240)
    # cached: files already read are not parsed again (junk now, still counted from the cache)
    for p in sd.glob("*.csv"):
        p.write_text("junk")
    assert len(Ledger(sd, tmp_path / "data").load().saved) == 4


def test_day_rolls_over_at_five():
    assert day_of(datetime(2026, 11, 27, 4, 59)) == date(2026, 11, 26)
    assert day_of(datetime(2026, 11, 27, 5, 0)) == date(2026, 11, 27)


def test_streak():
    t = 1800
    daily = {date(2026, 11, 22): t, date(2026, 11, 23): 10, date(2026, 11, 24): t, date(2026, 11, 25): t}
    assert streak(daily, date(2026, 11, 26), t) == 2              # today still open
    assert streak({**daily, date(2026, 11, 26): t}, date(2026, 11, 26), t) == 3
    assert streak(daily, date(2026, 11, 27), t) == 0


def test_restarts_count_once_and_never_double_count_saved_runs(tmp_path):
    sd = tmp_path / "stats"
    sd.mkdir()
    _stats(sd, "vts2", datetime(2026, 11, 27, 1, 0), 60)
    led = Ledger(sd, tmp_path / "data").load()
    t = datetime(2026, 11, 27, 1, 5)
    assert led.add_restart("log:1", t, t + timedelta(seconds=20)).seconds == 20
    assert led.add_restart("log:1", t, t + timedelta(seconds=20)) is None        # same key
    assert led.add_restart("log:2", t, t + timedelta(seconds=1)) is None         # a misclick
    saved = datetime(2026, 11, 27, 1, 0, 10)
    assert led.add_restart("log:3", saved, saved + timedelta(seconds=30)) is None  # that was the saved run
    s = led.summary(datetime(2026, 11, 27, 2), 30)
    assert s.today_s == 80 and s.scenario_today_s == 80       # restart credited to vts2 (ran just before)
    # restarts survive a restart of the overlay
    assert len(Ledger(sd, tmp_path / "data").load().restarts) == 1


def test_saved_run_reported_twice_counts_once(tmp_path):
    from spotter_analyze import kovaaks
    sd = tmp_path / "stats"
    sd.mkdir()
    p = _stats(sd, "vts2", datetime(2026, 11, 27, 1, 0), 60)
    led = Ledger(sd, tmp_path / "data").load()
    assert led.add_saved(kovaaks.parse_stats(p)) is None


# --- when the card may show --------------------------------------------------------------

def _frames(t0: float, n: int, targets: int, move: bool, shots: int = 0, hz: float = 100) -> np.ndarray:
    f = np.zeros((n, FRAME_COLS))
    f[:, 0] = (t0 + np.arange(n) / hz) * 1e6
    f[:, 1] = np.arange(n) * 0.5 if move else 10.0
    f[:, 7] = shots
    f[:, 9] = targets
    return f


def test_targets_mean_aiming_even_before_you_move():
    g = Gate()
    g.feed(_frames(0, 50, targets=4, move=False), now=100.0)
    assert g.state(False, True, now=100.1) is State.AIMING


def test_results_screen_shows_after_targets_clear():
    g = Gate()
    g.feed(_frames(0, 300, targets=4, move=True), now=100.0)
    g.feed(_frames(3, 100, targets=0, move=False), now=101.0)        # 1 s without targets: not yet
    assert g.state(False, True, now=101.1) is State.AIMING
    g.feed(_frames(4, 100, targets=0, move=False), now=102.0)        # 2 s: results screen
    assert g.state(False, True, now=102.1) is State.IDLE
    g.feed(_frames(5, 5, targets=4, move=False), now=102.2)          # restart: targets back
    assert g.state(False, True, now=102.3) is State.AIMING


def test_moving_without_targets_still_counts_as_aiming():
    g = Gate()
    g.feed(_frames(0, 300, targets=0, move=True), now=50.0)          # e.g. between clicking spawns
    assert g.state(False, True, now=50.1) is State.AIMING


def test_shots_count_as_activity():
    g = Gate()
    f = _frames(0, 300, targets=0, move=False)
    f[:, 7] = np.arange(300) // 10
    g.feed(f, now=10.0)
    assert g.state(False, True, now=10.1) is State.AIMING


def test_quiet_log_menu_vs_no_hook():
    g = Gate()
    g.feed(_frames(0, 10, targets=4, move=True), now=0.0)
    assert g.state(hook_paused=True, hook_alive=True, now=5.0) is State.IDLE       # a menu
    assert g.state(hook_paused=False, hook_alive=False, now=5.0) is State.UNKNOWN  # blind


def test_unknown_shot_counter_is_not_activity():
    g = Gate()
    f = _frames(0, 300, targets=0, move=False)
    f[:, 7] = NO_SHOTS
    g.feed(f, now=10.0)
    assert g.state(False, True, now=10.1) is State.IDLE


def test_hook_log_follows_state_lines(tmp_path):
    p = tmp_path / "spotter.log"
    p.write_text("00:00:01.000  in a scenario: sampling\n", encoding="utf-8")
    h = HookLog(p)
    assert h.status() == (True, False)
    with p.open("a", encoding="utf-8") as fh:
        fh.write("00:00:02.000  no player/world: paused\n" + "00:00:12.000  frames=1 targets=0\n" * 3000)
    assert h.status() == (True, True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write("00:01:00.000  frames=2\n")          # heartbeat only: still paused
    assert h.status() == (True, True)
    assert HookLog(tmp_path / "missing.log").status() == (False, False)
    assert h.status(now_wall=p.stat().st_mtime + gate_mod.HOOK_ALIVE_S + 1)[0] is False


# --- settings and drawing --------------------------------------------------------------

def test_config_defaults_and_bad_files(tmp_path):
    p = tmp_path / "v.json"
    cfg = Config.load(p)
    assert p.exists() and cfg.font == "SF Pro Text" and cfg.daily_target_min == 30
    p.write_text('{"font": "Segoe UI", "daily_target_min": 45, "unknown": 1}')
    cfg = Config.load(p)
    assert (cfg.font, cfg.daily_target_min) == ("Segoe UI", 45)
    p.write_text("{nope")
    assert Config.load(p).font == "SF Pro Text"


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_font_falls_back_when_missing(qapp):
    from spotter_volume.overlay import Fonts
    assert Fonts(Config(font="No Such Font 123")).family == "Segoe UI"


def test_squircle_fills_its_rect(qapp):
    from PySide6.QtCore import QPointF, QRectF
    from spotter_volume.overlay import squircle
    r = QRectF(10, 20, 296, 150)
    path = squircle(r, 22)
    b = path.boundingRect()
    assert abs(b.left() - 10) < 1e-6 and abs(b.right() - 306) < 1e-6
    assert abs(b.top() - 20) < 1e-6 and abs(b.bottom() - 170) < 1e-6
    assert path.contains(QPointF(158, 95))
    assert not path.contains(QPointF(11, 21))            # the very corner is cut away


def test_card_paints_and_hides_instantly(qapp, tmp_path):
    from spotter_volume.ledger import Summary
    from spotter_volume.overlay import VolumeCard, fmt_clock, fmt_hours
    card = VolumeCard(Config())
    card.set_summary(Summary(1120, 1800, 3, "voxTargetSwitch 2", 372, 77400, 357840), gain_s=62)
    img = card.grab().toImage()
    assert img.width() > 300 and not img.isNull()
    card.present()
    assert card.isVisible()
    card.conceal()
    assert not card.isVisible()
    assert (fmt_clock(62), fmt_clock(4325), fmt_hours(77400), fmt_hours(357840)) == \
        ("1:02", "1:12:05", "21.5 h", "99.4 h")
    card.close()


def test_app_hides_during_a_run(qapp, tmp_path, monkeypatch):
    """End to end on a synthetic log: targets on screen = card hidden, whatever else holds."""
    from spotter_analyze import synth
    from spotter_analyze.format import write_spot
    import spotter_volume.__main__ as m

    runs, stats, data = tmp_path / "runs", tmp_path / "stats", tmp_path / "data"
    for d in (runs, stats):
        d.mkdir()
    run = synth.clicking(kills=5, seed=3)
    write_spot(runs / "spotter-20261127-010000.spot", run)
    (runs / "spotter-20261127-010000.log").write_text("x  in a scenario: sampling\n")
    monkeypatch.setattr(m, "game_in_front", lambda: True)
    app = m.App(qapp, Config(hotkey=""), runs, stats, data)
    app._loading.result()
    for _ in range(5):
        app._tick()
    assert app.caught_up
    assert app.gate.state(False, True) is State.AIMING
    assert not app.card.isVisible()

    # "show now" works even mid-scenario, and ends the moment the hand moves
    app.toggle()
    assert app.card.isVisible()
    app.gate.last_move_t += 1.0
    app._apply(State.AIMING)
    assert not app.card.isVisible()

    # results screen: shows; Home hides it until after the next run
    app.gate.state = lambda *a, **k: State.IDLE
    app._apply(State.IDLE)
    assert app.card.isVisible()
    app.toggle()
    app._apply(State.IDLE)
    assert not app.card.isVisible()
    app._apply(State.AIMING)
    app._apply(State.IDLE)
    assert app.card.isVisible()
    app.timer.stop()
    app._tray.hide()
    app.card.close()


# --- glass -----------------------------------------------------------------------------

def _rect_alpha(h=120, w=200, pad=0):
    a = np.zeros((h, w), np.float32)
    a[pad:h - pad, pad:w - pad] = 1.0
    return a


def test_snell_shift_peaks_on_the_rim_and_vanishes_on_the_flat_top():
    from spotter_volume.glass import GlassParams, snell_shift
    shift, _ = snell_shift(256, GlassParams())
    assert shift.max() == 1.0 and shift[-1] == 0.0
    assert 0 < shift.argmax() < 128            # the bend lives in the outer half of the rim


def test_rim_pixels_sample_from_further_inside():
    from spotter_volume.glass import build_maps
    m = build_maps(_rect_alpha(), 1.0)
    # left edge, middle row: convex glass looks inward, i.e. to the right
    assert m.sx[60, 3] > 1.0 and abs(m.sy[60, 3]) < 0.5
    assert m.sx[60, 100] == 0 and m.sy[60, 100] == 0         # flat centre: straight through
    assert m.rim[2, 190] > m.rim[117, 10]                    # lit corner (up-right) brighter


def _gpu_or_skip():
    try:
        import moderngl
        moderngl.create_standalone_context().release()
    except Exception as e:                     # no OpenGL 3.3 here (CI, remote desktop)
        pytest.skip(f"no GPU context: {e}")


def test_gpu_glass_center_is_tinted_frost_and_the_lit_rim_is_brighter():
    _gpu_or_skip()
    from spotter_volume.glass import GlassParams, build_maps, looks_blank
    from spotter_volume.gpu_glass import GpuGlass
    m = build_maps(_rect_alpha(), 1.0)
    p = GlassParams()
    g = GpuGlass(m, 1.0, p)
    bg = np.full((200, 280, 3), 100, np.uint8)           # card (120x200) with a 40 px margin
    g.set_backdrop(bg, (40, 40))
    tint, ta = (0.0, 0.0, 0.0), 0.5
    out = g.compose(1.0, tint, ta)
    assert out.shape == (120, 200, 4) and out[60, 100, 3] == 255
    # flat centre of a flat backdrop: just the frost (= 100) under the tint
    assert abs(int(out[60, 100, 0]) - round(100 * (1 - ta))) <= 1
    assert out[3, 190, :3].mean() > out[116, 10, :3].mean()      # light from up-right
    # not materialized yet: the backdrop shows straight through, untinted
    assert abs(int(g.compose(0.0, tint, ta)[60, 100, 0]) - 100) <= 1
    assert looks_blank(np.zeros((120, 200, 3), np.uint8)) and not looks_blank(bg)
    g.release()


def test_appearance_flips_with_hysteresis_and_tint_tracks_legibility():
    from spotter_volume.glass import Appearance, legibility_tint
    a = Appearance()
    assert [a.update(x) for x in (0.3, 0.6, 0.7, 0.55, 0.5, 0.4)] == \
        [False, False, True, True, True, False]
    # white text needs more tint over a bright backdrop; dark text over a dark one
    assert legibility_tint(0.8, 0.0) > legibility_tint(0.1, 0.0)
    assert legibility_tint(0.1, 1.0) > legibility_tint(0.9, 1.0)


class _Frames:
    """Stands in for the screen: hands out whatever .frame is, like DXGI would."""
    def __init__(self, frame):
        self.frame = frame
        self.width, self.height = 4000, 4000

    def grab(self, region, new_frame_only=True):
        l, t, r, b = region
        return self.frame[t:b, l:r].copy()


def test_backdrop_only_reports_a_frame_when_the_area_changed():
    from spotter_volume.capture import Backdrop
    screen = np.full((600, 800, 3), 50, np.uint8)
    bd = Backdrop()
    cam = _Frames(screen)
    bd._cam, bd._cam_mon, bd._cam_origin, bd.source = cam, None, (0, 0), "dxgi"
    import spotter_volume.capture as cap
    cap_monitor = cap._monitor_of
    cap._monitor_of = lambda *a: None
    try:
        rect = (100, 100, 400, 300)
        assert bd.grab(rect) is not None                     # first look
        assert bd.grab(rect) is None                         # same pixels: no work
        screen[500, 700] = 255                               # far away from the card
        assert bd.grab(rect) is None
        screen[150, 150] = 255                               # inside the area the glass samples
        assert bd.grab(rect) is not None
        cam.grab = lambda *a, **k: None                      # DXGI: desktop not redrawn at all
        assert bd.grab(rect) is None
    finally:
        cap._monitor_of = cap_monitor


def _glass_card():
    from spotter_volume.overlay import VolumeCard
    card = VolumeCard(Config(style="glass"))
    card.prefs = {"transparency": True, "animations": False, "high_contrast": False}
    return card


def test_card_glass_follows_the_backdrop_and_falls_back_to_solid(qapp):
    _gpu_or_skip()
    from spotter_volume.overlay import VICINITY, WIDTH
    card = _glass_card()
    h = card._card_height()
    m = VICINITY
    card.set_backdrop(np.full((h + 2 * m, WIDTH + 2 * m, 3), 40, np.uint8), (m, m))
    assert card.glass_ok and card._glass is not None and card._light == 0.0    # dark backdrop
    card.set_backdrop(np.full((h + 2 * m, WIDTH + 2 * m, 3), 235, np.uint8), (m, m))
    assert card._light == 1.0                                                # bright: light glass
    img = card.grab().toImage()
    assert not img.isNull()
    card.set_backdrop(np.zeros((h, WIDTH, 3), np.uint8))     # black capture: solid card
    assert card.glass_ok is False and not card.glass_wanted
    card.close()


def test_glass_does_nothing_while_hidden_or_when_windows_says_no(qapp, monkeypatch):
    import spotter_volume.capture as cap
    from spotter_volume.overlay import VolumeCard
    calls = []
    monkeypatch.setattr(cap.Backdrop, "grab", lambda self, *a, **k: calls.append(1))
    monkeypatch.setattr(cap, "windows_prefs",
                        lambda: {"transparency": True, "animations": False, "high_contrast": False})
    card = _glass_card()
    card.present()
    assert card._live.isActive() and calls
    card.conceal()
    assert not card._live.isActive()
    card.close()

    monkeypatch.setattr(cap, "windows_prefs",
                        lambda: {"transparency": False, "animations": False, "high_contrast": False})
    calls.clear()
    card = _glass_card()
    card.present()                               # transparency effects off: solid, no capture
    assert not card._live.isActive() and not calls and not card.glass_wanted
    card.close()

    solid = VolumeCard(Config(style="solid"))
    solid.present()
    assert not solid._live.isActive()
    solid.close()


def test_restarts_go_to_the_run_they_lead_up_to():
    from spotter_volume.ledger import Run, attribute
    t = lambda m: datetime(2026, 11, 27, 1, 0) + timedelta(minutes=m)
    saved = [Run("A", t(0), t(1)), Run("B", t(5), t(6))]
    restarts = [Run(None, t(3), t(3.5)),      # between A and B: B comes next within 10 min
                Run(None, t(8), t(8.5)),      # after B, nothing next yet: back to B
                Run(None, t(30), t(30.5)),    # nothing within 10 min either side
                Run(None, t(-4), t(-3.5))]    # before A
    assert attribute(restarts, saved) == ["B", "B", None, "A"]
