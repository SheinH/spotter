"""The analyzer must recover errors planted in synthetic runs."""
import math
from pathlib import Path

import numpy as np
import pytest

from spotter_analyze import clicking, kovaaks, report, synth, tension, tracking
from spotter_analyze.__main__ import main, pick_mode
from spotter_analyze.format import read_spot, write_spot


def test_spot_roundtrip(tmp_path):
    run = synth.clicking(kills=5)
    write_spot(tmp_path / "a.spot", run)
    back = read_spot(tmp_path / "a.spot")
    assert len(back.t) == len(run.t)
    np.testing.assert_allclose(back.t, run.t, atol=1e-6)
    np.testing.assert_allclose(back.yaw, run.yaw, atol=1e-4)
    np.testing.assert_array_equal(back.shots, run.shots)
    np.testing.assert_array_equal(back.tgt_frame, run.tgt_frame)
    np.testing.assert_allclose(back.tgt_pos, run.tgt_pos, rtol=1e-6)


def test_truncated_log_reads_whole_frames(tmp_path):
    run = synth.clicking(kills=3)
    write_spot(tmp_path / "a.spot", run)
    data = (tmp_path / "a.spot").read_bytes()
    (tmp_path / "cut.spot").write_bytes(data[:-17])
    assert len(read_spot(tmp_path / "cut.spot").t) == len(run.t) - 1


@pytest.mark.parametrize("lag", [0.08, 0.15])
def test_tracking_recovers_planted_lag(lag):
    run = synth.tracking(lag=lag, speed=20.0, radius=60.0)
    assert pick_mode(run) == "tracking"
    res = tracking.analyze(run)
    assert res["lag_ms"] == pytest.approx(lag * 1000, abs=12)
    rev = [r for r in res["table"] if r["rev"]]
    assert len(rev) > 15
    assert np.nanmedian([r["react_ms"] for r in rev]) == pytest.approx(lag * 1000, abs=30)
    # Steady state: crosshair trails by speed*lag degrees; target radius is asin(60/1000).
    expected = -(20.0 * lag) / math.degrees(math.asin(0.06))
    steady = [r["lead_r"] for r in res["table"] if not r["rev"] or r["dur_ms"] > 800]
    assert np.nanmedian(steady) == pytest.approx(expected, abs=0.2)
    assert np.nanmedian([r["gain"] for r in res["table"]]) == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("lag", [0.06, 0.12])
def test_tracking_finds_smooth_reversals(lag):
    run = synth.tracking(lag=lag, speed=40.0, smooth_turns=True)
    res = tracking.analyze(run)
    rev = [r for r in res["table"] if r["rev"]]
    # every knot in a 30 s run with 0.6-1.5 s legs is a turnaround: about 28 of them
    assert len(rev) > 18
    assert np.nanmedian([r["react_ms"] for r in rev]) == pytest.approx(lag * 1000, abs=30)
    assert len(res["plot"]["prof_c"]) > 10


def test_tension_recovers_planted_tremor():
    calm = tension.analyze(synth.tracking(lag=0.1, speed=40.0, smooth_turns=True), "tracking", None)
    shaky = tension.analyze(synth.tracking(lag=0.1, speed=40.0, smooth_turns=True,
                                           tremor_deg=0.05, tremor_grow=1.0), "tracking", None)
    # 0.05 deg at 10 Hz, growing 1.0x -> 1.5x -> 2.0x amplitude: rms velocity 2*pi*10*A/sqrt(2)
    base = 2 * math.pi * 10 * 0.05 / math.sqrt(2)
    assert shaky["thirds"][0] == pytest.approx(base * 7 / 6, rel=0.15)
    assert shaky["thirds"][2] == pytest.approx(base * 11 / 6, rel=0.15)
    assert calm["jitter"] < 0.25 * shaky["jitter"]
    assert shaky["peak_hz"] == pytest.approx(10, abs=1)
    assert shaky["excess_db"] > calm["excess_db"] + 5


def test_tension_flags_flick_rebound():
    clean = tension.analyze(synth.clicking(kills=60), "clicking", None)
    clamped = tension.analyze(synth.clicking(kills=60, lockout_deg=0.15), "clicking", None)
    rb = tension.FLICK_COLS.index("rebound_pct")
    assert np.median(clamped["flicks"][:, rb]) > 2 * np.median(clean["flicks"][:, rb])
    dec, acc = tension.FLICK_COLS.index("decel_ms"), tension.FLICK_COLS.index("accel_ms")
    # minimum-jerk flicks are symmetric
    assert np.median(clean["flicks"][:, dec] / clean["flicks"][:, acc]) == pytest.approx(1.0, abs=0.1)


def test_tracking_on_target_drops_with_lag():
    good = tracking.analyze(synth.tracking(lag=0.05))["on_pct"]
    bad = tracking.analyze(synth.tracking(lag=0.25))["on_pct"]
    assert good > 95 and bad < good - 20


def test_clicking_recovers_planted_undershoot():
    run = synth.clicking(kills=40, react=0.18, reach=0.85, early_every=5)
    assert pick_mode(run) == "clicking"
    res = clicking.analyze(run)
    t = res["table"]
    assert res["kills"] == 40
    assert res["shots"] == 48
    assert res["misses"] == 8
    assert all(r["miss"] in ("0", "1E") for r in t)
    assert np.median([r["react_ms"] for r in t]) == pytest.approx(180, abs=25)
    assert np.median([r["land_r"] for r in t]) < -1.0          # short of the target
    assert np.mean([r["corr"] for r in t]) == pytest.approx(1.0, abs=0.2)
    assert np.median([r["confirm_ms"] for r in t]) == pytest.approx(50, abs=40)


def test_clicking_overshoot_has_positive_landing():
    res = clicking.analyze(synth.clicking(kills=20, reach=1.2, early_every=0))
    assert np.median([r["land_r"] for r in res["table"]]) > 1.0
    assert res["misses"] == 0
    t = res["table"]
    # synth is past the edge for the end of the flick, a 60 ms pause, and most of the 80 ms way back
    assert np.mean([r["over_r"] > 1.0 for r in t]) > 0.8
    assert np.median([r["past_ms"] for r in t if r["over_r"] > 1.0]) == pytest.approx(160, abs=40)
    assert res["metrics"]["over_pct"] > 80


def test_undershoot_has_no_overflick():
    res = clicking.analyze(synth.clicking(kills=20, reach=0.85, early_every=0))
    assert all(r["past_ms"] == 0 for r in res["table"])
    assert res["metrics"]["over_pct"] == 0


def test_overflick_counts_pass_through_but_not_later_drift():
    tt = np.arange(0, 400, 2.5)                      # 400 Hz, ms
    # pass through the target (on for 20 ms), 50 ms beyond the edge, come back and settle
    rem = np.interp(tt, [0, 40, 60, 110, 160, 400], [5, 1, -2.5, -2.5, 0, 0])
    on = np.abs(rem) <= 1
    over, past = clicking._overflick(rem, on, tt)
    assert over == pytest.approx(2.5)
    assert past == pytest.approx(80, abs=10)
    # settled on target first, then drifted past: tracking error, not overflick
    rem2 = np.where(tt < 200, 0.0, -3.0)
    assert clicking._overflick(rem2, np.abs(rem2) <= 1, tt) == (0.0, 0.0)


CSV = """Kill #,Timestamp,Bot,Weapon,TTK,Shots,Hits,Accuracy,Damage Done,Damage Possible,Efficiency,Cheated,OverShots
1,23:59:59.890,target,pistol,0.000000s,1,1,1.000000,25.000000,50.000000,0.500000,0,0
2,00:00:00.520,target,pistol,0.000000s,2,1,0.500000,25.000000,50.000000,0.500000,0,0

Weapon,Shots,Hits,Damage Done,Damage Possible,,Sens Scale,Horiz Sens
pistol,119,104,2600.0,5950.0,

Kills:,104
Score:,104.0
Scenario:,ww4t Voltaic
Challenge Start:,23:59:07.100

Horiz Sens:,30.0
FOV:,103.0
"""


def test_stats_csv_across_midnight(tmp_path):
    p = tmp_path / "ww4t Voltaic - Challenge - 2026.11.12-00.00.07 Stats.csv"
    p.write_text(CSV)
    st = kovaaks.parse_stats(p)
    assert st.scenario == "ww4t Voltaic"
    assert (st.shots, st.hits) == (119, 104)
    assert st.fields["FOV"] == "103.0"
    assert st.start.day == 11 and st.start.hour == 23
    assert [k.day for k in st.kill_times] == [11, 12]
    assert st.kill_shots == [1, 2]


REAL_PERF = kovaaks.DEFAULT_ROOT / "performances" / "ww4t Voltaic - Challenge - 2026.11.12-00.15.07 Performance.perf"


@pytest.mark.skipif(not REAL_PERF.exists(), reason="needs a real KovaaK's install")
def test_real_perf_totals():
    perf = kovaaks.parse_perf(REAL_PERF)
    assert perf["scenario"] == "ww4t Voltaic"
    assert perf["duration"] == 60.0
    assert (perf["shots"].sum(), perf["hits"].sum(), perf["kills"].sum()) == (119, 104, 104)


@pytest.mark.parametrize("kind", ["tracking", "clicking"])
def test_cli_writes_report(tmp_path, kind):
    write_spot(tmp_path / "x.spot", getattr(synth, kind)())
    assert main([str(tmp_path / "x.spot"), "--out", str(tmp_path / "r"), "--worst", "2",
                 "--stats-dir", str(tmp_path / "none")]) == 0
    out = tmp_path / "r" / "x"
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "## Events" in text and "## Worst 2 events" in text
    assert len(text) < 16000
    assert (out / "error_scatter.png").stat().st_size > 5000
    assert (out / "velocity_profiles.png").stat().st_size > 5000


def test_mode_uses_stats_kill_count():
    run = synth.tracking()
    st = kovaaks.Stats(path=Path("x"), scenario="s", end=None, start=None, fields={"Kills": "0"})
    assert pick_mode(run, st) == "tracking"
    # without stats: no target ever lost health, so nothing can have been killed
    assert pick_mode(run) == "tracking"


def _join(a, b, gap=0.5):
    """Two runs back to back in one log, with the shot counter reset between them."""
    from dataclasses import replace
    shift = a.t[-1] + gap
    return replace(
        a, t=np.r_[a.t, b.t + shift], yaw=np.r_[a.yaw, b.yaw], pitch=np.r_[a.pitch, b.pitch],
        fov=np.r_[a.fov, b.fov], cam=np.r_[a.cam, b.cam], shots=np.r_[a.shots, b.shots],
        buttons=np.r_[a.buttons, b.buttons], tgt_frame=np.r_[a.tgt_frame, b.tgt_frame + len(a.t)],
        tgt_id=np.r_[a.tgt_id, b.tgt_id + 1000], tgt_pos=np.r_[a.tgt_pos, b.tgt_pos],
        tgt_radius=np.r_[a.tgt_radius, b.tgt_radius], tgt_hh=np.r_[a.tgt_hh, b.tgt_hh],
        tgt_health=np.r_[a.tgt_health, b.tgt_health])


def test_runs_found_from_log_without_stats(tmp_path):
    from spotter_analyze import spans
    log = _join(synth.tracking(duration=30), synth.tracking(seed=1, duration=40))
    found = spans.find_runs(log, tmp_path / "none")
    assert [round(s.t1 - s.t0) for s in found] == [30, 40]
    # a clicking run's one-frame gaps between kills stay inside the run
    assert len(spans.target_spans(synth.clicking(kills=40))) == 1


def test_noise_verdict():
    from spotter_analyze import history
    hist = [{"name": f"r{i}", "scenario": "s", "mode": "tracking", "score": 100.0 + i % 3,
             "metrics": {"on_pct": 50.0 + (i % 3)}} for i in range(10)]
    me = {"name": "me", "scenario": "s", "mode": "tracking", "score": 101.0, "metrics": {"on_pct": 51.5}}
    cols, rows, n = history.noise_rows(me, hist)
    assert n == 10 and rows[1][5] == "within noise"
    me["metrics"]["on_pct"] = 60.0
    assert history.noise_rows(me, hist)[1][1][5].startswith("higher (better)")


def test_session_page(tmp_path):
    from spotter_analyze import history, session
    recs = []
    for i in range(6):
        res = tracking.analyze(synth.tracking(seed=i, lag=0.05 + 0.03 * i))
        st = kovaaks.Stats(path=Path("x"), scenario="Synth", end=__import__("datetime").datetime(2026, 1, 1, 10, i),
                           start=None, fields={"Score": str(1000 - 50 * i), "Horiz Sens": "30", "FOV": "103"})
        recs.append(history.make_record(f"run{i}", "tracking", st, res, None))
    text = session.write(tmp_path / "s", "Synth", recs, tmp_path / "none").read_text(encoding="utf-8")
    assert "## Pooled across runs" in text and "## Trend" in text and "What moves with score" in text
    assert "far behind" in text and "100-200" in text
