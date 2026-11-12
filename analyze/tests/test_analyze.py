"""The analyzer must recover errors planted in synthetic runs."""
import math
from pathlib import Path

import numpy as np
import pytest

from spotter_analyze import clicking, kovaaks, report, synth, tracking
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
