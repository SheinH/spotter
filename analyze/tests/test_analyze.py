"""The analyzer must recover errors planted in synthetic runs."""
import math
from pathlib import Path

import numpy as np
import pytest

from spotter_analyze import clicking, kovaaks, report, sens, synth, tension, tracking
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
    assert calm["activity"] < 0.25 * shaky["activity"]
    # on-target jitter sees the tremor too, and the tremor peak test finds it at 10 Hz
    # (calm still has synth's 0.02 deg per-frame noise, which reads ~0.85 deg/s)
    assert calm["hold_jitter"] < 0.4 * shaky["hold_jitter"]
    assert shaky["tremor"]["peak_hz"] == pytest.approx(10, abs=1)
    assert shaky["tremor"]["peak_db"] > 3


def test_hold_jitter_ignores_flicks_and_finds_tremor():
    def hold(**kw):
        return tension.hold_jitter(synth.clicking(kills=60, early_every=0, confirm=0.19, **kw), "switching")
    calm, shaky = hold(), hold(tremor_deg=0.05)
    planted = 2 * math.pi * 10 * 0.05 / math.sqrt(2)
    assert calm["value"] < 0.05                     # flicks and corrections don't leak in
    assert shaky["value"] == pytest.approx(planted, rel=0.35)   # trend removal costs some of it
    lo, hi = shaky["range90"]
    assert lo <= shaky["value"] <= hi


def test_hold_jitter_does_not_grow_with_flick_size():
    small = tension.hold_jitter(synth.clicking(kills=60, early_every=0, confirm=0.19, tremor_deg=0.03), "switching")
    big = tension.hold_jitter(synth.clicking(kills=60, early_every=0, confirm=0.19, tremor_deg=0.03,
                                             reach=0.6), "switching")      # bigger corrections
    assert big["value"] == pytest.approx(small["value"], rel=0.2)


def test_tension_flags_flick_rebound():
    clean = tension.analyze(synth.clicking(kills=60), "clicking", None)
    clamped = tension.analyze(synth.clicking(kills=60, lockout_deg=0.15), "clicking", None)
    rb = tension.FLICK_COLS.index("rebound_pct")
    assert np.median(clamped["flicks"][:, rb]) > 2 * np.median(clean["flicks"][:, rb])
    dec, acc = tension.FLICK_COLS.index("decel_ms"), tension.FLICK_COLS.index("accel_ms")
    # minimum-jerk flicks are symmetric
    assert np.median(clean["flicks"][:, dec] / clean["flicks"][:, acc]) == pytest.approx(1.0, abs=0.1)


def test_brake_ratio_tells_hard_stops_from_coasting():
    def brake(**kw):
        return clicking.analyze(synth.clicking(kills=40, early_every=0, **kw))["metrics"]["brake_ratio"]
    sym, hard, coast = brake(), brake(stop_skew=1.6), brake(stop_skew=0.6)
    assert sym == pytest.approx(1.0, abs=0.1)        # minimum jerk is symmetric
    assert hard > 1.3
    assert coast < 0.7


def test_post_flick_wobble_sees_shake_not_clean_corrections():
    clean = clicking.analyze(synth.clicking(kills=40, early_every=0, reach=0.7))   # big but clean corrections
    shaky = clicking.analyze(synth.clicking(kills=40, early_every=0, lockout_deg=0.15))
    assert clean["metrics"]["wobble_r"] < 0.02
    assert shaky["metrics"]["wobble_r"] > 0.08
    # a symmetric flick with a shake after it is still a symmetric flick
    assert shaky["metrics"]["brake_ratio"] == pytest.approx(1.0, abs=0.15)


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


def test_off_episodes_after_settling():
    tt = np.arange(0, 600, 2.5)                      # 400 Hz, ms
    miss_r = np.full(len(tt), 0.5)                   # on target (inside the edge)
    miss_r[tt < 100] = 3.0                           # flying in
    miss_r[(tt >= 105) & (tt < 115)] = 2.0           # pass-through bounce before settling: not counted
    miss_r[(tt >= 250) & (tt < 255)] = 1.02          # 2-frame graze: an episode, but under SLIP_MS
    miss_r[(tt >= 350) & (tt < 450)] = 1.6           # a real slip, 100 ms
    miss_r[tt >= 590] = 4.0                          # leaving as it dies: not counted
    eps = clicking._off_episodes(miss_r <= 1, miss_r, tt)
    assert len(eps) == 2
    assert eps[0][0] < clicking.SLIP_MS
    assert eps[1][0] == pytest.approx(102.5, abs=3)
    assert eps[1][1] == pytest.approx(0.6)


@pytest.mark.parametrize("reach", [0.85, 1.2])
def test_flick_gain_recovers_planted_reach(reach):
    res = clicking.analyze(synth.clicking(kills=40, reach=reach, early_every=0))
    fg = clicking.flick_gain(res["table"])
    # the first movement is cut where speed falls to 25% of peak: ~97.5% of a minimum-jerk path
    assert fg["gain"] == pytest.approx(0.975 * reach, abs=0.04)
    assert fg["slope"] == pytest.approx(0.975 * reach, abs=0.06)
    assert abs(fg["offset_deg"]) < 1.0
    assert fg["spread_pct"] < 5 and fg["side_deg"] < 0.3
    assert res["metrics"]["flick_gain"] == fg["gain"]


def test_time_budget_adds_up():
    res = clicking.analyze(synth.clicking(kills=40, react=0.18, reach=0.85, early_every=0), "switching")
    for r in res["table"]:
        assert sum(r["_budget"].values()) == pytest.approx(r["ttk_ms"], abs=0.5)
    assert np.median([r["_budget"]["react"] for r in res["table"]]) == pytest.approx(180, abs=25)
    assert all(r["_budget"]["slip"] == 0 for r in res["table"])
    cols, rows, share = clicking.time_budget(res["table"], True)
    assert sum(v[0] for v in share.values()) == pytest.approx(100, abs=0.5)
    assert math.isnan(rows[-1][5])          # on-target time can't be cut in switching
    assert any("time budget" in s for s in res["summary"])


@pytest.mark.parametrize("mode", ["clicking", "switching"])
def test_flick_micro_phases_add_up(mode):
    res = clicking.analyze(synth.clicking(kills=40, react=0.18, reach=0.85, early_every=0), mode)
    for r in res["table"]:
        assert sum(r["_phase"].values()) == pytest.approx(r["ttk_ms"], abs=0.5)
        assert min(r["_phase"].values()) >= 0
    m = res["metrics"]
    # synth: flick lasts 100 ms + 4 ms/deg (cut a little early at 25% of peak speed), then a 60 ms
    # pause and an 80 ms correction onto the centre
    assert m["flick_ms"] == pytest.approx(150, abs=30)
    assert m["micro_ms"] == pytest.approx(130, abs=30)
    assert m["flick_on_pct"] == 0
    assert m["flick_err_pct"] == pytest.approx(15, abs=3)
    assert any(s.startswith("flick verdict") for s in res["summary"])
    assert any(s.startswith("micro verdict") for s in res["summary"])
    assert any(t.startswith("Flick vs micro") for t, _c, _r in res["splits"])


def test_exact_flick_needs_no_micro():
    res = clicking.analyze(synth.clicking(kills=30, reach=1.0, early_every=0))
    m = res["metrics"]
    assert m["flick_on_pct"] == 100
    assert m["micro_ms"] == 0
    assert m["flick_ms"] == pytest.approx(150, abs=30)


def test_micro_blame_goes_to_the_flick_when_micro_is_steady():
    # every micro in synth is the same 60 ms pause + 80 ms correction, so none of it is the
    # micro being slower than its own best: it is all the flick's leftover
    m = clicking.analyze(synth.clicking(kills=40, reach=0.85, early_every=0))["metrics"]
    assert m["micro_saved_s"] < 0.15 * m["flick_cost_s"]


@pytest.mark.parametrize("lag", [0.08, 0.15])
def test_lag_shift_recovers_planted_delay(lag):
    # small target, so a pure delay pushes the crosshair off it; shifting it back fixes everything
    res = tracking.analyze(synth.tracking(lag=lag, speed=40.0, radius=25.0))
    m = res["metrics"]
    assert m["shift_on_pct"] > 95
    assert m["shift_gain_pts"] > 20
    line = [s for s in res["summary"] if s.startswith("lag-shift")][0]
    assert float(line.split("crosshair ")[1].split(" ms")[0]) == pytest.approx(lag * 1000, abs=12)


def test_clean_switching_run_has_no_slips():
    res = clicking.analyze(synth.clicking(kills=20, reach=0.85, early_every=0), "switching")
    assert all(r["slips"] == 0 for r in res["table"])
    assert res["metrics"]["slip_pct"] == 0


def _with_decoy(run, angles):
    """`run` plus one target that never dies, at yaw/pitch `angles` (one row per frame)."""
    from dataclasses import replace
    n = len(run.t)
    order = np.argsort(np.r_[run.tgt_frame, np.arange(n)], kind="stable")
    cat = lambda a, b: np.concatenate([a, b])[order]
    return replace(
        run, tgt_frame=cat(run.tgt_frame, np.arange(n)), tgt_id=cat(run.tgt_id, np.full(n, 9999)),
        tgt_pos=cat(run.tgt_pos, synth.DIST * synth._direction(angles[:, 0], angles[:, 1])),
        tgt_radius=cat(run.tgt_radius, np.full(n, 25.0)), tgt_hh=cat(run.tgt_hh, np.full(n, 25.0)),
        tgt_health=cat(run.tgt_health, np.ones(n)))


def test_target_choice_ranks_the_picked_target():
    run = synth.clicking(kills=20, early_every=0)
    n = len(run.t)
    alone = clicking.analyze(run)
    assert "nearest_pct" not in alone["metrics"]      # one target: nothing to choose
    assert "pick" not in alone["columns"]

    far = clicking.analyze(_with_decoy(run, np.tile([120.0, 0.0], (n, 1))))
    assert far["metrics"]["nearest_pct"] == 100
    assert far["metrics"]["extra_deg"] == 0
    assert far["columns"][-3:] == ["opts", "pick", "extra_deg"]

    # a decoy that rides 3 deg from the crosshair is always the nearer one (targets spawn 12+ away)
    near = clicking.analyze(_with_decoy(run, np.c_[run.yaw + 3.0, run.pitch]))
    assert near["metrics"]["nearest_pct"] == 0
    assert all(r["pick"] == 2 and r["opts"] == 2 for r in near["table"])
    assert all(r["extra_deg"] == pytest.approx(r["dist_deg"] - 3.0, abs=0.3) for r in near["table"])
    assert any(line.startswith("target choice") for line in near["summary"])


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


def test_align_kills_recovers_clock_offset_between_close_kills():
    # kills ~0.5 s apart with the log clock 247 ms behind: half way to the next kill, so plain
    # nearest-neighbour matching would pair many game kills with the wrong log kill
    rng = np.random.default_rng(3)
    game = 1000.0 + np.cumsum(rng.uniform(0.35, 0.75, 100))
    mine = np.sort(game - 0.247 + rng.normal(0, 0.002, len(game)))
    shift, matched = report.align_kills(mine, game)
    assert shift == pytest.approx(-247, abs=3)
    assert matched >= 98


# ---- live sensitivity (format v3) ----

def _with_sens(run, mult):
    """Configured 0.116633 (35 cm/360 @ 3200 dpi); live scale = configured / mult."""
    n = len(run.t)
    run.user_sens = np.full(n, 0.116633)
    run.yaw_scale = run.user_sens / np.broadcast_to(mult, (n,))
    return run


FIELDS_35 = {"Sens Scale": "cm/360", "Horiz Sens": "35.0", "DPI": "3200", "Sens Increment": "0.116633"}


def test_spot_v3_roundtrip_keeps_sens(tmp_path):
    run = _with_sens(synth.clicking(kills=3), np.linspace(0.75, 1.25, 1))
    run.yaw_scale[: len(run.t) // 2] *= 1.5
    write_spot(tmp_path / "a.spot", run)
    back = read_spot(tmp_path / "a.spot")
    np.testing.assert_allclose(back.yaw_scale, run.yaw_scale, rtol=1e-6)
    np.testing.assert_allclose(back.user_sens, run.user_sens, rtol=1e-6)
    sl = back.slice(back.t[10], back.t[50])
    assert len(sl.yaw_scale) == len(sl.t)


def test_v2_log_still_reads_without_sens(tmp_path):
    import struct
    from spotter_analyze import format as fmt
    run = synth.clicking(kills=3)
    write_spot(tmp_path / "a.spot", run)
    data = bytearray((tmp_path / "a.spot").read_bytes())
    # rebuild as v2: drop the 8 sens bytes after every frame record
    out = bytearray(data[:fmt.HEADER.size])
    struct.pack_into("<H", out, 4, 2)
    off = fmt.HEADER.size
    while off < len(data):
        f = fmt.FRAME.unpack_from(data, off)
        out += data[off:off + fmt.FRAME_V2.size]
        off += fmt.FRAME.size
        out += data[off:off + f[9] * fmt.TARGET.size]
        off += f[9] * fmt.TARGET.size
    (tmp_path / "v2.spot").write_bytes(bytes(out))
    back = read_spot(tmp_path / "v2.spot")
    assert len(back.t) == len(run.t)
    assert np.isnan(back.yaw_scale).all()
    assert sens.analyze(back, FIELDS_35) is None
    assert tension.mm_per_deg(FIELDS_35, back) == pytest.approx(35 * 10 / 360)


def test_sens_steady_is_not_flagged():
    run = _with_sens(synth.clicking(kills=3), 1.0)
    sv = sens.analyze(run, FIELDS_35)
    assert not sv["randomized"] and sv["cm360_median"] == pytest.approx(35.0)


def test_sens_randomizer_detected_and_reciprocal():
    run = synth.clicking(kills=3)
    n = len(run.t)
    # engine scale 0.155377 = turns further per count than configured -> fewer cm per 360
    run = _with_sens(run, 1.0)
    run.yaw_scale[n // 2:] = 0.155377
    sv = sens.analyze(run, FIELDS_35)
    assert sv["randomized"] and sv["varies"]
    assert sv["mult_min"] * 35 == pytest.approx(26.27, abs=0.01)
    assert sv["mult_max"] * 35 == pytest.approx(35.0)
    live = sens.cm360(run, FIELDS_35)
    assert live[0] == pytest.approx(35.0) and live[-1] == pytest.approx(26.27, abs=0.01)
    mmpd = tension.mm_per_deg(FIELDS_35, run)
    assert mmpd.shape == run.t.shape
    tension.analyze(run, "clicking", mmpd)   # per-frame scale runs end to end


def test_configured_cm360_from_increment_matches_stats():
    f = dict(FIELDS_35, **{"Sens Scale": "Overwatch"})
    assert sens.configured_cm360(f) == pytest.approx(35.0, rel=1e-3)
