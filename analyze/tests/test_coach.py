"""Tests for Spotter's streaming .spot tailer, live state machine, mid-run cues, debriefs, and overlay."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from spotter_analyze import synth
from spotter_analyze.format import read_spot, write_spot
from spotter_analyze.stream import (
    STATS_WAIT_S,
    LiveTracker,
    RunAborted,
    RunCompleted,
    RunStarted,
    SpotTailer,
)
from spotter_coach.controller import CoachController
from spotter_coach.cues import COOLDOWN_S, WARMUP_S, CueEngine, evaluate_cue_effect
from spotter_coach.debrief import DebriefToolHandler, build_debrief
from spotter_coach.overlay import CoachOverlay


def _write_stats_csv(stats_dir: Path, scenario: str, start: datetime, end: datetime,
                     score: float = 112.5, shots: int = 40, hits: int = 32) -> Path:
    stamp = end.strftime("%Y.%m.%d-%H.%M.%S")
    path = stats_dir / f"{scenario} - Challenge - {stamp} Stats.csv"
    text = (
        "Kill #,Timestamp,Bot,Weapon,TTK,Shots,Hits,Accuracy,Damage Done,Damage Possible,Efficiency,Cheated,OverShots\n"
        f"1,{end.strftime('%H:%M:%S.%f')[:-3]},target,rifle,0.35s,4,4,1.0,100,100,1.0,0,0\n\n"
        "Weapon,Shots,Hits,Damage Done,Damage Possible\n"
        f"rifle,{shots},{hits},1000,1000\n\n"
        f"Score:,{score}\n"
        f"Challenge Start:,{start.strftime('%H:%M:%S.%f')[:-3]}\n"
        "Horiz Sens:,35.0\n"
        "Sens Scale:,cm/360\n"
        "DPI:,1600\n"
        "FOV:,103\n"
        "FOVScale:, Overwatch\n"
        "Avg FPS:,360\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def test_spot_tailer_matches_read_spot_across_partial_byte_chunks(tmp_path: Path):
    run = synth.clicking(kills=12, seed=7)
    run.start_wall_us = 1_760_000_000_000_000
    spot = tmp_path / "chunked.spot"
    write_spot(spot, run)
    expected = read_spot(spot)

    # Drip uneven prime-sized byte slices (73 bytes cuts right across 40B frames and 28B targets)
    tailer = SpotTailer(spot)
    while tailer.offset < spot.stat().st_size:
        tailer.poll(max_bytes=73)

    got = tailer.to_run()
    assert np.allclose(got.t, expected.t)
    assert np.allclose(got.yaw, expected.yaw)
    assert np.allclose(got.pitch, expected.pitch)
    assert np.array_equal(got.shots, expected.shots)
    assert np.array_equal(got.tgt_frame, expected.tgt_frame)
    assert np.allclose(got.tgt_pos, expected.tgt_pos)

    # Window slice matches Run.slice bit-for-bit
    s_got = tailer.slice(1.5, 4.2)
    s_exp = expected.slice(1.5, 4.2)
    assert np.allclose(s_got.t, s_exp.t)
    assert np.array_equal(s_got.tgt_frame, s_exp.tgt_frame)
    assert np.allclose(s_got.tgt_pos, s_exp.tgt_pos)


def test_live_tracker_suppresses_early_restarts_and_detects_completed_runs(tmp_path: Path):
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()

    # Build a run from t=0..12s, followed by frozen score-screen frames from t=12..16s
    base = synth.clicking(kills=18, seed=1)
    wall0_dt = datetime(2026, 11, 17, 20, 0, 0)
    base.start_wall_us = int(wall0_dt.timestamp() * 1e6)

    # Append 4 seconds of frozen score-screen frames (targets still alive in arena, camera & shots still)
    t_end = float(base.t[-1])
    extra_t = np.arange(t_end + 0.01, t_end + 4.0, 0.01)
    m = len(extra_t)
    n0 = len(base.t)
    frozen_tgt_frames = np.arange(n0, n0 + m, dtype=np.int64)
    full = synth.Run(
        t=np.r_[base.t, extra_t],
        yaw=np.r_[base.yaw, np.full(m, base.yaw[-1])],
        pitch=np.r_[base.pitch, np.full(m, base.pitch[-1])],
        fov=np.r_[base.fov, np.full(m, base.fov[-1])],
        cam=np.vstack([base.cam, np.tile(base.cam[-1], (m, 1))]),
        shots=np.r_[base.shots, np.full(m, base.shots[-1], dtype=np.int64)],
        buttons=np.r_[base.buttons, np.zeros(m, dtype=np.uint8)],
        tgt_frame=np.r_[base.tgt_frame, frozen_tgt_frames],
        tgt_id=np.r_[base.tgt_id, np.full(m, 999, dtype=np.int64)],
        tgt_pos=np.vstack([base.tgt_pos, np.tile([400.0, 0.0, 0.0], (m, 1))]),
        tgt_radius=np.r_[base.tgt_radius, np.full(m, 25.0)],
        tgt_hh=np.r_[base.tgt_hh, np.full(m, 25.0)],
        tgt_health=np.r_[base.tgt_health, np.ones(m)],
        start_wall_us=base.start_wall_us,
        game="kovaaks",
    )

    # Case 1: No Stats.csv -> early restart / paused -> emits RunStarted then RunAborted, NEVER RunCompleted
    spot1 = tmp_path / "abort.spot"
    write_spot(spot1, full)
    tracker1 = LiveTracker(SpotTailer(spot1), stats_dir=stats_dir)
    ev1 = tracker1.step(now_monotonic=100.0)
    ev1 += tracker1.step(now_monotonic=100.0 + STATS_WAIT_S + 0.5)
    assert any(isinstance(e, RunStarted) for e in ev1)
    assert any(isinstance(e, RunAborted) for e in ev1)
    assert not any(isinstance(e, RunCompleted) for e in ev1)

    # Case 2: Stats.csv written when challenge completes -> emits RunCompleted, and frozen score-screen
    # frames afterwards do NOT falsely start a second run!
    spot2 = tmp_path / "done.spot"
    write_spot(spot2, full)
    _write_stats_csv(
        stats_dir,
        "voxTargetSwitch 2",
        start=wall0_dt,
        end=wall0_dt + timedelta(seconds=t_end),
    )
    tracker2 = LiveTracker(SpotTailer(spot2), stats_dir=stats_dir, watch_existing_stats=True)
    ev2 = tracker2.step(now_monotonic=200.0)
    starts = [e for e in ev2 if isinstance(e, RunStarted)]
    completed = [e for e in ev2 if isinstance(e, RunCompleted)]
    assert len(starts) == 1
    assert len(completed) == 1
    assert completed[0].stats.scenario == "voxTargetSwitch 2"
    assert not tracker2.in_run  # frozen score-screen targets did not re-enter IN_RUN


def test_cue_engine_enforces_warmup_and_cooldown():
    # Planted undershoot (reach=0.82 -> flick_gain ~0.80 < 0.87)
    run = synth.clicking(kills=30, reach=0.82, early_every=0, seed=4)
    engine = CueEngine(history_records=[], scenario_hint="1w6ts reload v2")

    # Warmup (<10 s) must stay completely quiet
    assert engine.evaluate(run, run_elapsed_s=WARMUP_S - 0.5) is None

    # Past warmup -> fires underflick cue
    cue1 = engine.evaluate(run, run_elapsed_s=14.0)
    assert cue1 is not None
    assert cue1.metric in ("flick_gain", "short_pct")
    assert "[IN_RUN_CUE]" in cue1.prompt

    # Within 20s cooldown -> suppressed
    assert engine.evaluate(run, run_elapsed_s=14.0 + COOLDOWN_S - 2.0) is None

    # Past cooldown -> allowed again
    cue2 = engine.evaluate(run, run_elapsed_s=14.0 + COOLDOWN_S + 1.0)
    assert cue2 is not None


def test_debrief_bundle_and_tools(tmp_path: Path):
    stats_dir = tmp_path / "stats"
    out_dir = tmp_path / "reports"
    stats_dir.mkdir()

    run = synth.clicking(kills=24, reach=0.78, early_every=0, seed=9)
    t_dur = float(run.t[-1])
    start_dt = datetime(2026, 11, 17, 21, 0, 0)
    end_dt = start_dt + timedelta(seconds=t_dur)
    run.start_wall_us = int(start_dt.timestamp() * 1e6)

    st_path = _write_stats_csv(stats_dir, "voxTargetSwitch 2", start_dt, end_dt, score=108.4)
    from spotter_analyze import kovaaks
    st = kovaaks.parse_stats(st_path)

    engine = CueEngine([], "voxTargetSwitch 2")
    cue = engine.evaluate(run, run_elapsed_s=15.0)
    assert cue is not None

    bundle = build_debrief(run, st, out_dir=out_dir, stats_dir=stats_dir, fired_cues=[cue])
    assert bundle.scenario == "voxTargetSwitch 2"
    assert bundle.scatter_png.is_file()
    assert bundle.profiles_png.is_file()
    assert bundle.report_path.is_file()
    assert bundle.time_budget is not None
    assert len(bundle.cue_notes) == 1
    assert "[POST_RUN_DEBRIEF]" in bundle.llm_prompt

    handler = DebriefToolHandler(out_dir=out_dir, stats_dir=stats_dir)
    handler.current = bundle
    thirds = handler.get_split_details("thirds")
    assert "rows" in thirds and len(thirds["rows"]) == 3


def test_overlay_and_controller_state_transitions(tmp_path: Path):
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    stats_dir = tmp_path / "stats"
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "reports"
    stats_dir.mkdir()
    runs_dir.mkdir()

    run = synth.clicking(kills=20, reach=0.88, early_every=0, seed=5)
    start_dt = datetime(2026, 11, 17, 21, 10, 0)
    end_dt = start_dt + timedelta(seconds=float(run.t[-1]))
    run.start_wall_us = int(start_dt.timestamp() * 1e6)

    spot = runs_dir / "test.spot"
    write_spot(spot, run)
    st_path = _write_stats_csv(stats_dir, "1w6ts reload v2", start_dt, end_dt, score=122.0)
    from spotter_analyze import kovaaks
    st = kovaaks.parse_stats(st_path)

    async def _exercise():
        loop = asyncio.get_running_loop()
        overlay = CoachOverlay()
        ctrl = CoachController(
            runs_dir=runs_dir,
            stats_dir=stats_dir,
            out_dir=out_dir,
            overlay=overlay,
            loop=loop,
            spot_path=spot,
            api_key="",  # visual-only mode
            enable_audio=False,
        )
        ctrl._ensure_tracker()
        ctrl._tailer.poll()

        # 1. Start in IDLE: compact pill (<95 px tall!), mic strictly off
        assert ctrl.state == "IDLE"
        assert overlay.isVisible()
        assert overlay.height() < 95
        assert not ctrl.audio.mic_active

        # Home key toggle hides/shows IDLE cleanly
        overlay.toggle_user_visibility()
        assert not overlay.isVisible()
        overlay.toggle_user_visibility()
        assert overlay.isVisible()

        # 2. Enter IN_RUN: window 100% hidden (voice-only mid-run), mic strictly off
        await ctrl._on_run_started(RunStarted(t0=0.0))
        assert ctrl.state == "IN_RUN"
        assert not overlay.isVisible()
        assert not ctrl.audio.mic_active

        # 3. Early restart (RunAborted): returns to IDLE without opening DEBRIEF
        await ctrl._on_run_aborted(RunAborted(t0=0.0, t1=5.0, reason="shot_reset"))
        assert ctrl.state == "IDLE"
        assert overlay.isVisible()

        # 4. Completed run: opens DEBRIEF card and allows AI tool calls to switch visuals & set reminders
        await ctrl._on_run_completed(RunCompleted(t0=0.0, t1=float(run.t[-1]), stats=st))
        assert ctrl.state == "DEBRIEF"
        assert overlay.isVisible()
        res_vis = await ctrl._handle_tool_call("show_visual", {"view": "trend"})
        assert res_vis["shown"] == "trend"
        res_focus = await ctrl._handle_tool_call("set_next_run_focus", {"cue": "Brake earlier", "metric": "over_pct"})
        assert ctrl.pinned_metric == "over_pct"
        assert res_focus["pinned_cue"] == "Brake earlier"

        res_rem = await ctrl._handle_tool_call(
            "configure_in_run_cues",
            {"reminder_text": "Relax grip and breathe", "reminder_every_s": 15},
        )
        assert len(res_rem["reminders"]) == 1
        rem_cue = ctrl._cues.evaluate(run, run_elapsed_s=16.0)
        assert rem_cue is not None and "Relax grip" in rem_cue.hint

        # 5. End Convo button / F8 / end_debrief() collapses back to compact IDLE & shuts off mic
        await ctrl.end_debrief()
        assert ctrl.state == "IDLE"
        assert overlay.height() < 95
        assert not ctrl.audio.mic_active
        await ctrl.stop()
        overlay.close()

    asyncio.run(_exercise())
    assert app is not None
