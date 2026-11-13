"""Synthetic runs with known, planted aim errors. Used by the tests and for trying the
analyzer without the game: `python -m spotter_analyze.synth tracking out.spot`."""
from __future__ import annotations

import sys

import numpy as np

from .format import BTN_FIRE, Run, write_spot

DIST = 1000.0  # target distance, world units


def _direction(yaw_deg, pitch_deg):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.stack([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)], axis=-1)


def _frame_times(rng, duration: float, fps: float) -> np.ndarray:
    dt = (1.0 / fps) * (1.0 + 0.05 * rng.standard_normal(int(duration * fps)))
    return np.cumsum(np.abs(dt))


def tracking(seed: int = 0, duration: float = 30.0, fps: float = 360.0, lag: float = 0.10,
             speed: float = 20.0, radius: float = 60.0, smooth_turns: bool = False,
             tremor_deg: float = 0.0, tremor_hz: float = 10.0, tremor_grow: float = 0.0) -> Run:
    """One target strafing left/right at `speed` deg/s, reversing at random. The crosshair
    replays the target's path `lag` seconds late. `smooth_turns` eases each reversal with a
    sinusoidal yaw path and a small vertical bob, like Smoothsphere's targets."""
    rng = np.random.default_rng(seed)
    t = _frame_times(rng, duration, fps)
    n = len(t)
    # target yaw as a piecewise-linear path
    knots_t, knots_y = [0.0], [0.0]
    direction = 1.0
    while knots_t[-1] < duration + 1:
        leg = rng.uniform(0.6, 1.5)
        knots_t.append(knots_t[-1] + leg)
        knots_y.append(knots_y[-1] + direction * speed * leg)
        direction = -direction
    def path_yaw(tt):
        if not smooth_turns:
            return np.interp(tt, knots_t, knots_y)
        i = np.clip(np.searchsorted(knots_t, tt) - 1, 0, len(knots_t) - 2)
        u = (tt - np.array(knots_t)[i]) / np.diff(knots_t)[i]
        y0, y1 = np.array(knots_y)[i], np.array(knots_y)[i + 1]
        return y0 + (y1 - y0) * (1 - np.cos(np.pi * u)) / 2

    def path_pitch(tt):
        return 3.0 * np.sin(2 * np.pi * 0.37 * tt) if smooth_turns else np.zeros_like(tt)

    tyaw, tpitch = path_yaw(t), path_pitch(t)
    # Hand tremor: a sinusoid of `tremor_deg` amplitude whose size grows by `tremor_grow`
    # (fraction) from the start to the end of the run, with a wandering phase.
    phase = 2 * np.pi * tremor_hz * t + np.cumsum(0.05 * rng.standard_normal(n))
    shake = tremor_deg * (1 + tremor_grow * t / duration) * np.sin(phase)
    yaw = path_yaw(t - lag) + shake + 0.02 * rng.standard_normal(n)
    pitch = path_pitch(t - lag) + 0.02 * rng.standard_normal(n)
    return Run(
        t=t, yaw=yaw, pitch=pitch, fov=np.full(n, 103.0), cam=np.zeros((n, 3)),
        shots=np.floor(t * 30).astype(np.int64), buttons=np.full(n, BTN_FIRE, dtype=np.uint8),
        tgt_frame=np.arange(n), tgt_id=np.full(n, 1, dtype=np.int64),
        tgt_pos=DIST * _direction(tyaw, tpitch), tgt_radius=np.full(n, radius),
        tgt_hh=np.full(n, radius),
        tgt_health=np.full(n, 1000.0), start_wall_us=1_794_460_000_000_000, game="synthetic",
    )


def _min_jerk(u):
    u = np.clip(u, 0.0, 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def clicking(seed: int = 0, kills: int = 40, fps: float = 360.0, react: float = 0.18,
             reach: float = 0.85, radius: float = 25.0, early_every: int = 5,
             lockout_deg: float = 0.0) -> Run:
    """One static target at a time. Each kill: wait `react`, flick `reach` of the way
    (0.85 = 15% short), pause, correct onto the centre, confirm 50 ms, shoot.
    Every `early_every`-th kill also fires once at the end of the short first flick.
    `lockout_deg`: after each flick stops, a 10 Hz shake that decays over ~80 ms, with this
    amplitude per 30 deg of flick (bigger flicks, bigger shake)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    cur = np.array([0.0, 0.0])        # crosshair yaw, pitch
    t_now, shots = 0.0, 0
    T, YAW, PITCH, SHOTS, BTN, TGT = [], [], [], [], [], []
    stops: list[tuple[float, float]] = []

    def emit(pos, target, fire=False):
        nonlocal t_now, shots
        t_now += dt * (1.0 + 0.05 * rng.standard_normal())
        if fire:
            shots += 1
        T.append(t_now); YAW.append(pos[0]); PITCH.append(pos[1])
        SHOTS.append(shots); BTN.append(BTN_FIRE if fire else 0); TGT.append(target)

    for k in range(kills):
        while True:
            tgt = np.array([rng.uniform(-30, 30), rng.uniform(-15, 15)])
            if np.hypot(*(tgt - cur)) >= 12.0:
                break
        target = (k + 1, tgt)
        start = cur.copy()
        landing = start + reach * (tgt - start)
        dist = np.hypot(*(tgt - start))
        move = 0.10 + 0.004 * dist
        for _ in range(int(react * fps)):
            emit(start, target)
        steps = int(move * fps)
        for i in range(1, steps + 1):
            emit(start + _min_jerk(i / steps) * (landing - start), target)
        stops.append((t_now, dist))
        if early_every and k % early_every == early_every - 1:
            emit(landing, target, fire=True)
        for _ in range(int(0.06 * fps)):
            emit(landing, target)
        steps = int(0.08 * fps)
        for i in range(1, steps + 1):
            emit(landing + _min_jerk(i / steps) * (tgt - landing), target)
        for _ in range(int(0.05 * fps)):
            emit(tgt, target)
        cur = tgt
        # The killing frame: shot counter ticks and the target is already gone;
        # the next target shows up from the following frame.
        emit(tgt, None, fire=True)

    n = len(T)
    if lockout_deg:
        tt = np.array(T)
        shake = np.zeros(n)
        for ts, dist in stops:
            u = tt - ts
            m = (u > 0) & (u < 0.4)
            shake[m] += lockout_deg * dist / 30 * np.exp(-u[m] / 0.08) * np.sin(2 * np.pi * 10 * u[m])
        YAW = list(np.array(YAW) + shake)
    rows = [(i, tg[0], tg[1]) for i, tg in enumerate(TGT) if tg is not None]
    angles = np.array([r[2] for r in rows])
    return Run(
        t=np.array(T), yaw=np.array(YAW), pitch=np.array(PITCH), fov=np.full(n, 103.0),
        cam=np.zeros((n, 3)), shots=np.array(SHOTS, dtype=np.int64),
        buttons=np.array(BTN, dtype=np.uint8),
        tgt_frame=np.array([r[0] for r in rows], dtype=np.int64),
        tgt_id=np.array([r[1] for r in rows], dtype=np.int64),
        tgt_pos=DIST * _direction(angles[:, 0], angles[:, 1]),
        tgt_radius=np.full(len(rows), radius), tgt_hh=np.full(len(rows), radius), tgt_health=np.full(len(rows), 1.0),
        start_wall_us=1_794_460_000_000_000, game="synthetic",
    )


if __name__ == "__main__":
    kind, out = sys.argv[1], sys.argv[2]
    write_spot(out, {"tracking": tracking, "clicking": clicking}[kind]())
    print(f"wrote {out}")
