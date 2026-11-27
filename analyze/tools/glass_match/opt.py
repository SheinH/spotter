"""CMA-ES fit of GlassParams (geometry, frost, rim light, shadow) to the GlassRef captures.

Starts from the repo's current GlassParams defaults, so re-running continues from where the
code is. Colour is not fitted here (fit_curves.py measures it directly).

    FIT_SCALE=1 WORKERS=3 python opt.py [start.json] [generations]

FIT_SCALE=1 renders at 1x (4x cheaper, ranks candidates the same as 2x); WORKERS sets the
process pool (each has its own GL context; keep it low on a laptop).
Writes out/best.json and out/sheet_{light,dark}_fit.png (real left, recreation right).
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace

import cma
import numpy as np
from PIL import Image

import render as R
from spotter_volume import glass

# name, lo, hi, start
SPACE = [
    ("bezel_px", 2, 64, 18), ("thickness_px", 2, 60, 26), ("max_shift_px", 0, 30, 16),
    ("blur_px", 1, 30, 7), ("chroma", 0, 0.4, 0.1),
    ("light_deg", -180, 0, -60), ("specular", 0, 1.5, 0.55), ("fresnel", 0, 1.0, 0.06),
    ("rim_power", 0.5, 12, 1.6), ("rim_echo", 0, 1, 0.22), ("rim_line_px", 0.3, 4, 1.0),
    ("rim_broad", 0, 1, 0.25), ("rim_broad_power", 0.5, 8, 3.0),
    ("rim_a", 0, 0.6, 0.18),
    ("shadow_offset_px", 0, 20, 10), ("shadow_blur_px", 2, 30, 14), ("shadow_opacity", 0, 0.6, 0.16),
    ("contact_offset_px", 0, 4, 1), ("contact_blur_px", 0.3, 6, 2), ("contact_opacity", 0, 0.6, 0.0),
]
GLASS_FIELDS = {f for f in glass.GlassParams.__dataclass_fields__}

Z = R.ZONE                                   # card + 24 logical px around it

DATA = {ap: R.dataset(ap) for ap in ("light", "dark")}


def decode(u):
    v = {}
    for (name, lo, hi, _), x in zip(SPACE, u):
        v[name] = lo + (hi - lo) * float(np.clip(x, 0, 1))
    return v


def encode(v):
    return [(v[name] - lo) / (hi - lo) for name, lo, hi, _ in SPACE]


def to_params(v):
    p = glass.GlassParams(**{k: v[k] for k in v if k in GLASS_FIELDS})
    return p


RND = R.Renderer()


def evaluate(v, keep=False):
    p = to_params(v)
    errs, outs = {}, {}
    for ap, light in (("light", 1.0), ("dark", 0.0)):
        extra = dict(rim_a=v["rim_a"])
        for n, bare, ref in DATA[ap]:
            out = RND.render(bare, p, light, extra)
            errs[(ap, n)] = float(np.abs(out[Z] - ref[Z]).mean()) * 255
            if keep:
                outs[(ap, n)] = out
    return float(np.mean(list(errs.values()))), errs, outs


def sheet(v, tag):
    _, errs, outs = evaluate(v, keep=True)
    for ap in ("light", "dark"):
        rows = []
        for n, bare, ref in DATA[ap]:
            if n in ("vstripes", "grid", "circles", "text"):
                rows.append(np.concatenate([ref, outs[(ap, n)]], 1)[R.Y0 - int(32 * R.S):R.Y0 + int(160 * R.S)])
        Image.fromarray((np.clip(np.concatenate(rows, 0), 0, 1)[::2, ::2] * 255).astype(np.uint8)).save(
            f"{R.OUT}/sheet_{ap}_{tag}.png")
    return errs


def _f(x):
    return evaluate(decode(x))[0]


if __name__ == "__main__":
    import multiprocessing as mp
    POOL = mp.get_context("spawn").Pool(int(os.environ.get("WORKERS", "3")))
    from spotter_volume import overlay
    d = glass.GlassParams()
    start = {n: (getattr(d, n) if n in GLASS_FIELDS else overlay.GLASS_RIM.alphaF()) for n, *_ in SPACE}
    start = {n: min(max(start[n], lo), hi) for n, lo, hi, _ in SPACE}
    if len(sys.argv) > 1 and sys.argv[1]:
        start.update({k: x for k, x in json.load(open(sys.argv[1])).items() if k in start})
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    l0, _, _ = evaluate(start)
    print("start loss", round(l0, 3), flush=True)
    es = cma.CMAEvolutionStrategy(encode(start), 0.15, {"bounds": [0, 1], "popsize": 21, "seed": 1, "verbose": -9})
    t = time.time()
    best = (l0, start)
    for it in range(iters):
        xs = es.ask()
        fs = POOL.map(_f, xs)
        es.tell(xs, fs)
        i = int(np.argmin(fs))
        if fs[i] < best[0]:
            best = (fs[i], decode(xs[i]))
        print(f"it {it:3d} best {best[0]:.3f}  ({time.time()-t:.0f}s)", flush=True)
        json.dump(best[1], open(f"{R.OUT}/best.json", "w"), indent=1)
    errs = sheet(best[1], "fit")
    print(json.dumps({k: round(v, 3) for k, v in best[1].items()}))
    for k, e in sorted(errs.items()):
        print(k, round(e, 2))
