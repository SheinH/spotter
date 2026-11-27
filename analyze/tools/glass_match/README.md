# glass_match

Measures how close `spotter_volume`'s Liquid Glass is to Apple's, and fits it.

macOS 26+ only. `GlassRef.swift` places a real `NSGlassEffectView` (296 × 128 pt, radius 22,
in a non-activating floating panel like the overlay card) over a set of test backdrops and
captures it at 2x with ScreenCaptureKit. `render.py` draws the same card headlessly with the
repo's own `glass.py` / `gpu_glass.py`, so the two can be compared pixel for pixel.

```sh
./run.sh            # capture light + dark, measure colour curves, score the current code
./run.sh fit 60     # ...and run the optimizer for 60 generations
```

The app that launches it needs Screen Recording permission. It never takes focus.

| file | does |
|---|---|
| `GlassRef.swift` | real glass + bare backdrop captures (`ACTIVE=1`: key-window state instead) |
| `gen_backdrops.py` | scenes, flat greys/colours (`f_*`), half/half splits (`s*_*`) |
| `fit_curves.py` | colour response, `out = luma(L, A) + chroma(L, A)·(c − L)`; prints the `LUMA_*`/`CHROMA_*` tables for `glass.py` |
| `render.py` | headless recreation; run alone it prints per-backdrop error and writes `out/sheet_*_current.png` |
| `opt.py` | CMA-ES over geometry, frost, rim light and shadow, starting from the current `GlassParams`; writes `out/best.json` |

Error is mean absolute difference per channel (0–255) over the card plus a 24 pt band
around it. At the time of fitting (macOS 27.2) the code scored about 1.0.

What the captures showed:
- The colour is not a tint: luma goes through a curve, chroma is mostly kept. Light glass
  lifts black to ~0.53; dark glass caps white near 0.45.
- No temporal adaptation: the glass is final within a frame.
- Dark glass is lighter in a focused key window (white 121, black 32) than in a panel that
  never takes focus (114, 20). The overlay never takes focus, so the fit uses the latter.
