# Matching spotter_volume's Liquid Glass to Apple's (2026-11-27)

The volume overlay's glass (`analyze/spotter_volume/glass.py`, `gpu_glass.py`) was ported from
open recreations (kube.io, Prismal, liquid-glass-react). It was then measured against macOS's
real `NSGlassEffectView` and refitted. It now matches to about 1/255 mean error per channel
over the card, down from 5.5 (light) and 12.5 (dark).

## How

`analyze/tools/glass_match/` (see its README). A Swift tool captures real glass over test
backdrops through ScreenCaptureKit. A headless renderer draws the repo's own glass over the
same backdrops. The loss is the pixel difference between the two, so tuning is numeric rather
than eyeballed. `./run.sh` captures and scores in about 40 s. `./run.sh fit N` also runs CMA-ES.

## What the real glass does

- **The colour is not a tint.** Regular glass maps each pixel's luma through a curve and keeps
  its chroma: `out = luma(L, A) + chroma(L, A)·(c − L)`. Light glass lifts black to ~0.53.
  Dark glass caps white near 0.45, and the curve bends back down above mid-grey. That is what
  keeps text readable, so the old `saturation` / `tint_*` / `legibility_tint` model was
  replaced by measured tables (`LUMA_*`, `CHROMA_*`). `A` is the backdrop's mean luma and
  matters only slightly, in dark mode.
- **No temporal adaptation.** The glass is final within a frame. An early "it adapts over
  ~4 s" reading was wrong: those captures mixed two focus states.
- **Focus changes dark glass.** In a key window of the active app, dark glass is lighter
  (white→121, black→32) than in a non-activating panel (114, 20). The overlay is
  `Tool | WindowDoesNotAcceptFocus`, i.e. a panel, so the fit uses the panel state. iOS has
  no such state and was not measured.
- **Fitted look.** Frost σ ≈ 15.5 pt. The bezel is wide and shallow (39.5 pt, thickness 8).
  The rim highlight is barely there, and there is no white stroke. The shadow is a soft dark
  hairline (contact shadow, σ 0.7 pt, 0.43), with no wide drop shadow.

## Open

- Not yet run on Windows (the shader is plain GLSL 3.30).
- The largest remaining error is on the stripe backdrops, from edge/refraction detail.
- Keep optimizer runs light (1x scale, ≤3 workers). 7 workers overheated the laptop.
