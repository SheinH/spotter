"""Liquid Glass, ported to numpy from the open recreations of Apple's material.

Sources (none is Apple's own code; Apple has not published the shader):
- kube.io, "Liquid Glass in the browser" (2025): the bezel surface profiles, the convex
  squircle y = (1 - (1 - x)^4)^(1/4) that keeps refracted rays inside the glass, Snell's law
  with n = 1.5, one 1D ray computation per distance from the border turned into a 2D map
  along the border normal, and the -60 degree specular angle.
- Prismal TECHNICAL.md: Rec. 709 luma for vibrancy, Schlick Fresnel, and a rim highlight that
  is strongest on the border facing the light with a weaker echo on the opposite border.
- liquid-glass-react: a small red/blue split (chromatic aberration).

The material's colour response (luma curve + chroma gain, per appearance) is not from any of
those: it is measured from macOS's own NSGlassEffectView (regular style) over flat greys and
colours, captured with ScreenCaptureKit and fitted per knot.

The maps depend only on the card's size and are built once here, in numpy. The per-frame work
(frost, refraction lookup, tint, light) runs on the GPU in gpu_glass.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


# rows: adaptation level A (mean luma behind the card) = 0, 1/4 .. 1; columns: pixel luma
# L = 0, 1/8 .. 1. Measured from NSGlassEffectView in a non-activating panel (the state a
# HUD that never takes focus is drawn in; a key window's dark glass is a little lighter).
LUMA_LIGHT = (
    (0.5216, 0.5758, 0.6304, 0.6840, 0.7369, 0.7891, 0.8410, 0.8924, 0.9436),
    (0.5215, 0.5759, 0.6308, 0.6844, 0.7366, 0.7894, 0.8415, 0.8923, 0.9424),
    (0.5217, 0.5764, 0.6309, 0.6845, 0.7366, 0.7897, 0.8419, 0.8922, 0.9414),
    (0.5224, 0.5768, 0.6308, 0.6840, 0.7366, 0.7897, 0.8419, 0.8923, 0.9412),
    (0.5233, 0.5770, 0.6305, 0.6838, 0.7369, 0.7898, 0.8418, 0.8924, 0.9412),
)
CHROMA_LIGHT = (
    (0.955, 0.949, 0.944, 0.942, 0.941, 0.940, 0.939, 0.939, 0.939),
    (0.955, 0.947, 0.938, 0.941, 0.941, 0.940, 0.938, 0.938, 0.939),
    (0.954, 0.948, 0.944, 0.945, 0.941, 0.943, 0.933, 0.934, 0.940),
    (0.954, 0.950, 0.947, 0.944, 0.938, 0.929, 0.923, 0.935, 0.944),
    (0.955, 0.951, 0.947, 0.943, 0.937, 0.934, 0.934, 0.939, 0.944),
)
LUMA_DARK = (
    (0.0789, 0.1988, 0.2983, 0.3758, 0.4300, 0.4629, 0.4792, 0.4853, 0.4875),
    (0.0785, 0.1982, 0.2980, 0.3781, 0.4331, 0.4663, 0.4808, 0.4782, 0.4671),
    (0.0810, 0.1968, 0.2975, 0.3770, 0.4336, 0.4683, 0.4823, 0.4734, 0.4497),
    (0.0977, 0.2009, 0.2957, 0.3740, 0.4328, 0.4688, 0.4827, 0.4731, 0.4470),
    (0.1174, 0.2074, 0.2938, 0.3698, 0.4291, 0.4670, 0.4815, 0.4733, 0.4474),
)
CHROMA_DARK = (
    (1.370, 1.270, 1.161, 1.056, 0.960, 0.871, 0.786, 0.704, 0.623),
    (1.364, 1.260, 1.142, 1.049, 0.959, 0.870, 0.785, 0.702, 0.620),
    (1.349, 1.246, 1.148, 1.065, 0.963, 0.862, 0.782, 0.698, 0.614),
    (1.340, 1.243, 1.149, 1.056, 0.961, 0.869, 0.779, 0.686, 0.604),
    (1.336, 1.242, 1.148, 1.054, 0.961, 0.866, 0.772, 0.684, 0.602),
)


@dataclass(frozen=True)
class GlassParams:
    # Geometry, light and shadow below are fitted (CMA-ES, pixel error) to NSGlassEffectView
    # captures over ten test backdrops, light and dark, card 296 x 128, radius 22.
    bezel_px: float = 39.5        # width of the curved rim, logical px
    thickness_px: float = 8.0     # glass height at the top of the rim, logical px
    ior: float = 1.5              # kube.io: "prefer 1.5"
    max_shift_px: float = 23.0    # strongest refraction, logical px (where the rim bends most)
    blur_px: float = 15.5         # frost (Gaussian sigma), logical px
    # Apple's regular glass does not lay a tint over the frost: it remaps each pixel's luma L
    # through a curve and keeps its chroma (c - L), scaled by a gain that also depends on L:
    #     out = luma(L) + chroma(L) * (c - L)
    # Light glass lifts black to ~0.53 so dark text always reads; dark glass caps white at
    # ~0.48 (and even bends back down) so white text always reads. Both curves shift a little
    # with how bright the whole backdrop is (sRGB, Rec. 709 luma).
    luma_light: tuple[tuple[float, ...], ...] = LUMA_LIGHT
    chroma_light: tuple[tuple[float, ...], ...] = CHROMA_LIGHT
    luma_dark: tuple[tuple[float, ...], ...] = LUMA_DARK
    chroma_dark: tuple[tuple[float, ...], ...] = CHROMA_DARK
    chroma: float = 0.15          # red/blue split as a share of the shift
    light_deg: float = -24.0      # specular angle (screen: 0 = right, -90 = up)
    specular: float = 0.03        # rim highlight strength (Apple's is barely there)
    fresnel: float = 0.02         # edge brightening; keep low or every corner glows white
    rim_power: float = 8.0        # how tightly the highlight hugs the border facing the light
    rim_echo: float = 0.32        # the weaker highlight on the opposite border
    rim_line_px: float = 0.87     # width of the crisp edge line, logical px
    rim_broad: float = 0.065      # soft glow inside the lit border
    rim_broad_power: float = 8.0
    # drop shadow: a wide soft one plus a tight contact one right under the edge. Apple's glass
    # (in a panel that never takes focus) has only the tight one: a soft dark hairline.
    shadow_offset_px: float = 10.0
    shadow_blur_px: float = 14.0
    shadow_opacity: float = 0.0
    contact_offset_px: float = 0.0
    contact_blur_px: float = 0.7
    contact_opacity: float = 0.43


def profile(x: np.ndarray) -> np.ndarray:
    """Convex squircle surface: 0 at the outer edge, 1 where the flat top starts."""
    x = np.clip(x, 0.0, 1.0)
    return (1.0 - (1.0 - x) ** 4) ** 0.25


def snell_shift(n: int, p: GlassParams) -> tuple[np.ndarray, np.ndarray]:
    """1D: for each distance across the rim (0..1), how far a straight-down ray is pushed
    sideways after refracting at the curved top surface, and the cosine between the view and
    the surface normal (for Fresnel). Shift is normalized to max 1; the caller scales it."""
    x = np.linspace(0.0, 1.0, n)
    dx = 1e-3
    slope = (profile(x + dx) - profile(np.maximum(x - dx, 0.0))) / (x + dx - np.maximum(x - dx, 0.0))
    slope *= p.thickness_px / p.bezel_px               # height units -> same units as width
    theta1 = np.arctan(slope)                           # incidence angle vs the surface normal
    theta2 = np.arcsin(np.clip(np.sin(theta1) / p.ior, -1, 1))
    depth = p.thickness_px * profile(x) + 1.0           # glass the bent ray travels through
    shift = depth * np.tan(theta1 - theta2)
    shift[-1] = 0.0                                      # flat top: no bend
    peak = shift.max()
    return (shift / peak if peak > 0 else shift), np.cos(theta1)


@dataclass
class GlassMaps:
    alpha: np.ndarray     # (h, w) coverage of the card shape, antialiased
    sy: np.ndarray        # sampling offset, rows (device px)
    sx: np.ndarray        # sampling offset, cols (device px)
    rim: np.ndarray       # specular highlight 0..1
    fres: np.ndarray      # Fresnel 0..1


def build_maps(alpha: np.ndarray, scale: float, p: GlassParams = GlassParams()) -> GlassMaps:
    """alpha: the card's shape at device resolution (1 inside, 0 outside, soft edge).
    scale: device pixels per logical pixel."""
    # the card fills its canvas edge to edge: pad with "outside" so the straight sides have a
    # border to measure from (otherwise only the corners would bend anything)
    P = 4
    inside = np.pad(alpha > 0.5, P)
    dist = ndimage.distance_transform_edt(inside) - 0.5          # device px from the border
    bezel = p.bezel_px * scale
    x = np.clip(dist / bezel, 0.0, 1.0)

    # border normal: gradient of a smoothed distance field points inward
    g = ndimage.gaussian_filter(np.minimum(dist, bezel * 1.5), 1.2 * scale)
    gy, gx = np.gradient(g)
    norm = np.hypot(gx, gy) + 1e-9
    ny, nx = gy / norm, gx / norm                                  # inward unit normal
    crop = (slice(P, -P), slice(P, -P))
    dist, x, ny, nx = dist[crop], x[crop], ny[crop], nx[crop]

    table, cos_t = snell_shift(256, p)
    idx = np.clip((x * 255).round().astype(int), 0, 255)
    shift = table[idx] * p.max_shift_px * scale * (x < 1.0)
    # convex glass keeps rays inside: each rim pixel shows the backdrop from further in
    sy, sx = ny * shift, nx * shift

    # rim light: strongest on the border facing the light, a weaker echo opposite (Prismal)
    a = np.deg2rad(p.light_deg)
    ly, lx = np.sin(a), np.cos(a)
    facing = -(nx * lx + ny * ly)                                 # outward normal . light
    lit = np.clip(facing, 0, None) ** p.rim_power                  # border facing the light
    echo = np.clip(-facing, 0, None) ** p.rim_power                # the opposite border, faint
    line = np.exp(-np.maximum(dist, 0.0) / (p.rim_line_px * scale))   # crisp edge line
    broad = (1.0 - x) ** p.rim_broad_power * (x < 1.0)            # soft glow just inside it
    rim = np.clip(line * (0.9 * lit + p.rim_echo * echo) + broad * p.rim_broad * lit, 0.0, 1.0)

    r0 = ((1 - p.ior) / (1 + p.ior)) ** 2
    fres = (r0 + (1 - r0) * (1 - cos_t[idx]) ** 5) * (x < 1.0)
    return GlassMaps(alpha=alpha.astype(np.float32), sy=sy.astype(np.float32),
                     sx=sx.astype(np.float32), rim=rim.astype(np.float32),
                     fres=fres.astype(np.float32))


def looks_blank(backdrop: np.ndarray) -> bool:
    """Some games hand screen capture a black frame; then glass would just be dark grey."""
    return float(backdrop.std()) < 1.5 and float(backdrop.mean()) < 4.0


def shadow(alpha: np.ndarray, scale: float, p: GlassParams = GlassParams()) -> np.ndarray:
    """The glass's drop shadow: a wide, soft one shifted down plus a tight contact one hugging
    the edge, the way NSGlassEffectView floats. alpha is the card shape on the full (padded)
    canvas; the result is the black shadow's coverage 0..1."""
    a = alpha.astype(np.float32)

    def layer(offset_px: float, blur_px: float, opacity: float) -> np.ndarray:
        if opacity <= 0:
            return np.zeros_like(a)
        s = ndimage.shift(a, (offset_px * scale, 0), order=1, mode="constant")
        return ndimage.gaussian_filter(s, max(blur_px * scale, 0.1)) * opacity

    wide = layer(p.shadow_offset_px, p.shadow_blur_px, p.shadow_opacity)
    contact = layer(p.contact_offset_px, p.contact_blur_px, p.contact_opacity)
    return np.clip(1 - (1 - wide) * (1 - contact), 0, 1)


def backdrop_luma(rgb: np.ndarray) -> float:
    """Mean Rec. 709 luma (0..1) of what is behind the card; a coarse sample is plenty."""
    s = rgb[::4, ::4].astype(np.float32).mean(axis=(0, 1)) / 255.0   # no BLAS: no worker threads
    return float(0.2126 * s[0] + 0.7152 * s[1] + 0.0722 * s[2])


class Appearance:
    """Light or dark glass, from how bright the backdrop is, with hysteresis so a backdrop
    that sits near the middle does not make the card flicker between the two."""

    TO_LIGHT, TO_DARK = 0.62, 0.48

    def __init__(self) -> None:
        self.light = False

    def update(self, luma: float) -> bool:
        if self.light and luma < self.TO_DARK:
            self.light = False
        elif not self.light and luma > self.TO_LIGHT:
            self.light = True
        return self.light



def color_curves(light: float, adapt: float = 0.5,
                 p: GlassParams = GlassParams()) -> tuple[list[float], list[float]]:
    """The 9 luma and chroma knots for an appearance between dark (0) and light (1), over a
    backdrop whose mean luma is adapt (0..1)."""
    def at(table) -> list[float]:
        t = min(max(adapt, 0.0), 1.0) * (len(table) - 1)
        i = min(int(t), len(table) - 2)
        return [a + (b - a) * (t - i) for a, b in zip(table[i], table[i + 1])]
    mix = lambda a, b: [x + (y - x) * light for x, y in zip(a, b)]
    return mix(at(p.luma_dark), at(p.luma_light)), mix(at(p.chroma_dark), at(p.chroma_light))
