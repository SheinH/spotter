"""Liquid Glass, ported to numpy from the open recreations of Apple's material.

Sources (none is Apple's own code; Apple has not published the shader):
- kube.io, "Liquid Glass in the browser" (2025): the bezel surface profiles, the convex
  squircle y = (1 - (1 - x)^4)^(1/4) that keeps refracted rays inside the glass, Snell's law
  with n = 1.5, one 1D ray computation per distance from the border turned into a 2D map
  along the border normal, and the -60 degree specular angle.
- Prismal TECHNICAL.md: Rec. 709 luma for vibrancy, Schlick Fresnel, and a rim highlight that
  is strongest on the border facing the light with a weaker echo on the opposite border.
- liquid-glass-react: 140% saturation and a small red/blue split (chromatic aberration).

The maps depend only on the card's size and are built once here, in numpy. The per-frame work
(frost, refraction lookup, tint, light) runs on the GPU in gpu_glass.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class GlassParams:
    bezel_px: float = 18.0        # width of the curved rim, logical px
    thickness_px: float = 26.0    # glass height at the top of the rim, logical px
    ior: float = 1.5              # kube.io: "prefer 1.5"
    max_shift_px: float = 16.0    # strongest refraction, logical px (where the rim bends most)
    blur_px: float = 7.0          # frost (Gaussian sigma), logical px
    saturation: float = 1.4       # liquid-glass-react default 140%
    # Apple's "regular" glass adapts to what is behind it: dark glass with white text over dark
    # content, light glass with dark text over bright content. The tint gets stronger the
    # closer the backdrop is to the text's own brightness, so labels always stay readable.
    tint_dark: tuple[int, int, int] = (18, 18, 20)
    tint_light: tuple[int, int, int] = (246, 246, 248)
    tint_alpha_min: float = 0.30
    tint_alpha_max: float = 0.62
    chroma: float = 0.10          # red/blue split as a share of the shift
    light_deg: float = -60.0      # kube.io specular angle (screen: 0 = right, -90 = up)
    specular: float = 0.55        # rim highlight strength
    fresnel: float = 0.06         # edge brightening; keep low or every corner glows white


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
    lit = np.clip(facing, 0, None) ** 1.6                          # border facing the light
    echo = np.clip(-facing, 0, None) ** 1.6                        # the opposite border, faint
    line = np.exp(-np.maximum(dist, 0.0) / (1.0 * scale))         # crisp edge line
    broad = (1.0 - x) ** 3 * (x < 1.0)                            # soft glow just inside it
    rim = np.clip(line * (0.9 * lit + 0.22 * echo) + broad * 0.25 * lit, 0.0, 1.0)

    r0 = ((1 - p.ior) / (1 + p.ior)) ** 2
    fres = (r0 + (1 - r0) * (1 - cos_t[idx]) ** 5) * (x < 1.0)
    return GlassMaps(alpha=alpha.astype(np.float32), sy=sy.astype(np.float32),
                     sx=sx.astype(np.float32), rim=rim.astype(np.float32),
                     fres=fres.astype(np.float32))


def looks_blank(backdrop: np.ndarray) -> bool:
    """Some games hand screen capture a black frame; then glass would just be dark grey."""
    return float(backdrop.std()) < 1.5 and float(backdrop.mean()) < 4.0


def shadow(alpha: np.ndarray, scale: float, offset_px: float = 10.0, blur_px: float = 14.0,
           opacity: float = 0.30) -> np.ndarray:
    """A real blurred drop shadow, the way iOS floats banners and popovers: wide, soft,
    shifted down, no visible edge. alpha is the card shape on the full (padded) canvas."""
    s = ndimage.shift(alpha.astype(np.float32), (offset_px * scale, 0), order=1, mode="constant")
    s = ndimage.gaussian_filter(s, blur_px * scale)
    return np.clip(s * opacity, 0, 1)


def backdrop_luma(rgb: np.ndarray) -> float:
    """Mean Rec. 709 luma (0..1) of what is behind the card; a coarse sample is plenty."""
    s = rgb[::4, ::4].astype(np.float32) / 255.0
    return float((s @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean())


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


def legibility_tint(luma: float, light: float, p: GlassParams = GlassParams()) -> float:
    """How strongly to tint the glass. White text (dark glass) needs more tint the brighter
    the backdrop is; dark text (light glass) the darker it is. light: 0 dark .. 1 light."""
    def need(closeness: float) -> float:
        t = min(max((closeness - 0.15) / 0.6, 0.0), 1.0)
        return p.tint_alpha_min + (p.tint_alpha_max - p.tint_alpha_min) * t * t * (3 - 2 * t)
    return need(luma) * (1 - light) + need(1 - luma) * light
