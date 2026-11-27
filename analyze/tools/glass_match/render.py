"""Headless recreation of spotter_volume's glass card (overlay.paintEvent minus the text), rendered
with the repo's own glass.py / gpu_glass.py, for comparing against the GlassRef captures."""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QRectF  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen  # noqa: E402

_app = QGuiApplication.instance() or QGuiApplication([])
from spotter_volume import glass, gpu_glass, overlay  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
S = float(os.environ.get("FIT_SCALE", "2"))
CW, CH, M = 296, 128, 64
W, H = int((CW + 2 * M) * S), int((CH + 2 * M) * S)
NAMES = ["black", "checker", "circles", "gradient", "grey", "grid", "hstripes", "text", "vstripes", "white"]


def load(path):
    a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    k = int(round(2 / S))                     # captures are 2x; box-average down to the fit scale
    return a.reshape(a.shape[0] // k, k, a.shape[1] // k, k, 3).mean((1, 3)) if k > 1 else a


def _qmask(draw) -> np.ndarray:
    img = QImage(W, H, QImage.Format.Format_Grayscale8)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(S, S)
    draw(p)
    p.end()
    buf = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes()).reshape(H, img.bytesPerLine())
    return buf[:, :W].astype(np.float32) / 255.0


CARD = QRectF(M, M, CW, CH)
FULL_ALPHA = _qmask(lambda p: p.fillPath(overlay.squircle(CARD, overlay.RADIUS), QColor(255, 255, 255)))
RIM_MASK = _qmask(lambda p: p.strokePath(overlay.squircle(CARD, overlay.RADIUS), QPen(QColor(255, 255, 255), 1.0)))
CARD_ALPHA = overlay._shape_alpha(QRectF(0, 0, CW, CH), int(CW * S), int(CH * S), S)
Y0, X0 = int(M * S), int(M * S)


class Renderer:
    """Renders with any GlassParams; extra knobs (shadow, rim stroke) passed as a dict."""

    def __init__(self):
        self._gpu = None
        self._key = None

    def render(self, bare: np.ndarray, p: glass.GlassParams, light: float, extra: dict) -> np.ndarray:
        key = (p.bezel_px, p.thickness_px, p.ior, p.max_shift_px, p.light_deg, p.rim_power, p.rim_echo,
               p.rim_line_px, p.rim_broad, p.rim_broad_power)
        if key != self._key:
            maps = glass.build_maps(CARD_ALPHA, S, p)
            if self._gpu is not None:
                self._gpu.release()
            self._gpu = gpu_glass.GpuGlass(maps, S, p)
            self._key = key
        g = self._gpu
        g.p = p
        g.set_backdrop((bare * 255 + 0.5).astype(np.uint8), (X0, Y0))
        adapt = float((bare[Y0:Y0 + CARD_ALPHA.shape[0], X0:X0 + CARD_ALPHA.shape[1]] @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean())
        card = g.compose(1.0, light, adapt).astype(np.float32) / 255.0
        out = bare.copy()
        sk = (p.shadow_offset_px, p.shadow_blur_px, p.shadow_opacity, p.contact_offset_px, p.contact_blur_px, p.contact_opacity)
        if getattr(self, "_sk", None) != sk:
            self._sk, self._sh = sk, glass.shadow(FULL_ALPHA, S, p)
        sh = self._sh
        out *= 1 - sh[..., None]
        ch, cw = card.shape[:2]
        reg = out[Y0:Y0 + ch, X0:X0 + cw]
        out[Y0:Y0 + ch, X0:X0 + cw] = card[..., :3] + reg * (1 - card[..., 3:4])
        ra = RIM_MASK[..., None] * extra.get("rim_a", overlay.GLASS_RIM.alphaF())
        out = out * (1 - ra) + ra
        return out


ZONE = np.zeros((H, W), bool)                 # the card plus 24 logical px around it
_m = int(24 * S)
ZONE[Y0 - _m:Y0 + int(CH * S) + _m, X0 - _m:X0 + int(CW * S) + _m] = True


def loss_one(a, b):
    """Mean abs error per channel (0..1), over the card and the band around it."""
    return float(np.abs(a[ZONE] - b[ZONE]).mean())


def dataset(appearance: str):
    d = f"{OUT}/cap_{appearance}"
    return [(n, load(f"{d}/{n}_bare.png"), load(f"{d}/{n}_glass.png")) for n in NAMES]


if __name__ == "__main__":
    r = Renderer()
    p = glass.GlassParams()
    for ap, light in (("light", 1.0), ("dark", 0.0)):
        tot = []
        rows = []
        for n, bare, ref in dataset(ap):
            out = r.render(bare, p, light, {})
            l = loss_one(out, ref)
            tot.append(l)
            print(f"{ap:5s} {n:9s} {l*255:6.2f}")
            if n in ("vstripes", "grid", "circles", "text"):
                rows.append(np.concatenate([ref, out], 1))
        print(ap, "mean", np.mean(tot) * 255)
        Image.fromarray((np.concatenate(rows, 0)[::2, ::2] * 255).astype(np.uint8)).save(f"{OUT}/sheet_{ap}_current.png")
