"""Test backdrops at 2x (device px), card 296x128 logical centred with a 64 px margin.

- scenes (stripes, grid, text, circles...): what the optimizer compares against
- f_*: flat greys and colours, s*_*: half/half grey splits: what fit_curves.py reads
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

S = 2
CW, CH, M = 296, 128, 64
W, H = (CW + 2 * M) * S, (CH + 2 * M) * S
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "backdrops")
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(3)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)


def save(name, a):
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(f"{out}/{name}.png")


cols = np.array([[230, 60, 60], [250, 200, 40], [60, 180, 90], [40, 120, 230], [150, 70, 200], [20, 20, 25], [245, 245, 245]])
save("vstripes", cols[((xx // 24) % len(cols)).astype(int)])
save("hstripes", cols[((yy // 18) % len(cols)).astype(int)])
chk = (((xx // 32) + (yy // 32)) % 2)[..., None]
save("checker", chk * 235 + (1 - chk) * 30 + np.zeros(3))
g = np.stack([xx / W * 255, yy / H * 255, (1 - xx / W) * 200 + 30], -1)
save("gradient", g)
save("grey", np.full((H, W, 3), 128.0))
save("white", np.full((H, W, 3), 250.0))
save("black", np.full((H, W, 3), 12.0))
# circles
img = Image.new("RGB", (W, H), (35, 40, 60))
d = ImageDraw.Draw(img)
for _ in range(40):
    r = rng.integers(10, 70); x, y = rng.integers(0, W), rng.integers(0, H)
    d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in rng.integers(0, 256, 3)))
img.save(f"{out}/circles.png")
# text
img = Image.new("RGB", (W, H), (250, 250, 250))
d = ImageDraw.Draw(img)
try:
    f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
except OSError:
    f = ImageFont.load_default()
for i, y in enumerate(range(6, H, 34)):
    d.text((8 - (i * 37) % 60, y), "The quick brown fox jumps over the lazy dog 0123456789 " * 2, fill=(20, 20, 20), font=f)
img.save(f"{out}/text.png")
# thin lines (refraction probe)
a = np.full((H, W, 3), 240.0)
a[(xx % 16) < 2] = 20; a[(yy % 16) < 2] = 20
save("grid", a)
# flats: greys every 32 levels + a few saturated colours (luma curve and chroma gain)
for v in list(range(0, 256, 32)) + [255]:
    save(f"f_g{v:03d}", np.full((H, W, 3), float(v)))
for k, c in {"red": (230, 40, 40), "green": (40, 200, 70), "blue": (30, 90, 230), "yellow": (250, 210, 40),
             "purple": (150, 60, 200), "teal": (20, 160, 170), "navy": (35, 40, 60), "pink": (250, 150, 190)}.items():
    save(f"f_c_{k}", np.zeros((H, W, 3)) + c)
# splits: left half a, right half b (pixel luma vs. the backdrop's mean luma)
lv = [0, 64, 128, 192, 255]
for a in lv:
    for b in lv:
        if a <= b:
            im = np.zeros((H, W, 3)); im[:, :W // 2] = a; im[:, W // 2:] = b
            save(f"s{a:03d}_{b:03d}", im)
print("backdrops ->", out)
