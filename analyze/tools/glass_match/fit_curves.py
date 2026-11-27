"""Measure the glass's colour response from the flat (f_*) and split (s*) captures:
    out = f(L, A) + g(L, A) * (c - L)   per appearance
L: the frosted pixel's luma; A: adaptation level = mean luma of the backdrop under the card.
f, g bilinear on 9 L-knots x 5 A-knots (sRGB 0..1, Rec. 709 luma). Prints the tables as the
LUMA_*/CHROMA_* constants for spotter_volume/glass.py and writes out/curves.json."""
import glob
import json
import os

import numpy as np
from PIL import Image
from scipy.optimize import least_squares

LK = np.linspace(0, 1, 9)
AK = np.linspace(0, 1, 5)
LW = np.array([0.2126, 0.7152, 0.0722])
X0, Y0, CW2, CH2 = 128, 128, 592, 256
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load(f):
    return np.asarray(Image.open(f).convert("RGB"), np.float64) / 255


def samples(ap):
    rows = []  # (c_in rgb, A, c_out rgb, name)
    for f in sorted(glob.glob(f"{OUT}/cap_{ap}/*_bare.png")):
        n = os.path.basename(f)[:-9]
        if not (n.startswith("f_") or n.startswith("s")):
            continue
        b, g = load(f), load(f.replace("_bare", "_glass"))
        A = float((b[Y0:Y0 + CH2, X0:X0 + CW2] @ LW).mean())
        cy = Y0 + CH2 // 2
        xs = [X0 + CW2 // 2] if n.startswith("f_") else [X0 + 116, X0 + CW2 - 116]
        for x in xs:
            rows.append((b[cy, x], A, g[cy - 20:cy + 20, x - 20:x + 20].mean((0, 1)), n))
    return rows


def bilerp(K, L, A):
    li = np.clip(L * 8, 0, 8 - 1e-9); i = li.astype(int); u = li - i
    ai = np.clip(A * 4, 0, 4 - 1e-9); j = ai.astype(int); v = ai - j
    return ((1 - u) * (1 - v) * K[j, i] + u * (1 - v) * K[j, i + 1]
            + (1 - u) * v * K[j + 1, i] + u * v * K[j + 1, i + 1])


def model(F, G, C, A):
    L = C @ LW
    return np.clip(bilerp(F, L, A)[:, None] + bilerp(G, L, A)[:, None] * (C - L[:, None]), 0, 1)


out = {}
for ap in ("light", "dark"):
    rows = samples(ap)
    C = np.array([r[0] for r in rows]); A = np.array([r[1] for r in rows]); Y = np.array([r[2] for r in rows])

    def unpack(v):
        return v[:45].reshape(5, 9), v[45:].reshape(5, 9)

    def res(v):
        F, G = unpack(v)
        r = (model(F, G, C, A) - Y).ravel() * 255
        reg = [0.5 * np.diff(F, 2, axis=1).ravel() * 255, 0.5 * np.diff(F, 2, axis=0).ravel() * 255,
               0.5 * np.diff(F, 1, axis=0).ravel() * 255 * 0.2,
               3 * np.diff(G, 2, axis=1).ravel(), 3 * np.diff(G, 1, axis=0).ravel(), 0.3 * (G - 1).ravel()]
        return np.concatenate([r] + reg)
    v0 = np.concatenate([np.tile(0.3 + 0.4 * LK, 5), np.ones(45)])
    sol = least_squares(res, v0)
    F, G = unpack(sol.x)
    err = np.abs(model(F, G, C, A) - Y) * 255
    print(ap, "n", len(rows), "mean abs err", err.mean().round(2), "max", err.max().round(1))
    for (c, a, y, n), e in zip(rows, err.max(1)):
        if e > 3:
            print("   ", n, "in", (c * 255).round(), "A", round(a, 2), "out", (y * 255).round(), "err", e.round(1))
    print("  F (rows A=0..1, cols L=0..1) x255:\n", (F * 255).round().astype(int))
    print("  G:\n", G.round(2))
    out[ap] = {"luma": [[round(x, 4) for x in r] for r in F], "chroma": [[round(x, 3) for x in r] for r in G]}
json.dump(out, open(f"{OUT}/curves.json", "w"), indent=1)


def table(name, t, nd):
    return f"{name} = (\n" + "".join("    (" + ", ".join(f"{x:.{nd}f}" for x in r) + "),\n" for r in t) + ")"


print("\n# paste into spotter_volume/glass.py")
for ap in ("light", "dark"):
    print(table(f"LUMA_{ap.upper()}", out[ap]["luma"], 4))
    print(table(f"CHROMA_{ap.upper()}", out[ap]["chroma"], 3))
