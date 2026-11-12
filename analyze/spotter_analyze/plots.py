"""The two images that go with a report. Static PNGs, light surface."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3, fixed order
SEQ = LinearSegmentedColormap.from_list("seq_blue", [SURFACE, "#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"])
CLIP = 4.0  # radii shown either side of the target centre


def _axes(title: str, subtitle: str, square: bool = False):
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(6.4, 6.4) if square else (7.6, 4.6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.suptitle(title, x=0.02, y=0.98, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.02, 0.925, subtitle, ha="left", va="top", fontsize=9, color=INK2)
    return fig, ax


def _save(fig, path: Path, top: float = 0.87):
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _target_frame(ax):
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, edgecolor=INK, linewidth=1.4, zorder=5))
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_xlim(-CLIP, CLIP)
    ax.set_ylim(-CLIP, CLIP)
    ax.set_aspect("equal")
    ax.set_xlabel("crosshair left / right of target centre (target radii)", color=INK2)
    ax.set_ylabel("crosshair below / above (target radii)", color=INK2)


def error_scatter(result: dict, path: Path) -> None:
    p = result["plot"]
    if result["mode"] == "tracking":
        x, y = p["px"], p["py"]
        inside = (np.abs(x) < CLIP) & (np.abs(y) < CLIP)
        fig, ax = _axes("Where the crosshair sat relative to the target",
                        f"Darker = more time. Circle = target edge. {100 * (1 - inside.mean()):.0f}% of time "
                        f"was further than {CLIP:.0f} radii out and is not shown.", square=True)
        ax.hist2d(x[inside], y[inside], bins=64, range=[[-CLIP, CLIP], [-CLIP, CLIP]], cmap=SEQ, zorder=2)
        _target_frame(ax)
    else:
        s = p["scatter"]
        if s is None or len(s["x"]) == 0:
            x = np.array([r["land_r"] for r in result["table"]], dtype=float)
            fig, ax = _axes("Where the first movement ended along the flick line",
                            "Target radii; negative = short of the target, positive = past it. "
                            "Band = on target.")
            ax.axvspan(-1, 1, color="#cde2fb", zorder=1)
            ax.hist(x[~np.isnan(x)], bins=np.linspace(-CLIP, CLIP, 41), color=SERIES[0], zorder=3,
                    edgecolor=SURFACE, linewidth=0.8)
            ax.set_xlabel("landing point along the flick (target radii)", color=INK2)
            ax.set_ylabel("kills", color=INK2)
        else:
            out = (np.abs(s["x"]) >= CLIP) | (np.abs(s["y"]) >= CLIP)
            fig, ax = _axes("Where each shot went relative to its target",
                            f"Circle = target edge. {int(out.sum())} shots further than {CLIP:.0f} radii out "
                            f"are drawn at the border.", square=True)
            x, y = np.clip(s["x"], -CLIP + 0.05, CLIP - 0.05), np.clip(s["y"], -CLIP + 0.05, CLIP - 0.05)
            for mask, color, marker, label in [(s["hit"], SERIES[0], "o", "hit"), (~s["hit"], SERIES[1], "X", "miss")]:
                ax.scatter(x[mask], y[mask], s=34, color=color, marker=marker, edgecolors=SURFACE,
                           linewidths=0.8, label=f"{label} ({int(mask.sum())})", zorder=4)
            _target_frame(ax)
            ax.legend(loc="upper right", frameon=False, labelcolor=INK2)
    _save(fig, path, top=0.9)


def velocity_profiles(result: dict, path: Path) -> None:
    p = result["plot"]
    if result["mode"] == "tracking":
        fig, ax = _axes("Crosshair speed through target direction changes",
                        f"Average of {len(p['prof_c'])} reversals, measured along the target's new direction. "
                        "0 ms = target turns.")
        if len(p["prof_c"]):
            t = p["prof_t"]
            for line in p["prof_c"][:60]:
                ax.plot(t, line, color=SERIES[0], linewidth=0.6, alpha=0.12, zorder=2)
            ax.plot(t, p["prof_tg"].mean(axis=0), color=INK2, linewidth=2, linestyle=(0, (4, 2)), label="target", zorder=3)
            ax.plot(t, p["prof_c"].mean(axis=0), color=SERIES[0], linewidth=2, label="crosshair", zorder=4)
            ax.axvline(0, color=AXIS, linewidth=0.8)
            ax.legend(loc="lower right", frameon=False, labelcolor=INK2)
        ax.set_xlabel("time from target reversal (ms)", color=INK2)
        ax.set_ylabel("speed along new direction (deg/s)", color=INK2)
    else:
        fig, ax = _axes("Flick speed profiles by distance",
                        "Bold = median flick in each distance band, from when the hand starts moving. "
                        "Extra bumps after the main peak are corrections.")
        grid = np.arange(0, 500, 5.0)
        for (name, members), color in zip(p["buckets"], SERIES):
            curves = []
            for i in members:
                tr = p["traces"][i]
                if np.isnan(tr["onset_ms"]) or len(tr["tt"]) < 3:
                    continue
                curves.append(np.interp(grid, tr["tt"] - tr["onset_ms"], tr["sp"], left=np.nan, right=0.0))
            if not curves:
                continue
            curves = np.array(curves)
            for line in curves[:25]:
                ax.plot(grid, line, color=color, linewidth=0.6, alpha=0.12, zorder=2)
            ax.plot(grid, np.nanmedian(curves, axis=0), color=color, linewidth=2, label=f"{name} ({len(curves)})", zorder=4)
        ax.legend(loc="upper right", frameon=False, labelcolor=INK2)
        ax.set_xlabel("time from movement start (ms)", color=INK2)
        ax.set_ylabel("crosshair speed (deg/s)", color=INK2)
        ax.set_xlim(0, 500)
        ax.set_ylim(bottom=0)
    _save(fig, path)
