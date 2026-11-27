"""User settings, kept in spotter-volume.json at the repo root so they are easy to edit by hand.

To go back to Segoe UI, set "font" to "Segoe UI". A font that is not installed falls back to
"fallback_font" automatically.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATH = REPO / "spotter-volume.json"


@dataclass
class Config:
    daily_target_min: float = 30.0
    font: str = "SF Pro Text"
    fallback_font: str = "Segoe UI"
    corner: str = "top-right"          # top-right | top-left | bottom-right | bottom-left
    margin_px: int = 20
    hotkey: str = "Home"               # show/hide by hand; "" to disable
    style: str = "solid"               # "solid" | "glass" (live Liquid Glass on the GPU; the
                                       # card is then left out of screenshots and recordings)

    @classmethod
    def load(cls, path: Path = PATH) -> "Config":
        """Read the file, filling in anything missing; writes a fresh file if there is none."""
        known = {f.name for f in fields(cls)}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            cfg = cls(**{k: v for k, v in raw.items() if k in known})
            if known - raw.keys():
                cfg.save(path)          # add settings that are new since the file was written
        except FileNotFoundError:
            cfg = cls()
            cfg.save(path)
        except (OSError, ValueError, TypeError):
            cfg = cls()                 # a broken file never stops the overlay
        if not cfg.daily_target_min or cfg.daily_target_min <= 0:
            cfg.daily_target_min = cls.daily_target_min
        return cfg

    def save(self, path: Path = PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
