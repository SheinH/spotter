"""spotter_coach CLI entrypoint: real-time voice aim coach & PySide6 overlay.

    uv run --extra coach python -m spotter_coach
    uv run --extra coach python -m spotter_coach --replay ../runs/spotter-20261116-220738.spot --replay-speed 20
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
import warnings

from PySide6.QtWidgets import QApplication
import qasync

from spotter_analyze import kovaaks

from .controller import CoachController
from .overlay import CoachOverlay

REPO = Path(__file__).resolve().parents[2]
warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
warnings.filterwarnings("ignore", "nperseg", UserWarning)
warnings.filterwarnings("ignore", "No artists with labels", UserWarning)


def _load_dotenv(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'\"")
        if k and k not in os.environ:
            os.environ[k] = v


def main(argv: list[str] | None = None) -> int:
    _load_dotenv(REPO / ".env")
    _load_dotenv(REPO / "analyze" / ".env")
    ap = argparse.ArgumentParser(prog="spotter_coach", description=__doc__.splitlines()[0])
    ap.add_argument("--spot", type=Path, help="Pin a specific .spot file (default: newest in runs/)")
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    ap.add_argument("--stats-dir", type=Path, default=kovaaks.DEFAULT_ROOT / "stats")
    ap.add_argument("--out", type=Path, default=REPO / "reports")
    ap.add_argument("--api-key", help="Gemini API key (default: GEMINI_API_KEY or GOOGLE_API_KEY env var)")
    ap.add_argument("--model", help="Gemini Live model ID override (default: auto-selects 3.8-live / native-audio)")
    ap.add_argument("--voice", default="Sadachbia",
                    help="Pinned Gemini voice name (e.g. Sadachbia, Aoede, Kore, Puck, Fenrir, Charon; default: Sadachbia)")
    ap.add_argument("--no-audio", action="store_true", help="Disable audio devices (visual/test mode)")
    ap.add_argument("--replay", type=Path, metavar="SPOT",
                    help="Replay an existing .spot file + matching Stats.csv files for offline testing")
    ap.add_argument("--replay-speed", type=float, default=15.0,
                    help="Playback multiplier for --replay (default: 15x)")
    ap.add_argument("--debrief-latest", action="store_true",
                    help="Immediately debrief any run completed in the last 10 minutes of the current sitting")
    args = ap.parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    overlay = CoachOverlay()
    controller = CoachController(
        runs_dir=args.runs_dir,
        stats_dir=args.stats_dir,
        out_dir=args.out,
        overlay=overlay,
        loop=loop,
        spot_path=args.spot,
        api_key=args.api_key,
        model=args.model,
        voice=args.voice,
        enable_audio=not args.no_audio,
        replay_path=args.replay,
        replay_speed=args.replay_speed,
        debrief_latest=args.debrief_latest,
    )

    async def _shutdown():
        await controller.stop()
        overlay.close()
        app.quit()

    overlay.quit_requested.connect(lambda: loop.create_task(_shutdown()))
    loop.call_soon(controller.start)

    with loop:
        loop.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
