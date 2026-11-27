#!/bin/sh
# Capture macOS's real Liquid Glass and score spotter_volume's recreation against it.
#   ./run.sh            capture (light + dark), measure the colour curves, score the current code
#   ./run.sh fit [N]    then also run the optimizer for N generations (default 60; warms a laptop)
set -e
cd "$(dirname "$0")"
PY="uv run --no-project -q --with numpy --with scipy --with pillow --with moderngl --with PySide6-Essentials --with cma python -W ignore"

mkdir -p out
swiftc -O GlassRef.swift -o out/GlassRef
$PY gen_backdrops.py
rm -rf out/cap_light out/cap_dark
out/GlassRef out/backdrops out/cap_light light
out/GlassRef out/backdrops out/cap_dark dark
$PY fit_curves.py
$PY render.py
if [ "$1" = "fit" ]; then
    FIT_SCALE=1 WORKERS=${WORKERS:-3} $PY opt.py "" "${2:-60}"
fi
