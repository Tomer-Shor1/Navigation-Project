#!/usr/bin/env bash
# Launch the interactive navigator.
#
#   ./run.sh
#
# Sets up the virtualenv on first use, then opens the entry window where you
# pick a flight video. Everything else -- decoding frames, building the map --
# happens when you press Start, into a temporary folder that is erased when you
# leave. Any arguments are passed through to nav_player.py.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
VENV=venv

if ! command -v ffmpeg > /dev/null; then
  echo "ffmpeg is not on your PATH -- it is what decodes the flight video."
  echo "  macOS:  brew install ffmpeg"
  echo "  Ubuntu: sudo apt install ffmpeg"
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "Setting up $VENV/ (first run only) ..."
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r requirements.txt
fi

# Confirm the dependencies are actually importable -- a half-built venv from an
# interrupted first run is otherwise a confusing traceback later.
if ! "$VENV/bin/python" -c "import cv2, numpy, matplotlib, utm" 2> /dev/null; then
  echo "Installing dependencies into $VENV/ ..."
  "$VENV/bin/pip" install --quiet -r requirements.txt
fi

exec "$VENV/bin/python" nav_player.py "$@"
