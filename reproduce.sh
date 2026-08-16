#!/usr/bin/env bash
# Regenerate every number in RESULTS.md.
#
#   ./reproduce.sh
#
# Needs: ffmpeg on the PATH, and each flight's video + matching .srt in data/raw/
# (they are too large to commit). Frames are extracted on the first run and
# cached in data/frames/, so later runs are much faster.
#
# If you only want to *read* the results, no data required:  python summarize.py
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}

# Reuse already-extracted frames when they exist, otherwise extract from video.
frame_flags () {
  if compgen -G "data/frames/$1/frame_*.jpg" > /dev/null; then
    echo "--use-cached-frames"
  fi
}

flights () {
  for srt in data/raw/*.srt; do
    [ -e "$srt" ] || continue
    basename "$srt" .srt
  done
}

found=0
echo "=============================================================="
echo " Main experiment: interleaved hold-out + self-calibration"
echo "=============================================================="
for stem in $(flights); do
  video="data/raw/$stem.mp4"
  flags=$(frame_flags "$stem")
  if [ -z "$flags" ] && [ ! -f "$video" ]; then
    echo; echo "--- $stem: skipped (no $video and no cached frames) ---"
    continue
  fi
  found=1
  echo; echo "--- $stem ---"
  $PY run_pipeline.py --video "$video" --srt "data/raw/$stem.srt" $flags
done

if [ "$found" -eq 0 ]; then
  echo
  echo "No flight data found in data/raw/. Put a flight video and its matching"
  echo ".srt there (e.g. data/raw/flight.mp4 + data/raw/flight.srt) and re-run."
  echo "To just read the committed results instead:  $PY summarize.py"
  exit 1
fi

# ---- ablation, on whichever flight is available ----------------------------
ABL=$(flights | head -1)
ABL_FLAGS=$(frame_flags "$ABL")
echo
echo "=============================================================="
echo " Ablation on $ABL (scratch output, not committed)"
echo "=============================================================="
run_ablation () {
  local label="$1"; shift
  echo; echo "--- $label ---"
  $PY run_pipeline.py --video "data/raw/$ABL.mp4" --srt "data/raw/$ABL.srt" \
      $ABL_FLAGS --results-dir results/_ablation "$@" | tail -6
}
run_ablation "interleaved hold-out + self-calibration (default)"
run_ablation "interleaved hold-out, modelled camera geometry" --no-calibrate
run_ablation "chronological tail split + self-calibration" --split tail
run_ablation "chronological tail split, modelled geometry (original MVP)" \
    --split tail --no-calibrate --exclusion-sec 0
rm -rf results/_ablation

echo
$PY summarize.py
