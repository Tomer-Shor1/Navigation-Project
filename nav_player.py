"""Watch the GPS-denied navigator work, one frame at a time.

    python nav_player.py

That opens an entry window listing the flight videos under `data/`, with a
Browse button for anything else on your computer. Pick one, press Start, and the
app decodes it, builds the map and opens the player.

Everything it needs is built when you press Start and erased when you leave:
frames go into a temporary folder, not `data/frames/`, so the app opens any video
you point it at and never shows you a stale run left over from a previous
session.

To skip the entry window and go straight to a flight -- handy for a demo, and the
way to reach the options the window does not expose -- name the flight on the
command line:

    python nav_player.py --video data/raw/flight.mp4 --srt data/raw/flight.srt
    python nav_player.py --video data/raw/flight.mp4 --srt data/raw/flight.srt \
        --map-source ortho --use-cached-frames

It accepts every flag `run_pipeline.py` accepts (they share `add_pipeline_args`),
so `--map-source`, `--split`, `--no-motion` and the rest mean exactly the same
thing here, which makes it an ablation viewer as well as a demo. Note that
`--use-cached-frames` is the one way to make it reuse `data/frames/` -- and then
you are responsible for those frames matching the video.

Nothing in the player re-implements the algorithm: `src/trace.py` drives
`localize_stream`, the same generator `localize_all` collects, so what you watch
is what `run_pipeline.py` measures. `tools/check_player.py` checks that.
"""

from __future__ import annotations

import argparse
import os
import sys
import tkinter as tk

from src.session import SessionInputError, add_pipeline_args, prepare_session
from src.trace import NavigationTrace
from src.workspace import Workspace, clean_up_on_termination
from ui.launcher import NavApp, WINDOW_SIZE
from ui.player import NavPlayer


def use_repo_as_working_directory() -> None:
    """Make `data/` and `results/` resolve however the app was launched.

    Every path in the project is relative to the repository root, which holds as
    long as you run it from a shell that is already there. Double-clicking the
    script, or running it by absolute path, is not that -- so if the current
    directory has no `data/` but the script's own directory does, work from
    there instead.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir("data") and os.path.isdir(os.path.join(here, "data")):
        os.chdir(here)


def names_a_flight(argv: list[str]) -> bool:
    """Did the caller say which flight to open? Both `--video x` and `--video=x`."""
    return any(arg.split("=")[0] in ("--video", "--srt") for arg in argv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive player for the GPS-denied visual navigation pipeline",
        epilog="Run with no arguments to choose a flight from a window instead.")
    add_pipeline_args(parser)
    parser.add_argument("--data-root", default="data",
                        help="Folder the entry window scans for flight videos (default: data)")
    parser.add_argument("--start", type=int, default=0,
                        help="Test frame index to open on (default: 0)")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Initial playback rate (default: 2)")
    return parser.parse_args()


def run_directly(args: argparse.Namespace) -> None:
    """Skip the entry window: prepare the named flight and open the player.

    Frames still go to a temporary folder unless `--frames-dir` or
    `--use-cached-frames` says otherwise, so the command-line path leaves no
    litter either.
    """
    workspace = None
    if args.frames_dir is None and not args.use_cached_frames:
        workspace = Workspace()
        args.frames_dir = workspace.frames_dir

    try:
        print("Preparing the map (the same preprocessing run_pipeline.py does) ...")
        session = prepare_session(args)
        print(f"      -> {len(session.test_frames)} test frames to navigate.\n")
        print("Opening the player. Space = play/pause, arrows = step, shift+arrows = jump 5.")

        root = tk.Tk()
        root.geometry(WINDOW_SIZE)
        player = NavPlayer(root, session, NavigationTrace(session),
                           start=args.start, fps=args.fps)

        def on_close():
            player.playing = False
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)
        root.mainloop()
    finally:
        if workspace is not None:
            workspace.close()


def main() -> None:
    use_repo_as_working_directory()
    clean_up_on_termination()
    args = parse_args()

    # No flight named -> let the user choose one in the window.
    if not names_a_flight(sys.argv[1:]):
        root = tk.Tk()
        app = NavApp(root, data_root=args.data_root)
        try:
            root.mainloop()
        finally:
            app.release()
        return

    try:
        run_directly(args)
    except SessionInputError as exc:
        print("=" * 70, file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
