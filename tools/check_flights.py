"""Drive the real app over every flight in data/, exactly as a user would.

    python tools/check_flights.py                 # quick: first 45 s of each flight
    python tools/check_flights.py --full          # every flight end to end
    python tools/check_flights.py --seconds 90

For each flight found under `data/` this selects it in the entry window, presses
Start, waits for preprocessing, steps through the player, switches to live
playback and runs it, then returns to the chooser -- checking at each stage that
the window is in the state it should be. Every map source the flight can offer is
exercised, so a basemap that exists gets the GIS and hybrid paths too.

It opens real Tk windows (briefly). The point is to catch the things a unit test
cannot: a view that fails to build, a map source that errors only on one flight,
a video whose frame count or telemetry breaks an assumption.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ui.launcher as launcher_module                                  # noqa: E402
from src.session import discover_flights                               # noqa: E402
from ui.launcher import ChooseFlightView, NavApp, basemap_for          # noqa: E402
from ui.live import LiveFlightView                                     # noqa: E402


class Failure(Exception):
    pass


def pump(root, seconds: float, until=None) -> bool:
    """Run the Tk event loop for a while, optionally stopping on a condition."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        if until is not None and until():
            return True
        time.sleep(0.02)
    return until is None


def exercise(app, root, flight, map_label, seconds, live_seconds, errors) -> str:
    view = app._view
    if not isinstance(view, ChooseFlightView):
        raise Failure("not on the chooser")

    # Select this flight the way a click would.
    index = [os.path.abspath(f.video_path) for f in view.flights].index(
        os.path.abspath(flight.video_path))
    children = view.tree.get_children()
    view.tree.selection_set(children[index])
    view._on_select()
    root.update()

    if map_label not in view.map_choice.cget("values"):
        return f"skipped ({map_label} not offered)"
    view.map_source.set(map_label)
    view.limit.set(str(int(seconds)) if seconds else "0")
    root.update()
    if "disabled" in view.start_button.state():
        return "skipped (Start disabled -- no telemetry)"

    errors.clear()
    app.last_refusal = None
    view.start()
    root.update()
    workspace = app._workspace.path if app._workspace else None

    if not pump(root, 900, lambda: app._player is not None or errors or app.last_refusal):
        raise Failure("preprocessing never finished")
    if app.last_refusal:
        # The app declined, with a reason. That is correct behaviour, not a bug.
        return f"declined: {app.last_refusal.splitlines()[0]}"
    if errors:
        raise Failure(f"preprocessing failed: {errors[0][1].splitlines()[0]}")

    player = app._player
    n = player.trace.n_frames
    if n < 1:
        raise Failure("player opened with no frames")
    for i in {0, min(1, n - 1), n // 2, n - 1}:
        player.goto(i)
        root.update()
    if not player.status.cget("text").strip():
        raise Failure("player status bar is empty")

    # Live playback.
    app.go_live()
    root.update()
    if errors:
        raise Failure(f"live view failed: {errors[0][1].splitlines()[0]}")
    live = app._view
    if not isinstance(live, LiveFlightView):
        raise Failure("live view did not open")
    live.speed_var.set("4x")
    live._set_speed()
    live.toggle_play()
    pump(root, live_seconds)
    if live.video_time <= 0.5:
        raise Failure("live video did not advance")
    localized = live.trace.n_computed
    live.toggle_play()

    app.show_chooser()
    root.update()
    if workspace and os.path.exists(workspace):
        raise Failure("scratch directory was not erased")
    return (f"ok  {n:>3} frames, player + live ({live.video_time:.0f}s played, "
            f"{localized} localized in real time)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seconds", type=float, default=45.0,
                        help="Clip each flight to its first N seconds (0 = whole flight)")
    parser.add_argument("--full", action="store_true", help="Same as --seconds 0")
    parser.add_argument("--live-seconds", type=float, default=5.0)
    parser.add_argument("--maps", default="all",
                        help="'all', or a comma-separated subset of the entry window's labels")
    args = parser.parse_args()
    seconds = 0 if args.full else args.seconds

    flights = discover_flights(args.data_root)
    if not flights:
        raise SystemExit(f"No videos found under {args.data_root}/")

    errors: list = []
    launcher_module.messagebox.showerror = lambda title, msg: errors.append((title, msg))
    launcher_module.messagebox.showinfo = lambda title, msg: None
    root = tk.Tk()
    root.geometry("1500x820")
    app = NavApp(root, data_root=args.data_root)
    root.update()

    labels = [label for label, _v, needs in launcher_module.MAP_SOURCES]
    if args.maps != "all":
        wanted = [m.strip().lower() for m in args.maps.split(",")]
        labels = [l for l in labels if any(w in l.lower() for w in wanted)]

    print(f"{len(flights)} flight(s) under {args.data_root}/, "
          f"{'whole flight' if not seconds else f'first {seconds:.0f}s'} each")
    print("  ok        = ran end to end")
    print("  declined  = app refused with a reason and stayed usable (not a bug)")
    print("  FAILED    = something actually broke\n")
    failures = 0
    for flight in flights:
        name = os.path.basename(flight.video_path)
        has_basemap = basemap_for(flight.video_path) is not None
        for label in labels:
            needs = next(n for l, _v, n in launcher_module.MAP_SOURCES if l == label)
            if needs and not has_basemap:
                continue
            print(f"  {name:<20} {label[:34]:<36}", end="", flush=True)
            try:
                print(exercise(app, root, flight, label, seconds, args.live_seconds, errors))
            except Failure as exc:
                print(f"FAILED: {exc}")
                failures += 1
                app.show_chooser()
                root.update()
            except Exception as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}")
                failures += 1
                app.show_chooser()
                root.update()

    app.quit()
    print(f"\n{'FAILURES: %d' % failures if failures else 'All flights passed.'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
