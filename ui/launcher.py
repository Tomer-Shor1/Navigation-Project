"""The entry window: choose a flight, watch it be prepared, then fly it.

Three views take turns in a single window, which is what `NavApp` coordinates:

    choose  ->  prepare  ->  play  ->  (choose again)

**Nothing is cached between runs.** Every flight the app opens is decoded into a
fresh `Workspace` -- a temporary directory that is erased when you pick another
flight or close the window. That costs a minute of ffmpeg up front and buys two
things worth more than the minute: the app opens *any* video you point it at,
and it can never quietly show you a stale run that a previous session left in
`data/frames/`.

Preparation runs on a worker thread. Tk is not thread-safe, so the worker only
ever puts messages on a queue and the GUI thread drains it on a timer -- no
widget is ever touched from off the main thread.
"""

from __future__ import annotations

import argparse
import glob
import os
import queue
import threading
import traceback
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from src.extract_frames import FrameExtractionCancelled, get_video_duration_sec
from src.session import (FlightSource, SessionInputError, VIDEO_EXTENSIONS,
                         add_pipeline_args, discover_flights, find_telemetry_for,
                         prepare_session)
from src.trace import NavigationTrace
from src.workspace import Workspace
from ui.live import LiveFlightView
from ui.player import NavPlayer

WINDOW_SIZE = "1620x820"
LAUNCHER_SIZE = "900x660"

# label -> --map-source value, and whether it needs a satellite basemap on disk.
MAP_SOURCES = (
    ("The flight's own earlier frames", "flight", False),
    ("A north-up orthomosaic built from them", "ortho", False),
    ("Satellite basemap only (GIS)", "gis", True),
    ("Both: previous flight + satellite", "hybrid", True),
)


def basemap_for(video_path: str, root: str = "data/basemap") -> Optional[str]:
    """The downloaded satellite basemap for this flight, if one has been fetched.

    `tools/fetch_basemap.py --flight <srt>` names it after the flight stem, so
    the entry window can offer the GIS map sources only when they would actually
    work rather than failing after a minute of frame extraction.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    image = os.path.join(root, f"{stem}.jpg")
    sidecar = os.path.splitext(image)[0] + ".json"
    return image if os.path.isfile(image) and os.path.isfile(sidecar) else None


@dataclass
class FlightRequest:
    """Everything the entry window collected about how to run one flight."""

    video_path: str
    srt_path: str
    rate_hz: float = 1.0
    holdout_every: int = 5
    map_source: str = "flight"
    basemap_path: Optional[str] = None
    max_seconds: Optional[float] = None
    playback_fps: float = 2.0

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.video_path))[0]

    def to_args(self, frames_dir: str, results_dir: str) -> argparse.Namespace:
        """Turn the choices into the same `args` the command line would produce.

        Starting from the parser's own defaults rather than a hand-written dict
        means a flag added to `add_pipeline_args` is automatically present here,
        with its documented default, instead of silently missing.
        """
        parser = argparse.ArgumentParser()
        add_pipeline_args(parser)
        args = parser.parse_args([])
        args.video = self.video_path
        args.srt = self.srt_path
        args.frames_dir = frames_dir
        args.results_dir = results_dir
        args.rate = self.rate_hz
        args.holdout_every = self.holdout_every
        args.map_source = self.map_source
        args.basemap = self.basemap_path
        args.max_seconds = self.max_seconds
        args.use_cached_frames = False   # the whole point: never reuse old frames
        return args


def _estimate_frames(request: FlightRequest) -> Optional[int]:
    """How many frames this video will yield, for the size warning and progress bar."""
    try:
        duration = get_video_duration_sec(request.video_path)
    except Exception:
        return None
    if request.max_seconds:
        duration = min(duration, request.max_seconds)
    return max(1, int(duration * request.rate_hz))


# ---------------------------------------------------------------------------
# View 1: choose a flight
# ---------------------------------------------------------------------------

class ChooseFlightView:
    """Lists the videos under `data/`, and takes any other video via Browse."""

    def __init__(self, root: tk.Tk, data_root: str, on_start: Callable[[FlightRequest], None]):
        self.root = root
        self.data_root = data_root
        self.on_start = on_start
        self.flights: list[FlightSource] = []
        self.selected: Optional[FlightSource] = None

        root.title("GPS-denied visual navigation")
        root.geometry(LAUNCHER_SIZE)

        self.frame = ttk.Frame(root, padding=(18, 14))
        self.frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self.frame, text="Choose a flight to navigate",
                  font=("TkDefaultFont", 17, "bold")).pack(anchor="w")
        ttk.Label(self.frame, wraplength=830, foreground="#5c6470",
                  text=f"Videos found under {data_root}/. Any video with a matching .srt "
                       "telemetry file will work — use Browse to open one from anywhere.").pack(
            anchor="w", pady=(3, 12))

        self._build_flight_list()
        self._build_options()
        self._build_footer()

        self.refresh()

    # -- widgets ----------------------------------------------------------
    def _build_flight_list(self) -> None:
        holder = ttk.Frame(self.frame)
        holder.pack(fill=tk.BOTH, expand=True)

        columns = ("video", "telemetry", "size", "folder")
        self.tree = ttk.Treeview(holder, columns=columns, show="headings", height=8)
        for column, heading, width, anchor in (
            ("video", "video", 230, "w"),
            ("telemetry", "telemetry", 130, "w"),
            ("size", "size", 80, "e"),
            ("folder", "folder", 250, "w"),
        ):
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, anchor=anchor)
        self.tree.tag_configure("no_srt", foreground="#b42318")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())
        self.tree.bind("<Double-1>", lambda _e: self.start())

        scroll = ttk.Scrollbar(holder, orient=tk.VERTICAL, command=self.tree.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scroll.set)

        buttons = ttk.Frame(self.frame)
        buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(buttons, text="Browse for a video…", command=self.browse_video).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Choose telemetry…", command=self.browse_telemetry).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Refresh", command=self.refresh).pack(side=tk.LEFT)

        self.summary = ttk.Label(self.frame, foreground="#5c6470", wraplength=830)
        self.summary.pack(anchor="w", pady=(10, 0))

    def _build_options(self) -> None:
        box = ttk.LabelFrame(self.frame, text="How to run it", padding=(12, 8))
        box.pack(fill=tk.X, pady=(12, 0))

        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        self.rate = tk.StringVar(value="1.0")
        self.holdout = tk.StringVar(value="5")
        self.limit = tk.StringVar(value="0")
        ttk.Label(row, text="Sample").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0.2, to=5.0, increment=0.2, width=5, textvariable=self.rate,
                    command=self._on_select).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(row, text="frames per second   ·   navigate every").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=2, to=20, width=4, textvariable=self.holdout,
                    command=self._on_select).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(row, text="th frame   ·   use first").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0, to=3600, increment=30, width=6, textvariable=self.limit,
                    command=self._on_select).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(row, text="seconds (0 = all)").pack(side=tk.LEFT)
        for var in (self.rate, self.holdout, self.limit):
            var.trace_add("write", lambda *_a: self._on_select())

        maprow = ttk.Frame(box)
        maprow.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(maprow, text="Map to navigate against:").pack(side=tk.LEFT)
        self.map_source = tk.StringVar(value=MAP_SOURCES[0][0])
        self.map_choice = ttk.Combobox(maprow, state="readonly", width=42,
                                       textvariable=self.map_source)
        self.map_choice.pack(side=tk.LEFT, padx=8)
        self.map_note = ttk.Label(maprow, foreground="#5c6470")
        self.map_note.pack(side=tk.LEFT)

    def _build_footer(self) -> None:
        footer = ttk.Frame(self.frame)
        footer.pack(fill=tk.X, pady=(14, 0))
        ttk.Label(footer, foreground="#5c6470", wraplength=560,
                  text="Frames are extracted to a temporary folder when you start, and "
                       "deleted as soon as you open another flight or close the window. "
                       "Nothing is left behind and nothing is reused.").pack(side=tk.LEFT)
        self.start_button = ttk.Button(footer, text="Start", command=self.start)
        self.start_button.pack(side=tk.RIGHT)
        self.start_button.state(["disabled"])

    # -- behaviour --------------------------------------------------------
    def refresh(self) -> None:
        self.flights = discover_flights(self.data_root) if os.path.isdir(self.data_root) else []
        self.tree.delete(*self.tree.get_children())
        for flight in self.flights:
            self.tree.insert("", "end", values=(
                os.path.basename(flight.video_path),
                os.path.basename(flight.srt_path) if flight.has_telemetry else "— none found —",
                f"{flight.size_mb:,.0f} MB",
                os.path.dirname(flight.video_path) or ".",
            ), tags=() if flight.has_telemetry else ("no_srt",))
        if self.flights:
            first = self.tree.get_children()[0]
            self.tree.selection_set(first)
            self.tree.focus(first)
        else:
            self.selected = None
            self.summary.configure(
                text=f"No videos found under {self.data_root}/. Use “Browse for a video…” "
                     "to open one from anywhere on this computer.")
            self.start_button.state(["disabled"])

    def _selected_index(self) -> Optional[int]:
        selection = self.tree.selection()
        if not selection:
            return None
        children = self.tree.get_children()
        return children.index(selection[0]) if selection[0] in children else None

    def _on_select(self) -> None:
        index = self._selected_index()
        if index is not None and index < len(self.flights):
            self.selected = self.flights[index]
        self._update_summary()

    def _refresh_map_choices(self) -> None:
        """Offer the satellite options only when a basemap has been downloaded."""
        basemap = basemap_for(self.selected.video_path) if self.selected else None
        allowed = [label for label, _v, needs in MAP_SOURCES if basemap or not needs]
        self.map_choice.configure(values=allowed)
        if self.map_source.get() not in allowed:
            self.map_source.set(allowed[0])
        self.map_note.configure(
            text="" if basemap else "  (fetch a basemap for the satellite options)")

    def _update_summary(self) -> None:
        self._refresh_map_choices()
        flight = self.selected
        if flight is None:
            self.start_button.state(["disabled"])
            return
        if not flight.has_telemetry:
            self.summary.configure(
                text=f"{os.path.basename(flight.video_path)} has no matching .srt telemetry. "
                     "The navigator needs it for the reference track — use “Choose telemetry…” "
                     "to point at the right file.")
            self.start_button.state(["disabled"])
            return

        request = self.build_request()
        estimate = _estimate_frames(request) if request else None
        if estimate is None:
            detail = "frame count unknown (ffprobe unavailable)"
        else:
            navigated = max(1, estimate // max(request.holdout_every, 2))
            detail = (f"≈ {estimate} frames to extract, ≈ {navigated} of them navigated "
                      f"against a map of ≈ {estimate - navigated}")
            if estimate > 600:
                detail += "  —  that is a lot; consider a lower rate or a shorter clip"
        self.summary.configure(
            text=f"{os.path.basename(flight.video_path)}  ·  telemetry "
                 f"{os.path.basename(flight.srt_path)}  ·  {detail}")
        self.start_button.state(["!disabled"])

    def browse_video(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)
        path = filedialog.askopenfilename(
            title="Choose a flight video",
            filetypes=[("Video files", patterns), ("All files", "*.*")],
            initialdir=self.data_root if os.path.isdir(self.data_root) else os.getcwd())
        if not path:
            return
        self._adopt(FlightSource(video_path=path, srt_path=find_telemetry_for(path)))

    def browse_telemetry(self) -> None:
        if self.selected is None:
            messagebox.showinfo("Choose a video first",
                                "Pick a video, then choose the .srt telemetry that goes with it.")
            return
        path = filedialog.askopenfilename(
            title="Choose the telemetry (.srt) for this video",
            filetypes=[("SubRip telemetry", "*.srt"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.selected.video_path))
        if not path:
            return
        self._adopt(FlightSource(video_path=self.selected.video_path, srt_path=path))

    def _adopt(self, flight: FlightSource) -> None:
        """Put a browsed-for flight at the top of the list and select it."""
        self.flights = [flight] + [f for f in self.flights
                                   if os.path.abspath(f.video_path) != os.path.abspath(flight.video_path)]
        self.tree.delete(*self.tree.get_children())
        for item in self.flights:
            self.tree.insert("", "end", values=(
                os.path.basename(item.video_path),
                os.path.basename(item.srt_path) if item.has_telemetry else "— none found —",
                f"{item.size_mb:,.0f} MB",
                os.path.dirname(item.video_path) or ".",
            ), tags=() if item.has_telemetry else ("no_srt",))
        first = self.tree.get_children()[0]
        self.tree.selection_set(first)
        self.tree.focus(first)
        self._on_select()

    def build_request(self) -> Optional[FlightRequest]:
        """The current choices, or None if they are not usable yet."""
        if self.selected is None or not self.selected.has_telemetry:
            return None

        def number(var, fallback, cast=float):
            try:
                return cast(var.get())
            except (TypeError, ValueError):
                return fallback

        limit = number(self.limit, 0.0)
        chosen = self.map_source.get()
        map_source = next((v for label, v, _n in MAP_SOURCES if label == chosen), "flight")
        return FlightRequest(
            basemap_path=basemap_for(self.selected.video_path),
            video_path=self.selected.video_path,
            srt_path=self.selected.srt_path,
            rate_hz=max(0.1, number(self.rate, 1.0)),
            holdout_every=max(2, number(self.holdout, 5, int)),
            map_source=map_source,
            max_seconds=limit if limit > 0 else None,
        )

    def start(self) -> None:
        request = self.build_request()
        if request is None:
            return
        self.on_start(request)

    def destroy(self) -> None:
        self.frame.destroy()
        self.rate = self.holdout = self.limit = self.map_source = None


# ---------------------------------------------------------------------------
# View 2: prepare the flight (on a worker thread)
# ---------------------------------------------------------------------------

class PrepareFlightView:
    """Runs `prepare_session` off the main thread and reports what it is doing."""

    POLL_MS = 120

    def __init__(self, root: tk.Tk, request: FlightRequest, workspace: Workspace,
                 on_ready: Callable, on_cancel: Callable, on_failed: Callable[[str], None],
                 on_refused: Optional[Callable[[str], None]] = None):
        self.root = root
        self.request = request
        self.workspace = workspace
        self.on_ready = on_ready
        self.on_cancel = on_cancel
        self.on_failed = on_failed
        self.on_refused = on_refused or on_failed

        self.messages: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.expected_frames = _estimate_frames(request)
        self._poll_id: Optional[str] = None
        self._finished = False

        root.title(f"Preparing {request.stem} …")
        self.frame = ttk.Frame(root, padding=(24, 20))
        self.frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self.frame, text=f"Preparing {os.path.basename(request.video_path)}",
                  font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        ttk.Label(self.frame, foreground="#5c6470", wraplength=780,
                  text="Decoding frames, indexing them into a map and self-calibrating the "
                       "camera scale. This happens once per flight and is thrown away "
                       "afterwards.").pack(anchor="w", pady=(4, 16))

        self.bar = ttk.Progressbar(self.frame, mode="determinate" if self.expected_frames else "indeterminate",
                                   maximum=self.expected_frames or 100, length=760)
        self.bar.pack(fill=tk.X)
        if not self.expected_frames:
            self.bar.start(12)

        self.step = ttk.Label(self.frame, font=("TkDefaultFont", 12))
        self.step.pack(anchor="w", pady=(12, 2))
        self.detail = ttk.Label(self.frame, foreground="#5c6470", font=("Menlo", 10))
        self.detail.pack(anchor="w")

        self.log = tk.Text(self.frame, height=11, wrap="word", relief="flat",
                           background="#f4f6f8", font=("Menlo", 10))
        self.log.pack(fill=tk.BOTH, expand=True, pady=(14, 12))
        self.log.configure(state="disabled")

        ttk.Button(self.frame, text="Cancel", command=self.cancel).pack(anchor="e")

        self.worker = threading.Thread(target=self._work, name="prepare-session", daemon=True)
        self.worker.start()
        # Deliberately not polled inline: preparation can fail before the
        # constructor returns (bad telemetry fails in the first step), and
        # finishing from in here would hand control to the next view while the
        # caller still believes it is building this one.
        self._poll_id = self.root.after(self.POLL_MS, self._poll)

    # -- worker thread ----------------------------------------------------
    def _work(self) -> None:
        """Runs off the main thread: only ever touches the queue, never a widget."""
        args = self.request.to_args(self.workspace.frames_dir,
                                    os.path.join(self.workspace.path, "results"))
        try:
            session = prepare_session(
                args,
                log=lambda message: self.messages.put(("log", message)),
                cancel_event=self.cancel_event,
            )
            self.messages.put(("ready", session))
        except FrameExtractionCancelled:
            self.messages.put(("cancelled", None))
        except SessionInputError as exc:
            # An expected, explicable refusal -- not a crash. Kept distinct so
            # the user is told "this flight cannot be run that way, because..."
            # rather than being shown something that looks like a bug.
            self.messages.put(("refused", str(exc)))
        except Exception as exc:                       # pragma: no cover - surfaced in the UI
            self.messages.put(("failed", f"{type(exc).__name__}: {exc}"))

    # -- main thread ------------------------------------------------------
    def _poll(self) -> None:
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append(payload)
            elif kind == "ready":
                self._finish(lambda: self.on_ready(payload))
                return
            elif kind == "cancelled":
                self._finish(self.on_cancel)
                return
            elif kind == "refused":
                self._finish(lambda: self.on_refused(payload))
                return
            elif kind == "failed":
                self._finish(lambda: self.on_failed(payload))
                return
        self._update_progress()
        self._poll_id = self.root.after(self.POLL_MS, self._poll)

    def _update_progress(self) -> None:
        """Count the frames on disk -- ffmpeg reports nothing else useful."""
        if not self.expected_frames or not self.workspace.is_open:
            return
        done = len(glob.glob(os.path.join(self.workspace.frames_dir, "frame_*.jpg")))
        if done:
            self.bar.configure(value=min(done, self.expected_frames))
            self.detail.configure(
                text=f"{done} / ≈{self.expected_frames} frames extracted"
                     f"   ({self.workspace.size_bytes() / 1e6:,.0f} MB of scratch)")

    def _append(self, message: str) -> None:
        text = message.rstrip()
        if not text:
            return
        if text.startswith("["):
            self.step.configure(text=text.split("...")[0].strip())
            if self.expected_frames and "Extracting" not in text:
                # Past extraction: the frame counter is finished, so stop nudging it.
                self.bar.configure(value=self.expected_frames)
        self.log.configure(state="normal")
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def cancel(self) -> None:
        if self._finished:
            return
        self.cancel_event.set()
        self.step.configure(text="Cancelling…")
        # The worker may be inside ORB indexing rather than ffmpeg, which is not
        # interruptible; either way it will notice and post 'cancelled'.
        self.messages.put(("cancelled", None))

    def _finish(self, action: Callable) -> None:
        """Hand over to the next view, but never from inside our own callback.

        `action` tears this view down, so it has to run on a clean stack rather
        than underneath the poll that triggered it.
        """
        self._finished = True
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        self.root.after(0, action)

    def destroy(self) -> None:
        """Stop the worker *before* returning, so the workspace is safe to erase.

        The caller deletes the scratch directory straight after this; if ffmpeg
        were still running it would keep writing frames into a directory that no
        longer exists.
        """
        self._finished = True
        self.cancel_event.set()
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        if self.worker.is_alive():
            self.worker.join(timeout=8)
        self.frame.destroy()


# ---------------------------------------------------------------------------
# The application: owns the window, the workspace, and which view is showing
# ---------------------------------------------------------------------------

class NavApp:
    """Coordinates the three views and guarantees the scratch space is erased."""

    def __init__(self, root: tk.Tk, data_root: str = "data"):
        self.root = root
        self.data_root = data_root
        self._view = None
        self._player: Optional[NavPlayer] = None
        self._workspace: Optional[Workspace] = None
        self._request: Optional[FlightRequest] = None
        self._session = None
        self.last_refusal: Optional[str] = None
        self.last_failure: Optional[str] = None

        root.protocol("WM_DELETE_WINDOW", self.quit)
        # Without this, an exception raised inside any Tk callback is printed to
        # a terminal the user may not be looking at and the window just stops
        # responding -- "it failed" with nothing to go on. Surface it instead.
        root.report_callback_exception = self._on_callback_error
        self.show_chooser()

    # -- views ------------------------------------------------------------
    def show_chooser(self) -> None:
        self._clear()
        self._view = ChooseFlightView(self.root, self.data_root, on_start=self.prepare)

    def prepare(self, request: FlightRequest) -> None:
        self._clear()
        self._request = request
        self._workspace = Workspace()
        self.root.geometry(WINDOW_SIZE)
        self._view = PrepareFlightView(
            self.root, request, self._workspace,
            on_ready=self.play, on_cancel=self.show_chooser,
            on_failed=self._report_failure, on_refused=self._report_refusal)
        self.root.update_idletasks()

    def play(self, session) -> None:
        self._drop_view()
        self._session = session
        self.root.title(f"GPS-denied visual navigation — {session.stem} "
                        f"(map: {session.map_source})")
        self._player = NavPlayer(
            self.root, session, NavigationTrace(session),
            fps=self._request.playback_fps if self._request else 2.0,
            on_choose_another=self.show_chooser,
            on_go_live=self.go_live)

    def go_live(self) -> None:
        """Swap the frame-by-frame player for live video playback.

        The scratch space and the prepared session are kept, so coming back is
        instant -- only the view changes.
        """
        session = self._session
        if session is None or not session.video_path:
            return
        if self._player is not None:
            self._player.destroy()
            self._player = None
        try:
            self._view = LiveFlightView(self.root, session, on_close=lambda: self.play(session))
        except Exception as exc:
            self._view = None
            messagebox.showerror("Could not play the video", str(exc))
            self.play(session)

    def _on_callback_error(self, exc_type, value, tb) -> None:
        detail = "".join(traceback.format_exception(exc_type, value, tb))
        print(detail, file=sys.stderr)
        messagebox.showerror(
            "Something went wrong",
            f"{exc_type.__name__}: {value}\n\n"
            f"The full traceback was printed to the terminal. The app is still "
            f"running -- use “Open another flight” to start over.")

    def _report_refusal(self, message: str) -> None:
        """An explicable "no" -- the settings do not suit this flight."""
        self.last_refusal = message
        self.show_chooser()
        messagebox.showinfo("This flight cannot be run that way", message)

    def _report_failure(self, message: str) -> None:
        self.last_failure = message
        self.show_chooser()
        messagebox.showerror("Could not prepare this flight", message)

    # -- lifecycle --------------------------------------------------------
    def _drop_view(self) -> None:
        if self._view is not None:
            self._view.destroy()
            self._view = None

    def _clear(self) -> None:
        """Return the window to empty, and erase the scratch space with it."""
        self._drop_view()
        if self._player is not None:
            self._player.destroy()
            self._player = None
        if self._workspace is not None:
            self._workspace.close()
            self._workspace = None

    def release(self) -> None:
        """Erase the scratch space. Safe to call more than once."""
        self._clear()

    def quit(self) -> None:
        self.release()
        self.root.destroy()
