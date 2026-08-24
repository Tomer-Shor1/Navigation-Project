"""Live flight: the real video playing at wall-clock speed, navigated as it runs.

The stepping player is for studying one decision. This is the other half of the
story -- what the system looks like actually operating. The flight video plays at
the rate it was flown, and the navigator runs beside it: every time playback
reaches a frame the navigator localizes, the fix lands and the route grows.

**Nothing is computed before it is due.** A frame is localized only once playback
actually reaches its moment in the flight, and the fix appears on the route when
the computation finishes -- so what you see is the true latency between the drone
being somewhere and the navigator knowing it. Running the work ahead and
replaying it on a timer would look identical and prove nothing.

The status bar reports that latency, and whether the navigator is keeping up: at
~300 ms of work against a 5 s frame spacing there is plenty of slack at 1x, and
raising the playback speed eats it -- at 8x the same flight gives the navigator
0.6 s per frame and it starts to fall behind, which the readout shows rather than
hides.

Tk is not thread-safe, so the worker only ever advances the trace and the GUI
thread only ever reads frames it has already finished.
"""

from __future__ import annotations

import gc

import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
import utm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.reference_source import as_reference_source
from src.trace import NavigationTrace

COL_TRUE = "#1f77b4"
COL_FIX = "#d62728"
COL_DR = "#ff7f0e"
COL_GIS = "#7a3cc4"

VIDEO_W = 840                 # displayed video width; the source is 1920 wide
TICK_MS = 40                  # ~25 display updates per second
SPEEDS = ("0.5x", "1x", "2x", "4x")


def to_photo(rgb: np.ndarray) -> tk.PhotoImage:
    """Wrap an RGB array as a Tk image via PPM -- no PIL dependency needed."""
    height, width = rgb.shape[:2]
    header = f"P6 {width} {height} 255 ".encode()
    return tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")


class LiveFlightView:
    """Video on the left, the route building on the right."""

    def __init__(self, root: tk.Tk, session, on_close):
        self.root = root
        self.session = session
        self.on_close = on_close

        self.capture = cv2.VideoCapture(session.video_path)
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open {session.video_path}")
        self.duration = (self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
                         / max(self.capture.get(cv2.CAP_PROP_FPS), 1e-6))

        self.trace = NavigationTrace(session)
        self.frame_times = [f.timestamp_sec for f in session.test_frames]
        self.latencies: list = []       # seconds of compute per landed fix
        self.behind_by = 0.0            # video seconds the navigator is lagging
        self.video_time = 0.0
        self.playing = False
        self.speed = 1.0
        self._photo: Optional[tk.PhotoImage] = None
        self._drawn = -1               # route redraws only when a new fix lands
        self._grabbed = False          # retrieve() needs a grab() to have happened
        self._tick_id: Optional[str] = None
        self._last_wall = time.perf_counter()
        self._stop = threading.Event()

        first = session.telemetry[0]
        e0, n0, zn, zl = utm.from_latlon(first.latitude, first.longitude)
        self._origin = (e0, n0, zn, zl)
        self._true_track = np.array([self._xy(r.latitude, r.longitude) for r in session.telemetry])
        self._bounds = self._square_bounds()

        root.title(f"Live flight — {session.stem}")
        self._build_widgets()

        # The navigator runs off the GUI thread, but strictly in step with
        # playback -- it blocks until the video reaches each frame.
        self.worker = threading.Thread(target=self._navigate, daemon=True,
                                       name="live-navigator")
        self.worker.start()
        self._show_frame()
        self._draw_route()

    # -- geometry ---------------------------------------------------------
    def _xy(self, lat, lon):
        e0, n0, zn, zl = self._origin
        e, n, _, _ = utm.from_latlon(lat, lon, force_zone_number=zn, force_zone_letter=zl)
        return e - e0, n - n0

    def _square_bounds(self):
        pts = self._true_track
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()
        y0, y1 = pts[:, 1].min(), pts[:, 1].max()
        half = max(x1 - x0, y1 - y0, 50.0) * 0.6
        return ((x0 + x1) / 2 - half, (x0 + x1) / 2 + half,
                (y0 + y1) / 2 - half, (y0 + y1) / 2 + half)

    # -- layout -----------------------------------------------------------
    def _build_widgets(self) -> None:
        self._panels = []
        self.frame = ttk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self._panels.append(self.frame)

        left = ttk.Frame(self.frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(left, background="#101418", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW)

        right = ttk.Frame(self.frame, width=560)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        self.fig = Figure(figsize=(5.5, 7.4), dpi=100, layout="constrained")
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0])
        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.ax_err = self.fig.add_subplot(gs[1, 0])
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status = ttk.Label(self.root, anchor="w", font=("Menlo", 11), padding=(10, 5))
        self.status.pack(side=tk.TOP, fill=tk.X)
        self._panels.append(self.status)

        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._panels.append(bar)
        self.play_button = ttk.Button(bar, text="Play", width=7, command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT)
        ttk.Button(bar, text="Restart", width=8, command=self.restart).pack(side=tk.LEFT, padx=4)
        ttk.Label(bar, text="speed").pack(side=tk.LEFT, padx=(12, 3))
        self.speed_var = tk.StringVar(value="1x")
        speed = ttk.Combobox(bar, values=list(SPEEDS), textvariable=self.speed_var,
                             state="readonly", width=5)
        speed.pack(side=tk.LEFT)
        speed.bind("<<ComboboxSelected>>", lambda _e: self._set_speed())
        self.progress = ttk.Progressbar(bar, maximum=max(self.duration, 1.0), length=340)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=12)
        ttk.Button(bar, text="Back to frame-by-frame",
                   command=self._close).pack(side=tk.RIGHT)

    def _set_speed(self) -> None:
        self.speed = float(self.speed_var.get().rstrip("x"))

    # -- the navigator, running against the clock -------------------------
    def _navigate(self) -> None:
        """Localize each frame only once playback has reached it.

        This is the whole point of the mode: the work happens when the drone is
        actually there, so the delay before a fix appears is the real one.
        """
        for index in range(self.trace.n_frames):
            due = self.frame_times[index]
            while not self._stop.is_set() and self.video_time < due:
                time.sleep(0.02)
            if self._stop.is_set():
                return
            started = time.perf_counter()
            try:
                self.trace.ensure(index)
            except Exception:
                return
            self.latencies.append(time.perf_counter() - started)
            # How far the video has run past the moment this fix describes.
            self.behind_by = max(0.0, self.video_time - due)

    # -- playback ---------------------------------------------------------
    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        self._last_wall = time.perf_counter()
        if self.playing:
            self._tick()
        elif self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None

    def restart(self) -> None:
        """Rewind the video *and* the navigator -- nothing carries over.

        The trace is forward-only, so replaying means a fresh one; that is also
        what keeps the demo honest, since a second run re-does the work rather
        than replaying the first run's answers.
        """
        was_playing = self.playing
        if was_playing:
            self.toggle_play()
        self._stop.set()
        if self.worker.is_alive():
            self.worker.join(timeout=3)

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._grabbed = False
        self.video_time = 0.0
        self._drawn = -1
        self.latencies = []
        self.behind_by = 0.0
        self.trace = NavigationTrace(self.session)
        self._stop = threading.Event()
        self.worker = threading.Thread(target=self._navigate, daemon=True,
                                       name="live-navigator")
        self.worker.start()

        self._last_wall = time.perf_counter()
        self._show_frame()
        self._draw_route()
        if was_playing:
            self.toggle_play()

    def _tick(self) -> None:
        if not self.playing:
            return
        now = time.perf_counter()
        self.video_time += (now - self._last_wall) * self.speed
        self._last_wall = now

        if self.video_time >= self.duration:
            self.playing = False
            self.play_button.configure(text="Play")
            self._show_frame()
            return

        self._show_frame()
        if self.trace.n_computed != self._drawn:
            self._drawn = self.trace.n_computed
            self._draw_route()
        self._tick_id = self.root.after(TICK_MS, self._tick)

    def _show_frame(self) -> None:
        """Advance the decoder to the current playback time and blit the frame."""
        target_ms = self.video_time * 1000.0
        ok = True
        # grab() skips decode work; only the frame we actually show is decoded.
        while ok and (not self._grabbed or self.capture.get(cv2.CAP_PROP_POS_MSEC) < target_ms):
            ok = self.capture.grab()
            self._grabbed = self._grabbed or ok
        if not self._grabbed:
            return
        ok, frame = self.capture.retrieve()
        if not ok or frame is None:
            return
        scale = VIDEO_W / frame.shape[1]
        frame = cv2.resize(frame, (VIDEO_W, int(frame.shape[0] * scale)))
        self._annotate(frame)
        self.last_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._photo = to_photo(rgb)
        self.canvas.itemconfig(self.image_id, image=self._photo)
        self.progress.configure(value=min(self.video_time, self.duration))
        self._update_status()

    def _annotate(self, frame) -> None:
        """Draw the navigator's current verdict onto the video itself."""
        latest = self._latest()
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, 34), (16, 20, 24), -1)
        clock = f"t = {self.video_time:5.1f}s / {self.duration:.0f}s"
        cv2.putText(frame, clock, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1,
                    cv2.LINE_AA)
        if latest is None:
            return
        if latest.used_gis:
            label, colour = "GIS FALLBACK (satellite)", (196, 60, 122)
        elif latest.mode == "map_fix":
            label, colour = "VISUAL FIX", (60, 200, 90)
        else:
            label, colour = f"DEAD RECKONING (coast #{latest.result.coast_steps})", (40, 150, 240)
        cv2.putText(frame, label, (width - 12 - 9 * len(label), 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
        # The frame centre is the point being localized.
        cv2.drawMarker(frame, (width // 2, height // 2), (0, 190, 255),
                       cv2.MARKER_CROSS, 26, 2)

    def _latest(self):
        return self.trace.computed_frames()[self.trace.n_computed - 1] if self.trace.n_computed else None

    # -- the route, building ----------------------------------------------
    def _draw_route(self) -> None:
        done = self.trace.computed_frames()[: self.trace.n_computed]
        ax = self.ax_map
        ax.clear()
        ax.plot(self._true_track[::10, 0], self._true_track[::10, 1], "-", lw=1.3,
                color=COL_TRUE, alpha=0.5, label="true GPS track")

        points, colours = [], []
        for t in done:
            if not t.result.has_estimate:
                continue
            points.append(self._xy(t.result.estimated_latitude, t.result.estimated_longitude))
            colours.append(COL_GIS if t.used_gis else
                           (COL_FIX if t.mode == "map_fix" else COL_DR))
        if points:
            pts = np.array(points)
            ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.6, color="#8a8f98", alpha=0.9, zorder=3)
            ax.scatter(pts[:, 0], pts[:, 1], c=colours, s=34, zorder=4)
            ax.plot(pts[-1, 0], pts[-1, 1], "*", ms=20, color=colours[-1], zorder=5)
        latest = self._latest()
        if latest is not None:
            tx, ty = self._xy(latest.true_latitude, latest.true_longitude)
            ax.plot([tx], [ty], "o", ms=9, color=COL_TRUE, zorder=5)

        ax.set_xlim(self._bounds[0], self._bounds[1])
        ax.set_ylim(self._bounds[2], self._bounds[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"Route built live — {len(points)} positions", fontsize=11)
        ax.set_xlabel("East (m)", fontsize=9)
        ax.set_ylabel("North (m)", fontsize=9)

        ax = self.ax_err
        ax.clear()
        errors = [t.error_m or 0.0 for t in done]
        if errors:
            ax.bar(range(len(errors)), errors, width=0.8,
                   color=[COL_GIS if t.used_gis else (COL_FIX if t.mode == "map_fix" else COL_DR)
                          for t in done])
            ax.set_ylim(0, max(max(errors) * 1.3, 1.0))
        ax.set_xlim(-0.6, self.trace.n_frames - 0.4)
        ax.set_ylabel("error (m)", fontsize=9)
        ax.set_xlabel("fix", fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        self.fig_canvas.draw_idle()

    def _update_status(self) -> None:
        done = self.trace.computed_frames()[: self.trace.n_computed]
        fixes = sum(1 for t in done if t.mode == "map_fix")
        coasted = len(done) - fixes
        gis = sum(1 for t in done if t.used_gis)
        errs = sorted(t.error_m for t in done if t.mode == "map_fix" and t.error_m is not None)
        median = f"{errs[len(errs) // 2]:.1f} m" if errs else "--"

        # The real-time readout: how long the last fix took to compute, and how
        # far playback had moved on by the time it landed. Both are measured, not
        # simulated -- the work only starts when the video reaches the frame.
        ready = self.trace.n_computed
        if self.latencies:
            last_ms = self.latencies[-1] * 1000.0
            typical = sorted(self.latencies)[len(self.latencies) // 2] * 1000.0
            budget = self._frame_budget()
            verdict = ("keeping up" if typical <= budget * 1000.0 else "FALLING BEHIND")
            pace = (f"last fix {last_ms:4.0f} ms, median {typical:4.0f} ms vs "
                    f"{budget * 1000.0:4.0f} ms available -- {verdict}")
            if self.behind_by > 1.0:
                pace += f" ({self.behind_by:.1f}s late)"
        else:
            pace = "waiting for the first frame to come due"
        self.status.configure(
            text=f"  video {self.video_time:5.1f}s   |   {fixes} visual fixes, {gis} via GIS, "
                 f"{coasted} coasted, median error {median}   |   {ready}/{self.trace.n_frames} "
                 f"localized   |   {pace}")

    def _frame_budget(self) -> float:
        """Wall-clock seconds available per frame at the current playback speed."""
        if len(self.frame_times) < 2:
            return 1.0
        spacing = self.frame_times[1] - self.frame_times[0]
        return max(spacing / max(self.speed, 1e-6), 1e-6)

    # -- teardown ---------------------------------------------------------
    def _close(self) -> None:
        self.destroy()
        self.on_close()

    def destroy(self) -> None:
        self.playing = False
        self._stop.set()
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        # Drop the Tk image while the interpreter still has a main loop; letting
        # the garbage collector reach it later raises a noisy (harmless)
        # "main thread is not in main loop" at shutdown.
        self.canvas.itemconfig(self.image_id, image="")
        self._photo = None
        self.fig.clear()
        # Collect the Tk photo images now, while there is still a
        # main loop for them to unregister from.
        gc.collect()
        # Release Tk variables while the interpreter
        # still has a main loop, to keep shutdown quiet.
        self.speed_var = None
        for panel in self._panels:
            panel.destroy()
        self._panels.clear()
