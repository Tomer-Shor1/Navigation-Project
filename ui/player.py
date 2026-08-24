"""The player window: four panels that show one step of the navigation loop.

Owns no algorithm and no preprocessing -- it is handed a prepared `NavSession`
and a `NavigationTrace` and draws whatever they report. See `nav_player.py` for
how to launch it and `src/trace.py` for where the per-frame detail comes from.

What the panels show
--------------------
* **Current frame** -- the image the matcher is looking at, every ORB keypoint it
  found, and the subset that survived RANSAC as geometric inliers.
* **Best-matching map view** -- the reference frame (or map tile) it chose, with
  the current frame's footprint projected onto it through the homography. That
  gold outline *is* the localization: its centre is the estimated position.
* **Route** -- the true GPS track, and the trajectory the navigator has built so
  far, drawn as it goes. Red = trusted visual fix, orange = dead-reckoned. The
  dashed circle is the motion model's search gate around its prediction.
* **Error** -- metres from ground truth per frame, filling in as it runs.

The candidate table beside them is the reason the frame went the way it did:
every map view considered, its match strength, and if it lost, why.
"""

from __future__ import annotations

import gc

import tkinter as tk
from collections import OrderedDict
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
import utm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch, Polygon

from src.trace import NavigationTrace, entry_label, render_ref_image, render_test_image

# One palette, used by every panel, so a colour means the same thing everywhere.
COL_TRUE = "#1f77b4"      # ground truth
COL_FIX = "#d62728"       # trusted visual map fix
COL_DR = "#ff7f0e"        # dead-reckoned (coasted) position
COL_KP = "#00a5c8"        # ORB keypoints
COL_IN = "#12a150"        # RANSAC inliers
COL_OUTLINE = "#e8a400"   # projected frame footprint / localized centre
COL_MAP = "#b0b8c1"       # map views available
COL_GATE = "#7b61ff"      # motion model prediction + search gate
COL_GIS = "#7a3cc4"       # a fix that came from the GIS / satellite map

MAX_DISPLAY_PX = 900        # downscale big frames for display; keeps redraws snappy
MAX_INLIER_MARKERS = 220    # a good overlap yields >1000 inliers; drawing them all
MAX_MATCH_LINES = 26        # hides the imagery, so both are thinned for legibility


def _fit_for_display(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrink an image for on-screen use; returns it with the scale applied, so
    keypoint coordinates can be scaled to match."""
    if image is None:
        return None, 1.0
    longest = max(image.shape[:2])
    if longest <= MAX_DISPLAY_PX:
        return image, 1.0
    s = MAX_DISPLAY_PX / longest
    small = cv2.resize(image, (int(image.shape[1] * s), int(image.shape[0] * s)),
                       interpolation=cv2.INTER_AREA)
    return small, s


class NavPlayer:
    """The window: a matplotlib figure of four panels, plus Tk transport controls."""

    def __init__(self, root: tk.Tk, session, trace: NavigationTrace, start: int = 0,
                 fps: float = 2.0, on_choose_another=None, on_go_live=None):
        self._inlier_idx = np.zeros(0, int)
        self._n_inliers = 0
        self._frame_scale = self._ref_scale = 1.0
        # Scrubbing revisits the same handful of frames constantly; decoding and
        # resampling them again each time is what makes a scrubber feel laggy.
        self._display_cache: OrderedDict = OrderedDict()
        self.root = root
        self.session = session
        self.trace = trace
        self.index = max(0, min(start, trace.n_frames - 1))
        self.playing = False
        self._after_id: Optional[str] = None
        self._scrubbing = False
        self._scrub_after: Optional[str] = None
        self._pending_index: Optional[int] = None
        # Set when the player is running inside the app rather than standalone;
        # enables the button that hands control back to the entry window.
        self._on_choose_another = on_choose_another
        # Offered only when the source video is on hand to play.
        self._on_go_live = on_go_live if session.video_path else None
        self._panels: list = []          # top-level containers, for teardown
        self._key_bindings: list = []    # sequences bound on the root, likewise

        # Local metric frame for the route panel: metres east/north of the first
        # telemetry fix, via UTM. Same convention as src/visualize.py.
        from src.reference_source import as_reference_source
        self._map_views = as_reference_source(session.reference_index).all()
        self._n_map_views = len(self._map_views)

        first = session.telemetry[0]
        e0, n0, zn, zl = utm.from_latlon(first.latitude, first.longitude)
        self._origin = (e0, n0, zn, zl)
        self._true_track = np.array([self._xy(r.latitude, r.longitude) for r in session.telemetry])
        self._map_xy = np.array([self._xy(e.latitude, e.longitude) for e in self._map_views])
        self._true_test = np.array([self._xy(f.latitude, f.longitude) for f in session.test_frames])
        self._map_extent = (self._basemap_extent(session.basemap)
                            if session.basemap is not None else None)
        # Both are identical on every frame; rebuilding them per redraw is waste.
        self._map_raster = (np.ma.masked_equal(session.basemap.image, 0)
                            if session.basemap is not None else None)
        step = max(1, len(self._true_track) // 1500)
        self._true_track_draw = self._true_track[::step]
        self._map_bounds = self._square_bounds()

        root.title(f"GPS-denied visual navigation -- {session.stem} "
                   f"(map: {session.map_source})")
        self._build_widgets()
        self._bind_keys()
        self.goto(self.index)

    def _display(self, key, produce) -> tuple:
        """Fetch a display-ready (image, scale) pair, computing it at most once."""
        hit = self._display_cache.get(key)
        if hit is None:
            hit = _fit_for_display(produce())
            self._display_cache[key] = hit
            if len(self._display_cache) > 12:
                self._display_cache.popitem(last=False)
        else:
            self._display_cache.move_to_end(key)
        return hit

    # -- geometry ---------------------------------------------------------
    def _xy(self, lat: float, lon: float) -> tuple[float, float]:
        e0, n0, zn, zl = self._origin
        e, n, _, _ = utm.from_latlon(lat, lon, force_zone_number=zn, force_zone_letter=zl)
        return e - e0, n - n0

    def _square_bounds(self) -> tuple:
        """A fixed, square window over the whole flight.

        The map panel must not autoscale to whatever is drawn on it: after a run
        of dropouts the motion model's search gate inflates to well over a
        kilometre, and letting that set the limits shrinks the entire trajectory
        to a dot. So the view is computed once, from the ground truth and the map
        itself, and never moves.
        """
        pts = [self._true_track, self._map_xy, self._true_test]
        if self._map_extent is not None:
            x0, x1, y0, y1 = self._map_extent
            pts.append(np.array([[x0, y0], [x1, y1]]))
        pts = np.vstack(pts)
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()
        y0, y1 = pts[:, 1].min(), pts[:, 1].max()
        half = max(x1 - x0, y1 - y0, 50.0) * 0.58   # 0.5 plus ~16% margin
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return cx - half, cx + half, cy - half, cy + half

    # -- layout -----------------------------------------------------------
    def _build_widgets(self) -> None:
        content = ttk.Frame(self.root)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._panels.append(content)

        self.fig = Figure(figsize=(12.8, 5.1), dpi=100, layout="constrained")
        # Ratios chosen so the two 16:9 image panels nearly fill their row -- an
        # over-tall row just letterboxes them in dead white space.
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1.22, 1.22, 1.05],
                                   height_ratios=[1.6, 1.0])
        self.ax_frame = self.fig.add_subplot(gs[0, 0])
        self.ax_ref = self.fig.add_subplot(gs[0, 1])
        self.ax_map = self.fig.add_subplot(gs[:, 2])
        self.ax_err = self.fig.add_subplot(gs[1, 0:2])

        self.canvas = FigureCanvasTkAgg(self.fig, master=content)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = ttk.Frame(content, width=372)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)
        ttk.Label(side, text="Map views considered this frame",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=8, pady=(10, 4))
        cols = ("map", "view", "good", "in", "verdict")
        self.tree = ttk.Treeview(side, columns=cols, show="headings", height=9)
        for col, text, width in (("map", "map", 62), ("view", "view", 96),
                                 ("good", "good", 44), ("in", "inl", 40),
                                 ("verdict", "outcome", 122)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="w" if col == "view" else "e")
        self.tree.tag_configure("best", background="#dff3e4")
        self.tree.tag_configure("gis_best", background="#ece0fa", foreground=COL_GIS)
        self.tree.tag_configure("rejected", foreground="#98a2b3")
        self.tree.pack(fill=tk.X, padx=8)
        self.tree.bind("<<TreeviewSelect>>", self._on_candidate_select)

        self.verdict = tk.Text(side, height=7, wrap="word", relief="flat",
                               background="#f4f6f8", font=("TkDefaultFont", 10))
        self.verdict.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.verdict.configure(state="disabled")

        self.status = ttk.Label(self.root, anchor="w", font=("Menlo", 11), padding=(10, 5))
        self.status.pack(side=tk.TOP, fill=tk.X)
        self._panels.append(self.status)

        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._panels.append(bar)
        for text, cmd, width in (("|<", self.go_start, 3), ("<", self.step_back, 3),
                                 ("Play", self.toggle_play, 6), (">", self.step_forward, 3),
                                 (">|", self.go_end, 3)):
            btn = ttk.Button(bar, text=text, width=width, command=cmd)
            btn.pack(side=tk.LEFT, padx=2)
            if text == "Play":
                self.play_button = btn

        self.scrub = ttk.Scale(bar, from_=0, to=max(1, self.trace.n_frames - 1),
                               orient=tk.HORIZONTAL, command=self._on_scrub)
        self.scrub.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.frame_label = ttk.Label(bar, width=16, font=("Menlo", 11))
        self.frame_label.pack(side=tk.LEFT)

        ttk.Label(bar, text="fps").pack(side=tk.LEFT, padx=(12, 2))
        self.fps_var = tk.StringVar(value="2")
        ttk.Spinbox(bar, from_=1, to=20, width=3, textvariable=self.fps_var).pack(side=tk.LEFT)

        self.show_kp = tk.BooleanVar(value=True)
        self.show_lines = tk.BooleanVar(value=True)
        self.follow = tk.BooleanVar(value=False)
        for text, var in (("keypoints", self.show_kp), ("match lines", self.show_lines),
                          ("follow", self.follow)):
            ttk.Checkbutton(bar, text=text, variable=var,
                            command=self.redraw).pack(side=tk.LEFT, padx=(10, 0))

        if self._on_choose_another is not None:
            ttk.Button(bar, text="Open another flight",
                       command=self._choose_another).pack(side=tk.RIGHT, padx=(12, 0))
        if self._on_go_live is not None:
            ttk.Button(bar, text="▶ Live flight",
                       command=self._on_go_live).pack(side=tk.RIGHT, padx=(12, 0))

    def _bind_keys(self) -> None:
        def key(handler):
            def wrapped(_event):
                handler()
                return "break"
            return wrapped

        for sequence, handler in (
            ("<space>", self.toggle_play),
            ("<Right>", self.step_forward),
            ("<Left>", self.step_back),
            ("<Shift-Right>", lambda: self.goto(self.index + 5)),
            ("<Shift-Left>", lambda: self.goto(self.index - 5)),
            ("<Home>", self.go_start),
            ("<End>", self.go_end),
        ):
            self.root.bind_all(sequence, key(handler))
            self._key_bindings.append(sequence)

    def _choose_another(self) -> None:
        callback = self._on_choose_another
        if callback is not None:
            callback()

    def destroy(self) -> None:
        """Tear the player down so the window can be reused.

        Bindings made with `bind_all` live on the root, not on the widgets, so
        destroying the panels is not enough -- the arrow keys would keep driving
        a player that no longer exists. Same for the pending `after` callbacks.
        """
        self.playing = False
        for attr in ("_after_id", "_scrub_after"):
            pending = getattr(self, attr, None)
            if pending is not None:
                try:
                    self.root.after_cancel(pending)
                except Exception:
                    pass
                setattr(self, attr, None)
        for sequence in self._key_bindings:
            self.root.unbind_all(sequence)
        self._key_bindings.clear()
        for panel in self._panels:
            panel.destroy()
        self._panels.clear()
        self._display_cache.clear()
        self.fig.clear()
        # Collect the Tk photo images now, while there is still a
        # main loop for them to unregister from.
        gc.collect()
        # Release Tk variables while the interpreter
        # still has a main loop, to keep shutdown quiet.
        self.show_kp = None
        self.show_lines = None
        self.follow = None
        self.fps_var = None

    # -- transport --------------------------------------------------------
    def goto(self, index: int) -> None:
        index = max(0, min(index, self.trace.n_frames - 1))
        if index >= self.trace.n_computed:
            # Beyond what has been navigated: run the real algorithm to get there,
            # narrating progress so a multi-frame jump isn't a silent freeze.
            self.trace.ensure(index, progress=self._report_progress)
        self.index = index
        self._sync_scrub()
        self.redraw()

    def _report_progress(self, done: int, target: int) -> None:
        self.status.configure(
            text=f"  localizing frame {done + 1} of {target + 1} ... "
                 f"(matching against {self._n_map_views} map views)")
        self.root.update_idletasks()

    def _sync_scrub(self) -> None:
        self._scrubbing = True
        self.scrub.set(self.index)
        self._scrubbing = False
        self.frame_label.configure(text=f" {self.index + 1:>3} / {self.trace.n_frames}")

    def _on_scrub(self, value: str) -> None:
        """Dragging the scrubber fires continuously, and a full redraw is ~0.3 s.

        Rendering every intermediate position would queue up work faster than it
        can be done and the drag would lag behind the mouse, so only the position
        the drag settles on is drawn.
        """
        if self._scrubbing:
            return
        index = int(round(float(value)))
        if index == self.index:
            return
        self._pending_index = index
        if self._scrub_after is not None:
            self.root.after_cancel(self._scrub_after)
        self._scrub_after = self.root.after(60, self._apply_scrub)

    def _apply_scrub(self) -> None:
        self._scrub_after = None
        index, self._pending_index = self._pending_index, None
        if index is not None and index != self.index:
            self.goto(index)

    def step_forward(self) -> None:
        self.goto(self.index + 1)

    def step_back(self) -> None:
        self.goto(self.index - 1)

    def go_start(self) -> None:
        self.goto(0)

    def go_end(self) -> None:
        self.goto(self.trace.n_frames - 1)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._tick()
        elif self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _tick(self) -> None:
        if not self.playing:
            return
        if self.index >= self.trace.n_frames - 1:
            self.playing = False
            self.play_button.configure(text="Play")
            return
        self.goto(self.index + 1)
        try:
            fps = max(0.5, min(20.0, float(self.fps_var.get())))
        except ValueError:
            fps = 2.0
        self._after_id = self.root.after(int(1000 / fps), self._tick)

    # -- drawing ----------------------------------------------------------
    def redraw(self) -> None:
        trace = self.trace[self.index]
        # Pick the inliers to draw once, so the frame panel, the map panel and
        # the connecting lines all highlight the *same* correspondences.
        idx = np.flatnonzero(trace.is_inlier) if trace.is_inlier.size else np.zeros(0, int)
        self._n_inliers = len(idx)
        self._inlier_idx = idx[:: max(1, len(idx) // MAX_INLIER_MARKERS)] if len(idx) else idx
        self._draw_frame_panel(trace)
        self._draw_ref_panel(trace)
        self._draw_match_lines(trace)
        self._draw_map_panel(trace)
        self._draw_error_panel(trace)
        self._update_status(trace)
        self._update_candidates(trace)
        self.canvas.draw_idle()

    def _draw_frame_panel(self, trace) -> None:
        ax = self.ax_frame
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        image, scale = self._display(("test", trace.index), lambda: render_test_image(trace))
        if image is None:
            ax.text(0.5, 0.5, "frame unavailable", ha="center", va="center", transform=ax.transAxes)
            return
        ax.imshow(image, interpolation="bilinear")
        self._frame_scale = scale

        if self.show_kp.get() and len(trace.test_kp_xy):
            kp = trace.test_kp_xy * scale
            ax.plot(kp[:, 0], kp[:, 1], ".", ms=1.6, color=COL_KP, alpha=0.5)
        if len(self._inlier_idx):
            inl = trace.src_xy[self._inlier_idx] * scale
            ax.plot(inl[:, 0], inl[:, 1], "o", ms=3.2, mfc="none", mec=COL_IN, mew=0.9)
        h, w = image.shape[:2]
        ax.plot([w / 2], [h / 2], marker="+", ms=20, mew=2.2, color=COL_OUTLINE)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        detail = f"{len(trace.test_kp_xy)} ORB keypoints"
        if self._n_inliers:
            detail += f", {self._n_inliers} inliers"
        ax.set_title(f"Current frame  --  {trace.name}   t={trace.timestamp_sec:.0f}s\n{detail}",
                     fontsize=10)

    def _draw_ref_panel(self, trace) -> None:
        ax = self.ax_ref
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        self._ref_scale = 1.0
        if trace.best_entry is None:
            ax.set_facecolor("#f2f4f6")
            reason = trace.result.failure_reason or "no trusted match"
            ax.text(0.5, 0.55, "no trusted match", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13, color=COL_DR)
            ax.text(0.5, 0.44, reason, ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#667085")
            ax.text(0.5, 0.34, "position coasted on the motion model", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#667085")
            ax.set_xlim(0, 16)
            ax.set_ylim(9, 0)
            ax.set_aspect("equal")
            ax.set_title("Best-matching map view  --  none\n ", fontsize=10)
            return

        image, scale = self._display(
            ("ref", trace.best_entry.frame_path),
            lambda: render_ref_image(trace.best_entry, self.session.basemap))
        if image is None:
            ax.text(0.5, 0.5, "map view unavailable", ha="center", va="center", transform=ax.transAxes)
            return
        ax.imshow(image, interpolation="bilinear")
        self._ref_scale = scale

        if len(self._inlier_idx):
            inl = trace.dst_xy[self._inlier_idx] * scale
            ax.plot(inl[:, 0], inl[:, 1], "o", ms=3.2, mfc="none", mec=COL_IN, mew=0.9)

        outline = trace.projected_outline * scale
        ax.add_patch(Polygon(outline, closed=True, fill=False, ec=COL_OUTLINE, lw=2.2))
        centre = trace.projected_centre * scale
        ax.plot([centre[0]], [centre[1]], marker="+", ms=20, mew=2.2, color=COL_OUTLINE)

        # The footprint can fall partly outside the map view; show all of it.
        h, w = image.shape[:2]
        xs = np.concatenate([outline[:, 0], [0, w]])
        ys = np.concatenate([outline[:, 1], [0, h]])
        pad = 0.04 * max(np.ptp(xs), np.ptp(ys))
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.max() + pad, ys.min() - pad)
        ax.set_aspect("equal")
        if trace.used_gis:
            # The whole point of the fallback is that you can see it happen, and
            # see the actual imagery the position was measured against.
            ax.set_title(f"GIS FALLBACK  --  satellite basemap, {entry_label(trace.best_entry)}\n"
                         f"previous-flight map could not explain this frame",
                         fontsize=10, color=COL_GIS, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_edgecolor(COL_GIS)
                spine.set_linewidth(2.5)
        else:
            ax.set_title(f"Best-matching map view  --  {entry_label(trace.best_entry)}\n"
                         f"gold outline = this frame, projected through the homography",
                         fontsize=10)

    def _draw_match_lines(self, trace) -> None:
        """Join corresponding inliers across the two image panels."""
        if not self.show_lines.get() or trace.best_entry is None or not len(self._inlier_idx):
            return
        idx = self._inlier_idx
        step = max(1, len(idx) // MAX_MATCH_LINES)
        for i in idx[::step]:
            cp = ConnectionPatch(
                xyA=tuple(trace.src_xy[i] * self._frame_scale), coordsA="data", axesA=self.ax_frame,
                xyB=tuple(trace.dst_xy[i] * self._ref_scale), coordsB="data", axesB=self.ax_ref,
                color=COL_IN, lw=0.6, alpha=0.45,
            )
            cp.set_clip_on(False)
            cp.set_in_layout(False)
            self.ax_ref.add_artist(cp)

    def _draw_map_panel(self, trace) -> None:
        ax = self.ax_map
        ax.clear()
        done = self.trace.computed_frames()[: self.index + 1]

        if self.session.basemap is not None:
            extent = self._map_extent
            if extent is not None:
                # An orthomosaic is a rotated footprint on a north-up canvas, so
                # the corners are unwritten black; mask them out rather than
                # drawing a big grey box around the flight.
                ax.imshow(self._map_raster, cmap="gray", extent=extent, origin="upper",
                          alpha=0.7, zorder=0, interpolation="nearest")

        ax.plot(self._map_xy[:, 0], self._map_xy[:, 1], ".", ms=2.5, color=COL_MAP,
                alpha=0.8, zorder=1, label=f"map views ({len(self._map_xy)})")
        ax.plot(self._true_track_draw[:, 0], self._true_track_draw[:, 1], "-", lw=1.4,
                color=COL_TRUE, alpha=0.85, zorder=2, label="true GPS track")
        ax.plot(self._true_test[:, 0], self._true_test[:, 1], "o", ms=3,
                mfc="none", mec=COL_TRUE, alpha=0.35, zorder=3)

        # The route as the navigator has built it: a continuous polyline, with
        # each leg coloured by how that position was obtained.
        est = [(self._xy(t.result.estimated_latitude, t.result.estimated_longitude), t.mode)
               for t in done if t.result.has_estimate]
        if est:
            pts = np.array([p for p, _ in est])
            # Each leg is coloured by how the position it *arrives at* was
            # obtained; NaN breaks keep that to one Line2D per mode.
            for mode, colour in (("map_fix", COL_FIX), ("dead_reckon", COL_DR)):
                legs = np.full((3 * max(len(pts) - 1, 1), 2), np.nan)
                for i in range(1, len(pts)):
                    if est[i][1] == mode:
                        legs[3 * (i - 1)] = pts[i - 1]
                        legs[3 * (i - 1) + 1] = pts[i]
                ax.plot(legs[:, 0], legs[:, 1], "-", lw=1.6, zorder=4, color=colour)
            fix = np.array([p for p, m in est if m == "map_fix"]) if any(m == "map_fix" for _, m in est) else None
            dr = np.array([p for p, m in est if m != "map_fix"]) if any(m != "map_fix" for _, m in est) else None
            if fix is not None:
                ax.plot(fix[:, 0], fix[:, 1], "x", ms=6, mew=1.4, color=COL_FIX, zorder=5,
                        label="visual map fix")
            if dr is not None:
                ax.plot(dr[:, 0], dr[:, 1], "^", ms=5, color=COL_DR, zorder=5,
                        label="dead-reckoned")

        tx, ty = self._xy(trace.true_latitude, trace.true_longitude)
        ax.plot([tx], [ty], "o", ms=8, color=COL_TRUE, zorder=7, label="true position now")
        if trace.result.has_estimate:
            ex, ey = self._xy(trace.result.estimated_latitude, trace.result.estimated_longitude)
            ax.plot([tx, ex], [ty, ey], "-", lw=1.2, color="#5c6470", alpha=0.9, zorder=7)
            colour = (COL_GIS if trace.used_gis
                      else (COL_FIX if trace.mode == "map_fix" else COL_DR))
            ax.plot([ex], [ey], "*", ms=15, zorder=8, color=colour,
                    label="estimate now (GIS)" if trace.used_gis else "estimate now")

        # Limits are fixed *before* the gate circle is added, so an inflated gate
        # is simply clipped instead of rescaling the whole panel.
        ax.set_aspect("equal", adjustable="box")
        if self.follow.get():
            span = max(120.0, (trace.gate_radius_m or 0) * 1.6)
            ax.set_xlim(tx - span, tx + span)
            ax.set_ylim(ty - span, ty + span)
        else:
            ax.set_xlim(self._map_bounds[0], self._map_bounds[1])
            ax.set_ylim(self._map_bounds[2], self._map_bounds[3])

        if trace.predicted_latlon is not None:
            px, py = self._xy(*trace.predicted_latlon)
            ax.plot([px], [py], marker="+", ms=11, mew=1.6, color=COL_GATE, zorder=6,
                    label="motion prediction")
            if trace.gate_radius_m:
                view_half = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2
                offscreen = trace.gate_radius_m > view_half * 1.6
                ax.add_patch(Circle((px, py), trace.gate_radius_m, fill=False, ls="--",
                                    lw=1.1, ec=COL_GATE, alpha=0.8, zorder=6, clip_on=True,
                                    label=f"search gate {trace.gate_radius_m:.0f} m"
                                          + (" (off view)" if offscreen else "")))

        ax.grid(True, alpha=0.25)
        ax.set_xlabel("East (m)", fontsize=9)
        ax.set_ylabel("North (m)", fontsize=9)
        ax.set_title("Route built so far", fontsize=10)
        # Equal aspect makes this panel square, which leaves slack under it in a
        # tall column -- so the legend goes there rather than over the route.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
                  fontsize=7, framealpha=0.9, borderaxespad=0.0)

    def _basemap_extent(self, basemap) -> Optional[tuple]:
        """Local-metre extent of a raster map, for drawing it under the route."""
        try:
            corners = [basemap.pixel_to_latlon(c, r)
                       for r, c in ((0, 0), (0, basemap.width), (basemap.height, 0),
                                    (basemap.height, basemap.width))]
        except Exception:
            return None
        xy = np.array([self._xy(lat, lon) for lat, lon in corners])
        return xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()

    def _draw_error_panel(self, trace) -> None:
        ax = self.ax_err
        ax.clear()
        done = self.trace.computed_frames()[: self.index + 1]
        errs = [t.error_m if t.error_m is not None else 0.0 for t in done]
        colors = [(COL_GIS if t.used_gis else COL_FIX) if t.mode == "map_fix" else COL_DR
                  for t in done]
        if errs:
            ax.bar(range(len(errs)), errs, color=colors, width=0.75)
            ax.bar([self.index], [errs[self.index]], color="none", edgecolor="#111827",
                   lw=1.4, width=0.75)
            ax.set_ylim(0, max(max(errs) * 1.45, 1.0))   # headroom for the legend
            fixes = sorted(t.error_m for t in done if t.mode == "map_fix" and t.error_m is not None)
            if fixes:
                median = fixes[len(fixes) // 2]
                ax.axhline(median, ls=":", lw=1.1, color="#374151")
                ax.text(0.995, 0.9, f"median fix {median:.1f} m", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, color="#374151")
        ax.set_xlim(-0.6, self.trace.n_frames - 0.4)
        ax.set_ylabel("error (m)", fontsize=9)
        ax.set_xlabel("test frame", fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title("Position error vs. GPS ground truth", fontsize=10)
        handles = [Line2D([], [], color=COL_FIX, lw=6, label="fix from flight video"),
                   Line2D([], [], color=COL_DR, lw=6, label="dead-reckoned")]
        if any(t.used_gis for t in done):
            handles.insert(1, Line2D([], [], color=COL_GIS, lw=6, label="fix from GIS"))
        ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.85)

    def _update_status(self, trace) -> None:
        done = self.trace.computed_frames()[: self.index + 1]
        n_fix = sum(1 for t in done if t.mode == "map_fix")
        n_dr = sum(1 for t in done if t.mode == "dead_reckon")
        fixes = sorted(t.error_m for t in done if t.mode == "map_fix" and t.error_m is not None)
        median = f"{fixes[len(fixes) // 2]:.1f} m" if fixes else "--"

        if trace.mode == "map_fix":
            if trace.gate_radius_m is None:
                how = "global search"       # --no-motion: there is no gate to be inside
            else:
                how = "re-acquired" if trace.reacquisition else "in gate"
            verdict = (f"MAP FIX   good {trace.result.num_good_matches:>4}  "
                       f"inliers {trace.result.num_inliers:>4}  ratio {trace.result.inlier_ratio:.2f}"
                       f"   {how}")
        elif trace.mode == "dead_reckon":
            verdict = f"DEAD-RECKON  coast #{trace.result.coast_steps}  ({trace.result.failure_reason})"
        else:
            verdict = f"FAILED  ({trace.result.failure_reason})"

        error = f"{trace.error_m:.2f} m" if trace.error_m is not None else "--"
        gate = f"{trace.gate_radius_m:.0f} m" if trace.gate_radius_m else "off"
        alt = f"{trace.altitude:.0f} m" if trace.altitude is not None else "?"
        # What the decision cost, against the 1 Hz frame budget it has to fit in.
        cost = ""
        if trace.compute_ms:
            cost = f"   {trace.compute_ms:>5.0f} ms ({1000.0 / trace.compute_ms:.1f}x real time)"
        origin = ""
        if trace.used_gis:
            origin = "  << GIS FALLBACK (satellite) >>"
        elif trace.matched_source and len(self.session.source_names) > 1:
            origin = "   via previous flight video"
        self.status.configure(
            text=f"  t={trace.timestamp_sec:>6.1f}s  alt {alt:>5}   {verdict}{origin}   "
                 f"error {error:>9}   gate {gate:>6} over {trace.n_searched}/{self._n_map_views} views"
                 f"{cost}   |   so far: {n_fix} fixes, {n_dr} coasted, median {median}")

    def _update_candidates(self, trace) -> None:
        self.tree.delete(*self.tree.get_children())
        self._candidate_by_iid = {}
        for cand in trace.candidates:
            if cand.is_best:
                tags = ("gis_best",) if cand.map_label == "satellite" else ("best",)
            else:
                tags = ("rejected",) if cand.n_inliers == 0 else ()
            iid = self.tree.insert("", "end", values=(
                cand.map_label, cand.label, cand.n_good, cand.n_inliers,
                cand.short_verdict), tags=tags)
            self._candidate_by_iid[iid] = cand
        self._set_verdict(self._frame_explanation(trace))

    def _frame_explanation(self, trace) -> str:
        gate = (f"The motion model predicted a position and allowed a search radius of "
                f"{trace.gate_radius_m:.0f} m, which held {trace.n_in_gate} of "
                f"{self._n_map_views} map views"
                if trace.gate_radius_m else
                "Motion gating is off, so every map view was searched")
        if trace.n_searched != trace.n_in_gate and trace.gate_radius_m:
            gate += f" ({trace.n_searched} after the temporal exclusion)"
        gate += "."
        if trace.reacquisition:
            gate += " The gate came up empty, so it re-acquired globally under a stricter bar."

        if trace.mode == "map_fix" and trace.gis_was_fallback:
            body = (f"\n\nNothing in the previous flight's video explained this frame, so it "
                    f"fell back to the GIS map and matched the satellite tile "
                    f"{entry_label(trace.best_entry)} with {trace.result.num_good_matches} good "
                    f"matches, {trace.result.num_inliers} geometrically consistent "
                    f"({trace.result.inlier_ratio:.0%}). That tile is the image shown above, and "
                    f"the position was measured from it -- {trace.error_m:.1f} m from the true GPS."
                    f"\n\nThis is the case the GIS map exists for: ground the drone has no "
                    f"previous footage of.")
        elif trace.mode == "map_fix":
            body = (f"\n\nIt matched {entry_label(trace.best_entry)} with "
                    f"{trace.result.num_good_matches} good matches, "
                    f"{trace.result.num_inliers} of them geometrically consistent "
                    f"({trace.result.inlier_ratio:.0%}). The frame centre projected through that "
                    f"homography gives the estimate, {trace.error_m:.1f} m from the true GPS.")
        elif trace.mode == "dead_reckon":
            body = (f"\n\nNothing cleared the acceptance bar ({trace.result.failure_reason}), so the "
                    f"position was coasted on the motion model -- consecutive coast #"
                    f"{trace.result.coast_steps}, now {trace.error_m:.1f} m off.")
        else:
            body = f"\n\nNo position at all: {trace.result.failure_reason}."
        return gate + body

    def _set_verdict(self, text: str) -> None:
        self.verdict.configure(state="normal")
        self.verdict.delete("1.0", tk.END)
        self.verdict.insert("1.0", text)
        self.verdict.configure(state="disabled")

    def _on_candidate_select(self, _event) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        cand = getattr(self, "_candidate_by_iid", {}).get(selection[0])
        if cand is None:
            return
        self._set_verdict(
            f"{cand.label}\n\n{cand.n_good} good matches, {cand.n_inliers} RANSAC inliers "
            f"({cand.inlier_ratio:.0%}).\n\nVerdict: {cand.verdict}."
            + ("\n\nThis is the match the position was computed from." if cand.is_best else ""))
