"""Real-time motion model + spatial gating for the navigation stage.

The single biggest error source in the pure image-matching MVP was *perceptual
aliasing*: a test frame matching a visually-similar reference frame from a
completely different part of the flight (see the old error report -- e.g. a
frame matching one 60s away, giving 200m+ error). A drone can't teleport, so a
cheap physical prior fixes most of this: track the drone's position with a
constant-velocity model and only consider reference frames whose known position
is physically reachable from the current estimate.

Crucially this uses *no GNSS from the test frames* -- it is seeded from the
reference track (which had GPS during preprocessing) and thereafter propagates
using only its own previous visual estimates plus elapsed time. That keeps the
GPS-denied evaluation honest and matches the real-time navigation problem: the
"predefined data contains the current position" (the assignment's wording), and
from there the drone dead-reckons + visually corrects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import utm

from .geo import approx_distance_m


@dataclass
class MotionState:
    """Tracks the drone's estimated position + planar velocity over time.

    Velocity is stored in UTM East/North meters-per-second, smoothed with an
    EMA so a single noisy estimate doesn't whip the prediction around.
    """

    latitude: float
    longitude: float
    timestamp_sec: float
    vel_east_mps: float = 0.0
    vel_north_mps: float = 0.0
    # How many consecutive frames we've failed to confidently localize. Used to
    # inflate the gating radius so we can re-acquire after a dropout instead of
    # staying locked to a stale prediction.
    misses: int = 0

    # Tunables. max_speed_mps caps plausible drone ground speed -- kept a little
    # above a Mini 3 Pro's ~19 m/s spec (the raw GPS shows brief 40 m/s spikes,
    # but those are jitter; gating to them just re-admits aliasing). Lowering
    # this from an over-generous value was what actually let the gate reject the
    # far-jump outliers. base_margin_m absorbs per-frame localization error.
    max_speed_mps: float = 22.0
    base_margin_m: float = 30.0
    vel_smoothing: float = 0.5   # EMA weight on the newest velocity sample
    max_miss_inflation: int = 2  # cap gate widening after a run of dropouts

    def predict(self, timestamp_sec: float) -> tuple[float, float]:
        """Constant-velocity dead-reckoned (lat, lon) at `timestamp_sec`."""
        dt = timestamp_sec - self.timestamp_sec
        easting, northing, zone_number, zone_letter = utm.from_latlon(self.latitude, self.longitude)
        pred_e = easting + self.vel_east_mps * dt
        pred_n = northing + self.vel_north_mps * dt
        return utm.to_latlon(pred_e, pred_n, zone_number, zone_letter)

    def gating_radius_m(self, timestamp_sec: float) -> float:
        """Radius around the prediction within which a reference frame is
        considered physically reachable. Grows with elapsed time and, after
        missed frames, widens further so we can re-acquire.
        """
        dt = max(timestamp_sec - self.timestamp_sec, 0.0)
        reach = self.max_speed_mps * dt + self.base_margin_m
        return reach * (1 + min(self.misses, self.max_miss_inflation))  # inflate after dropouts, capped

    def is_plausible(self, lat: float, lon: float, timestamp_sec: float) -> bool:
        """Is a candidate position reachable from the current estimate in the
        elapsed time (within the gating radius)?
        """
        pred_lat, pred_lon = self.predict(timestamp_sec)
        return approx_distance_m(pred_lat, pred_lon, lat, lon) <= self.gating_radius_m(timestamp_sec)

    def is_output_plausible(self, lat: float, lon: float, timestamp_sec: float) -> bool:
        """Acceptance ('innovation') gate: could the drone actually be at this
        newly-estimated position, given the last *confirmed* fix and its max
        speed? Unlike the search gate this uses the un-inflated reach from the
        last accepted fix, so a match found in a widened post-dropout search
        still has to be physically reachable to be believed -- this is what
        rejects a high-inlier-but-wrong match after a run of missed frames.
        """
        dt = max(timestamp_sec - self.timestamp_sec, 0.0)
        reach = self.max_speed_mps * dt + self.base_margin_m
        return approx_distance_m(self.latitude, self.longitude, lat, lon) <= reach

    def update(self, lat: float, lon: float, timestamp_sec: float) -> None:
        """Fold a newly-accepted position estimate into the state, refreshing
        the smoothed velocity.
        """
        dt = timestamp_sec - self.timestamp_sec
        if dt > 0:
            e0, n0, zn, zl = utm.from_latlon(self.latitude, self.longitude)
            e1, n1, _, _ = utm.from_latlon(lat, lon, force_zone_number=zn, force_zone_letter=zl)
            new_ve = (e1 - e0) / dt
            new_vn = (n1 - n0) / dt
            a = self.vel_smoothing
            self.vel_east_mps = a * new_ve + (1 - a) * self.vel_east_mps
            self.vel_north_mps = a * new_vn + (1 - a) * self.vel_north_mps
        self.latitude, self.longitude, self.timestamp_sec = lat, lon, timestamp_sec
        self.misses = 0

    def mark_miss(self) -> None:
        """Record a frame we couldn't confidently localize (coast on the
        prediction; widen the next gate)."""
        self.misses += 1

    def dead_reckon_report(self, timestamp_sec: float, extrapolate_cap_sec: float = 2.0) -> tuple[float, float]:
        """A best-effort reported position when no visual fix is trustworthy --
        rely on the drone's motion instead of rejecting the frame.

        Crucially this does NOT mutate the state (the matching gate stays
        anchored to the last *confirmed* fix, so a dropout doesn't drift the
        search off the real location) and it **caps** how long the last velocity
        is extrapolated. Without an IMU the only motion cue is the velocity from
        the last visual fix; extrapolating it indefinitely diverges the moment
        the drone hovers or turns (measured here: uncapped constant-velocity
        coasting ran to 500 m over a long gap). Capping to a couple of seconds
        means: extrapolate briefly, then hold near the last fix -- a far safer
        prior for this hover-heavy flight. Dead reckoning is thus a short-gap
        bridge between fixes, not a substitute for them.
        """
        dt = max(timestamp_sec - self.timestamp_sec, 0.0)
        eff = min(dt, extrapolate_cap_sec)
        easting, northing, zone_number, zone_letter = utm.from_latlon(self.latitude, self.longitude)
        return utm.to_latlon(
            easting + self.vel_east_mps * eff, northing + self.vel_north_mps * eff, zone_number, zone_letter
        )


def seed_from_reference_track(
    reference_index: list, start_time_sec: Optional[float] = None
) -> Optional[MotionState]:
    """Initialise a MotionState from the reference track.

    The reference set had GPS during preprocessing, so a known position +
    velocity from it is a legitimate, GNSS-free starting point for navigating
    the test frames -- this is the assignment's "the predefined data contains
    the current position".

    `start_time_sec` is the timestamp of the first test frame: we seed from the
    last reference view recorded *before* it. With the chronological tail split
    that is simply the final reference frame; with an interleaved hold-out it is
    the map frame immediately preceding the first test frame, which is the one
    the drone would actually have been over.
    """
    if not reference_index:
        return None
    ordered = sorted(reference_index, key=lambda e: e.timestamp_sec)
    if start_time_sec is not None:
        earlier = [e for e in ordered if e.timestamp_sec <= start_time_sec]
        ordered = earlier if len(earlier) >= 1 else ordered[:1]
    last = ordered[-1]
    reference_index = ordered
    state = MotionState(latitude=last.latitude, longitude=last.longitude, timestamp_sec=last.timestamp_sec)
    # Estimate initial velocity from the last two reference positions if we can.
    for prev in reversed(reference_index[:-1]):
        dt = last.timestamp_sec - prev.timestamp_sec
        if dt > 0:
            e0, n0, zn, zl = utm.from_latlon(prev.latitude, prev.longitude)
            e1, n1, _, _ = utm.from_latlon(last.latitude, last.longitude, force_zone_number=zn, force_zone_letter=zl)
            state.vel_east_mps = (e1 - e0) / dt
            state.vel_north_mps = (n1 - n0) / dt
            break
    return state
