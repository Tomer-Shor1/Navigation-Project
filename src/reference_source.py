"""Reference-map access behind a radius query -- the seam for not holding the
whole map in memory.

The navigator never needs *all* reference views at once: given where it thinks
it is, it only needs the views within a radius of interest. Expressing map
access as `query(lat, lon, radius)` means:

* the same interface serves the current in-memory same-flight index and a future
  tiled / Google-Earth backend that lazily loads (and evicts) only the tiles
  near the drone -- constant memory regardless of map size;
* matching cost stays bounded by the local neighborhood, not the whole map,
  which is what makes it real-time as the map grows.

`InMemoryReferenceSource` is the concrete implementation used today.
`CompositeReferenceSource` puts several maps behind that one query, which is how
the final project uses a satellite basemap *and* previously-flown video at the
same time. A `TiledMapReferenceSource` (load georeferenced tiles from
disk/network within the radius, cache by tile id, drop far ones) would implement
the same two methods and nothing downstream would change.
"""

from __future__ import annotations

from typing import Protocol

from .build_reference import ReferenceEntry
from .geo import approx_distance_m


class ReferenceSource(Protocol):
    """Anything the localizer can pull reference views from."""

    def query(self, latitude: float, longitude: float, radius_m: float) -> list[ReferenceEntry]:
        """Reference views whose position is within `radius_m` of (lat, lon)."""
        ...

    def all(self) -> list[ReferenceEntry]:
        """Every reference view (used only for global re-acquisition)."""
        ...


class InMemoryReferenceSource:
    """Holds the full reference index in memory and answers radius queries by a
    linear scan. Fine for the same-flight MVP (tens-to-hundreds of frames); a
    tiled backend would replace the scan with a spatial index + lazy loading.
    """

    def __init__(self, entries: list[ReferenceEntry]):
        self._entries = entries

    def query(self, latitude: float, longitude: float, radius_m: float) -> list[ReferenceEntry]:
        return [
            e for e in self._entries
            if approx_distance_m(latitude, longitude, e.latitude, e.longitude) <= radius_m
        ]

    def all(self) -> list[ReferenceEntry]:
        return self._entries


class CompositeReferenceSource:
    """Several maps answering one radius query -- the project brief's "and".

    A satellite basemap and a previously-flown video are complementary, not
    alternatives. The raster covers everything, including ground this drone has
    never flown, and its scale and heading are exact. The flight frames cover
    only the old corridor, but where they exist they are the *same camera on the
    same scene*, so they match far more strongly than cross-sensor satellite
    imagery does.

    Merging them here rather than choosing between them upstream means the
    localizer needs no new logic: it already verifies every candidate by RANSAC
    inliers and keeps the best, so it simply picks whichever map explains this
    frame better -- per frame, not per flight. `ReferenceEntry.source` records
    which one won, so the split can be reported afterwards.
    """

    def __init__(self, sources: list):
        assert sources, "Sanity check failed: a composite needs at least one source."
        self._sources = [as_reference_source(s) for s in sources]

    def query(self, latitude: float, longitude: float, radius_m: float) -> list[ReferenceEntry]:
        found: list[ReferenceEntry] = []
        for source in self._sources:
            found.extend(source.query(latitude, longitude, radius_m))
        return found

    def all(self) -> list[ReferenceEntry]:
        every: list[ReferenceEntry] = []
        for source in self._sources:
            every.extend(source.all())
        return every


def as_reference_source(reference: object) -> ReferenceSource:
    """Accept either a ReferenceSource or a plain list of entries (wrapping the
    latter), so callers can keep passing a list.
    """
    if isinstance(reference, list):
        return InMemoryReferenceSource(reference)
    return reference  # type: ignore[return-value]
