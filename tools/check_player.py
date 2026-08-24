"""Check that the interactive player shows the run the report measured.

    python tools/check_player.py --use-cached-frames

`nav_player.py` is only worth anything as evidence if what it draws is the same
navigation `run_pipeline.py` scores. That is meant to be true by construction --
`src/trace.py` drives `localize_stream`, the identical generator `localize_all`
collects -- but "by construction" is a claim, so this checks it: it runs the
batch pipeline and the traced one over the same session and asserts they agree
frame for frame, then confirms every drawable the player needs is present and
lines up with the image it will be drawn on.

Headless: no window is opened, so it is safe to run over SSH or in CI.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.localize import localize_all                    # noqa: E402
from src.reference_source import as_reference_source     # noqa: E402
from src.session import add_pipeline_args, prepare_session  # noqa: E402
from src.trace import NavigationTrace, render_ref_image, render_test_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_pipeline_args(parser)
    args = parser.parse_args()

    session = prepare_session(args, log=lambda msg: None)
    # A hybrid map is a CompositeReferenceSource, not a list -- ask it.
    n_views = len(as_reference_source(session.reference_index).all())
    print(f"session: {len(session.test_frames)} test frames against "
          f"{n_views} map views ({session.map_source}, "
          f"{'+'.join(sorted(session.source_names))})")

    batch = localize_all(session.test_frames, session.reference_index, **session.localize_kwargs)
    trace = NavigationTrace(session)
    trace.ensure(trace.n_frames - 1)
    assert trace.n_computed == len(batch), \
        f"traced {trace.n_computed} frames but the pipeline produced {len(batch)}"

    for i, (expected, ft) in enumerate(zip(batch, trace.computed_frames())):
        where = f"frame {i} ({ft.name})"
        got = ft.result
        assert got.mode == expected.mode, f"{where}: mode {got.mode} != {expected.mode}"
        assert got.num_inliers == expected.num_inliers, f"{where}: inlier count differs"
        assert got.matched_ref_frame == expected.matched_ref_frame, f"{where}: matched a different view"
        assert got.matched_source == expected.matched_source, f"{where}: matched a different map"
        assert got.estimated_latitude == expected.estimated_latitude, f"{where}: position differs"
        assert got.estimated_longitude == expected.estimated_longitude, f"{where}: position differs"

        # Everything the player draws must exist and be in the right frame of
        # reference -- keypoints are useless if they don't land on their image.
        image = render_test_image(ft)
        assert image is not None, f"{where}: current-frame image could not be rebuilt"
        assert image.shape[:2] == ft.image_shape, \
            f"{where}: display image {image.shape[:2]} != matched image {ft.image_shape}"
        if len(ft.test_kp_xy):
            h, w = ft.image_shape
            assert ft.test_kp_xy[:, 0].max() <= w and ft.test_kp_xy[:, 1].max() <= h, \
                f"{where}: keypoints fall outside the image they were computed from"
        if got.mode == "map_fix":
            assert ft.best_entry is not None and ft.homography is not None, f"{where}: no winning match recorded"
            assert ft.projected_outline is not None and ft.projected_outline.shape == (4, 2), \
                f"{where}: no projected footprint to draw"
            assert ft.src_xy.shape[0] == ft.dst_xy.shape[0] == ft.is_inlier.shape[0] == expected.num_good_matches, \
                f"{where}: correspondence arrays disagree with the reported match count"
            assert ft.n_inliers == expected.num_inliers, f"{where}: inlier mask disagrees with the count"
            ref = render_ref_image(ft.best_entry, session.basemap)
            assert ref is not None, f"{where}: matched map view could not be rebuilt"
            assert ref.shape[:2] == (ft.best_entry.image_height, ft.best_entry.image_width), \
                f"{where}: map view {ref.shape[:2]} is not the size it was indexed at"
        assert ft.candidates, f"{where}: no candidate list to explain the decision"

    n_fix = sum(1 for r in batch if r.mode == "map_fix")
    n_dr = sum(1 for r in batch if r.mode == "dead_reckon")
    print(f"OK: player trace matches the pipeline on all {len(batch)} frames "
          f"({n_fix} visual fixes, {n_dr} dead-reckoned).")


if __name__ == "__main__":
    main()
