"""A scratch directory for one run's intermediate files, erased afterwards.

Extracting frames from a flight video produces a few hundred JPEGs -- ~170 MB
for the flights here. The batch pipeline keeps them under `data/frames/` on
purpose, because re-running an experiment should not re-decode a multi-gigabyte
video every time.

The interactive app wants the opposite: it should be able to open any video the
user points it at, leave nothing behind, and never depend on something a previous
run happened to leave lying around. A `Workspace` is that -- a temporary
directory that is deleted when the run ends, however it ends.

    with Workspace() as ws:
        session = prepare_session(request.to_args(ws.frames_dir))
        ...
    # frames are gone here

`close()` is idempotent and never raises: failing to clean up a scratch directory
should not be able to take down the application on its way out.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import sys
import tempfile


def clean_up_on_termination() -> None:
    """Make a `kill` behave like a window close, as far as scratch space goes.

    `atexit` runs on a normal exit and on an unhandled exception, but not when
    the process is terminated -- so a killed app would leave a few hundred
    megabytes of frames behind. Turning the signal into a `SystemExit` puts the
    normal teardown back in charge. Any handler already installed is left alone.
    """
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            if signal.getsignal(sig) in (signal.SIG_DFL, None):
                signal.signal(sig, terminate)
        except (ValueError, OSError):
            pass          # not the main thread, or the platform lacks the signal


class Workspace:
    """A self-deleting directory for the frames extracted from one video."""

    def __init__(self, prefix: str = "visual-nav-") -> None:
        self._path: str | None = tempfile.mkdtemp(prefix=prefix)
        os.makedirs(self.frames_dir, exist_ok=True)
        # A backstop for the paths that skip __exit__ entirely -- an unhandled
        # exception, or the window manager killing the process.
        atexit.register(self.close)

    @property
    def path(self) -> str:
        if self._path is None:
            raise RuntimeError("this workspace has already been closed")
        return self._path

    @property
    def frames_dir(self) -> str:
        """Where extracted frames go."""
        return os.path.join(self.path, "frames")

    @property
    def is_open(self) -> bool:
        return self._path is not None

    def size_bytes(self) -> int:
        """How much disk this workspace is currently holding."""
        if self._path is None:
            return 0
        total = 0
        for root, _dirs, files in os.walk(self._path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        return total

    def close(self) -> None:
        path, self._path = self._path, None
        if path is None:
            return
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception as exc:            # pragma: no cover - defensive
            print(f"[workspace] could not remove {path}: {exc}", file=sys.stderr)

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
