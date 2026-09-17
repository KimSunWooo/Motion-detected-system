"""Frame capture helpers: rotation, FPS sanitization, latest-frame queue."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


ROTATE_CHOICES = (0, 90, 180, 270)


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate BGR frame before YOLO / viz / VideoWriter. degrees in {0,90,180,270}."""
    import cv2

    deg = int(degrees) % 360
    if deg == 0:
        return frame
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported rotate degrees: {degrees}")


def sanitize_fps(raw: float | None, fallback: float = 20.0) -> float:
    """Use source FPS when sane; otherwise fallback (default 20)."""
    try:
        fps = float(raw) if raw is not None else float("nan")
    except (TypeError, ValueError):
        return float(fallback)
    if not np.isfinite(fps) or fps <= 1.0 or fps > 240.0:
        return float(fallback)
    return float(fps)


@dataclass
class CapturedFrame:
    frame: np.ndarray
    index: int
    timestamp: float


class LatestFrameBuffer:
    """Keep at most one frame. New captures overwrite unread frames (drop)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: CapturedFrame | None = None
        self._dropped = 0
        self._closed = False

    @property
    def dropped(self) -> int:
        with self._lock:
            return int(self._dropped)

    def close(self) -> None:
        """Hard close: drop buffered frame (used on capture shutdown)."""
        with self._lock:
            self._closed = True
            self._item = None

    def mark_ended(self) -> None:
        """Soft EOF: stop accepting frames but keep last unread frame for drain."""
        with self._lock:
            self._closed = True

    def put(self, frame: np.ndarray, index: int, timestamp: float) -> None:
        with self._lock:
            if self._closed:
                return
            if self._item is not None:
                self._dropped += 1
            self._item = CapturedFrame(frame=frame, index=index, timestamp=timestamp)

    def get(self, timeout: float = 0.5) -> CapturedFrame | None:
        deadline = time.monotonic() + max(timeout, 0.0)
        while True:
            with self._lock:
                if self._closed and self._item is None:
                    return None
                if self._item is not None:
                    item = self._item
                    self._item = None
                    return item
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.005)

    def depth(self) -> int:
        with self._lock:
            return 0 if self._item is None else 1


class LatestFrameCapture:
    """Alias / thin wrapper: always keeps at most one frame (live sources)."""

    def __init__(self, source: str | int, *, rotate: int = 0) -> None:
        self._inner = FrameCapture(source, rotate=rotate, latest_frame=True)

    @property
    def latest_frame(self):
        return self._inner._buf._item

    @property
    def latest_frame_index(self) -> int | None:
        item = self._inner._buf._item
        return None if item is None else int(item.index)

    @property
    def latest_timestamp(self) -> float | None:
        item = self._inner._buf._item
        return None if item is None else float(item.timestamp)

    def start(self) -> None:
        self._inner.start()

    def read(self):
        return self._inner.read()

    def dropped_frames(self) -> int:
        return self._inner.dropped_frames()

    def buffer_depth(self) -> int:
        return self._inner.buffer_depth()

    def close(self) -> None:
        self._inner.close()

    def __enter__(self) -> "LatestFrameCapture":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def source_fps(self) -> float:
        return self._inner.source_fps

    @property
    def raw_fps_metadata(self) -> float:
        return self._inner.raw_fps_metadata


class FrameCapture:
    """OpenCV capture with optional latest-frame thread and rotation."""

    def __init__(
        self,
        source: str | int,
        *,
        rotate: int = 0,
        latest_frame: bool = False,
    ) -> None:
        import cv2

        self.source = source
        self.rotate = int(rotate)
        self.latest_frame = bool(latest_frame)
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open video source: {source!r}")
        raw_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.source_fps = sanitize_fps(raw_fps, fallback=20.0)
        self.raw_fps_metadata = raw_fps
        self._buf = LatestFrameBuffer()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._capture_index = 0
        self._error: str | None = None

    def start(self) -> None:
        if not self.latest_frame:
            return
        self._thread = threading.Thread(target=self._capture_loop, name="latest-frame-capture", daemon=True)
        self._thread.start()

    def _read_one(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        if self.rotate:
            frame = rotate_frame(frame, self.rotate)
        return True, frame

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                ok, frame = self._read_one()
                if not ok or frame is None:
                    self._buf.mark_ended()
                    break
                self._buf.put(frame, self._capture_index, time.monotonic())
                self._capture_index += 1
        except Exception as exc:  # pragma: no cover — surfaced via error
            self._error = str(exc)
            self._buf.mark_ended()

    def read(self) -> CapturedFrame | None:
        if self.latest_frame:
            return self._buf.get(timeout=0.5)
        ok, frame = self._read_one()
        if not ok or frame is None:
            return None
        item = CapturedFrame(frame=frame, index=self._capture_index, timestamp=time.monotonic())
        self._capture_index += 1
        return item

    def dropped_frames(self) -> int:
        return self._buf.dropped if self.latest_frame else 0

    def buffer_depth(self) -> int:
        return self._buf.depth() if self.latest_frame else 0

    def close(self) -> None:
        self._stop.set()
        self._buf.close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameCapture":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
