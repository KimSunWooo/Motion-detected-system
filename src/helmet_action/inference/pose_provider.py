from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

from helmet_action.pose.types import PoseObservation


class PoseProvider(ABC):
    """Video frame → zero or more PoseObservation (one per tracked person)."""

    @abstractmethod
    def iter_frames(self, source: str | int) -> Iterator[tuple[np.ndarray, list[PoseObservation]]]:
        raise NotImplementedError


class NumpySequenceProvider(PoseProvider):
    """Replay a (T,17,2) array as a single track — used by tests and synthetic demo."""

    def __init__(
        self,
        sequence: np.ndarray,
        confidence: np.ndarray | None = None,
        track_id: int = 1,
        fps: float = 20.0,
        frame_shape: tuple[int, int] = (720, 960),
    ) -> None:
        self.sequence = np.asarray(sequence, dtype=np.float64)[..., :2]
        t = self.sequence.shape[0]
        self.confidence = (
            np.ones((t, 17), dtype=np.float64)
            if confidence is None
            else np.asarray(confidence, dtype=np.float64)
        )
        self.track_id = track_id
        self.fps = fps
        self.frame_shape = frame_shape

    def iter_frames(self, source: str | int = 0) -> Iterator[tuple[np.ndarray, list[PoseObservation]]]:
        h, w = self.frame_shape
        for i, kpts in enumerate(self.sequence):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            xs = kpts[:, 0]
            ys = kpts[:, 1]
            finite = np.isfinite(xs) & np.isfinite(ys)
            if finite.any():
                bbox = (
                    float(xs[finite].min() - 8),
                    float(ys[finite].min() - 8),
                    float(xs[finite].max() + 8),
                    float(ys[finite].max() + 8),
                )
            else:
                bbox = (0.0, 0.0, 1.0, 1.0)
            obs = PoseObservation(
                timestamp=i / self.fps,
                frame_index=i,
                track_id=self.track_id,
                bbox=bbox,
                keypoints=kpts,
                keypoint_confidence=self.confidence[i],
                detection_confidence=1.0,
                source_fps=self.fps,
            )
            yield frame, [obs]
