"""ROI helpers and PoseEstimator protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from helmet_action.inference.tracker import HumanTrack
from helmet_action.pose.types import PoseObservation


def expand_bbox(
    bbox: tuple[float, float, float, float],
    *,
    margin: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """Expand bbox by margin fraction and clamp to image bounds. Returns int xyxy."""
    x1, y1, x2, y2 = bbox
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)
    mx = w * float(margin)
    my = h * float(margin)
    nx1 = int(np.floor(x1 - mx))
    ny1 = int(np.floor(y1 - my))
    nx2 = int(np.ceil(x2 + mx))
    ny2 = int(np.ceil(y2 + my))
    nx1 = max(0, min(nx1, max(frame_w - 1, 0)))
    ny1 = max(0, min(ny1, max(frame_h - 1, 0)))
    nx2 = max(nx1 + 1, min(nx2, frame_w))
    ny2 = max(ny1 + 1, min(ny2, frame_h))
    return nx1, ny1, nx2, ny2


def crop_roi(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    margin: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = int(frame.shape[0]), int(frame.shape[1])
    xyxy = expand_bbox(bbox, margin=margin, frame_w=w, frame_h=h)
    x1, y1, x2, y2 = xyxy
    roi = frame[y1:y2, x1:x2].copy()
    return roi, xyxy


def local_to_global_keypoints(
    keypoints: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    """Map ROI-local keypoint xy to original-frame coordinates."""
    x1, y1, _, _ = roi_xyxy
    out = np.asarray(keypoints, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] < 2:
        raise ValueError(f"keypoints must be (N,2+), got {out.shape}")
    out[:, 0] = out[:, 0] + float(x1)
    out[:, 1] = out[:, 1] + float(y1)
    return out


class PoseEstimator(Protocol):
    def estimate(
        self,
        roi: np.ndarray,
        track: HumanTrack,
        frame_index: int,
        timestamp: float,
    ) -> PoseObservation | None:
        ...
