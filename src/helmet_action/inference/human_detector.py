"""Human detection abstractions (person class only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class HumanDetection:
    """Person detection in original-frame coordinates."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0


class HumanDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[HumanDetection]:
        """Return person detections for the given frame (any resolution)."""
        ...


def rescale_bbox(
    bbox: tuple[float, float, float, float],
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


def downscale_frame(
    frame: np.ndarray,
    imgsz: int,
) -> tuple[np.ndarray, float, float]:
    """Resize so the longer side is ``imgsz``. Returns (resized, scale_x, scale_y)
    where scales map detection coords → original coords.
    """
    import cv2

    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h <= 0 or w <= 0:
        return frame, 1.0, 1.0
    target = max(int(imgsz), 32)
    long_side = max(h, w)
    if long_side <= target:
        return frame, 1.0, 1.0
    scale = target / float(long_side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    scale_x = w / float(nw)
    scale_y = h / float(nh)
    return resized, scale_x, scale_y
