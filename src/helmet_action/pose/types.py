from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class KeypointStatus(str, Enum):
    VALID = "VALID"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MISSING = "MISSING"


class PoseQuality(str, Enum):
    OK = "OK"
    INSUFFICIENT_POSE = "INSUFFICIENT_POSE"


@dataclass
class PoseObservation:
    timestamp: float
    frame_index: int
    track_id: int
    bbox: tuple[float, float, float, float]
    keypoints: np.ndarray
    keypoint_confidence: np.ndarray
    detection_confidence: float
    source_fps: float = 20.0

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float64)[..., :2]
        self.keypoint_confidence = np.asarray(self.keypoint_confidence, dtype=np.float64).reshape(-1)
        if self.keypoints.shape != (17, 2):
            raise ValueError(f"keypoints must be (17, 2), got {self.keypoints.shape}")
        if self.keypoint_confidence.shape != (17,):
            raise ValueError("keypoint_confidence must be length 17")


@dataclass
class NormalizeInfo:
    origin: np.ndarray
    scale: float


@dataclass
class PoseQualityReport:
    quality: PoseQuality = PoseQuality.OK
    wrist_valid_ratio: float = 1.0
    shoulder_valid_ratio: float = 1.0
    head_valid_ratio: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.quality is PoseQuality.OK
