from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from helmet_action.config import load_config
from helmet_action.pose.constants import (
    CENTER_RADIUS,
    EAR_RADIUS,
    HEAD_CENTER_OFFSET_Y,
    HEAD_HALF_HEIGHT,
    HEAD_HALF_WIDTH,
    L_EAR,
    L_ELBOW,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_EAR,
    R_ELBOW,
    R_SHOULDER,
    R_WRIST,
)


def _geo() -> dict:
    return load_config().section("geometry")


@dataclass
class HeadRegions:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    center: np.ndarray
    center_r: float
    left_ear: np.ndarray
    right_ear: np.ndarray
    ear_r: float

    def as_xywh(self) -> tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max - self.x_min, self.y_max - self.y_min

    def in_bbox(self, p: np.ndarray) -> bool:
        if not np.isfinite(p).all():
            return False
        return self.x_min <= float(p[0]) <= self.x_max and self.y_min <= float(p[1]) <= self.y_max

    def in_center(self, p: np.ndarray) -> bool:
        if not np.isfinite(p).all():
            return False
        return float(np.linalg.norm(p - self.center)) <= self.center_r

    def in_left_ear(self, p: np.ndarray) -> bool:
        if not np.isfinite(p).all():
            return False
        return float(np.linalg.norm(p - self.left_ear)) <= self.ear_r

    def in_right_ear(self, p: np.ndarray) -> bool:
        if not np.isfinite(p).all():
            return False
        return float(np.linalg.norm(p - self.right_ear)) <= self.ear_r

    def in_brim(self, p: np.ndarray) -> bool:
        """Front/top of the helmet: near head center, not required to sit on an ear."""
        if not np.isfinite(p).all():
            return False
        if self.in_center(p):
            return True
        # Slightly in front of the crown, still inside the head bbox.
        brim = self.center + np.array([0.0, 0.04])
        return float(np.linalg.norm(p - brim)) <= self.center_r * 1.15


def compute_neck(kpts: np.ndarray) -> np.ndarray:
    return 0.5 * (kpts[..., L_SHOULDER, :] + kpts[..., R_SHOULDER, :])


def compute_shoulder_width(kpts: np.ndarray) -> np.ndarray:
    return np.linalg.norm(kpts[..., L_SHOULDER, :] - kpts[..., R_SHOULDER, :], axis=-1)


def head_center_norm(pose: np.ndarray) -> np.ndarray:
    ears = np.stack([pose[L_EAR], pose[R_EAR]], axis=0)
    if np.isfinite(ears).all():
        return 0.5 * (pose[L_EAR] + pose[R_EAR])
    if np.isfinite(pose[NOSE]).all():
        geo = _geo()
        return np.array([float(pose[NOSE, 0]), float(geo.get("head_center_offset_y", HEAD_CENTER_OFFSET_Y))])
    geo = _geo()
    return np.array([0.0, float(geo.get("head_center_offset_y", HEAD_CENTER_OFFSET_Y))])


def head_scale_norm(pose: np.ndarray) -> float:
    if not (np.isfinite(pose[L_EAR]).all() and np.isfinite(pose[R_EAR]).all()):
        return float("nan")
    return float(np.linalg.norm(pose[L_EAR] - pose[R_EAR]))


def compute_head_regions(pose_norm: np.ndarray) -> HeadRegions:
    geo = _geo()
    offset_y = float(geo.get("head_center_offset_y", HEAD_CENTER_OFFSET_Y))
    half_w = float(geo.get("head_half_width", HEAD_HALF_WIDTH))
    half_h = float(geo.get("head_half_height", HEAD_HALF_HEIGHT))
    center_r = float(geo.get("center_radius", CENTER_RADIUS))
    ear_r = float(geo.get("ear_radius", EAR_RADIUS))

    cx = float(pose_norm[NOSE, 0]) if np.isfinite(pose_norm[NOSE, 0]) else 0.0
    cy = offset_y
    center = np.array([cx, cy], dtype=np.float64)
    left_ear = pose_norm[L_EAR].copy()
    right_ear = pose_norm[R_EAR].copy()
    if not np.isfinite(left_ear).all():
        left_ear = np.array([cx - half_w + 0.12, cy])
    if not np.isfinite(right_ear).all():
        right_ear = np.array([cx + half_w - 0.12, cy])
    return HeadRegions(
        x_min=cx - half_w,
        y_min=cy - half_h,
        x_max=cx + half_w,
        y_max=cy + half_h,
        center=center,
        center_r=center_r,
        left_ear=left_ear,
        right_ear=right_ear,
        ear_r=ear_r,
    )


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        return float("nan")
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-8 or nc < 1e-8:
        return float("nan")
    cos = float(np.dot(ba, bc) / (na * nc))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def elbow_angles(pose: np.ndarray) -> tuple[float, float]:
    left = angle_at(pose[L_SHOULDER], pose[L_ELBOW], pose[L_WRIST])
    right = angle_at(pose[R_SHOULDER], pose[R_ELBOW], pose[R_WRIST])
    return left, right


def upper_arm_angles(pose: np.ndarray) -> tuple[float, float]:
    left = _segment_angle(pose[L_SHOULDER], pose[L_ELBOW])
    right = _segment_angle(pose[R_SHOULDER], pose[R_ELBOW])
    return left, right


def forearm_angles(pose: np.ndarray) -> tuple[float, float]:
    left = _segment_angle(pose[L_ELBOW], pose[L_WRIST])
    right = _segment_angle(pose[R_ELBOW], pose[R_WRIST])
    return left, right


def _segment_angle(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    d = b - a
    if np.linalg.norm(d) < 1e-8:
        return float("nan")
    return float(np.degrees(np.arctan2(d[1], d[0])))


def wrist_head_direction_angles(pose: np.ndarray) -> tuple[float, float]:
    head = head_center_norm(pose)
    return _segment_angle(head, pose[L_WRIST]), _segment_angle(head, pose[R_WRIST])


def shoulder_orientation(pose: np.ndarray) -> float:
    return _segment_angle(pose[L_SHOULDER], pose[R_SHOULDER])
