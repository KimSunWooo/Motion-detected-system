"""COCO-17 keypoint indices and default head-region geometry."""

from __future__ import annotations

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

COCO_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

UPPER_BONES = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW, R_WRIST),
    (NOSE, L_EYE),
    (NOSE, R_EYE),
    (L_EYE, L_EAR),
    (R_EYE, R_EAR),
]
SKELETON_BONES = UPPER_BONES
FACE_IDX = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
UPPER_JOINTS = (
    NOSE,
    L_EYE,
    R_EYE,
    L_EAR,
    R_EAR,
    L_SHOULDER,
    R_SHOULDER,
    L_ELBOW,
    R_ELBOW,
    L_WRIST,
    R_WRIST,
)
WRIST_IDX = (L_WRIST, R_WRIST)
SHOULDER_IDX = (L_SHOULDER, R_SHOULDER)
HEAD_IDX = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)

HEAD_CENTER_OFFSET_Y = -0.30
HEAD_HALF_WIDTH = 0.52
HEAD_HALF_HEIGHT = 0.42
CENTER_RADIUS = 0.24
EAR_RADIUS = 0.24
