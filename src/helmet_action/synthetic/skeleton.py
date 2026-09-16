from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from helmet_action.pose.constants import (
    FACE_IDX,
    L_ANKLE,
    L_EAR,
    L_ELBOW,
    L_EYE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_ANKLE,
    R_EAR,
    R_ELBOW,
    R_EYE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)
from helmet_action.synthetic.kinematics import apply_torso_transform, limb_lengths, two_bone_ik


def canonical_pose_3d() -> np.ndarray:
    """Standing worker in world metres. Origin = floor between feet, Y = height."""
    p = np.zeros((17, 3), dtype=np.float64)
    p[NOSE] = (0.00, 1.66, 3.44)
    p[L_EYE] = (-0.035, 1.68, 3.45)
    p[R_EYE] = (0.035, 1.68, 3.45)
    p[L_EAR] = (-0.090, 1.64, 3.54)
    p[R_EAR] = (0.090, 1.64, 3.54)
    p[L_SHOULDER] = (-0.205, 1.42, 3.52)
    p[R_SHOULDER] = (0.205, 1.42, 3.52)
    p[L_ELBOW] = (-0.27, 1.12, 3.54)
    p[R_ELBOW] = (0.27, 1.12, 3.54)
    p[L_WRIST] = (-0.25, 0.86, 3.55)
    p[R_WRIST] = (0.25, 0.86, 3.55)
    p[L_HIP] = (-0.13, 0.94, 3.50)
    p[R_HIP] = (0.13, 0.94, 3.50)
    p[L_KNEE] = (-0.13, 0.50, 3.52)
    p[R_KNEE] = (0.13, 0.50, 3.52)
    p[L_ANKLE] = (-0.12, 0.07, 3.50)
    p[R_ANKLE] = (0.12, 0.07, 3.50)
    return p


@dataclass
class BodyParams:
    height_scale: float = 1.0
    shoulder_width_scale: float = 1.0
    arm_length_scale: float = 1.0
    forearm_scale: float = 1.0
    head_size_scale: float = 1.0
    neck_length_scale: float = 1.0
    body_yaw_deg: float = 0.0
    body_pitch_deg: float = 0.0
    torso_lean: float = 0.0
    handedness: str = "right"

    def to_dict(self) -> dict:
        return asdict(self)


def apply_body_params(pose: np.ndarray, body: BodyParams) -> np.ndarray:
    p = pose.copy()
    pelvis = 0.5 * (p[L_HIP] + p[R_HIP])
    p[:, 1] = pelvis[1] + (p[:, 1] - pelvis[1]) * body.height_scale
    mid_sh = 0.5 * (p[L_SHOULDER] + p[R_SHOULDER])
    for idx in (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST):
        p[idx, 0] = mid_sh[0] + (p[idx, 0] - mid_sh[0]) * body.shoulder_width_scale
    head_c = 0.5 * (p[L_EAR] + p[R_EAR])
    for idx in FACE_IDX:
        p[idx] = head_c + (p[idx] - head_c) * body.head_size_scale
    p = apply_torso_transform(
        p,
        yaw_deg=body.body_yaw_deg,
        pitch_deg=body.body_pitch_deg,
        lean=body.torso_lean,
        neck_length_scale=body.neck_length_scale,
    )
    return p


def place_arm_3d(pose: np.ndarray, side: str, wrist: np.ndarray, body: BodyParams | None = None) -> None:
    sh_i, el_i, wr_i = (L_SHOULDER, L_ELBOW, L_WRIST) if side == "left" else (R_SHOULDER, R_ELBOW, R_WRIST)
    arm_s = 1.0 if body is None else body.arm_length_scale
    fore_s = 1.0 if body is None else body.forearm_scale
    upper, fore = limb_lengths(arm_s, fore_s)
    sign = -1.0 if side == "left" else 1.0
    hint = np.array([sign, -0.15, -0.55], dtype=np.float64)
    wr, el = two_bone_ik(pose[sh_i], wrist, upper, fore, hint)
    pose[wr_i] = wr
    pose[el_i] = el


def scale_head_toward_camera(pose: np.ndarray, amount: float) -> None:
    c = 0.5 * (pose[L_EAR] + pose[R_EAR])
    cam_dir = np.array([0.04, 0.89, -0.45], dtype=np.float64)
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    for idx in FACE_IDX:
        radial = pose[idx] - c
        pose[idx] = c + radial * (1.0 + 0.85 * amount) + cam_dir * (0.18 * amount)
