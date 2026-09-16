"""Canonical demo sequences (original seeds/math) plus named scenario catalogue."""

from __future__ import annotations

import math

import numpy as np

from helmet_action.pose.constants import L_EAR, L_WRIST, R_EAR, R_WRIST
from helmet_action.synthetic.camera import HighAngleCamera
from helmet_action.synthetic.skeleton import canonical_pose_3d, place_arm_3d, scale_head_toward_camera

SCENARIOS = [
    "IDLE",
    "HEAD_SCRATCH",
    "ONE_HAND_HEAD_TOUCH",
    "TWO_HAND_HEAD_TOUCH",
    "FACE_TOUCH",
    "HELMET_ADJUST",
    "WIPE_SWEAT",
    "LOOK_DOWN",
    "RAISE_ARMS",
    "STRETCH",
    "PHONE_NEAR_HEAD",
    "HELMET_REMOVE",
    "HELMET_PUT_ON",
    "UNKNOWN_RANDOM_MOTION",
]


def _lerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    return (1.0 - u) * a + u * b


def _ease(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def generate_scratch_sequence(
    n_frames: int = 60,
    fps: int = 30,
    seed: int = 7,
    pixel_scale: float = 1.0,
    pixel_shift: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cam = HighAngleCamera.factory_ceiling()
    rest = canonical_pose_3d()
    scalp = 0.5 * (rest[L_EAR] + rest[R_EAR]) + np.array([0.00, 0.012, 0.015])
    seq = np.zeros((n_frames, 17, 2), dtype=np.float64)
    raise_end, scratch_end = 12, 50
    for i in range(n_frames):
        pose = rest.copy()
        if i < raise_end:
            u = _ease(i / max(raise_end - 1, 1))
            wrist = _lerp(rest[R_WRIST], scalp, u)
        elif i < scratch_end:
            t = (i - raise_end) / fps
            wrist = scalp + np.array(
                [
                    0.018 * math.sin(2 * math.pi * 3.6 * t),
                    0.006 * math.cos(2 * math.pi * 3.6 * t),
                    0.008 * math.sin(2 * math.pi * 4.1 * t + 0.4),
                ]
            )
        else:
            u = _ease((i - scratch_end) / max(n_frames - scratch_end - 1, 1))
            wrist = _lerp(scalp, rest[R_WRIST], u)
        place_arm_3d(pose, "right", wrist)
        pose += rng.normal(0.0, 0.0015, size=pose.shape)
        pix = cam.project(pose) * pixel_scale + np.asarray(pixel_shift)
        seq[i] = pix
    return seq


def generate_helmet_off_sequence(
    n_frames: int = 64,
    seed: int = 11,
    pixel_scale: float = 1.0,
    pixel_shift: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cam = HighAngleCamera.factory_ceiling()
    rest = canonical_pose_3d()
    grab_l = rest[L_EAR] + np.array([-0.03, 0.01, 0.02])
    grab_r = rest[R_EAR] + np.array([0.03, 0.01, 0.02])
    seq = np.zeros((n_frames, 17, 2), dtype=np.float64)
    raise_end, pause_end, lift_end = 11, 22, 52
    for i in range(n_frames):
        pose = rest.copy()
        amount = 0.0
        if i < raise_end:
            u = _ease(i / max(raise_end - 1, 1))
            wl = _lerp(rest[L_WRIST], grab_l, u)
            wr = _lerp(rest[R_WRIST], grab_r, u)
        elif i < pause_end:
            wl = grab_l + rng.normal(0.0, 0.0012, size=3)
            wr = grab_r + rng.normal(0.0, 0.0012, size=3)
        else:
            u = _ease(min(1.0, (i - pause_end) / max(lift_end - pause_end - 1, 1)))
            wl = grab_l + np.array([-0.08 * u, 0.22 * u, -0.10 * u])
            wr = grab_r + np.array([0.08 * u, 0.22 * u, -0.10 * u])
            amount = 0.55 * u
        place_arm_3d(pose, "left", wl)
        place_arm_3d(pose, "right", wr)
        scale_head_toward_camera(pose, amount)
        pose += rng.normal(0.0, 0.0014, size=pose.shape)
        seq[i] = cam.project(pose) * pixel_scale + np.asarray(pixel_shift)
    return seq


def generate_idle_sequence(n_frames: int = 40, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cam = HighAngleCamera.factory_ceiling()
    rest = canonical_pose_3d()
    seq = np.zeros((n_frames, 17, 2), dtype=np.float64)
    for i in range(n_frames):
        pose = rest.copy()
        phase = 2 * math.pi * i / 16.0
        wr = rest[R_WRIST] + np.array([0.03 * math.sin(phase), 0.04 * math.sin(phase), 0.0])
        place_arm_3d(pose, "right", wr)
        pose += rng.normal(0.0, 0.0015, size=pose.shape)
        seq[i] = cam.project(pose)
    return seq
