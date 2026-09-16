"""3D human kinematics helpers: interpolation, IK, limb constraints."""

from __future__ import annotations

import math

import numpy as np

from helmet_action.pose.constants import (
    FACE_IDX,
    L_EAR,
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_EAR,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
)


def clamp01(u: float) -> float:
    return float(np.clip(u, 0.0, 1.0))


def lerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    u = clamp01(u)
    return (1.0 - u) * np.asarray(a, dtype=np.float64) + u * np.asarray(b, dtype=np.float64)


def smoothstep(u: float) -> float:
    u = clamp01(u)
    return u * u * (3.0 - 2.0 * u)


def ease_in_out_cubic(u: float) -> float:
    u = clamp01(u)
    if u < 0.5:
        return 4.0 * u * u * u
    return 1.0 - ((-2.0 * u + 2.0) ** 3) / 2.0


def ease_in_quad(u: float) -> float:
    u = clamp01(u)
    return u * u


def ease_out_quad(u: float) -> float:
    u = clamp01(u)
    return 1.0 - (1.0 - u) * (1.0 - u)


def bezier_ease(u: float, p1: float = 0.25, p2: float = 0.75) -> float:
    """Cubic Bezier from (0,0) to (1,1) with inner x=y control points (ease)."""
    u = clamp01(u)
    omu = 1.0 - u
    return (3.0 * omu * omu * u * p1) + (3.0 * omu * u * u * p2) + (u * u * u)


def interp_ease(u: float, kind: str = "cubic") -> float:
    if kind == "linear":
        return clamp01(u)
    if kind == "smoothstep":
        return smoothstep(u)
    if kind == "ease_in":
        return ease_in_quad(u)
    if kind == "ease_out":
        return ease_out_quad(u)
    if kind == "bezier":
        return bezier_ease(u)
    return ease_in_out_cubic(u)


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return ry @ rx @ rz


def rotate_around(points: np.ndarray, origin: np.ndarray, r: np.ndarray) -> np.ndarray:
    rel = np.asarray(points, dtype=np.float64) - origin
    return origin + rel @ r.T


def two_bone_ik(
    shoulder: np.ndarray,
    wrist_target: np.ndarray,
    upper_len: float,
    fore_len: float,
    out_hint: np.ndarray,
    min_deg: float = 25.0,
    max_deg: float = 165.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (wrist, elbow) with reachable wrist and bounded elbow angle."""
    sh = np.asarray(shoulder, dtype=np.float64)
    wr = np.asarray(wrist_target, dtype=np.float64)
    delta = wr - sh
    dist = float(np.linalg.norm(delta))
    max_reach = upper_len + fore_len - 1e-4
    min_reach = abs(upper_len - fore_len) + 1e-4
    if dist < 1e-8:
        wr = sh + np.array([0.0, -min_reach, 0.0])
        dist = min_reach
        delta = wr - sh
    if dist > max_reach:
        wr = sh + delta * (max_reach / dist)
        dist = max_reach
        delta = wr - sh
    elif dist < min_reach:
        wr = sh + delta * (min_reach / dist)
        dist = min_reach
        delta = wr - sh

    # Law of cosines on elbow interior angle.
    cos_el = (upper_len**2 + fore_len**2 - dist**2) / (2.0 * upper_len * fore_len)
    el_deg = math.degrees(math.acos(float(np.clip(cos_el, -1.0, 1.0))))
    el_deg = float(np.clip(el_deg, min_deg, max_deg))
    # Recompute reachable distance from clamped elbow.
    el_rad = math.radians(el_deg)
    dist = math.sqrt(max(upper_len**2 + fore_len**2 - 2 * upper_len * fore_len * math.cos(el_rad), 1e-8))
    dir_mid = delta / (np.linalg.norm(delta) + 1e-9)
    wr = sh + dir_mid * dist

    hint = np.asarray(out_hint, dtype=np.float64)
    n = np.cross(dir_mid, hint)
    if float(np.linalg.norm(n)) < 1e-6:
        n = np.cross(dir_mid, np.array([0.0, 0.0, 1.0]))
    if float(np.linalg.norm(n)) < 1e-6:
        n = np.array([1.0, 0.0, 0.0])
    n = n / (np.linalg.norm(n) + 1e-9)
    a = (upper_len**2 - fore_len**2 + dist**2) / (2.0 * dist)
    h = math.sqrt(max(upper_len**2 - a * a, 0.0))
    elbow = sh + dir_mid * a + n * h
    return wr, elbow


def constrain_path(
    path: np.ndarray,
    dt: float,
    max_vel: float = 2.4,
    max_acc: float = 14.0,
) -> np.ndarray:
    """Clip per-frame velocity / acceleration so joints cannot teleport."""
    out = np.asarray(path, dtype=np.float64).copy()
    if out.shape[0] < 2:
        return out
    max_step = max_vel * max(dt, 1e-3)
    max_dvel = max_acc * max(dt, 1e-3)
    prev_vel = np.zeros(out.shape[1:], dtype=np.float64)
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        step = float(np.linalg.norm(delta))
        if step > max_step:
            delta = delta * (max_step / (step + 1e-9))
            out[i] = out[i - 1] + delta
        vel = delta / max(dt, 1e-3)
        dvel = vel - prev_vel
        mag = float(np.linalg.norm(dvel))
        if mag > max_dvel:
            vel = prev_vel + dvel * (max_dvel / (mag + 1e-9))
            out[i] = out[i - 1] + vel * dt
        prev_vel = vel
    return out


def limb_lengths(body_arm: float = 1.0, body_fore: float = 1.0) -> tuple[float, float]:
    return 0.30 * float(body_arm), 0.26 * float(body_fore)


def apply_torso_transform(
    pose: np.ndarray,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    lean: float = 0.0,
    neck_length_scale: float = 1.0,
) -> np.ndarray:
    """Yaw/pitch/lean in 3D, then scale neck (shoulders→head)."""
    p = pose.copy()
    pelvis = 0.5 * (p[L_HIP] + p[R_HIP])
    r = rotation_matrix(
        yaw=math.radians(yaw_deg),
        pitch=math.radians(pitch_deg + lean * 25.0),
        roll=0.0,
    )
    p = rotate_around(p, pelvis, r)
    mid_sh = 0.5 * (p[L_SHOULDER] + p[R_SHOULDER])
    for idx in FACE_IDX:
        p[idx] = mid_sh + (p[idx] - mid_sh) * neck_length_scale
    return p


def rotate_head(pose: np.ndarray, yaw_deg: float, pitch_deg: float = 0.0) -> None:
    mid = 0.5 * (pose[L_EAR] + pose[R_EAR])
    r = rotation_matrix(math.radians(yaw_deg), math.radians(pitch_deg), 0.0)
    for idx in FACE_IDX:
        pose[idx] = rotate_around(pose[idx][None, ...], mid, r)[0]


def shoulder_bound(pose: np.ndarray, rest: np.ndarray, max_shift: float = 0.08) -> None:
    for idx in (L_SHOULDER, R_SHOULDER):
        delta = pose[idx] - rest[idx]
        mag = float(np.linalg.norm(delta))
        if mag > max_shift:
            pose[idx] = rest[idx] + delta * (max_shift / mag)
