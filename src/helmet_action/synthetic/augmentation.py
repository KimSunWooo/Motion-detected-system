from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from helmet_action.pose.constants import (
    L_EAR,
    L_ELBOW,
    L_SHOULDER,
    L_WRIST,
    R_EAR,
    R_ELBOW,
    R_SHOULDER,
    R_WRIST,
)

# Detector-like relative noise: wrists/ears less stable than shoulders.
JOINT_NOISE_SCALE = np.ones(17, dtype=np.float64)
JOINT_NOISE_SCALE[[L_WRIST, R_WRIST]] = 1.85
JOINT_NOISE_SCALE[[L_ELBOW, R_ELBOW]] = 1.25
JOINT_NOISE_SCALE[[L_EAR, R_EAR]] = 1.55
JOINT_NOISE_SCALE[[L_SHOULDER, R_SHOULDER]] = 0.55
JOINT_NOISE_SCALE[0:3] = 1.15  # nose / eyes


@dataclass
class NoiseParams:
    gaussian_sigma: float = 0.0016
    dropout_prob: float = 0.02
    low_conf_prob: float = 0.04
    occlusion_prob: float = 0.03
    jitter: float = 0.002
    scale_jitter: float = 0.03
    translation_jitter: float = 4.0
    frame_drop_prob: float = 0.02
    temporal_stretch: float = 1.0
    consecutive_drop_prob: float = 0.04
    conf_flicker_prob: float = 0.05
    keypoint_swap_prob: float = 0.015
    track_interrupt_prob: float = 0.02
    bbox_jitter: float = 3.0
    ear_conf_degrade: float = 0.18
    person_scale_jitter: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _consecutive_drop(seq: np.ndarray, conf: np.ndarray, rng: np.random.Generator, joint: int, length: int) -> None:
    t = seq.shape[0]
    if t < 3:
        return
    start = int(rng.integers(0, max(t - 2, 1)))
    end = min(t, start + length)
    seq[start:end, joint] = np.nan
    conf[start:end, joint] = 0.0


def apply_detector_noise(
    seq_px: np.ndarray,
    rng: np.random.Generator,
    noise: NoiseParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate YOLO-pose detector errors on already-projected 2D keypoints."""
    seq = np.asarray(seq_px, dtype=np.float64).copy()
    t, j, _ = seq.shape
    conf = np.ones((t, j), dtype=np.float64)

    sigma = noise.gaussian_sigma * 40.0
    jitter = noise.jitter * 40.0
    joint_scale = JOINT_NOISE_SCALE[None, :, None]
    seq += rng.normal(0.0, sigma, size=seq.shape) * joint_scale
    seq += rng.normal(0.0, jitter, size=seq.shape) * joint_scale

    scale = 1.0 + rng.uniform(-noise.scale_jitter, noise.scale_jitter)
    if noise.person_scale_jitter:
        scale *= 1.0 + rng.uniform(-noise.person_scale_jitter, noise.person_scale_jitter)
    shift = rng.uniform(-noise.translation_jitter, noise.translation_jitter, size=2)
    bbox_shift = rng.normal(0.0, noise.bbox_jitter, size=2)
    seq = seq * scale + shift + bbox_shift

    # Ear confidence is systematically worse than shoulders.
    conf[:, [L_EAR, R_EAR]] *= max(0.35, 1.0 - noise.ear_conf_degrade)
    conf[:, [L_WRIST, R_WRIST]] *= 0.92

    drop = rng.random((t, j)) < noise.dropout_prob
    # Wrists drop more often than shoulders.
    drop[:, [L_WRIST, R_WRIST]] |= rng.random((t, 2)) < (noise.dropout_prob * 1.4)
    low = rng.random((t, j)) < noise.low_conf_prob
    if low.any():
        conf[low] = rng.uniform(0.12, 0.28, size=int(low.sum()))
    conf[drop] = 0.0
    seq[drop] = np.nan

    if rng.random() < noise.occlusion_prob:
        joint = int(rng.choice([L_WRIST, R_WRIST, L_EAR, R_EAR, L_ELBOW, R_ELBOW, 0]))
        start = int(rng.integers(0, max(t - 4, 1)))
        length = int(rng.integers(2, 8))
        seq[start : start + length, joint] = np.nan
        conf[start : start + length, joint] = 0.0

    if rng.random() < noise.consecutive_drop_prob:
        joint = int(rng.choice([L_WRIST, R_WRIST, L_EAR, R_EAR]))
        _consecutive_drop(seq, conf, rng, joint, int(rng.integers(2, 6)))

    if rng.random() < noise.conf_flicker_prob:
        flicker = rng.random((t, j)) < 0.12
        conf[flicker] = rng.uniform(0.08, 0.35, size=int(flicker.sum())) if flicker.any() else conf[flicker]

    if rng.random() < noise.keypoint_swap_prob and t > 4:
        a, b = (L_WRIST, R_WRIST) if rng.random() < 0.6 else (L_EAR, R_EAR)
        s = int(rng.integers(0, max(t - 2, 1)))
        e = min(t, s + int(rng.integers(1, 4)))
        seq[s:e, a], seq[s:e, b] = seq[s:e, b].copy(), seq[s:e, a].copy()
        conf[s:e, a], conf[s:e, b] = conf[s:e, b].copy(), conf[s:e, a].copy()

    if rng.random() < noise.track_interrupt_prob and t > 10:
        gap = int(rng.integers(3, 8))
        s = int(rng.integers(2, t - gap - 1))
        seq[s : s + gap] = np.nan
        conf[s : s + gap] = 0.0

    keep = rng.random(t) >= noise.frame_drop_prob
    if keep.sum() < 8:
        keep[:] = True
    seq = seq[keep]
    conf = conf[keep]

    stretch = float(np.clip(noise.temporal_stretch, 0.6, 1.6))
    if abs(stretch - 1.0) > 0.05 and seq.shape[0] >= 8:
        n_new = max(8, int(round(seq.shape[0] * stretch)))
        src = np.linspace(0, seq.shape[0] - 1, seq.shape[0])
        dst = np.linspace(0, seq.shape[0] - 1, n_new)
        out = np.empty((n_new, j, 2), dtype=np.float64)
        c_out = np.empty((n_new, j), dtype=np.float64)
        for ji in range(j):
            for axis in (0, 1):
                col = seq[:, ji, axis]
                finite = np.isfinite(col)
                if finite.sum() < 2:
                    out[:, ji, axis] = col[0] if col.size else np.nan
                else:
                    out[:, ji, axis] = np.interp(dst, src[finite], col[finite])
            c_out[:, ji] = np.interp(dst, src, np.nan_to_num(conf[:, ji], nan=0.0))
        seq, conf = out, c_out
    return seq, conf


def apply_pose_noise(
    seq_px: np.ndarray,
    rng: np.random.Generator,
    noise: NoiseParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-compat alias used by the generator and tests."""
    return apply_detector_noise(seq_px, rng, noise)


def apply_targeted_occlusion(
    seq: np.ndarray,
    conf: np.ndarray,
    joints: list[int],
    dropout_rate: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic-ish targeted dropout used by occlusion stress tests."""
    rng = rng or np.random.default_rng(0)
    out_s = np.asarray(seq, dtype=np.float64).copy()
    out_c = np.asarray(conf, dtype=np.float64).copy()
    t = out_s.shape[0]
    n_drop = int(round(t * float(np.clip(dropout_rate, 0.0, 1.0))))
    if n_drop <= 0:
        return out_s, out_c
    idx = rng.choice(t, size=min(n_drop, t), replace=False)
    for j in joints:
        out_s[idx, j] = np.nan
        out_c[idx, j] = 0.0
    return out_s, out_c


def apply_consecutive_occlusion(
    seq: np.ndarray,
    conf: np.ndarray,
    joints: list[int],
    n_frames: int,
    start: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop a consecutive block of frames for the given joints. Not random speckles."""
    rng = rng or np.random.default_rng(0)
    out_s = np.asarray(seq, dtype=np.float64).copy()
    out_c = np.asarray(conf, dtype=np.float64).copy()
    t = out_s.shape[0]
    gap = int(np.clip(n_frames, 0, t))
    if gap <= 0:
        return out_s, out_c
    if start is None:
        start = int(rng.integers(0, max(t - gap, 1)))
    start = int(np.clip(start, 0, max(t - 1, 0)))
    end = min(t, start + gap)
    for j in joints:
        out_s[start:end, j] = np.nan
        out_c[start:end, j] = 0.0
    return out_s, out_c
