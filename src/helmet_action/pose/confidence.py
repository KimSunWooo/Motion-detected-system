from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from helmet_action.config import load_config
from helmet_action.pose.constants import HEAD_IDX, L_WRIST, R_WRIST, SHOULDER_IDX, WRIST_IDX
from helmet_action.pose.types import KeypointStatus, PoseQuality, PoseQualityReport


def interpolation_max_gap(cfg=None) -> int:
    cfg = cfg or load_config()
    nested = cfg.get("pose.interpolation.max_gap_frames")
    if nested is not None:
        return int(nested)
    return int(cfg.get("pose.short_gap_frames", 3))


def classify_keypoint(conf: float, cfg=None) -> KeypointStatus:
    cfg = cfg or load_config()
    missing = float(cfg.get("pose.keypoint_missing", 0.10))
    low = float(cfg.get("pose.keypoint_low_confidence", 0.25))
    if not np.isfinite(conf) or conf < missing:
        return KeypointStatus.MISSING
    if conf < low:
        return KeypointStatus.LOW_CONFIDENCE
    return KeypointStatus.VALID


def mask_invalid(keypoints: np.ndarray, confidence: np.ndarray | None) -> np.ndarray:
    seq = np.asarray(keypoints, dtype=np.float64)[..., :2]
    if confidence is None:
        return seq
    conf = np.asarray(confidence, dtype=np.float64)
    if conf.ndim == 1:
        conf = conf[None, :]
    missing = float(load_config().get("pose.keypoint_missing", 0.10))
    out = seq.copy()
    out[conf < missing] = np.nan
    return out


def interpolate_short_gaps(seq: np.ndarray, max_gap: int | None = None) -> np.ndarray:
    """Linear-fill NaN runs of length <= max_gap. Longer runs stay MISSING."""
    max_gap = interpolation_max_gap() if max_gap is None else int(max_gap)
    out = np.asarray(seq, dtype=np.float64).copy()
    t, j, _ = out.shape
    for ji in range(j):
        for axis in (0, 1):
            y = out[:, ji, axis]
            nans = ~np.isfinite(y)
            if not nans.any() or nans.all():
                continue
            i = 0
            while i < t:
                if not nans[i]:
                    i += 1
                    continue
                start = i
                while i < t and nans[i]:
                    i += 1
                gap = i - start
                if gap > max_gap:
                    continue
                left = start - 1
                right = i
                if left >= 0 and right < t:
                    for k in range(start, right):
                        u = (k - left) / (right - left)
                        y[k] = (1.0 - u) * y[left] + u * y[right]
                elif left >= 0:
                    y[start:i] = y[left]
                elif right < t:
                    y[start:i] = y[right]
            out[:, ji, axis] = y
    return out


def hold_last_valid(seq: np.ndarray) -> np.ndarray:
    out = np.asarray(seq, dtype=np.float64).copy()
    t, j, d = out.shape
    for ji in range(j):
        last = None
        for i in range(t):
            if np.isfinite(out[i, ji]).all():
                last = out[i, ji].copy()
            elif last is not None:
                out[i, ji] = last
    return out


def assess_pose_quality(
    keypoints: np.ndarray,
    confidence: np.ndarray | None = None,
) -> PoseQualityReport:
    cfg = load_config()
    seq = np.asarray(keypoints, dtype=np.float64)[..., :2]
    if seq.ndim == 2:
        seq = seq[None, ...]
    t = seq.shape[0]
    if confidence is None:
        conf = np.ones((t, 17), dtype=np.float64)
        conf[~np.isfinite(seq).all(axis=-1)] = 0.0
    else:
        conf = np.asarray(confidence, dtype=np.float64)
        if conf.ndim == 1:
            conf = np.repeat(conf[None, :], t, axis=0)
    low = float(cfg.get("pose.keypoint_low_confidence", 0.25))
    wrist_ok = (conf[:, list(WRIST_IDX)] >= low).all(axis=1)
    shoulder_ok = (conf[:, list(SHOULDER_IDX)] >= low).all(axis=1)
    head_ok = (conf[:, list(HEAD_IDX)].max(axis=1) >= low)
    report = PoseQualityReport(
        wrist_valid_ratio=float(wrist_ok.mean()) if t else 0.0,
        shoulder_valid_ratio=float(shoulder_ok.mean()) if t else 0.0,
        head_valid_ratio=float(head_ok.mean()) if t else 0.0,
    )
    min_wrist = float(cfg.get("decision.insufficient_min_wrist_ratio", 0.50))
    min_sh = float(cfg.get("decision.insufficient_min_shoulder_ratio", 0.70))
    notes = []
    if report.shoulder_valid_ratio < min_sh:
        notes.append("shoulders unstable — cannot normalize")
    if report.wrist_valid_ratio < min_wrist:
        notes.append("wrists missing/occluded too long — refuse to decide")
    if report.head_valid_ratio < 0.4:
        notes.append("head keypoints unstable")
    if notes:
        report.quality = PoseQuality.INSUFFICIENT_POSE
        report.notes = notes
    return report


def prepare_sequence(
    keypoints: np.ndarray,
    confidence: np.ndarray | None = None,
) -> tuple[np.ndarray, PoseQualityReport]:
    seq = mask_invalid(keypoints, confidence)
    if seq.ndim == 2:
        seq = seq[None, ...]
    quality = assess_pose_quality(seq, confidence)
    repaired = interpolate_short_gaps(seq)
    # Do NOT hold-last-valid across gaps longer than max_gap — that hid hard occlusion.
    filled = (~np.isfinite(seq).all(axis=-1)) & np.isfinite(repaired).all(axis=-1)
    quality.interpolation_ratio = float(filled.mean()) if filled.size else 0.0
    return repaired, quality
