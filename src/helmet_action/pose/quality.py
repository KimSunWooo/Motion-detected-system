"""PoseQualityScore in [0, 1] — not a substitute for ML probability."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from helmet_action.config import load_config
from helmet_action.pose.confidence import interpolation_max_gap, mask_invalid
from helmet_action.pose.constants import HEAD_IDX, L_EAR, L_WRIST, R_EAR, R_WRIST, SHOULDER_IDX, WRIST_IDX
from helmet_action.pose.types import DecisionStatus


def _conf_matrix(keypoints: np.ndarray, confidence: np.ndarray | None) -> np.ndarray:
    seq = np.asarray(keypoints, dtype=np.float64)[..., :2]
    if seq.ndim == 2:
        seq = seq[None, ...]
    t = seq.shape[0]
    if confidence is None:
        conf = np.ones((t, 17), dtype=np.float64)
        conf[~np.isfinite(seq).all(axis=-1)] = 0.0
        return conf
    conf = np.asarray(confidence, dtype=np.float64)
    if conf.ndim == 1:
        conf = np.repeat(conf[None, :], t, axis=0)
    conf = conf.copy()
    conf[~np.isfinite(seq).all(axis=-1)] = np.minimum(conf[~np.isfinite(seq).all(axis=-1)], 0.0)
    return conf


def longest_missing_streak(missing_mask: np.ndarray) -> int:
    """Longest consecutive True run. missing_mask is (T,) or (T, J)."""
    arr = np.asarray(missing_mask, dtype=bool)
    if arr.ndim == 2:
        arr = arr.any(axis=1)
    if arr.size == 0 or not arr.any():
        return 0
    longest = cur = 0
    for flag in arr.tolist():
        if flag:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return int(longest)


def interpolation_fill_ratio(before: np.ndarray, after: np.ndarray) -> float:
    b = ~np.isfinite(np.asarray(before, dtype=np.float64)[..., :2]).all(axis=-1)
    a = np.isfinite(np.asarray(after, dtype=np.float64)[..., :2]).all(axis=-1)
    filled = b & a
    return float(filled.mean()) if filled.size else 0.0


@dataclass
class PoseQualityScore:
    """0 = unusable pose, 1 = complete high-confidence sequence."""

    score: float
    required_availability: float
    mean_confidence: float
    longest_missing_streak: int
    longest_wrist_streak: int
    longest_ear_streak: int
    both_wrist_streak: int
    buffer_completeness: float
    interpolation_ratio: float
    decision_status: str
    wrist_quality: float = 1.0
    ear_quality: float = 1.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": float(self.score),
            "required_availability": float(self.required_availability),
            "mean_confidence": float(self.mean_confidence),
            "longest_missing_streak": int(self.longest_missing_streak),
            "longest_wrist_streak": int(self.longest_wrist_streak),
            "longest_ear_streak": int(self.longest_ear_streak),
            "both_wrist_streak": int(self.both_wrist_streak),
            "buffer_completeness": float(self.buffer_completeness),
            "interpolation_ratio": float(self.interpolation_ratio),
            "decision_status": self.decision_status,
            "wrist_quality": float(self.wrist_quality),
            "ear_quality": float(self.ear_quality),
            "notes": list(self.notes),
        }


def compute_pose_quality_score(
    keypoints: np.ndarray,
    confidence: np.ndarray | None = None,
    buffer_completeness: float = 1.0,
    interpolation_ratio: float | None = None,
    repaired: np.ndarray | None = None,
) -> PoseQualityScore:
    cfg = load_config()
    seq = np.asarray(keypoints, dtype=np.float64)[..., :2]
    if seq.ndim == 2:
        seq = seq[None, ...]
    conf = _conf_matrix(seq, confidence)
    low = float(cfg.get("pose.keypoint_low_confidence", 0.25))
    required = list(cfg.get("pose.required_keypoints", [5, 6, 9, 10, 0]))
    max_gap = interpolation_max_gap()

    req_ok = (conf[:, required] >= low).all(axis=1) if required else np.ones(seq.shape[0], dtype=bool)
    wrist_missing = (conf[:, list(WRIST_IDX)] < low).any(axis=1)
    both_wrists_missing = (conf[:, list(WRIST_IDX)] < low).all(axis=1)
    ear_missing = (conf[:, [L_EAR, R_EAR]] < low).any(axis=1)
    both_ears_missing = (conf[:, [L_EAR, R_EAR]] < low).all(axis=1)
    required_missing = ~req_ok

    streak = longest_missing_streak(required_missing)
    wrist_streak = longest_missing_streak(wrist_missing)
    both_wrist_streak = longest_missing_streak(both_wrists_missing)
    ear_streak = longest_missing_streak(ear_missing)
    both_ear_streak = longest_missing_streak(both_ears_missing)
    availability = float(req_ok.mean()) if req_ok.size else 0.0
    mean_conf = float(np.nanmean(conf[:, required])) if required else float(np.nanmean(conf))
    if not np.isfinite(mean_conf):
        mean_conf = 0.0
    wrist_conf_mean = float(np.nanmean(conf[:, list(WRIST_IDX)]))
    ear_conf_mean = float(np.nanmean(conf[:, [L_EAR, R_EAR]]))
    if not np.isfinite(wrist_conf_mean):
        wrist_conf_mean = 0.0
    if not np.isfinite(ear_conf_mean):
        ear_conf_mean = 0.0
    wrist_quality = float(
        np.clip(
            0.55 * wrist_conf_mean
            + 0.25 * (1.0 - min(1.0, wrist_streak / 20.0))
            + 0.20 * (1.0 - min(1.0, both_wrist_streak / 15.0)),
            0.0,
            1.0,
        )
    )
    ear_quality = float(
        np.clip(
            0.50 * ear_conf_mean
            + 0.25 * (1.0 - min(1.0, ear_streak / 20.0))
            + 0.25 * (1.0 - min(1.0, both_ear_streak / 20.0)),
            0.0,
            1.0,
        )
    )

    if interpolation_ratio is None and repaired is not None:
        masked = mask_invalid(seq, confidence)
        interpolation_ratio = interpolation_fill_ratio(masked, repaired)
    interp = float(interpolation_ratio or 0.0)
    completeness = float(np.clip(buffer_completeness, 0.0, 1.0))

    t = max(seq.shape[0], 1)
    streak_pen = min(1.0, streak / max(t * 0.5, 1.0))
    wrist_pen = min(1.0, wrist_streak / max(float(max_gap * 4), 1.0))
    score = (
        0.34 * availability
        + 0.22 * float(np.clip(mean_conf, 0.0, 1.0))
        + 0.18 * completeness
        + 0.14 * (1.0 - streak_pen)
        + 0.12 * (1.0 - min(1.0, interp * 2.0))
    )
    score = float(np.clip(score - 0.15 * wrist_pen, 0.0, 1.0))

    notes: list[str] = []
    if availability < 0.5:
        notes.append("required keypoints often missing")
    if wrist_streak > max_gap:
        notes.append(f"wrist missing streak {wrist_streak} > max_gap {max_gap}")
    if both_wrist_streak > max_gap:
        notes.append(f"both wrists missing {both_wrist_streak} frames consecutively")
    if completeness < 0.45:
        notes.append(f"buffer completeness {completeness:.2f}")
    if interp > 0.25:
        notes.append(f"interpolation filled {interp:.0%} of joints")
    if ear_streak >= 20:
        notes.append(f"ear missing streak {ear_streak}")
    if both_ear_streak >= 20:
        notes.append(f"both ears missing {both_ear_streak} frames consecutively")

    score = float(np.clip(score - 0.08 * min(1.0, both_ear_streak / 30.0), 0.0, 1.0))

    if score < 0.32 or availability < 0.35 or both_wrist_streak >= 15:
        status = DecisionStatus.INSUFFICIENT_POSE.value
    elif score < 0.52 or wrist_streak >= 8 or both_wrist_streak > max_gap:
        status = DecisionStatus.UNKNOWN.value
    elif score < 0.70 or wrist_streak > max_gap or both_ear_streak >= 20:
        status = DecisionStatus.LOW_CONFIDENCE.value
    else:
        status = DecisionStatus.VALID.value

    # Long ear occlusion must not look "VALID" — SAFE confirmation is refused downstream.
    if both_ear_streak >= 30 or ear_streak >= 30:
        if status == DecisionStatus.VALID.value:
            status = DecisionStatus.LOW_CONFIDENCE.value
        notes.append("long ear occlusion: insufficient to confirm a SAFE class")

    return PoseQualityScore(
        score=score,
        required_availability=availability,
        mean_confidence=mean_conf,
        longest_missing_streak=streak,
        longest_wrist_streak=wrist_streak,
        longest_ear_streak=int(ear_streak),
        both_wrist_streak=int(both_wrist_streak),
        buffer_completeness=completeness,
        interpolation_ratio=interp,
        decision_status=status,
        wrist_quality=wrist_quality,
        ear_quality=ear_quality,
        notes=notes,
    )


def phase_confidence(saw_approach: bool, saw_grasp: bool, saw_lift: bool, ordered: bool) -> float:
    if saw_grasp and saw_lift and ordered:
        return 0.95
    if saw_grasp and saw_lift:
        return 0.70
    if saw_grasp:
        return 0.45
    if saw_approach:
        return 0.25
    return 0.08


def phase_score_from_confidences(
    grasp_confidence: float,
    lift_confidence: float,
    separation_confidence: float,
    ordered: bool,
) -> float:
    """Continuous phase evidence in [0, 1]. One weak boolean does not zero the score."""
    score = (
        0.38 * float(np.clip(grasp_confidence, 0.0, 1.0))
        + 0.40 * float(np.clip(lift_confidence, 0.0, 1.0))
        + 0.22 * float(np.clip(separation_confidence, 0.0, 1.0))
    )
    if ordered:
        score = min(1.0, score + 0.06)
    return float(np.clip(score, 0.0, 1.0))
