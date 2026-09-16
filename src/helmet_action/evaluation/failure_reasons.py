"""Automatic HELMET_REMOVE false-negative reason tags. Multiple tags per sample."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from helmet_action.models.labels import ActionClass
from helmet_action.models.rule_based import RuleBasedActionClassifier
from helmet_action.pose.confidence import interpolation_max_gap, mask_invalid
from helmet_action.pose.constants import L_EAR, L_WRIST, R_EAR, R_WRIST
from helmet_action.pose.normalizer import normalize_keypoints
from helmet_action.pose.quality import longest_missing_streak
from helmet_action.state.action_state_machine import infer_phases

FAILURE_REASONS = (
    "NO_APPROACH",
    "NO_GRASP",
    "WEAK_GRASP",
    "NO_LIFT",
    "WEAK_LIFT",
    "NO_SEPARATION",
    "WEAK_SEPARATION",
    "LOW_WRIST_CONFIDENCE",
    "LOW_EAR_CONFIDENCE",
    "SHORT_SEQUENCE",
    "TRACK_FRAGMENTATION",
    "STUTTER_MOTION",
    "LATERAL_MOTION",
    "PHASE_INCONSISTENCY",
    "ML_DISAGREES_WITH_RULE",
    "RULE_DISAGREES_WITH_ML",
    "UNKNOWN",
)


def tag_failure_reasons(
    keypoints: np.ndarray,
    confidence: np.ndarray | None,
    y_pred: str,
    ml_proba: dict[str, float] | None = None,
    rule_label: str | None = None,
    meta: dict[str, Any] | None = None,
) -> list[str]:
    seq = np.asarray(keypoints, dtype=np.float64)
    conf = np.ones((seq.shape[0], 17), dtype=np.float64) if confidence is None else np.asarray(confidence, dtype=np.float64)
    phase = infer_phases(seq)
    tags: list[str] = []
    meta = meta or {}
    action = meta.get("action") or {}

    if seq.shape[0] < 16:
        tags.append("SHORT_SEQUENCE")

    if not phase.saw_approach:
        tags.append("NO_APPROACH")
    if not phase.saw_grasp:
        tags.append("NO_GRASP")
    elif phase.history.count("HELMET_GRASP") < 6:
        tags.append("WEAK_GRASP")
    if not phase.saw_lift:
        tags.append("NO_LIFT")
    elif phase.history.count("LIFT_OR_SEPARATE") < 4:
        tags.append("WEAK_LIFT")
    if not phase.ordered:
        tags.append("PHASE_INCONSISTENCY")

    seq_n, _ = normalize_keypoints(seq)
    lw, rw = seq_n[:, L_WRIST], seq_n[:, R_WRIST]
    sep = np.linalg.norm(lw - rw, axis=1)
    if np.isfinite(sep).sum() >= 4:
        early = float(np.nanmean(sep[: max(seq.shape[0] // 4, 1)]))
        late = float(np.nanmean(sep[-max(seq.shape[0] // 4, 1) :]))
        if late - early < 0.04:
            tags.append("NO_SEPARATION" if late - early < 0.015 else "WEAK_SEPARATION")

    low = 0.25
    if float(np.nanmean(conf[:, [L_WRIST, R_WRIST]])) < low:
        tags.append("LOW_WRIST_CONFIDENCE")
    if float(np.nanmean(conf[:, [L_EAR, R_EAR]])) < low:
        tags.append("LOW_EAR_CONFIDENCE")

    masked = mask_invalid(seq, conf)
    miss = ~np.isfinite(masked).all(axis=-1)
    if longest_missing_streak(miss) > interpolation_max_gap() * 2:
        tags.append("TRACK_FRAGMENTATION")

    if bool(action.get("stutter")) or str(action.get("pause_mode", "")) == "stutter":
        tags.append("STUTTER_MOTION")
    if str(action.get("lift_dir", "up")) in ("left", "right") or "lateral" in str(meta.get("variant", "")).lower():
        tags.append("LATERAL_MOTION")

    ml_label = None
    if ml_proba:
        ml_label = max(ml_proba, key=ml_proba.get)
    if rule_label is None:
        rule_label = RuleBasedActionClassifier().predict(seq).label.value
    if ml_label and ml_label == ActionClass.HELMET_REMOVE.value and rule_label != "helmet_off":
        tags.append("RULE_DISAGREES_WITH_ML")
    if rule_label == "helmet_off" and (ml_label is not None and ml_label != ActionClass.HELMET_REMOVE.value):
        tags.append("ML_DISAGREES_WITH_RULE")

    if not tags:
        tags.append("UNKNOWN")
    # unique, stable order
    seen = set()
    ordered = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def summarize_reasons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    n = max(len(rows), 1)
    for row in rows:
        for reason in row.get("reasons", []):
            counts[str(reason)] += 1
    ratios = {k: float(v) / float(n) for k, v in counts.most_common()}
    return {
        "n_false_negatives": len(rows),
        "reason_counts": dict(counts),
        "reason_ratio": ratios,
        "ranked": [{"reason": k, "count": int(v), "ratio": ratios[k]} for k, v in counts.most_common()],
        "samples": rows,
    }
