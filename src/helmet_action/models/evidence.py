"""Hybrid decision evidence — never bury the final call in a single opaque if/else."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from helmet_action.models.labels import ActionClass, ActionLabel, RemovalPhase, RiskLevel


REJECTION_REASONS = (
    "LOW_WRIST_CONFIDENCE",
    "NO_GRASP",
    "NO_LIFT",
    "PHASE_INCONSISTENCY",
    "TRACK_FRAGMENTATION",
    "LOW_POSE_QUALITY",
    "LOW_BUFFER_COMPLETENESS",
    "RULE_ML_DISAGREEMENT",
    "TEMPORAL_INCOMPLETE",
    "BOTH_WRISTS_MISSING",
    "ID_SWITCH",
    "SHORT_SEQUENCE",
    "NO_PHASE_HISTORY",
    "LOW_EAR_QUALITY",
    "POSE_CRITICAL",
    "HARD_SAFETY_GATE",
)


@dataclass
class HybridDecisionEvidence:
    ml_remove_probability: float = 0.0
    ml_label: str | None = None
    rule_remove_score: float = 0.0
    rule_label: str = ""
    rule_confidence: float = 0.0
    phase: str = RemovalPhase.IDLE.value
    phase_confidence: float = 0.0
    grasp_confidence: float = 0.0
    lift_confidence: float = 0.0
    separation_confidence: float = 0.0
    pose_quality: float = 1.0
    wrist_quality: float = 1.0
    ear_quality: float = 1.0
    buffer_completeness: float = 1.0
    tracking_quality: float = 1.0
    motion_score: float = 0.0
    ml_score: float = 0.0
    rule_score: float = 0.0
    phase_score: float = 0.0
    removal_evidence: float = 0.0
    fusion_remove_score: float = 0.0
    n_frames: int = 0
    both_wrist_streak: int = 0
    longest_ear_streak: int = 0
    longest_wrist_streak: int = 0
    id_switched: bool = False
    track_fragmented: bool = False
    phase_ordered: bool = False
    saw_approach: bool = False
    saw_grasp: bool = False
    saw_partial_grasp: bool = False
    saw_lift: bool = False
    brim_grasp: bool = False
    one_then_two: bool = False
    lateral_lift: bool = False
    physically_ok: bool = False
    safety_gate_triggered: bool = False
    hybrid_version: str = "v1"
    risk_level: str = RiskLevel.UNKNOWN.value
    rejection_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        payload["notes"] = list(self.notes)
        return payload


@dataclass
class SafetyGateResult:
    triggered: bool
    force_insufficient: bool = False
    force_unknown: bool = False
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def rule_remove_score(rule_label: str, rule_confidence: float) -> float:
    if rule_label in (ActionLabel.HELMET_OFF.value, "helmet_off"):
        return float(np.clip(rule_confidence, 0.0, 1.0))
    if rule_label in (ActionLabel.UNKNOWN_CONTACT.value, "unknown_contact"):
        return float(np.clip(0.35 * rule_confidence, 0.0, 1.0))
    return 0.0


def motion_score_from_features(features: Any) -> float:
    dx = float(getattr(features, "dx_spread", 0.0) or 0.0)
    rise = float(getattr(features, "co_rise_y", 0.0) or 0.0)
    radial = float(getattr(features, "radial_expand", 0.0) or 0.0)
    spread = float(getattr(features, "wrist_spread", 0.0) or 0.0)
    return float(
        np.clip(
            0.30 * min(1.0, max(dx, 0.0) / 0.12)
            + 0.30 * min(1.0, max(rise, 0.0) / 0.08)
            + 0.20 * min(1.0, max(radial, 0.0) / 0.10)
            + 0.20 * min(1.0, max(spread, 0.0) / 0.12),
            0.0,
            1.0,
        )
    )


def evaluate_safety_gates(
    evidence: HybridDecisionEvidence,
    *,
    pose_critical: float = 0.32,
    min_frames: int = 12,
    both_wrist_long: int = 15,
) -> SafetyGateResult:
    """Hard safety: never emit REMOVE_CONFIRMED / ALERT regardless of ML probability."""
    reasons: list[str] = []
    notes: list[str] = []
    force_insufficient = False
    force_unknown = False

    if evidence.n_frames < min_frames:
        reasons.append("SHORT_SEQUENCE")
        notes.append(f"sequence length {evidence.n_frames} < {min_frames}")
        force_unknown = True
    if evidence.pose_quality < pose_critical:
        reasons.append("POSE_CRITICAL")
        notes.append(f"pose_quality {evidence.pose_quality:.2f} < critical {pose_critical:.2f}")
        force_insufficient = True
    if evidence.both_wrist_streak >= both_wrist_long:
        reasons.append("BOTH_WRISTS_MISSING")
        notes.append(f"both wrists missing {evidence.both_wrist_streak} frames")
        force_insufficient = True
    if evidence.id_switched:
        reasons.append("ID_SWITCH")
        notes.append("tracking ID switch")
        force_unknown = True
    if evidence.track_fragmented or evidence.tracking_quality < 0.25:
        reasons.append("TRACK_FRAGMENTATION")
        notes.append("tracking fragmentation")
        force_unknown = True
    if evidence.buffer_completeness < 0.35:
        reasons.append("LOW_BUFFER_COMPLETENESS")
        notes.append(f"buffer completeness {evidence.buffer_completeness:.2f}")
        force_unknown = True
    phase_history = (
        evidence.saw_approach
        or evidence.saw_grasp
        or evidence.saw_partial_grasp
        or evidence.saw_lift
        or evidence.grasp_confidence > 0.15
        or evidence.lift_confidence > 0.15
    )
    if evidence.n_frames >= min_frames and not phase_history and evidence.ml_remove_probability >= 0.75:
        # High ML without any phase trace: still allow fusion to WATCH, but cannot CONFIRM.
        reasons.append("NO_PHASE_HISTORY")
        notes.append("no phase history — refuse REMOVE_CONFIRMED")

    triggered = bool(reasons) and (force_insufficient or force_unknown)
    if triggered:
        reasons.append("HARD_SAFETY_GATE")
    return SafetyGateResult(
        triggered=triggered,
        force_insufficient=force_insufficient,
        force_unknown=force_unknown or force_insufficient,
        reasons=_unique(reasons),
        notes=notes,
    )


def collect_rejection_reasons(evidence: HybridDecisionEvidence, action: ActionClass) -> list[str]:
    """Human-readable why this window is UNKNOWN (or why confirmation was refused)."""
    reasons: list[str] = list(evidence.rejection_reasons)
    if action not in (ActionClass.UNKNOWN, ActionClass.INSUFFICIENT_POSE):
        return _unique(reasons)

    if evidence.wrist_quality < 0.40 or evidence.longest_wrist_streak >= 8:
        reasons.append("LOW_WRIST_CONFIDENCE")
    if evidence.grasp_confidence < 0.35 and not evidence.saw_grasp:
        reasons.append("NO_GRASP")
    if evidence.lift_confidence < 0.35 and not evidence.saw_lift:
        reasons.append("NO_LIFT")
    if (evidence.saw_grasp or evidence.saw_lift or evidence.saw_partial_grasp) and not evidence.phase_ordered:
        reasons.append("PHASE_INCONSISTENCY")
    if evidence.track_fragmented or evidence.tracking_quality < 0.40:
        reasons.append("TRACK_FRAGMENTATION")
    if evidence.pose_quality < 0.52:
        reasons.append("LOW_POSE_QUALITY")
    if evidence.buffer_completeness < 0.45:
        reasons.append("LOW_BUFFER_COMPLETENESS")
    if evidence.ml_remove_probability >= 0.75 and evidence.rule_remove_score < 0.40:
        reasons.append("RULE_ML_DISAGREEMENT")
    if evidence.n_frames < 16 or evidence.buffer_completeness < 0.55:
        reasons.append("TEMPORAL_INCOMPLETE")
    if evidence.ear_quality < 0.35 or evidence.longest_ear_streak >= 20:
        reasons.append("LOW_EAR_QUALITY")
    if evidence.both_wrist_streak >= 8:
        reasons.append("BOTH_WRISTS_MISSING")
    if evidence.id_switched:
        reasons.append("ID_SWITCH")
    if evidence.n_frames < 12:
        reasons.append("SHORT_SEQUENCE")
    if not (
        evidence.saw_approach
        or evidence.saw_grasp
        or evidence.saw_partial_grasp
        or evidence.saw_lift
    ):
        reasons.append("NO_PHASE_HISTORY")
    return _unique(reasons)


def fused_remove_score(
    evidence: HybridDecisionEvidence,
    w_ml: float,
    w_rule: float,
    w_phase: float,
    w_motion: float,
) -> float:
    weights = np.array([w_ml, w_rule, w_phase, w_motion], dtype=np.float64)
    s = float(weights.sum())
    if s <= 0:
        weights = np.array([0.40, 0.20, 0.30, 0.10])
        s = 1.0
    weights = weights / s
    score = (
        weights[0] * evidence.ml_score
        + weights[1] * evidence.rule_score
        + weights[2] * evidence.phase_score
        + weights[3] * evidence.motion_score
    )
    return float(np.clip(score, 0.0, 1.0))


def _unique(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
