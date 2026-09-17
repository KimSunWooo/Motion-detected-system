from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from helmet_action.config import load_config
from helmet_action.models.evidence import (
    HybridDecisionEvidence,
    collect_rejection_reasons,
    evaluate_safety_gates,
    fused_remove_score,
    motion_score_from_features,
    rule_remove_score,
)
from helmet_action.models.labels import (
    ActionClass,
    ActionLabel,
    BaselineActionLabel,
    ClassificationResult,
    LABEL_KO,
    RemovalPhase,
    RiskLevel,
    SAFE_CONFIRMED_ACTIONS,
    action_to_baseline,
    baseline_to_action,
)
from helmet_action.models.risk import RiskEscalator, apply_escalation_to_action, propose_risk_level
from helmet_action.models.rule_based import RuleBasedActionClassifier
from helmet_action.models.temporal_classifier import SklearnActionClassifier
from helmet_action.pose.confidence import interpolation_max_gap, prepare_sequence
from helmet_action.pose.quality import compute_pose_quality_score, phase_confidence, phase_score_from_confidences
from helmet_action.pose.types import DecisionStatus
from helmet_action.state.action_state_machine import infer_phases
from helmet_action.state.helmet_state import (
    ActionEvent,
    AlertGate,
    DummyHelmetPresenceDetector,
    HelmetState,
    HelmetStateMachine,
)


@dataclass
class HybridDecision:
    action: ActionClass
    baseline: BaselineActionLabel
    confidence: float
    rule_label: str
    rule_confidence: float
    ml_label: str | None
    ml_proba: dict[str, float]
    phase: RemovalPhase
    phase_ordered: bool
    helmet_state: HelmetState
    event: ActionEvent
    risk: float
    alert: bool
    quality: str
    explanation: list[str] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    frame_labels: list[str] = field(default_factory=list)
    action_probability: float = 0.0
    pose_quality: float = 1.0
    phase_confidence: float = 0.0
    decision_status: str = "VALID"
    hybrid_version: str = "v1"
    risk_level: str = RiskLevel.UNKNOWN.value
    rejection_reasons: list[str] = field(default_factory=list)
    evidence: HybridDecisionEvidence | None = None

    def to_dict(self) -> dict:
        return {
            "final_prediction": self.action.value,
            "final_ko": {a: a.value for a in ActionClass}.get(self.action, self.action.value),
            "label": self.baseline.value,
            "label_ko": LABEL_KO.get(self.baseline, self.action.value),
            "confidence": self.confidence,
            "action": self.action.value,
            "action_probability": self.action_probability,
            "pose_quality": self.pose_quality,
            "phase_confidence": self.phase_confidence,
            "decision_status": self.decision_status,
            "hybrid_version": self.hybrid_version,
            "risk_level": self.risk_level,
            "rejection_reasons": list(self.rejection_reasons),
            "evidence": None if self.evidence is None else self.evidence.to_dict(),
            "rule_prediction": self.rule_label,
            "rule_confidence": self.rule_confidence,
            "ml_prediction": self.ml_label,
            "ml_proba": self.ml_proba,
            "phase": self.phase.value,
            "phase_ordered": self.phase_ordered,
            "helmet_state": self.helmet_state.value,
            "event": self.event.value,
            "risk": self.risk,
            "alert": self.alert,
            "quality": self.quality,
            "explanation": self.explanation,
            "features": self.features,
            "frame_labels": self.frame_labels,
        }


def _ear_blocks_safe(pq, cfg) -> bool:
    streak = int(getattr(pq, "longest_ear_streak", 0) or 0)
    ear_q = float(getattr(pq, "ear_quality", 1.0) or 1.0)
    refuse_streak = int(cfg.get("decision.ear_refuse_safe_streak", interpolation_max_gap() + 1))
    return streak >= refuse_streak or ear_q < 0.35


def _gather_evidence(
    *,
    seq: np.ndarray,
    pq,
    quality,
    rule: ClassificationResult,
    phase,
    ml_proba: dict[str, float],
    ml_label: str | None,
    buffer_completeness: float,
    id_switched: bool,
    track_fragmented: bool,
    version: str,
) -> HybridDecisionEvidence:
    p_remove = float(ml_proba.get(ActionClass.HELMET_REMOVE.value, 0.0))
    r_score = rule_remove_score(rule.label.value, rule.confidence)
    m_score = motion_score_from_features(rule.features)
    pconf = float(getattr(phase, "phase_confidence", 0.0) or 0.0)
    if pconf <= 0:
        pconf = phase_confidence(phase.saw_approach, phase.saw_grasp, phase.saw_lift, phase.ordered)
    phase_score = phase_score_from_confidences(
        float(getattr(phase, "grasp_confidence", 0.0) or 0.0),
        float(getattr(phase, "lift_confidence", 0.0) or 0.0),
        float(getattr(phase, "separation_confidence", 0.0) or 0.0),
        bool(phase.ordered),
    )
    tracking = 1.0
    if id_switched:
        tracking = 0.0
    elif track_fragmented:
        tracking = 0.20
    physically_ok = bool(phase.saw_grasp and phase.saw_lift and phase.ordered)
    return HybridDecisionEvidence(
        ml_remove_probability=p_remove,
        ml_label=ml_label,
        rule_remove_score=r_score,
        rule_label=rule.label.value,
        rule_confidence=float(rule.confidence),
        phase=phase.phase.value,
        phase_confidence=pconf,
        grasp_confidence=float(getattr(phase, "grasp_confidence", 0.0) or 0.0),
        lift_confidence=float(getattr(phase, "lift_confidence", 0.0) or 0.0),
        separation_confidence=float(getattr(phase, "separation_confidence", 0.0) or 0.0),
        pose_quality=float(pq.score),
        wrist_quality=float(getattr(pq, "wrist_quality", 1.0) or 1.0),
        ear_quality=float(getattr(pq, "ear_quality", 1.0) or 1.0),
        buffer_completeness=float(buffer_completeness),
        tracking_quality=float(tracking),
        motion_score=m_score,
        ml_score=p_remove,
        rule_score=r_score,
        phase_score=phase_score,
        n_frames=int(seq.shape[0]),
        both_wrist_streak=int(getattr(pq, "both_wrist_streak", 0) or 0),
        longest_ear_streak=int(getattr(pq, "longest_ear_streak", 0) or 0),
        longest_wrist_streak=int(getattr(pq, "longest_wrist_streak", 0) or 0),
        id_switched=bool(id_switched),
        track_fragmented=bool(track_fragmented),
        phase_ordered=bool(phase.ordered),
        saw_approach=bool(phase.saw_approach),
        saw_grasp=bool(phase.saw_grasp),
        saw_partial_grasp=bool(getattr(phase, "saw_partial_grasp", False)),
        saw_lift=bool(phase.saw_lift),
        brim_grasp=bool(getattr(phase, "brim_grasp", False)),
        one_then_two=bool(getattr(phase, "one_then_two", False)),
        lateral_lift=bool(getattr(phase, "lateral_lift", False)),
        physically_ok=physically_ok,
        hybrid_version=version,
    )


def _decide_v1_hard_gate(
    *,
    cfg,
    rule: ClassificationResult,
    phase,
    ml_proba: dict[str, float],
    ml_label: str | None,
    pq,
    quality,
    buffer_completeness: float,
    ml_present: bool,
) -> tuple[ActionClass, ActionEvent, float, float, list[str], str]:
    """Existing AND-gate baseline. Do not delete."""
    notes: list[str] = []
    notes.extend(pq.notes)
    notes.extend(rule.explanation)
    notes.append(f"phase={phase.phase.value} ordered={phase.ordered} grasp={phase.saw_grasp} lift={phase.saw_lift}")
    p_remove = float(ml_proba.get(ActionClass.HELMET_REMOVE.value, 0.0))
    p_puton = float(ml_proba.get(ActionClass.HELMET_PUT_ON.value, 0.0))
    p_scratch = float(ml_proba.get(ActionClass.HEAD_SCRATCH.value, 0.0))
    p_adjust = float(ml_proba.get(ActionClass.HELMET_ADJUST.value, 0.0))
    high = float(cfg.get("decision.ml_high_threshold", 0.75))
    mid = float(cfg.get("decision.ml_remove_threshold", 0.55))
    if ml_label:
        notes.append(f"ML argmax={ml_label} P(remove)={p_remove:.2f} P(adjust)={p_adjust:.2f} P(scratch)={p_scratch:.2f}")

    action = baseline_to_action(rule.label)
    conf = rule.confidence
    event = ActionEvent.NONE
    risk = 0.0
    physically_ok = phase.saw_grasp and phase.saw_lift and phase.ordered
    physically_intent = phase.saw_approach or phase.saw_grasp or bool(getattr(phase, "saw_partial_grasp", False))

    if not ml_present:
        if rule.label is ActionLabel.HELMET_OFF:
            action = ActionClass.HELMET_REMOVE
            risk = rule.confidence
            event = ActionEvent.REMOVE_CONFIRMED if physically_ok else ActionEvent.REMOVE_INTENT
        elif rule.label is ActionLabel.SCRATCH:
            action = ActionClass.HEAD_SCRATCH
            risk = 0.05
        else:
            action = baseline_to_action(rule.label)
            risk = 0.1 if rule.label is ActionLabel.UNKNOWN_CONTACT else 0.0
    else:
        if p_remove >= high and physically_ok:
            action = ActionClass.HELMET_REMOVE
            conf = max(p_remove, rule.confidence if rule.label is ActionLabel.HELMET_OFF else p_remove)
            risk = p_remove
            event = ActionEvent.REMOVE_CONFIRMED
            notes.append("Rule grasp+lift 와 ML 고확률이 일치 → REMOVE_CONFIRMED.")
        elif p_remove >= high and not physically_ok:
            action = ActionClass.UNKNOWN
            conf = min(p_remove, 0.45)
            risk = 0.35
            event = ActionEvent.REMOVE_INTENT if physically_intent else ActionEvent.NONE
            notes.append("ML P(remove)는 높지만 phase 순서가 물리적으로 불완전 → UNKNOWN.")
        elif rule.label is ActionLabel.HELMET_OFF and p_remove >= mid and (physically_ok or phase.saw_lift):
            action = ActionClass.HELMET_REMOVE
            conf = 0.5 * (rule.confidence + p_remove)
            risk = max(rule.confidence, p_remove)
            event = ActionEvent.REMOVE_CONFIRMED if physically_ok else ActionEvent.REMOVE_INTENT
            notes.append("Rule HELMET_OFF + ML 중확률 + lift 신호.")
        elif p_puton >= high and phase.saw_grasp:
            action = ActionClass.HELMET_PUT_ON
            conf = p_puton
            event = ActionEvent.PUT_ON_CONFIRMED
            risk = 0.05
        elif p_adjust >= p_remove and p_adjust >= 0.45 and not physically_ok:
            action = ActionClass.HELMET_ADJUST
            conf = p_adjust
            risk = 0.15
            notes.append("고쳐 쓰기 쪽이 더 높고 lift 확정이 없음.")
        elif p_scratch >= 0.45 and p_remove < mid:
            action = ActionClass.HEAD_SCRATCH
            conf = max(p_scratch, rule.confidence if rule.label is ActionLabel.SCRATCH else p_scratch)
            risk = 0.05
        elif ml_label:
            action = ActionClass(ml_label) if ml_label in ActionClass._value2member_map_ else ActionClass.UNKNOWN
            conf = ml_proba.get(ml_label, 0.3)
            risk = p_remove * 0.5
        if action is ActionClass.HELMET_REMOVE and p_adjust > p_remove + 0.10 and not physically_ok:
            action = ActionClass.HELMET_ADJUST
            risk = 0.2
            notes.append("ADJUST가 REMOVE보다 높고 양손 리프트가 없어 강등.")

    min_comp = float(cfg.get("decision.min_buffer_completeness", 0.45))
    reject_margin = float(cfg.get("decision.reject_top_margin", 0.06))
    reject_inconsistent = bool(cfg.get("decision.reject_inconsistent_phase", True))
    if buffer_completeness < min_comp:
        notes.append(f"PoseBuffer completeness {buffer_completeness:.2f} < {min_comp:.2f} → UNKNOWN.")
        if action is ActionClass.HELMET_REMOVE:
            event = ActionEvent.REMOVE_INTENT if physically_intent else ActionEvent.NONE
        action = ActionClass.UNKNOWN
        conf = min(conf, 0.35)
        risk = min(risk, 0.25)
    if ml_proba:
        ordered_p = sorted(ml_proba.values(), reverse=True)
        max_p = float(ordered_p[0]) if ordered_p else 0.0
        margin = float(ordered_p[0] - ordered_p[1]) if len(ordered_p) > 1 else max_p
        if action is ActionClass.HELMET_REMOVE and not physically_ok:
            action = ActionClass.UNKNOWN
            event = ActionEvent.REMOVE_INTENT if physically_intent else ActionEvent.NONE
            conf = min(conf, 0.45)
            notes.append("HELMET_REMOVE without complete grasp→lift history → UNKNOWN / REMOVE_INTENT.")
        if reject_inconsistent and action is ActionClass.HELMET_REMOVE and not phase.ordered:
            action = ActionClass.UNKNOWN
            event = ActionEvent.REMOVE_INTENT if physically_intent else ActionEvent.NONE
            notes.append("phase sequence inconsistent → UNKNOWN.")
        if action not in (ActionClass.IDLE, ActionClass.INSUFFICIENT_POSE) and margin < reject_margin and not physically_ok:
            notes.append(f"top1-top2 margin {margin:.3f} < {reject_margin:.3f} → UNKNOWN.")
            action = ActionClass.UNKNOWN
            conf = min(conf, 0.40)
            if event is ActionEvent.REMOVE_CONFIRMED:
                event = ActionEvent.REMOVE_INTENT if physically_intent else ActionEvent.NONE

    status = pq.decision_status
    if status == DecisionStatus.UNKNOWN.value:
        notes.append("PoseQualityScore → UNKNOWN (do not confirm SAFE or REMOVE).")
        if action is ActionClass.HELMET_REMOVE:
            event = ActionEvent.REMOVE_INTENT if physically_intent else ActionEvent.NONE
        action = ActionClass.UNKNOWN
        conf = min(conf, 0.35)
        risk = min(risk, 0.25)
    elif status == DecisionStatus.LOW_CONFIDENCE.value:
        notes.append("PoseQualityScore → LOW_CONFIDENCE.")
        if action is ActionClass.HELMET_REMOVE and not physically_ok:
            action = ActionClass.UNKNOWN
            event = ActionEvent.REMOVE_INTENT if physically_intent else ActionEvent.NONE
            notes.append("low pose quality + incomplete phase → UNKNOWN, not a safe confirmation.")
        if action.value in SAFE_CONFIRMED_ACTIONS and pq.longest_wrist_streak > 3:
            action = ActionClass.UNKNOWN
            conf = min(conf, 0.40)
            notes.append("wrist gaps remain after max_gap interpolation → refuse SAFE confirmation.")

    if _ear_blocks_safe(pq, cfg) and action.value in SAFE_CONFIRMED_ACTIONS:
        notes.append("long ear occlusion / low ear quality → refuse SAFE, prefer UNKNOWN.")
        action = ActionClass.UNKNOWN
        conf = min(conf, 0.40)
        risk = min(risk, 0.25)

    if action is ActionClass.UNKNOWN:
        status = DecisionStatus.UNKNOWN.value
    elif action is ActionClass.INSUFFICIENT_POSE:
        status = DecisionStatus.INSUFFICIENT_POSE.value
    elif status == DecisionStatus.VALID.value and quality.quality.value != "OK":
        status = DecisionStatus.LOW_CONFIDENCE.value
    return action, event, float(conf), float(risk), notes, status


def _decide_v2_fusion(
    *,
    cfg,
    evidence: HybridDecisionEvidence,
    rule: ClassificationResult,
    phase,
    ml_proba: dict[str, float],
    ml_label: str | None,
    pq,
    quality,
) -> tuple[ActionClass, ActionEvent, float, float, list[str], str, RiskLevel]:
    notes: list[str] = []
    notes.extend(pq.notes)
    notes.extend(rule.explanation)
    notes.append(
        f"V2 fusion evidence: ml={evidence.ml_score:.2f} rule={evidence.rule_score:.2f} "
        f"phase={evidence.phase_score:.2f} motion={evidence.motion_score:.2f} "
        f"grasp={evidence.grasp_confidence:.2f} lift={evidence.lift_confidence:.2f} "
        f"sep={evidence.separation_confidence:.2f}"
    )
    fusion_cfg = cfg.section("decision.fusion") or {}
    w_ml = float(fusion_cfg.get("w_ml", cfg.get("decision.fusion.w_ml", 0.40)))
    w_rule = float(fusion_cfg.get("w_rule", cfg.get("decision.fusion.w_rule", 0.20)))
    w_phase = float(fusion_cfg.get("w_phase", cfg.get("decision.fusion.w_phase", 0.30)))
    w_motion = float(fusion_cfg.get("w_motion", cfg.get("decision.fusion.w_motion", 0.10)))
    alert_th = float(fusion_cfg.get("alert_threshold", cfg.get("decision.fusion.alert_threshold", 0.62)))
    watch_th = float(fusion_cfg.get("watch_threshold", cfg.get("decision.fusion.watch_threshold", 0.48)))
    pose_critical = float(cfg.get("decision.pose_critical", 0.32))
    min_frames = int(cfg.get("window.min_frames", 12))

    gate = evaluate_safety_gates(
        evidence,
        pose_critical=pose_critical,
        min_frames=min_frames,
        both_wrist_long=15,
        both_wrist_unknown=interpolation_max_gap() + 1,
    )
    evidence.safety_gate_triggered = gate.triggered
    evidence.rejection_reasons.extend(gate.reasons)
    notes.extend(gate.notes)

    remove_score = fused_remove_score(evidence, w_ml, w_rule, w_phase, w_motion)
    evidence.fusion_remove_score = remove_score
    evidence.removal_evidence = remove_score
    p_remove = evidence.ml_remove_probability
    p_puton = float(ml_proba.get(ActionClass.HELMET_PUT_ON.value, 0.0))
    p_scratch = float(ml_proba.get(ActionClass.HEAD_SCRATCH.value, 0.0))
    p_adjust = float(ml_proba.get(ActionClass.HELMET_ADJUST.value, 0.0))
    p_touch = float(ml_proba.get(ActionClass.HEAD_TOUCH.value, 0.0))
    p_idle = float(ml_proba.get(ActionClass.IDLE.value, 0.0))
    high = float(cfg.get("decision.ml_high_threshold", 0.75))

    if gate.force_insufficient:
        notes.append("safety gate → INSUFFICIENT_POSE (both-wrist / critical pose).")
        return (
            ActionClass.INSUFFICIENT_POSE,
            ActionEvent.NONE,
            0.0,
            0.0,
            notes,
            DecisionStatus.INSUFFICIENT_POSE.value,
            RiskLevel.UNKNOWN,
        )
    if gate.force_unknown:
        notes.append("safety gate → UNKNOWN; ML probability is ignored for confirmation.")
        return (
            ActionClass.UNKNOWN,
            ActionEvent.NONE,
            min(0.35, remove_score),
            min(0.25, remove_score),
            notes,
            DecisionStatus.UNKNOWN.value,
            RiskLevel.UNKNOWN,
        )

    action = baseline_to_action(rule.label)
    conf = float(max(remove_score, rule.confidence * 0.5))
    event = ActionEvent.NONE
    risk = float(remove_score)
    status = pq.decision_status
    physically_intent = evidence.saw_approach or evidence.saw_grasp or evidence.saw_partial_grasp

    phase_support = evidence.phase_score >= 0.38 or (
        evidence.grasp_confidence >= 0.40 and evidence.lift_confidence >= 0.35
    )
    pose_ok = evidence.pose_quality >= 0.52 and evidence.wrist_quality >= 0.28

    if remove_score >= alert_th and phase_support and pose_ok:
        action = ActionClass.HELMET_REMOVE
        event = ActionEvent.REMOVE_CONFIRMED
        conf = max(remove_score, p_remove)
        risk = max(remove_score, p_remove)
        notes.append("V2 evidence fusion ≥ alert threshold with phase support → REMOVE_CONFIRMED.")
    elif remove_score >= watch_th and (p_remove >= 0.45 or evidence.rule_score >= 0.40 or evidence.phase_score >= 0.40):
        action = ActionClass.UNKNOWN
        event = ActionEvent.REMOVE_INTENT if physically_intent else ActionEvent.NONE
        conf = min(remove_score, 0.55)
        risk = remove_score
        notes.append("V2 evidence is partial → WATCH / UNKNOWN (not SAFE, not ALERT).")
    elif p_puton >= high and evidence.saw_grasp and p_puton > p_remove + 0.05:
        action = ActionClass.HELMET_PUT_ON
        conf = p_puton
        event = ActionEvent.PUT_ON_CONFIRMED
        risk = 0.05
    elif p_adjust >= 0.45 and p_adjust >= p_remove and remove_score < alert_th:
        action = ActionClass.HELMET_ADJUST
        conf = p_adjust
        risk = 0.15
    elif p_scratch >= 0.45 and p_remove < 0.55 and remove_score < alert_th:
        action = ActionClass.HEAD_SCRATCH
        conf = p_scratch
        risk = 0.05
    elif p_touch >= 0.45 and p_remove < 0.50 and remove_score < watch_th:
        action = ActionClass.HEAD_TOUCH
        conf = p_touch
        risk = 0.08
    elif p_idle >= 0.50 and p_remove < 0.40 and remove_score < watch_th:
        action = ActionClass.IDLE
        conf = p_idle
        risk = 0.0
    elif ml_label:
        mapped = ActionClass(ml_label) if ml_label in ActionClass._value2member_map_ else ActionClass.UNKNOWN
        if mapped is ActionClass.HELMET_REMOVE and remove_score < alert_th:
            action = ActionClass.UNKNOWN
            event = ActionEvent.REMOVE_INTENT if physically_intent else ActionEvent.NONE
            notes.append("ML argmax REMOVE but fused evidence below alert → UNKNOWN.")
        else:
            action = mapped
            conf = ml_proba.get(ml_label, 0.3)
            risk = p_remove * 0.5

    # Ear occlusion: never confirm SAFE when ear evidence is gone.
    if _ear_blocks_safe(pq, cfg) and action.value in SAFE_CONFIRMED_ACTIONS:
        notes.append("long ear occlusion → UNKNOWN over SAFE.")
        action = ActionClass.UNKNOWN
        conf = min(conf, 0.40)
        risk = min(risk, 0.30)
        evidence.rejection_reasons.append("LOW_EAR_QUALITY")

    # Pose LOW_CONFIDENCE: still allow WATCH/UNKNOWN, refuse SAFE if wrists/ears are weak.
    if status == DecisionStatus.LOW_CONFIDENCE.value and action.value in SAFE_CONFIRMED_ACTIONS:
        if evidence.longest_wrist_streak > 3 or evidence.longest_ear_streak >= 20:
            action = ActionClass.UNKNOWN
            notes.append("low pose quality: refuse SAFE confirmation.")

    proposed = propose_risk_level(
        action=action,
        remove_score=remove_score,
        pose_quality=evidence.pose_quality,
        phase_score=evidence.phase_score,
        ml_remove=p_remove,
        watch_threshold=watch_th,
        alert_threshold=alert_th,
        safety_unknown=action in (ActionClass.UNKNOWN, ActionClass.INSUFFICIENT_POSE) and remove_score < watch_th,
    )
    if action is ActionClass.UNKNOWN and remove_score >= watch_th:
        proposed = RiskLevel.WATCH
    if action is ActionClass.HELMET_REMOVE:
        proposed = RiskLevel.ALERT if (phase_support and pose_ok) else RiskLevel.WATCH

    if action is ActionClass.UNKNOWN:
        status = DecisionStatus.UNKNOWN.value
    elif action is ActionClass.INSUFFICIENT_POSE:
        status = DecisionStatus.INSUFFICIENT_POSE.value
    elif status == DecisionStatus.VALID.value and quality.quality.value != "OK":
        status = DecisionStatus.LOW_CONFIDENCE.value
    return action, event, float(conf), float(risk), notes, status, proposed


_ML_AUTOLOAD = object()


class HybridActionClassifier:
    def __init__(
        self,
        ml: SklearnActionClassifier | None | object = _ML_AUTOLOAD,
        helmet_sm: HelmetStateMachine | None = None,
        alert_gate: AlertGate | None = None,
        version: str | None = None,
        escalator: RiskEscalator | None = None,
    ) -> None:
        self.rule = RuleBasedActionClassifier()
        if ml is _ML_AUTOLOAD:
            self.ml = SklearnActionClassifier.try_load()
        else:
            self.ml = ml  # type: ignore[assignment]
        self.helmet_sm = helmet_sm or HelmetStateMachine()
        cfg = load_config()
        self.gate = alert_gate or AlertGate(
            enter=float(cfg.get("decision.alert_enter", 0.75)),
            exit=float(cfg.get("decision.alert_exit", 0.45)),
            window=int(cfg.get("decision.confirm_window", 5)),
            hits=int(cfg.get("decision.confirm_hits", 3)),
        )
        self.detector = DummyHelmetPresenceDetector()
        self.version = (version or str(cfg.get("decision.hybrid_version", "v1"))).lower()
        min_win = int(cfg.get("decision.fusion.min_windows_for_alert", 2))
        self.escalator = escalator or RiskEscalator(min_windows_for_alert=min_win)

    def predict_v1(self, keypoints: np.ndarray, confidence: np.ndarray | None = None, **kwargs) -> HybridDecision:
        return self.predict(keypoints, confidence, version="v1", **kwargs)

    def predict_v2(self, keypoints: np.ndarray, confidence: np.ndarray | None = None, **kwargs) -> HybridDecision:
        return self.predict(keypoints, confidence, version="v2", **kwargs)

    def predict(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray | None = None,
        buffer_completeness: float = 1.0,
        *,
        version: str | None = None,
        track_id: int = 0,
        id_switched: bool = False,
        track_fragmented: bool = False,
        streaming: bool = False,
    ) -> HybridDecision:
        cfg = load_config()
        version = (version or self.version or "v1").lower()
        seq, quality = prepare_sequence(keypoints, confidence)
        pq = compute_pose_quality_score(
            keypoints,
            confidence,
            buffer_completeness=buffer_completeness,
            interpolation_ratio=quality.interpolation_ratio,
            repaired=seq,
        )
        notes: list[str] = []
        notes.extend(pq.notes)
        if (not quality.usable) or pq.decision_status == DecisionStatus.INSUFFICIENT_POSE.value:
            notes.extend(quality.notes)
            notes.append("관절 신뢰도 부족 → UNKNOWN / INSUFFICIENT_POSE. 정상·위험 모두 확정하지 않습니다.")
            evidence = HybridDecisionEvidence(
                pose_quality=pq.score,
                wrist_quality=float(getattr(pq, "wrist_quality", 0.0) or 0.0),
                ear_quality=float(getattr(pq, "ear_quality", 0.0) or 0.0),
                buffer_completeness=float(buffer_completeness),
                n_frames=int(np.asarray(keypoints).shape[0]),
                both_wrist_streak=int(getattr(pq, "both_wrist_streak", 0) or 0),
                longest_ear_streak=int(getattr(pq, "longest_ear_streak", 0) or 0),
                longest_wrist_streak=int(getattr(pq, "longest_wrist_streak", 0) or 0),
                id_switched=id_switched,
                track_fragmented=track_fragmented,
                safety_gate_triggered=True,
                hybrid_version=version,
                risk_level=RiskLevel.UNKNOWN.value,
                rejection_reasons=["POSE_CRITICAL", "HARD_SAFETY_GATE"],
            )
            alert = self.gate.update(0.0)
            if streaming:
                self.escalator.update(RiskLevel.UNKNOWN, track_id=track_id)
            return HybridDecision(
                action=ActionClass.INSUFFICIENT_POSE,
                baseline=ActionLabel.UNKNOWN_CONTACT,
                confidence=0.0,
                rule_label="insufficient",
                rule_confidence=0.0,
                ml_label=None,
                ml_proba={},
                phase=RemovalPhase.IDLE,
                phase_ordered=False,
                helmet_state=self.helmet_sm.state,
                event=ActionEvent.NONE,
                risk=0.0,
                alert=alert,
                quality=quality.quality.value,
                explanation=notes,
                action_probability=0.0,
                pose_quality=pq.score,
                phase_confidence=0.0,
                decision_status=DecisionStatus.INSUFFICIENT_POSE.value,
                hybrid_version=version,
                risk_level=RiskLevel.UNKNOWN.value,
                rejection_reasons=list(evidence.rejection_reasons),
                evidence=evidence,
            )

        rule: ClassificationResult = self.rule.predict(seq)
        phase = infer_phases(seq)
        ml_proba: dict[str, float] = {}
        ml_label = None
        if self.ml is not None:
            ml_proba = self.ml.predict_proba(seq, confidence)
            ml_label = max(ml_proba, key=ml_proba.get)

        evidence = _gather_evidence(
            seq=seq,
            pq=pq,
            quality=quality,
            rule=rule,
            phase=phase,
            ml_proba=ml_proba,
            ml_label=ml_label,
            buffer_completeness=buffer_completeness,
            id_switched=id_switched,
            track_fragmented=track_fragmented,
            version=version,
        )
        proposed_risk = RiskLevel.UNKNOWN
        if version == "v2":
            action, event, conf, risk, extra_notes, status, proposed_risk = _decide_v2_fusion(
                cfg=cfg,
                evidence=evidence,
                rule=rule,
                phase=phase,
                ml_proba=ml_proba,
                ml_label=ml_label,
                pq=pq,
                quality=quality,
            )
        else:
            action, event, conf, risk, extra_notes, status = _decide_v1_hard_gate(
                cfg=cfg,
                rule=rule,
                phase=phase,
                ml_proba=ml_proba,
                ml_label=ml_label,
                pq=pq,
                quality=quality,
                buffer_completeness=buffer_completeness,
                ml_present=self.ml is not None,
            )
            evidence.removal_evidence = 1.0 if action is ActionClass.HELMET_REMOVE else (0.45 if event is ActionEvent.REMOVE_INTENT else 0.0)
            proposed_risk = propose_risk_level(
                action=action,
                remove_score=float(evidence.ml_remove_probability if action is ActionClass.HELMET_REMOVE else evidence.removal_evidence),
                pose_quality=evidence.pose_quality,
                phase_score=evidence.phase_score,
                ml_remove=evidence.ml_remove_probability,
                watch_threshold=float(cfg.get("decision.fusion.watch_threshold", 0.48)),
                alert_threshold=float(cfg.get("decision.fusion.alert_threshold", 0.62)),
                safety_unknown=action in (ActionClass.UNKNOWN, ActionClass.INSUFFICIENT_POSE),
            )
        notes.extend(extra_notes)

        risk_level = proposed_risk
        if streaming:
            risk_level = self.escalator.update(proposed_risk, track_id=track_id)
            action, risk_level = apply_escalation_to_action(action, risk_level, proposed_risk)
            if risk_level is RiskLevel.WATCH and event is ActionEvent.REMOVE_CONFIRMED:
                event = ActionEvent.REMOVE_INTENT
                notes.append("streaming debounce: WATCH, not yet ALERT.")

        evidence.rejection_reasons = collect_rejection_reasons(evidence, action)
        evidence.risk_level = risk_level.value
        evidence.notes = list(notes)

        if action is ActionClass.UNKNOWN:
            status = DecisionStatus.UNKNOWN.value
        elif action is ActionClass.INSUFFICIENT_POSE:
            status = DecisionStatus.INSUFFICIENT_POSE.value

        helmet = self.helmet_sm.update(event)
        alert = self.gate.update(risk)
        if alert and action is ActionClass.HELMET_REMOVE:
            notes.append("hysteresis: ALERT (enter/exit).")
        elif not alert:
            notes.append("hysteresis: CLEAR.")

        notes.append(
            "HelmetState는 검출기 없이 UNKNOWN을 유지합니다. "
            "Pose만으로 착용/미착용 정지 상태는 구분할 수 없습니다."
        )
        action_p = float(ml_proba.get(action.value, conf)) if ml_proba else float(conf)
        return HybridDecision(
            action=action,
            baseline=action_to_baseline(action),
            confidence=float(np.clip(conf, 0.0, 0.99)),
            rule_label=rule.label.value,
            rule_confidence=rule.confidence,
            ml_label=ml_label,
            ml_proba=ml_proba,
            phase=phase.phase,
            phase_ordered=phase.ordered,
            helmet_state=helmet,
            event=event,
            risk=float(np.clip(risk, 0.0, 1.0)),
            alert=alert,
            quality=quality.quality.value,
            explanation=notes,
            features=rule.features.__dict__,
            frame_labels=rule.frame_labels,
            action_probability=float(np.clip(action_p, 0.0, 1.0)),
            pose_quality=float(pq.score),
            phase_confidence=float(evidence.phase_confidence),
            decision_status=status,
            hybrid_version=version,
            risk_level=risk_level.value,
            rejection_reasons=list(evidence.rejection_reasons),
            evidence=evidence,
        )


# Explicit aliases requested by the arbitration spec.
HybridDecisionV1 = HybridDecision
HybridDecisionV2 = HybridDecision
