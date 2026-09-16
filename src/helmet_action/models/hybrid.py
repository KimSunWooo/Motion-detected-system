from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from helmet_action.config import load_config
from helmet_action.models.labels import (
    ActionClass,
    ActionLabel,
    BaselineActionLabel,
    ClassificationResult,
    LABEL_KO,
    RemovalPhase,
    action_to_baseline,
    baseline_to_action,
)
from helmet_action.models.rule_based import RuleBasedActionClassifier
from helmet_action.models.temporal_classifier import SklearnActionClassifier
from helmet_action.pose.confidence import prepare_sequence
from helmet_action.pose.types import PoseQuality
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

    def to_dict(self) -> dict:
        return {
            "final_prediction": self.action.value,
            "final_ko": {a: a.value for a in ActionClass}.get(self.action, self.action.value),
            "label": self.baseline.value,
            "label_ko": LABEL_KO.get(self.baseline, self.action.value),
            "confidence": self.confidence,
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


class HybridActionClassifier:
    def __init__(
        self,
        ml: SklearnActionClassifier | None = None,
        helmet_sm: HelmetStateMachine | None = None,
        alert_gate: AlertGate | None = None,
    ) -> None:
        self.rule = RuleBasedActionClassifier()
        self.ml = ml if ml is not None else SklearnActionClassifier.try_load()
        self.helmet_sm = helmet_sm or HelmetStateMachine()
        cfg = load_config()
        self.gate = alert_gate or AlertGate(
            enter=float(cfg.get("decision.alert_enter", 0.75)),
            exit=float(cfg.get("decision.alert_exit", 0.45)),
            window=int(cfg.get("decision.confirm_window", 5)),
            hits=int(cfg.get("decision.confirm_hits", 3)),
        )
        self.detector = DummyHelmetPresenceDetector()

    def predict(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray | None = None,
        buffer_completeness: float = 1.0,
    ) -> HybridDecision:
        cfg = load_config()
        seq, quality = prepare_sequence(keypoints, confidence)
        notes: list[str] = []
        if not quality.usable:
            notes.extend(quality.notes)
            notes.append("관절 신뢰도 부족 → UNKNOWN / INSUFFICIENT_POSE. 정상·위험 모두 확정하지 않습니다.")
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
                alert=self.gate.update(0.0),
                quality=quality.quality.value,
                explanation=notes,
            )

        rule: ClassificationResult = self.rule.predict(seq)
        phase = infer_phases(seq)
        ml_proba: dict[str, float] = {}
        ml_label = None
        if self.ml is not None:
            ml_proba = self.ml.predict_proba(seq, confidence)
            ml_label = max(ml_proba, key=ml_proba.get)

        p_remove = float(ml_proba.get(ActionClass.HELMET_REMOVE.value, 0.0))
        p_puton = float(ml_proba.get(ActionClass.HELMET_PUT_ON.value, 0.0))
        p_scratch = float(ml_proba.get(ActionClass.HEAD_SCRATCH.value, 0.0))
        p_adjust = float(ml_proba.get(ActionClass.HELMET_ADJUST.value, 0.0))
        high = float(cfg.get("decision.ml_high_threshold", 0.75))
        mid = float(cfg.get("decision.ml_remove_threshold", 0.55))

        notes.extend(rule.explanation)
        notes.append(f"phase={phase.phase.value} ordered={phase.ordered} grasp={phase.saw_grasp} lift={phase.saw_lift}")
        if ml_label:
            notes.append(f"ML argmax={ml_label} P(remove)={p_remove:.2f} P(adjust)={p_adjust:.2f} P(scratch)={p_scratch:.2f}")

        action = baseline_to_action(rule.label)
        conf = rule.confidence
        event = ActionEvent.NONE
        risk = 0.0

        physically_ok = phase.saw_grasp and phase.saw_lift and phase.ordered
        physically_intent = phase.saw_approach or phase.saw_grasp

        if self.ml is None:
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

        # UNKNOWN rejection: never confirm REMOVE without phase history or usable pose.
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
            if (
                reject_inconsistent
                and action is ActionClass.HELMET_REMOVE
                and not phase.ordered
            ):
                action = ActionClass.UNKNOWN
                event = ActionEvent.REMOVE_INTENT if physically_intent else ActionEvent.NONE
                notes.append("phase sequence inconsistent → UNKNOWN.")
            if action not in (ActionClass.IDLE, ActionClass.INSUFFICIENT_POSE) and margin < reject_margin and not physically_ok:
                notes.append(f"top1-top2 margin {margin:.3f} < {reject_margin:.3f} → UNKNOWN.")
                action = ActionClass.UNKNOWN
                conf = min(conf, 0.40)
                if event is ActionEvent.REMOVE_CONFIRMED:
                    event = ActionEvent.REMOVE_INTENT if physically_intent else ActionEvent.NONE

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
        )
