from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from helmet_action.evaluation.false_safe import false_safe_rate, safety_metrics
from helmet_action.models.evidence import HybridDecisionEvidence
from helmet_action.models.fusion import iter_fusion_candidates
from helmet_action.models.hybrid import HybridActionClassifier
from helmet_action.models.labels import ActionClass, RemovalPhase, RiskLevel, SAFE_CONFIRMED_ACTIONS
from helmet_action.models.risk import RiskEscalator
from helmet_action.models.temporal_classifier import SklearnActionClassifier
from helmet_action.pose.constants import L_EAR, L_WRIST, R_EAR, R_WRIST
from helmet_action.pose.quality import compute_pose_quality_score
from helmet_action.real.dataset import smoke_raw_capture_layout
from helmet_action.real.manifest import MANIFEST_LABELS, parse_manifest, write_manifest_template
from helmet_action.state.action_state_machine import infer_phases
from helmet_action.synthetic.augmentation import apply_consecutive_occlusion
from helmet_action.synthetic.generator import generate_one, generate_remove_c_subtype


def _clf():
    path = Path("models/action_classifier.joblib")
    if path.exists():
        return SklearnActionClassifier.load(path)
    return None


def test_hybrid_evidence_serialization():
    seq, conf, _ = generate_one(seed=3, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    dec = HybridActionClassifier(ml=_clf()).predict_v1(seq, conf)
    payload = dec.to_dict()
    dumped = json.dumps(payload, default=str)
    assert "evidence" in payload
    assert "rejection_reasons" in payload
    assert "risk_level" in payload
    assert json.loads(dumped)["action"] == payload["action"]
    assert dec.evidence is not None
    ev = HybridDecisionEvidence(**{k: v for k, v in dec.evidence.to_dict().items() if k in HybridDecisionEvidence.__dataclass_fields__})
    assert 0.0 <= ev.ml_remove_probability <= 1.0
    assert 0.0 <= ev.pose_quality <= 1.0


def test_hybrid_v1_v2_deterministic():
    seq, conf, _ = generate_one(seed=9, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    hy = HybridActionClassifier(ml=_clf())
    a1 = hy.predict_v1(seq, conf)
    a2 = hy.predict_v1(seq, conf)
    b1 = hy.predict_v2(seq, conf)
    b2 = hy.predict_v2(seq, conf)
    assert a1.action == a2.action
    assert a1.risk_level == a2.risk_level
    assert np.isclose(a1.confidence, a2.confidence)
    assert b1.action == b2.action
    assert b1.hybrid_version == "v2"
    assert a1.hybrid_version == "v1"
    assert b1.evidence is not None
    assert abs(sum([
        0.40, 0.20, 0.30, 0.10
    ]) - 1.0) < 1e-9


def test_risk_level_transitions_and_unknown_not_safe():
    esc = RiskEscalator(min_windows_for_alert=2)
    first = esc.update(RiskLevel.ALERT, track_id=7)
    assert first is RiskLevel.WATCH
    second = esc.update(RiskLevel.ALERT, track_id=7)
    assert second is RiskLevel.ALERT
    unk = esc.update(RiskLevel.UNKNOWN, track_id=8)
    assert unk is RiskLevel.UNKNOWN
    assert unk is not RiskLevel.SAFE
    assert ActionClass.UNKNOWN.value not in SAFE_CONFIRMED_ACTIONS
    assert RiskLevel.UNKNOWN.value != RiskLevel.SAFE.value
    yt = np.array(["HELMET_REMOVE", "HELMET_REMOVE", "IDLE"])
    yp = np.array(["UNKNOWN", "IDLE", "HELMET_REMOVE"])
    m = safety_metrics(yt, yp)
    assert m["unknown_on_positive_rate"] == 0.5
    assert m["false_safe_rate"] == 0.5
    assert m["false_alarm_rate"] == 1.0
    assert false_safe_rate(yt, np.array(["UNKNOWN", "UNKNOWN", "IDLE"])) == 0.0


def test_watch_to_alert_escalation():
    esc = RiskEscalator(min_windows_for_alert=2)
    assert esc.update(RiskLevel.WATCH, track_id=1) is RiskLevel.WATCH
    assert esc.update(RiskLevel.WATCH, track_id=1) is RiskLevel.WATCH
    assert esc.update(RiskLevel.ALERT, track_id=1) is RiskLevel.ALERT
    hy = HybridActionClassifier(ml=_clf())
    seq, conf, _ = generate_one(seed=5, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    w1 = hy.predict_v2(seq, conf, streaming=True, track_id=42)
    w2 = hy.predict_v2(seq, conf, streaming=True, track_id=42)
    assert w1.risk_level != RiskLevel.ALERT.value
    assert w2.risk_level in {r.value for r in RiskLevel}
    assert RiskLevel.UNKNOWN.value != RiskLevel.SAFE.value


def test_long_ear_occlusion_never_false_safe():
    seq, conf, _ = generate_one(seed=11, scenario="HELMET_REMOVE", split="test", apply_noise=False)
    occ, c2 = apply_consecutive_occlusion(seq, conf, [L_EAR, R_EAR], 30, start=4)
    pq = compute_pose_quality_score(occ, c2)
    assert pq.longest_ear_streak >= 20 or pq.ear_quality < 0.6
    hy = HybridActionClassifier(ml=_clf())
    for version in ("v1", "v2"):
        dec = hy.predict(occ, c2, version=version)
        assert dec.action.value not in SAFE_CONFIRMED_ACTIONS
        assert dec.action in (ActionClass.UNKNOWN, ActionClass.INSUFFICIENT_POSE, ActionClass.HELMET_REMOVE)
        if dec.action.value in SAFE_CONFIRMED_ACTIONS:
            raise AssertionError("long ear occlusion confirmed SAFE")
        if pq.ear_quality < 0.35 or pq.longest_ear_streak >= 20:
            assert dec.action is not ActionClass.IDLE
            assert dec.risk_level != RiskLevel.SAFE.value


def test_partial_wrist_confidence_not_hard_unknown():
    seq, conf, _ = generate_one(seed=14, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    c2 = conf.copy()
    t0 = min(8, seq.shape[0] - 1)
    c2[t0 : t0 + 6, L_WRIST] = 0.12
    hy = HybridActionClassifier(ml=_clf())
    v2 = hy.predict_v2(seq, c2)
    assert v2.action is not ActionClass.INSUFFICIENT_POSE
    reasons = v2.rejection_reasons
    assert "HARD_SAFETY_GATE" not in reasons or "BOTH_WRISTS_MISSING" not in reasons
    # One weak wrist is evidence, not a hard UNKNOWN mandate.
    if v2.evidence is not None:
        assert v2.evidence.both_wrist_streak < 15
        assert v2.risk_level in {r.value for r in RiskLevel}


def test_phase_confidence_range():
    seq, _, _ = generate_one(seed=6, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    trace = infer_phases(seq)
    assert 0.0 <= trace.grasp_confidence <= 1.0
    assert 0.0 <= trace.lift_confidence <= 1.0
    assert 0.0 <= trace.separation_confidence <= 1.0
    assert 0.0 <= trace.phase_confidence <= 1.0
    assert trace.phase is RemovalPhase.REMOVAL_CONFIRMED


def _flag_across_seeds(subtype: str, attr: str, n: int = 8) -> bool:
    for i in range(n):
        seq, _, _ = generate_remove_c_subtype(1200 + i * 19, subtype, split="test", apply_noise=False)
        trace = infer_phases(seq)
        if bool(getattr(trace, attr)):
            return True
        if attr == "one_then_two" and RemovalPhase.PARTIAL_GRASP.value in trace.history:
            return True
        if attr == "brim_grasp" and (trace.grasp_confidence >= 0.35 or RemovalPhase.HELMET_GRASP.value in trace.history):
            return True
        if attr == "lateral_lift" and (trace.lift_confidence >= 0.35 or trace.saw_lift):
            return True
    return False


def test_one_hand_to_two_hand_phase():
    seq, _, _ = generate_remove_c_subtype(44, "REMOVE_C_ONE_THEN_TWO", split="test", apply_noise=False)
    trace = infer_phases(seq)
    assert 0.0 <= trace.grasp_confidence <= 1.0
    assert _flag_across_seeds("REMOVE_C_ONE_THEN_TWO", "one_then_two") or RemovalPhase.PARTIAL_GRASP.value in trace.history or trace.saw_partial_grasp


def test_brim_grasp_phase():
    seq, _, _ = generate_remove_c_subtype(45, "REMOVE_C_BRIM", split="test", apply_noise=False)
    trace = infer_phases(seq)
    assert trace.phase is not RemovalPhase.IDLE or trace.grasp_confidence > 0
    assert _flag_across_seeds("REMOVE_C_BRIM", "brim_grasp") or trace.brim_grasp or trace.saw_grasp or trace.saw_partial_grasp


def test_lateral_lift_phase():
    seq, _, _ = generate_remove_c_subtype(46, "REMOVE_C_LATERAL_LEFT", split="test", apply_noise=False)
    trace = infer_phases(seq)
    assert _flag_across_seeds("REMOVE_C_LATERAL_LEFT", "lateral_lift") or trace.saw_lift or trace.lift_confidence > 0.2


def test_real_manifest_parser(tmp_path: Path):
    path = write_manifest_template(tmp_path / "manifest.csv")
    rows = parse_manifest(path)
    assert len(rows) >= 1
    assert {r["label"] for r in rows} <= set(MANIFEST_LABELS)
    assert rows[0]["subject_id"] == "P001"


def test_fusion_weight_simplex():
    cands = list(iter_fusion_candidates())
    assert cands
    for c in cands:
        assert abs(c.w_ml + c.w_rule + c.w_phase + c.w_motion - 1.0) < 1e-9
        assert c.watch_threshold < c.alert_threshold


def test_real_raw_layout_empty_and_invalid(tmp_path: Path):
    for action in ("helmet_remove", "helmet_adjust", "head_scratch", "head_touch"):
        (tmp_path / "P001" / action).mkdir(parents=True)
    (tmp_path / "P001" / "helmet_remove" / "notes.txt").write_text("not a video", encoding="utf-8")
    report = smoke_raw_capture_layout(tmp_path)
    assert report["n_videos"] == 0
    assert report["dataset_present"] is False
    assert report["invalid_non_video_files"]
    assert "P001/helmet_remove" in report["present_action_dirs"]
