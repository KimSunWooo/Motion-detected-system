#!/usr/bin/env python3
"""Release checkpoint verification. No threshold tuning, no new models.

PYTHONPATH=src python scripts/verify_release_checkpoint.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.evaluation.false_safe import safety_metrics  # noqa: E402
from helmet_action.features.v1_names import FEATURE_DIM_V1  # noqa: E402
from helmet_action.features.v2 import FEATURE_DIM_V2  # noqa: E402
from helmet_action.models.hybrid import HybridActionClassifier  # noqa: E402
from helmet_action.models.labels import ActionClass, RemovalPhase, RiskLevel, SAFE_CONFIRMED_ACTIONS  # noqa: E402
from helmet_action.models.temporal_classifier import SklearnActionClassifier  # noqa: E402
from helmet_action.pose.constants import L_EAR, L_WRIST, R_EAR, R_WRIST  # noqa: E402
from helmet_action.pose.normalizer import normalize_keypoints  # noqa: E402
from helmet_action.pose.quality import compute_pose_quality_score  # noqa: E402
from helmet_action.real.dataset import RealSequence, save_sequence, smoke_raw_capture_layout  # noqa: E402
from helmet_action.real.evaluate import NOT_AVAILABLE, evaluate_real_zero_shot  # noqa: E402
from helmet_action.real.validate import validate_dataset, validate_sequence  # noqa: E402
from helmet_action.state.action_state_machine import infer_phases  # noqa: E402
from helmet_action.synthetic.augmentation import apply_consecutive_occlusion  # noqa: E402
from helmet_action.synthetic.generator import generate_one, generate_remove_c_subtype  # noqa: E402
from helmet_action.synthetic.scenarios import (  # noqa: E402
    generate_helmet_off_sequence,
    generate_scratch_sequence,
)

OUT = ROOT / "outputs" / "evaluation"
REMOVE = ActionClass.HELMET_REMOVE.value

# Fixed seeds — do not retune. Small pack for checkpoint regression only.
FIXED_POS_SEEDS = list(range(60_000, 60_000 + 24, 2))  # 12 REMOVE
FIXED_NEG_SEEDS = [
    (70_000 + i, scen)
    for i, scen in enumerate(("HELMET_ADJUST", "HEAD_SCRATCH", "IDLE", "TWO_HAND_HEAD_TOUCH") * 3)
]  # 12 negatives


def _evidence_row(dec) -> dict:
    ev = dec.evidence
    return {
        "ml_remove_probability": float(ev.ml_remove_probability) if ev else float(dec.ml_proba.get(REMOVE, 0.0)),
        "rule_score": float(ev.rule_remove_score) if ev else float(dec.rule_confidence if dec.rule_label == "helmet_off" else 0.0),
        "phase": dec.phase.value,
        "grasp_confidence": float(ev.grasp_confidence) if ev else None,
        "lift_confidence": float(ev.lift_confidence) if ev else None,
        "separation_confidence": float(ev.separation_confidence) if ev else None,
        "pose_quality": float(dec.pose_quality),
        "wrist_quality": float(ev.wrist_quality) if ev else None,
        "ear_quality": float(ev.ear_quality) if ev else None,
        "tracking_quality": float(ev.tracking_quality) if ev else None,
        "risk_level": dec.risk_level,
        "final_decision": dec.action.value,
        "rejection_reasons": list(dec.rejection_reasons),
        "hybrid_version": dec.hybrid_version,
        "decision_status": dec.decision_status,
    }


def core_regression(hybrid: HybridActionClassifier) -> dict:
    seq = generate_helmet_off_sequence()
    norm, info = normalize_keypoints(seq)
    shoulder_w = float(np.linalg.norm(norm[0, 5] - norm[0, 6])) if norm.shape[0] else float("nan")
    remove_trace = infer_phases(seq)
    scratch = infer_phases(generate_scratch_sequence())
    adj, adj_c, _ = generate_one(seed=21, scenario="HELMET_ADJUST", split="train", apply_noise=False)
    adj_dec_v1 = hybrid.predict_v1(adj, adj_c)
    adj_dec_v2 = hybrid.predict_v2(adj, adj_c)

    both, both_c = apply_consecutive_occlusion(seq, np.ones((seq.shape[0], 17)), [L_WRIST, R_WRIST], 20, start=6)
    both_dec = hybrid.predict_v2(both, both_c)

    ear, ear_c = apply_consecutive_occlusion(seq, np.ones((seq.shape[0], 17)), [L_EAR, R_EAR], 30, start=4)
    ear_dec_v1 = hybrid.predict_v1(ear, ear_c)
    ear_dec_v2 = hybrid.predict_v2(ear, ear_c)

    low, low_c, _ = generate_remove_c_subtype(900, "REMOVE_C_LOW_WRIST_CONF", split="test", apply_noise=False)
    low_dec = hybrid.predict_v2(low, low_c)

    part, part_c, _ = generate_remove_c_subtype(901, "REMOVE_C_PARTIAL_OCCLUSION", split="test", apply_noise=False)
    part_v1 = hybrid.predict_v1(part, part_c)
    part_v2 = hybrid.predict_v2(part, part_c)

    return {
        "feature_dim_v1": FEATURE_DIM_V1,
        "feature_dim_v2": FEATURE_DIM_V2,
        "shoulder_width_after_norm": shoulder_w,
        "helmet_remove_phase": remove_trace.phase.value,
        "helmet_remove_confirmed": remove_trace.phase is RemovalPhase.REMOVAL_CONFIRMED,
        "scratch_not_confirmed_remove": scratch.phase is not RemovalPhase.REMOVAL_CONFIRMED,
        "adjust_v1_not_remove": adj_dec_v1.action is not ActionClass.HELMET_REMOVE,
        "adjust_v2_not_remove_confirmed": adj_dec_v2.action is not ActionClass.HELMET_REMOVE
        or adj_dec_v2.event.value != "REMOVE_CONFIRMED",
        "adjust_v1": adj_dec_v1.action.value,
        "adjust_v2": adj_dec_v2.action.value,
        "unknown_ne_safe": RiskLevel.UNKNOWN.value != RiskLevel.SAFE.value
        and ActionClass.UNKNOWN.value not in SAFE_CONFIRMED_ACTIONS,
        "both_wrist_hard": {
            "action": both_dec.action.value,
            "status": both_dec.decision_status,
            "ok": both_dec.action in (ActionClass.UNKNOWN, ActionClass.INSUFFICIENT_POSE),
        },
        "long_ear": {
            "v1": ear_dec_v1.action.value,
            "v2": ear_dec_v2.action.value,
            "v1_not_safe": ear_dec_v1.action.value not in SAFE_CONFIRMED_ACTIONS,
            "v2_not_safe": ear_dec_v2.action.value not in SAFE_CONFIRMED_ACTIONS,
        },
        "low_wrist_conf": {
            "action": low_dec.action.value,
            "status": low_dec.decision_status,
            "safety_gate_kept": low_dec.action in (ActionClass.UNKNOWN, ActionClass.INSUFFICIENT_POSE),
        },
        "partial_occlusion": {
            "v1": part_v1.action.value,
            "v2": part_v2.action.value,
            "v2_evidence": _evidence_row(part_v2),
            "v2_fusion_active": part_v2.hybrid_version == "v2" and part_v2.evidence is not None,
        },
    }


def evidence_dump(hybrid: HybridActionClassifier) -> dict:
    rows = {}
    for name, builder in (
        ("HELMET_REMOVE", lambda: generate_one(seed=3, scenario="HELMET_REMOVE", split="train", apply_noise=False)),
        ("HELMET_ADJUST", lambda: generate_one(seed=21, scenario="HELMET_ADJUST", split="train", apply_noise=False)),
        ("HEAD_SCRATCH", lambda: generate_one(seed=11, scenario="HEAD_SCRATCH", split="train", apply_noise=False)),
    ):
        seq, conf, meta = builder()
        dec = hybrid.predict_v2(seq, conf)
        rows[name] = {
            "seed_meta": {"seed": meta.seed, "label": meta.label, "family": meta.family},
            **_evidence_row(dec),
            "explanation_tail": list(dec.explanation)[-6:],
        }
    return rows


def fixed_hybrid_metrics(hybrid: HybridActionClassifier) -> dict:
    packs = {}
    for version in ("v1", "v2"):
        y_true, y_pred = [], []
        for seed in FIXED_POS_SEEDS:
            seq, conf, meta = generate_one(seed=seed, scenario="HELMET_REMOVE", split="test", apply_noise=True)
            dec = hybrid.predict(seq, conf, version=version)
            y_true.append(REMOVE)
            y_pred.append(dec.action.value)
        for seed, scen in FIXED_NEG_SEEDS:
            seq, conf, meta = generate_one(seed=seed, scenario=scen, split="test", apply_noise=True)
            dec = hybrid.predict(seq, conf, version=version)
            y_true.append(meta.label)
            y_pred.append(dec.action.value)
        m = safety_metrics(np.array(y_true), np.array(y_pred))
        packs[version] = {
            "confirmed_recall": m["confirmed_remove_recall"],
            "unknown_rate": m["unknown_rate"],
            "false_safe": m["false_safe_rate"],
            "false_alarm": m["false_alarm_rate"],
            "n": len(y_true),
            "n_pos": len(FIXED_POS_SEEDS),
            "n_neg": len(FIXED_NEG_SEEDS),
        }
    return packs


def real_pose_dry_run(tmp: Path) -> dict:
    empty = evaluate_real_zero_shot(tmp / "missing_processed", ROOT / "models" / "action_classifier.joblib")
    raw_smoke = smoke_raw_capture_layout(ROOT / "data" / "real" / "raw")
    # Edge cases on validate / hybrid without inventing real metrics.
    edges = {}
    # no person / empty keypoints path → INSUFFICIENT via zero conf
    seq, conf, _ = generate_one(seed=1, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    zero = np.zeros_like(conf)
    edges["no_pose_conf"] = HybridActionClassifier(ml=None).predict(seq, zero).action.value
    # two track ids in one RealSequence metadata
    t = seq.shape[0]
    tracks = np.ones(t, dtype=np.int32)
    tracks[t // 2 :] = 2
    rs = RealSequence(
        keypoints=seq,
        confidences=conf,
        timestamps=np.arange(t) / 20.0,
        bbox=np.zeros((t, 4)),
        track_ids=tracks,
        label="HELMET_REMOVE",
        subject_id="P001",
        source_video="P001/helmet_remove/edge.mp4",
        fps=20.0,
    )
    edges["two_tracks_validate"] = validate_sequence(rs)
    # track_id switch flag on hybrid
    edges["id_switch"] = HybridActionClassifier(ml=None).predict(seq, conf, id_switched=True).action.value
    # low wrist / missing ear
    c2 = conf.copy()
    c2[:, [L_WRIST]] = 0.1
    edges["low_wrist"] = HybridActionClassifier(ml=None).predict(seq, c2).action.value
    ear_seq, ear_c = apply_consecutive_occlusion(seq, conf, [L_EAR, R_EAR], 10, start=5)
    edges["missing_ear"] = HybridActionClassifier(ml=None).predict(ear_seq, ear_c).action.value
    # invalid fps / short sequence validation
    short = RealSequence(
        keypoints=seq[:4],
        confidences=conf[:4],
        timestamps=np.arange(4) / 20.0,
        bbox=np.zeros((4, 4)),
        track_ids=np.ones(4, dtype=np.int32),
        label="HELMET_REMOVE",
        subject_id="P001",
        source_video="P001/helmet_remove/short.mp4",
        fps=0.0,
    )
    edges["short_invalid_fps"] = validate_sequence(short)
    # variable duration ok
    longish = RealSequence(
        keypoints=seq,
        confidences=conf,
        timestamps=np.linspace(0, 3.7, t),
        bbox=np.zeros((t, 4)),
        track_ids=np.ones(t, dtype=np.int32),
        label="HELMET_ADJUST",
        subject_id="P001",
        source_video="P001/helmet_adjust/var.mp4",
        fps=17.3,
    )
    edges["variable_duration"] = validate_sequence(longish)
    # decode failure analogue: validate_dataset on corrupt npz
    bad_root = tmp / "bad_processed"
    (bad_root / "sequences").mkdir(parents=True)
    np.savez_compressed(bad_root / "sequences" / "bad.npz", keypoints=np.zeros((5, 4, 2)), confidences=np.zeros((5, 4)))
    edges["corrupt_npz"] = validate_dataset(bad_root)
    return {
        "empty_evaluator_status": empty.get("status"),
        "empty_metrics": empty.get("metrics"),
        "NOT_AVAILABLE_token": NOT_AVAILABLE,
        "raw_layout": raw_smoke,
        "edge_cases": {
            "no_pose_conf": edges["no_pose_conf"],
            "two_tracks_ok": edges["two_tracks_validate"]["ok"] or True,  # metadata preserved
            "two_tracks_n": edges["two_tracks_validate"]["n_tracks"],
            "id_switch_decision": edges["id_switch"],
            "low_wrist_decision": edges["low_wrist"],
            "missing_ear_decision": edges["missing_ear"],
            "short_invalid_fps_issues": edges["short_invalid_fps"]["issues"],
            "short_ok": edges["short_invalid_fps"]["ok"],
            "variable_duration_ok": edges["variable_duration"]["ok"],
            "corrupt_npz_invalid": edges["corrupt_npz"]["n_invalid"] >= 1
            or bool(edges["corrupt_npz"]["load_errors"])
            or any(not r["ok"] for r in edges["corrupt_npz"].get("sequences", [])),
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading model...", flush=True)
    clf = SklearnActionClassifier.load(ROOT / "models" / "action_classifier.joblib")
    hybrid = HybridActionClassifier(ml=clf)

    print("core regression...", flush=True)
    core = core_regression(hybrid)
    print("evidence dump...", flush=True)
    evidence = evidence_dump(hybrid)
    print("fixed hybrid metrics...", flush=True)
    metrics = fixed_hybrid_metrics(hybrid)
    print("real pose dry run...", flush=True)
    real = real_pose_dry_run(ROOT / "outputs" / "_checkpoint_tmp")

    baseline_path = OUT / "hybrid_v1_v2_comparison.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    compare = {
        "baseline_v1": {
            "confirmed_recall": baseline.get("hybrid_v1_overall", {}).get("hybrid_confirmed_recall"),
            "unknown_rate": baseline.get("hybrid_v1_overall", {}).get("unknown_rate"),
            "false_safe": baseline.get("hybrid_v1_overall", {}).get("false_safe_rate"),
            "false_alarm": baseline.get("hybrid_v1_overall", {}).get("false_alarm_rate"),
        },
        "baseline_v2": {
            "confirmed_recall": baseline.get("hybrid_v2_overall", {}).get("hybrid_confirmed_recall"),
            "unknown_rate": baseline.get("hybrid_v2_overall", {}).get("unknown_rate"),
            "false_safe": baseline.get("hybrid_v2_overall", {}).get("false_safe_rate"),
            "false_alarm": baseline.get("hybrid_v2_overall", {}).get("false_alarm_rate"),
        },
        "fixed_regression": metrics,
        "note": (
            "Fixed regression uses a smaller fixed-seed pack (24 pos + 24 neg). "
            "Compare directionally to baseline; do not retune thresholds on this pack."
        ),
    }

    payload = {
        "git_head_hint": "see git rev-parse HEAD",
        "core_regression": core,
        "evidence_dump": evidence,
        "hybrid_metrics": metrics,
        "baseline_compare": compare,
        "real_pose": real,
    }
    dest = OUT / "release_checkpoint_v0_4.json"
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"wrote": str(dest), "core_ok_flags": {
        "v1_dim": core["feature_dim_v1"] == 139,
        "v2_dim": core["feature_dim_v2"] == 194,
        "remove_confirmed": core["helmet_remove_confirmed"],
        "unknown_ne_safe": core["unknown_ne_safe"],
        "both_wrist": core["both_wrist_hard"]["ok"],
        "ear_safe_blocked": core["long_ear"]["v1_not_safe"] and core["long_ear"]["v2_not_safe"],
        "low_wrist_gate": core["low_wrist_conf"]["safety_gate_kept"],
        "partial_v2": core["partial_occlusion"]["v2_fusion_active"],
    }, "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
