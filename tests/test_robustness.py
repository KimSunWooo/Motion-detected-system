from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from helmet_action.evaluation.failures import dump_failure
from helmet_action.evaluation.metrics import aggregate_seed_results, permutation_importance_report
from helmet_action.features.temporal_features import extract_feature_vector, feature_names
from helmet_action.features.v1_names import FEATURE_DIM_V1, FEATURE_NAMES_V1
from helmet_action.features.v2 import FEATURE_DIM_V2, FEATURE_NAMES_V2, extract_feature_dict_v2, extract_feature_vector_v2
from helmet_action.models.hybrid import HybridActionClassifier
from helmet_action.models.labels import ActionClass, RemovalPhase
from helmet_action.models.temporal_classifier import train_sklearn_classifier
from helmet_action.models.training import vectorize_dataset
from helmet_action.pose.confidence import prepare_sequence
from helmet_action.pose.constants import L_WRIST, R_WRIST
from helmet_action.pose.normalizer import normalize_keypoints
from helmet_action.pose.pose_buffer import TrackPoseBuffer
from helmet_action.pose.types import PoseObservation
from helmet_action.state.action_state_machine import infer_phases
from helmet_action.synthetic.augmentation import apply_targeted_occlusion
from helmet_action.synthetic.families import choose_family
from helmet_action.synthetic.generator import generate_counterfactual_pair, generate_one
from helmet_action.synthetic.scenarios import generate_helmet_off_sequence


def test_v1_feature_dimension_frozen():
    assert FEATURE_DIM_V1 == 139
    assert len(feature_names()) == 139
    assert feature_names() == FEATURE_NAMES_V1
    seq = generate_helmet_off_sequence()
    vec = extract_feature_vector(normalize_keypoints(seq)[0])
    assert vec.shape == (139,)


def test_phase_aware_feature_dimension_fixed():
    assert FEATURE_DIM_V2 == len(FEATURE_NAMES_V2)
    assert FEATURE_DIM_V2 > FEATURE_DIM_V1
    assert FEATURE_NAMES_V2[:139] == FEATURE_NAMES_V1
    seq = generate_helmet_off_sequence()
    d = extract_feature_dict_v2(normalize_keypoints(seq)[0])
    vec = extract_feature_vector_v2(normalize_keypoints(seq)[0])
    assert vec.shape == (FEATURE_DIM_V2,)
    assert set(FEATURE_NAMES_V2) <= set(d)


def test_counterfactual_pair_deterministic():
    a1, b1 = generate_counterfactual_pair(77, kind="adjust_vs_remove")
    a2, b2 = generate_counterfactual_pair(77, kind="adjust_vs_remove")
    assert np.allclose(a1[0], a2[0])
    assert np.allclose(b1[0], b2[0])
    assert a1[2].pair_id == a2[2].pair_id
    assert a1[2].label == "HELMET_ADJUST"
    assert b1[2].label == "HELMET_REMOVE"
    share = min(a1[0].shape[0], b1[0].shape[0])
    cut = min(12, max(6, share // 3))
    assert np.allclose(a1[0][:cut], b1[0][:cut])
    assert not np.allclose(a1[0][-8:], b1[0][-8:])


def test_counterfactual_remove_vs_adjust_features():
    seps_rm, seps_adj = [], []
    for i in range(6):
        (neg, cneg, _), (rm, crm, _) = generate_counterfactual_pair(11 + i * 5, kind="adjust_vs_remove")
        n_neg = normalize_keypoints(prepare_sequence(neg, cneg)[0])[0]
        n_rm = normalize_keypoints(prepare_sequence(rm, crm)[0])[0]
        seps_adj.append(extract_feature_dict_v2(n_neg)["late_sep_mean"])
        seps_rm.append(extract_feature_dict_v2(n_rm)["late_sep_mean"])
    assert float(np.mean(seps_rm)) > float(np.mean(seps_adj))


def test_trajectory_family_holdout_no_leakage():
    train, test = set(), set()
    for i in range(30):
        train.add(choose_family("HELMET_REMOVE", "train", np.random.default_rng(i)))
        test.add(choose_family("HELMET_REMOVE", "test", np.random.default_rng(1000 + i)))
        _, _, mtr = generate_one(seed=200 + i, scenario="HELMET_REMOVE", split="train", apply_noise=False)
        _, _, mte = generate_one(seed=9000 + i, scenario="HELMET_REMOVE", split="test", apply_noise=False)
        train.add(mtr.family)
        test.add(mte.family)
    assert "REMOVE_C" not in train
    assert "REMOVE_C" in test
    assert not (train & {"REMOVE_C"})


def test_reverse_sequence_changes_v2_timing():
    seq = generate_helmet_off_sequence()
    fwd = extract_feature_dict_v2(normalize_keypoints(seq)[0])
    rev = extract_feature_dict_v2(normalize_keypoints(seq[::-1])[0])
    assert abs(fwd["peak_wrist_sep_t"] - rev["peak_wrist_sep_t"]) > 0.08
    v1_fwd = extract_feature_vector(normalize_keypoints(seq)[0])
    v1_rev = extract_feature_vector(normalize_keypoints(seq[::-1])[0])
    # V1 aggregates can be similar; V2 peak timing must move.
    assert fwd["peak_vert_lift_t"] != rev["peak_vert_lift_t"] or abs(v1_fwd - v1_rev).mean() > 1e-6


def test_fps_resampling_duration():
    seq, conf, _ = generate_one(seed=5, scenario="HELMET_REMOVE", split="train", apply_noise=False, fps=30.0, n_frames=60)
    buf = TrackPoseBuffer(window_seconds=4.0, target_fps=20)
    for i in range(len(seq)):
        buf.push(
            PoseObservation(
                timestamp=i / 30.0,
                frame_index=i,
                track_id=3,
                bbox=(0, 0, 1, 1),
                keypoints=seq[i],
                keypoint_confidence=conf[i],
                detection_confidence=1.0,
                source_fps=30.0,
            )
        )
    packed = buf.get_arrays(3)
    assert packed is not None
    k, c, ts = packed
    assert k.shape[1:] == (17, 2)
    assert abs((ts[-1] - ts[0]) - (len(seq) - 1) / 30.0) < 0.05
    assert extract_feature_vector(normalize_keypoints(k)[0]).shape == (139,)


def test_targeted_wrist_dropout_does_not_crash():
    seq, conf, _ = generate_one(seed=9, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    s2, c2 = apply_targeted_occlusion(seq, conf, [L_WRIST, R_WRIST], 0.4, np.random.default_rng(0))
    s2 = np.where(np.isfinite(s2), s2, 0.0)
    dec = HybridActionClassifier(ml=None).predict(s2, c2)
    assert dec.action in (
        ActionClass.HELMET_REMOVE,
        ActionClass.UNKNOWN,
        ActionClass.INSUFFICIENT_POSE,
        ActionClass.HEAD_TOUCH,
        ActionClass.HELMET_ADJUST,
        ActionClass.IDLE,
    )


def test_track_fragmentation_and_id_switch_isolation():
    seq, conf, _ = generate_one(seed=3, scenario="HELMET_REMOVE", split="train", apply_noise=False, fps=20.0)
    buf = TrackPoseBuffer(window_seconds=5.0, target_fps=20)
    gap_from = 20
    for i in range(len(seq)):
        if 20 <= i < 30:
            continue
        buf.push(
            PoseObservation(
                timestamp=i / 20.0,
                frame_index=i,
                track_id=8,
                bbox=(0, 0, 1, 1),
                keypoints=seq[i],
                keypoint_confidence=conf[i],
                detection_confidence=1.0,
                source_fps=20.0,
            )
        )
    assert buf.has_long_gap(8, gap_seconds=0.25)
    assert buf.completeness(8, expected_fps=20.0) < 0.95

    idle, ic, _ = generate_one(seed=4, scenario="IDLE", split="train", apply_noise=False)
    for i in range(min(len(idle), 25)):
        buf.push(
            PoseObservation(
                timestamp=i / 20.0,
                frame_index=i,
                track_id=9,
                bbox=(0, 0, 1, 1),
                keypoints=idle[i],
                keypoint_confidence=ic[i],
                detection_confidence=1.0,
                source_fps=20.0,
            )
        )
    a = buf.get_arrays(8)
    b = buf.get_arrays(9)
    assert a is not None and b is not None
    assert not np.allclose(a[0][:4], b[0][:4])
    da = HybridActionClassifier(ml=None).predict(a[0], a[1], buffer_completeness=buf.completeness(8))
    db = HybridActionClassifier(ml=None).predict(b[0], b[1], buffer_completeness=buf.completeness(9))
    assert da.action != db.action or db.action is ActionClass.IDLE


def test_partial_remove_not_forced_confirmed():
    seq = generate_helmet_off_sequence()
    early = infer_phases(seq[:16])
    assert early.phase is not RemovalPhase.REMOVAL_CONFIRMED
    late = infer_phases(seq[40:])
    if not late.saw_grasp:
        assert late.phase is not RemovalPhase.REMOVAL_CONFIRMED


def test_failure_artifact_serialization(tmp_path: Path):
    seq, conf, meta = generate_one(seed=12, scenario="HELMET_REMOVE", split="test", apply_noise=False)
    payload = dump_failure(tmp_path, "seed_12_remove_fn_001", seq, conf, meta.to_dict(), clf=None, y_pred="HEAD_TOUCH")
    js = json.loads((tmp_path / "seed_12_remove_fn_001.json").read_text(encoding="utf-8"))
    for key in (
        "camera",
        "body",
        "action",
        "noise",
        "fps",
        "ml_proba",
        "rule_prediction",
        "phase_history",
        "final_prediction",
        "feature_vector",
        "series",
    ):
        assert key in js
    for series_key in (
        "lw_head",
        "rw_head",
        "wrist_sep",
        "wrist_sep_v",
        "vertical_wrist_v",
        "radial_v",
        "motion_energy",
        "left_elbow",
        "phase",
    ):
        assert series_key in js["series"]
    assert (tmp_path / "seed_12_remove_fn_001.npz").exists()
    assert payload["y_pred"] == "HEAD_TOUCH"


def test_benchmark_aggregation():
    rows = [
        {
            "seed": 42,
            "accuracy": 0.8,
            "macro_precision": 0.7,
            "macro_recall": 0.7,
            "macro_f1": 0.7,
            "helmet_remove_precision": 1.0,
            "helmet_remove_recall": 0.8,
            "helmet_remove_f1": 0.88,
            "helmet_remove_fnr": 0.2,
            "helmet_remove_fpr": 0.0,
            "helmet_remove_pr_auc": 0.9,
            "fn_destinations": ["HEAD_TOUCH"],
        },
        {
            "seed": 101,
            "accuracy": 1.0,
            "macro_precision": 0.9,
            "macro_recall": 0.9,
            "macro_f1": 0.9,
            "helmet_remove_precision": 1.0,
            "helmet_remove_recall": 1.0,
            "helmet_remove_f1": 1.0,
            "helmet_remove_fnr": 0.0,
            "helmet_remove_fpr": 0.0,
            "helmet_remove_pr_auc": 1.0,
            "fn_destinations": [],
        },
    ]
    summary = aggregate_seed_results(rows)
    assert abs(summary["accuracy"]["mean"] - 0.9) < 1e-9
    assert summary["helmet_remove_fnr"]["min"] == 0.0
    assert summary["helmet_remove_fnr"]["max"] == 0.2
    assert "mean" in summary["helmet_remove_recall_ci95"]
    assert summary["fn_destinations"]["HEAD_TOUCH"] == 1


def test_permutation_importance_execution():
    scenarios = ["IDLE", "HEAD_SCRATCH", "HELMET_REMOVE", "HELMET_ADJUST"] * 6
    kpts, confs, labels = [], [], []
    for i, sc in enumerate(scenarios):
        seq, conf, meta = generate_one(seed=3000 + i, scenario=sc, split="train", apply_noise=False)
        kpts.append(seq)
        confs.append(conf)
        labels.append(meta.label)
    X = vectorize_dataset(kpts, confs, feature_version="v1")
    y = np.array(labels)
    clf = train_sklearn_classifier(X, y, feature_version="v1")
    rows = permutation_importance_report(clf.estimator, X, y, clf.feature_names, clf.encoder, n_repeats=2, seed=0)
    assert len(rows) == 139
    assert "feature" in rows[0]
    assert "importance_mean" in rows[0]
