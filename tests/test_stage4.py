from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from helmet_action.evaluation.failure_reasons import tag_failure_reasons
from helmet_action.evaluation.false_safe import false_safe_rate, unknown_rate
from helmet_action.evaluation.metrics import paired_delta_report
from helmet_action.evaluation.temporal_order import temporal_variants
from helmet_action.features.v1_names import FEATURE_DIM_V1
from helmet_action.features.v2 import FEATURE_DIM_V2, extract_feature_vector_v2
from helmet_action.models.hybrid import HybridActionClassifier
from helmet_action.models.labels import ActionClass
from helmet_action.models.temporal_classifier import train_sklearn_classifier
from helmet_action.models.training import vectorize_dataset
from helmet_action.pose.confidence import interpolate_short_gaps, interpolation_max_gap, mask_invalid, prepare_sequence
from helmet_action.pose.constants import L_WRIST, R_WRIST
from helmet_action.pose.normalizer import normalize_keypoints
from helmet_action.pose.quality import compute_pose_quality_score
from helmet_action.real.dataset import RealSequence, anonymize_subject, load_sequence, parse_raw_video_path, save_sequence
from helmet_action.real.evaluate import NOT_AVAILABLE, evaluate_real_zero_shot
from helmet_action.real.validate import validate_dataset, validate_sequence
from helmet_action.synthetic.augmentation import apply_consecutive_occlusion
from helmet_action.synthetic.families import REMOVE_C_SUBTYPES
from helmet_action.synthetic.generator import generate_one, generate_remove_c_subtype


def test_v1_v2_same_split(tmp_path: Path):
    import generate_dataset

    generate_dataset.main(["--samples", "80", "--quick", "--output", str(tmp_path), "--seed", "7"])
    npz_tr = np.load(tmp_path / "train.npz", allow_pickle=True)
    npz_te = np.load(tmp_path / "test.npz", allow_pickle=True)
    y_true = np.array([str(x) for x in npz_te["labels"]])
    ktr = [npz_tr["keypoints"][i, : int(npz_tr["lengths"][i])] for i in range(len(npz_tr["labels"]))]
    ctr = [npz_tr["confidences"][i, : int(npz_tr["lengths"][i])] for i in range(len(npz_tr["labels"]))]
    kte = [npz_te["keypoints"][i, : int(npz_te["lengths"][i])] for i in range(len(npz_te["labels"]))]
    cte = [npz_te["confidences"][i, : int(npz_te["lengths"][i])] for i in range(len(npz_te["labels"]))]
    preds = {}
    for version in ("v1", "v2"):
        Xtr = vectorize_dataset(ktr, ctr, feature_version=version)
        clf = train_sklearn_classifier(Xtr, npz_tr["labels"], feature_version=version)
        yp = [clf.predict(k, c) for k, c in zip(kte, cte)]
        preds[version] = np.array(yp)
        assert Xtr.shape[0] == len(npz_tr["labels"])
    assert len(preds["v1"]) == len(preds["v2"]) == len(y_true)
    assert FEATURE_DIM_V1 == 139
    assert FEATURE_DIM_V2 == 194


def test_feature_v2_deterministic_dimension():
    assert FEATURE_DIM_V2 == 194
    seq, conf, _ = generate_one(seed=3, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    vec = extract_feature_vector_v2(normalize_keypoints(prepare_sequence(seq, conf)[0])[0])
    assert vec.shape == (194,)
    vec2 = extract_feature_vector_v2(normalize_keypoints(prepare_sequence(seq, conf)[0])[0])
    assert np.allclose(vec, vec2)


def test_remove_c_subtype_generation():
    for i, subtype in enumerate(REMOVE_C_SUBTYPES):
        seq, conf, meta = generate_remove_c_subtype(900 + i, subtype, split="test", apply_noise=False)
        assert seq.shape[1:] == (17, 2)
        assert meta.label == "HELMET_REMOVE"
        assert meta.family == subtype
        assert seq.shape[0] == conf.shape[0] >= 16


def test_failure_reason_tagging():
    seq, conf, meta = generate_one(seed=21, scenario="HELMET_REMOVE", split="test", apply_noise=False)
    tags = tag_failure_reasons(seq[::-1], conf[::-1], y_pred="HEAD_TOUCH", ml_proba={"HEAD_TOUCH": 0.8, "HELMET_REMOVE": 0.1}, meta=meta.to_dict())
    assert tags
    assert all(t in {
        "NO_APPROACH", "NO_GRASP", "WEAK_GRASP", "NO_LIFT", "WEAK_LIFT", "NO_SEPARATION",
        "WEAK_SEPARATION", "LOW_WRIST_CONFIDENCE", "LOW_EAR_CONFIDENCE", "SHORT_SEQUENCE",
        "TRACK_FRAGMENTATION", "STUTTER_MOTION", "LATERAL_MOTION", "PHASE_INCONSISTENCY",
        "ML_DISAGREES_WITH_RULE", "RULE_DISAGREES_WITH_ML", "UNKNOWN",
    } for t in tags)
    short = tag_failure_reasons(seq[:6], conf[:6], y_pred="IDLE", meta=meta.to_dict())
    assert "SHORT_SEQUENCE" in short


def test_hard_occlusion_max_gap():
    seq, conf, _ = generate_one(seed=4, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    max_gap = interpolation_max_gap()
    assert max_gap == 3
    soft, c_soft = apply_consecutive_occlusion(seq, conf, [L_WRIST], 2, start=10)
    masked_s = mask_invalid(soft, c_soft)
    repaired_s = interpolate_short_gaps(masked_s)
    assert np.isfinite(repaired_s[10:12, L_WRIST]).all()
    hard, c_hard = apply_consecutive_occlusion(seq, conf, [L_WRIST, R_WRIST], 10, start=10)
    masked_h = mask_invalid(hard, c_hard)
    repaired_h = interpolate_short_gaps(masked_h)
    assert not np.isfinite(repaired_h[10:20, L_WRIST]).all()
    assert np.isnan(mask_invalid(hard, c_hard)[12, L_WRIST]).all()
    repaired_full, _ = prepare_sequence(hard, c_hard)
    # Long gaps must remain missing (no hold-last-valid fill).
    assert not np.isfinite(repaired_full[12:18, L_WRIST]).all()


def test_pose_quality_score_range():
    seq, conf, _ = generate_one(seed=8, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    good = compute_pose_quality_score(seq, conf)
    assert 0.0 <= good.score <= 1.0
    assert good.decision_status in {"VALID", "LOW_CONFIDENCE", "INSUFFICIENT_POSE", "UNKNOWN"}
    bad_c = np.zeros_like(conf)
    bad = compute_pose_quality_score(seq, bad_c)
    assert bad.score < good.score
    assert bad.decision_status in {"INSUFFICIENT_POSE", "UNKNOWN"}


def test_false_safe_calculation():
    yt = np.array(["HELMET_REMOVE"] * 10)
    yp = np.array(["IDLE", "HEAD_TOUCH", "UNKNOWN", "INSUFFICIENT_POSE", "HELMET_REMOVE"] * 2)
    fs = false_safe_rate(yt, yp)
    # 2 IDLE + 2 HEAD_TOUCH out of 10
    assert abs(fs - 0.4) < 1e-9
    assert unknown_rate(yp) == 0.2
    yp2 = np.array(["UNKNOWN"] * 10)
    assert false_safe_rate(yt, yp2) == 0.0


def test_hard_occlusion_does_not_false_safe():
    seq, conf, _ = generate_one(seed=11, scenario="HELMET_REMOVE", split="test", apply_noise=False)
    occ, c2 = apply_consecutive_occlusion(seq, conf, [L_WRIST, R_WRIST], 20, start=6)
    dec = HybridActionClassifier(ml=None).predict(occ, c2)
    assert dec.action in (ActionClass.UNKNOWN, ActionClass.INSUFFICIENT_POSE)
    assert dec.decision_status in {"UNKNOWN", "INSUFFICIENT_POSE"}
    assert "pose_quality" in dec.to_dict()
    assert "action_probability" in dec.to_dict()


def test_soft_occlusion_two_frames_interpolates():
    seq, conf, _ = generate_one(seed=12, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    occ, c2 = apply_consecutive_occlusion(seq, conf, [L_WRIST], 2, start=8)
    masked = mask_invalid(occ, c2)
    repaired = interpolate_short_gaps(masked)
    assert np.isfinite(repaired[8:10, L_WRIST]).all()
    dec = HybridActionClassifier(ml=None).predict(occ, c2)
    assert dec.action is not ActionClass.INSUFFICIENT_POSE or dec.pose_quality < 0.4


def test_temporal_variants_shapes():
    seq, conf, _ = generate_one(seed=13, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    pack = temporal_variants(seq, conf, seed=0)
    assert set(pack) == {"NORMAL", "REVERSED", "SHUFFLED", "PARTIAL_START", "PARTIAL_END"}
    assert pack["REVERSED"][0].shape == seq.shape
    assert pack["PARTIAL_START"][0].shape[0] < seq.shape[0]
    assert not np.allclose(pack["REVERSED"][0][:5], seq[:5])


def test_real_pose_dataset_serialization(tmp_path: Path):
    seq, conf, _ = generate_one(seed=14, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    t = seq.shape[0]
    real = RealSequence(
        keypoints=seq,
        confidences=conf,
        timestamps=np.arange(t) / 20.0,
        bbox=np.tile(np.array([10.0, 20.0, 80.0, 200.0]), (t, 1)),
        track_ids=np.full(t, 3, dtype=np.int32),
        label="helmet_remove",
        subject_id="person_01",
        source_video="P001/helmet_remove/clip.mp4",
        fps=20.0,
        camera_id="high_angle",
    )
    assert real.subject_id == "P001"
    assert real.label == "HELMET_REMOVE"
    path = save_sequence(tmp_path, real, 0)
    loaded = load_sequence(path)
    assert loaded.subject_id == "P001"
    assert loaded.label == "HELMET_REMOVE"
    assert loaded.keypoints.shape == seq.shape
    assert loaded.camera_id == "high_angle"
    assert np.allclose(loaded.track_ids, 3)


def test_real_dataset_validation_and_empty(tmp_path: Path):
    empty = validate_dataset(tmp_path)
    assert empty["available"] is False
    assert empty["n_sequences"] == 0
    seq, conf, _ = generate_one(seed=15, scenario="HELMET_ADJUST", split="train", apply_noise=False)
    t = seq.shape[0]
    good = RealSequence(
        keypoints=seq,
        confidences=conf,
        timestamps=np.arange(t) / 20.0,
        bbox=np.zeros((t, 4)),
        track_ids=np.ones(t, dtype=np.int32),
        label="HELMET_ADJUST",
        subject_id="P002",
        source_video="P002/helmet_adjust/a.mp4",
        fps=20.0,
    )
    save_sequence(tmp_path, good, 1)
    report = validate_dataset(tmp_path)
    assert report["available"] is True
    assert report["subject_distribution"]["P002"] == 1
    assert report["subject_ids_preserved"] is True


def test_invalid_real_sample_handling(tmp_path: Path):
    seq_dir = tmp_path / "sequences"
    seq_dir.mkdir(parents=True)
    np.savez_compressed(seq_dir / "bad.npz", keypoints=np.zeros((5, 4, 2)), confidences=np.zeros((5, 4)))
    report = validate_dataset(tmp_path)
    assert report["n_invalid"] >= 1 or report["load_errors"] or any(not r["ok"] for r in report["sequences"])


def test_real_zero_shot_empty_dataset(tmp_path: Path):
    model = Path("models/action_classifier.joblib")
    if not model.exists():
        seqs = ["IDLE", "HEAD_SCRATCH", "HELMET_REMOVE", "HELMET_ADJUST"] * 5
        kpts, confs, labels = [], [], []
        for i, sc in enumerate(seqs):
            seq, conf, meta = generate_one(seed=400 + i, scenario=sc, split="train", apply_noise=False)
            kpts.append(seq)
            confs.append(conf)
            labels.append(meta.label)
        X = vectorize_dataset(kpts, confs, feature_version="v1")
        clf = train_sklearn_classifier(X, np.array(labels), feature_version="v1")
        model = tmp_path / "mini.joblib"
        clf.save(model)
    report = evaluate_real_zero_shot(tmp_path / "missing", model)
    assert report["dataset_available"] is False
    assert report["metrics"] is None
    assert NOT_AVAILABLE in report["status"]


def test_real_zero_shot_evaluator_on_serialized_pose(tmp_path: Path):
    seqs = ["IDLE", "HEAD_SCRATCH", "HELMET_REMOVE", "HELMET_ADJUST"] * 4
    kpts, confs, labels = [], [], []
    for i, sc in enumerate(seqs):
        seq, conf, meta = generate_one(seed=500 + i, scenario=sc, split="train", apply_noise=False)
        kpts.append(seq)
        confs.append(conf)
        labels.append(meta.label)
    X = vectorize_dataset(kpts, confs, feature_version="v1")
    clf = train_sklearn_classifier(X, np.array(labels), feature_version="v1")
    model = tmp_path / "model.joblib"
    clf.save(model)
    processed = tmp_path / "processed"
    for i, sc in enumerate(["HELMET_REMOVE", "HELMET_ADJUST"]):
        seq, conf, _ = generate_one(seed=800 + i, scenario=sc, split="test", apply_noise=False)
        t = seq.shape[0]
        save_sequence(
            processed,
            RealSequence(
                keypoints=seq,
                confidences=conf,
                timestamps=np.arange(t) / 20.0,
                bbox=np.zeros((t, 4)),
                track_ids=np.ones(t, dtype=np.int32),
                label=sc,
                subject_id=f"P{i+1:03d}",
                source_video=f"P{i+1:03d}/{sc.lower()}/x.mp4",
                fps=20.0,
            ),
            i,
        )
    report = evaluate_real_zero_shot(processed, model)
    assert report["dataset_available"] is True
    assert report["metrics"] is not None
    assert "helmet_remove_fnr" in report["metrics"]
    assert "P001" in report["subject_metrics"]


def test_subject_metadata_preservation():
    assert anonymize_subject("person_01") == "P001"
    assert anonymize_subject("P7") == "P007"
    info = parse_raw_video_path(
        Path("/data/real/raw/person_03/near/helmet_remove/a.mp4"),
        Path("/data/real/raw"),
    )
    assert info["subject_id"] == "P003"
    assert info["label"] == "HELMET_REMOVE"
    assert info["camera_id"] == "near"


def test_raw_video_gitignore():
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "data/real/raw/" in text or "data/real/raw/*" in text
    assert "data/real/processed/" in text or "data/real/processed/*" in text
    assert ".gitkeep" in text


def test_paired_delta_report():
    d = paired_delta_report([0.9, 0.8, 0.85], [0.8, 0.8, 0.7])
    assert d["mean"] > 0
    assert d["n_v2_better"] >= 2
