"""Zero-shot evaluation: synthetic-trained model on real pose sequences. Never retrains."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from helmet_action.evaluation.false_safe import false_safe_rate, insufficient_rate, unknown_rate
from helmet_action.evaluation.metrics import flatten_eval
from helmet_action.models.hybrid import HybridActionClassifier
from helmet_action.models.temporal_classifier import SklearnActionClassifier
from helmet_action.models.training import REMOVE, evaluate_predictions
from helmet_action.pose.quality import compute_pose_quality_score
from helmet_action.real.dataset import RealSequence, load_processed_dataset
from helmet_action.real.validate import validate_dataset

NOT_AVAILABLE = "REAL DATASET: NOT AVAILABLE"


def _predict_one(clf: SklearnActionClassifier | None, hybrid: HybridActionClassifier, seq: RealSequence) -> dict[str, Any]:
    k, c = seq.keypoints, seq.confidences
    pq = compute_pose_quality_score(k, c)
    ml_proba = clf.predict_proba(k, c) if clf is not None else {}
    ml_pred = max(ml_proba, key=ml_proba.get) if ml_proba else None
    dec = hybrid.predict(k, c)
    return {
        "ml_pred": ml_pred,
        "ml_p_remove": float(ml_proba.get(REMOVE, 0.0)),
        "hybrid_pred": dec.action.value,
        "decision_status": dec.decision_status,
        "pose_quality": float(dec.pose_quality),
        "phase_confidence": float(dec.phase_confidence),
        "action_probability": float(dec.action_probability),
        "quality_status": pq.decision_status,
    }


def evaluate_real_zero_shot(
    processed_root: Path,
    model_path: Path,
    feature_version: str | None = None,
) -> dict[str, Any]:
    processed_root = Path(processed_root)
    validation = validate_dataset(processed_root)
    if not validation.get("available"):
        return {
            "dataset_available": False,
            "status": NOT_AVAILABLE,
            "metrics": None,
            "note": "No processed real sequences. Do not invent real metrics.",
            "validation": validation,
        }

    clf = SklearnActionClassifier.load(model_path)
    if feature_version:
        clf.feature_version = feature_version
    hybrid = HybridActionClassifier(ml=clf)
    sequences = load_processed_dataset(processed_root)

    y_true, y_pred, scores = [], [], []
    hybrid_pred = []
    rows = []
    quality_scores = []
    by_subject: dict[str, dict[str, list]] = defaultdict(lambda: {"y_true": [], "y_pred": [], "scores": []})
    by_camera: dict[str, dict[str, list]] = defaultdict(lambda: {"y_true": [], "y_pred": [], "scores": []})

    for seq in sequences:
        pred = _predict_one(clf, hybrid, seq)
        yt = seq.ml_label
        yp = pred["ml_pred"] or pred["hybrid_pred"]
        y_true.append(yt)
        y_pred.append(yp)
        scores.append(pred["ml_p_remove"])
        hybrid_pred.append(pred["hybrid_pred"])
        quality_scores.append(pred["pose_quality"])
        row = {
            "subject_id": seq.subject_id,
            "label": seq.label,
            "ml_label": seq.ml_label,
            "source_video": seq.source_video,
            "camera_id": seq.camera_id,
            **pred,
        }
        rows.append(row)
        by_subject[seq.subject_id]["y_true"].append(yt)
        by_subject[seq.subject_id]["y_pred"].append(yp)
        by_subject[seq.subject_id]["scores"].append(pred["ml_p_remove"])
        by_camera[seq.camera_id]["y_true"].append(yt)
        by_camera[seq.camera_id]["y_pred"].append(yp)
        by_camera[seq.camera_id]["scores"].append(pred["ml_p_remove"])

    report = evaluate_predictions(
        np.array(y_true),
        np.array(y_pred),
        y_score_remove=np.array(scores, dtype=np.float64),
    )
    flat = flatten_eval(report, np.array(y_true), np.array(y_pred))
    hy = np.array(hybrid_pred)

    def _group(table: dict[str, dict[str, list]]) -> dict[str, Any]:
        out = {}
        for key, bundle in table.items():
            yt = np.array(bundle["y_true"])
            yp = np.array(bundle["y_pred"])
            sc = np.array(bundle["scores"], dtype=np.float64)
            g = evaluate_predictions(yt, yp, y_score_remove=sc)
            out[key] = flatten_eval(g, yt, yp)
            out[key]["n"] = int(len(yt))
        return out

    return {
        "dataset_available": True,
        "status": "evaluated",
        "feature_version": clf.feature_version,
        "model_path": str(model_path),
        "n_sequences": len(sequences),
        "metrics": flat,
        "confusion_matrix": report.get("confusion_matrix"),
        "labels": report.get("labels"),
        "unknown_rate": unknown_rate(hy),
        "insufficient_pose_rate": insufficient_rate(hy),
        "false_safe_rate": false_safe_rate(np.array(y_true), hy),
        "pose_quality": {
            "mean": float(np.mean(quality_scores)) if quality_scores else 0.0,
            "std": float(np.std(quality_scores, ddof=1)) if len(quality_scores) > 1 else 0.0,
            "min": float(np.min(quality_scores)) if quality_scores else 0.0,
            "max": float(np.max(quality_scores)) if quality_scores else 0.0,
        },
        "subject_metrics": _group(by_subject),
        "camera_metrics": _group(by_camera),
        "per_sequence": rows,
        "validation": {k: validation[k] for k in validation if k != "sequences"},
        "note": "Synthetic-trained model evaluated zero-shot. Not real-world CCTV performance.",
    }
