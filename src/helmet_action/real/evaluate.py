"""Zero-shot evaluation: synthetic-trained model on real pose sequences. Never retrains."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from helmet_action.evaluation.false_safe import (
    false_alarm_rate,
    false_safe_rate,
    insufficient_rate,
    safety_metrics,
    unknown_rate,
)
from helmet_action.evaluation.metrics import flatten_eval
from helmet_action.models.hybrid import HybridActionClassifier
from helmet_action.models.labels import ActionClass, RiskLevel
from helmet_action.models.temporal_classifier import SklearnActionClassifier
from helmet_action.models.training import REMOVE, evaluate_predictions
from helmet_action.pose.constants import L_EAR, L_WRIST, R_EAR, R_WRIST
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
    low = 0.25
    missing_wrist = float((c[:, [L_WRIST, R_WRIST]] < low).any(axis=1).mean()) if c.size else 0.0
    missing_ear = float((c[:, [L_EAR, R_EAR]] < low).any(axis=1).mean()) if c.size else 0.0
    return {
        "ml_pred": ml_pred,
        "ml_p_remove": float(ml_proba.get(REMOVE, 0.0)),
        "hybrid_pred": dec.action.value,
        "decision_status": dec.decision_status,
        "risk_level": getattr(dec, "risk_level", RiskLevel.UNKNOWN.value),
        "hybrid_version": getattr(dec, "hybrid_version", "v1"),
        "rejection_reasons": list(getattr(dec, "rejection_reasons", [])),
        "pose_quality": float(dec.pose_quality),
        "phase_confidence": float(dec.phase_confidence),
        "action_probability": float(dec.action_probability),
        "quality_status": pq.decision_status,
        "wrist_quality": float(getattr(pq, "wrist_quality", 1.0)),
        "ear_quality": float(getattr(pq, "ear_quality", 1.0)),
        "missing_wrist_ratio": missing_wrist,
        "missing_ear_ratio": missing_ear,
    }


def _empty_schema() -> dict[str, Any]:
    return {
        "overall": None,
        "helmet_remove": None,
        "decision": None,
        "safety": None,
        "pose": None,
        "per_subject": {},
        "per_condition": {"distance": {}, "speed": {}},
    }


def evaluate_real_zero_shot(
    processed_root: Path,
    model_path: Path,
    feature_version: str | None = None,
    hybrid_version: str | None = None,
) -> dict[str, Any]:
    processed_root = Path(processed_root)
    validation = validate_dataset(processed_root)
    if not validation.get("available"):
        return {
            "dataset_available": False,
            "status": NOT_AVAILABLE,
            "metrics": None,
            "REAL METRICS": NOT_AVAILABLE,
            "overall": None,
            "helmet_remove": None,
            "decision": None,
            "safety": None,
            "pose": None,
            "per_subject": {},
            "per_condition": {"distance": {}, "speed": {}},
            "note": "No processed real sequences. Do not invent real metrics.",
            "validation": validation,
        }

    clf = SklearnActionClassifier.load(model_path)
    if feature_version:
        clf.feature_version = feature_version
    hybrid = HybridActionClassifier(ml=clf, version=hybrid_version)
    sequences = load_processed_dataset(processed_root)

    y_true, y_pred, scores = [], [], []
    hybrid_pred = []
    risk_levels = []
    rows = []
    quality_scores = []
    missing_wrist = []
    missing_ear = []
    by_subject: dict[str, dict[str, list]] = defaultdict(lambda: {"y_true": [], "y_pred": [], "scores": [], "hybrid": [], "risk": []})
    by_camera: dict[str, dict[str, list]] = defaultdict(lambda: {"y_true": [], "y_pred": [], "scores": [], "hybrid": []})
    by_speed: dict[str, dict[str, list]] = defaultdict(lambda: {"y_true": [], "y_pred": [], "scores": [], "hybrid": []})

    for seq in sequences:
        pred = _predict_one(clf, hybrid, seq)
        yt = seq.ml_label
        yp = pred["ml_pred"] or pred["hybrid_pred"]
        y_true.append(yt)
        y_pred.append(yp)
        scores.append(pred["ml_p_remove"])
        hybrid_pred.append(pred["hybrid_pred"])
        risk_levels.append(pred["risk_level"])
        quality_scores.append(pred["pose_quality"])
        missing_wrist.append(pred["missing_wrist_ratio"])
        missing_ear.append(pred["missing_ear_ratio"])
        speed = "unknown"
        notes = " ".join(seq.notes).lower()
        for token in ("slow", "normal", "fast"):
            if token in notes or token in str(seq.source_video).lower():
                speed = token
                break
        row = {
            "subject_id": seq.subject_id,
            "label": seq.label,
            "ml_label": seq.ml_label,
            "source_video": seq.source_video,
            "camera_id": seq.camera_id,
            "speed": speed,
            **pred,
        }
        rows.append(row)
        by_subject[seq.subject_id]["y_true"].append(yt)
        by_subject[seq.subject_id]["y_pred"].append(yp)
        by_subject[seq.subject_id]["scores"].append(pred["ml_p_remove"])
        by_subject[seq.subject_id]["hybrid"].append(pred["hybrid_pred"])
        by_subject[seq.subject_id]["risk"].append(pred["risk_level"])
        dist = seq.camera_id if seq.camera_id in {"near", "medium", "far"} else seq.camera_id
        by_camera[dist]["y_true"].append(yt)
        by_camera[dist]["y_pred"].append(yp)
        by_camera[dist]["scores"].append(pred["ml_p_remove"])
        by_camera[dist]["hybrid"].append(pred["hybrid_pred"])
        by_speed[speed]["y_true"].append(yt)
        by_speed[speed]["y_pred"].append(yp)
        by_speed[speed]["scores"].append(pred["ml_p_remove"])
        by_speed[speed]["hybrid"].append(pred["hybrid_pred"])

    report = evaluate_predictions(
        np.array(y_true),
        np.array(y_pred),
        y_score_remove=np.array(scores, dtype=np.float64),
    )
    flat = flatten_eval(report, np.array(y_true), np.array(y_pred))
    hy = np.array(hybrid_pred)
    yt = np.array(y_true)
    risk_counts = Counter(risk_levels)
    n = max(len(risk_levels), 1)
    safety = safety_metrics(yt, hy)

    def _group(table: dict[str, dict[str, list]]) -> dict[str, Any]:
        out = {}
        for key, bundle in table.items():
            g_yt = np.array(bundle["y_true"])
            g_yp = np.array(bundle["y_pred"])
            sc = np.array(bundle["scores"], dtype=np.float64)
            g = evaluate_predictions(g_yt, g_yp, y_score_remove=sc)
            out[key] = flatten_eval(g, g_yt, g_yp)
            out[key]["n"] = int(len(g_yt))
            if bundle.get("hybrid"):
                out[key]["safety"] = safety_metrics(g_yt, np.array(bundle["hybrid"]))
        return out

    b = report.get("binary_helmet_remove", {})
    pose_block = {
        "mean": float(np.mean(quality_scores)) if quality_scores else 0.0,
        "std": float(np.std(quality_scores, ddof=1)) if len(quality_scores) > 1 else 0.0,
        "min": float(np.min(quality_scores)) if quality_scores else 0.0,
        "max": float(np.max(quality_scores)) if quality_scores else 0.0,
        "missing_wrist_ratio_mean": float(np.mean(missing_wrist)) if missing_wrist else 0.0,
        "missing_ear_ratio_mean": float(np.mean(missing_ear)) if missing_ear else 0.0,
    }
    decision_block = {
        "ALERT_rate": float(risk_counts.get(RiskLevel.ALERT.value, 0) / n),
        "WATCH_rate": float(risk_counts.get(RiskLevel.WATCH.value, 0) / n),
        "UNKNOWN_rate": float(risk_counts.get(RiskLevel.UNKNOWN.value, 0) / n),
        "SAFE_rate": float(risk_counts.get(RiskLevel.SAFE.value, 0) / n),
        "counts": dict(risk_counts),
    }
    helmet_remove_block = {
        "precision": float(b.get("precision", 0.0)),
        "recall": float(b.get("recall", 0.0)),
        "f1": float(b.get("f1", 0.0)),
        "fnr": float(b.get("fnr", 0.0)),
        "fpr": float(b.get("fpr", 0.0)),
    }
    overall = {
        "accuracy": float(flat.get("accuracy", 0.0)),
        "macro_f1": float(flat.get("macro_f1", 0.0)),
        "n_sequences": len(sequences),
    }

    return {
        "dataset_available": True,
        "status": "evaluated",
        "feature_version": clf.feature_version,
        "model_path": str(model_path),
        "n_sequences": len(sequences),
        "metrics": flat,
        "overall": overall,
        "helmet_remove": helmet_remove_block,
        "decision": decision_block,
        "safety": {
            "false_safe_rate": safety["false_safe_rate"],
            "false_alarm_rate": safety["false_alarm_rate"],
            "confirmed_remove_recall": safety["confirmed_remove_recall"],
            "unknown_rate": safety["unknown_rate"],
            "insufficient_pose_rate": insufficient_rate(hy),
        },
        "pose": pose_block,
        "confusion_matrix": report.get("confusion_matrix"),
        "labels": report.get("labels"),
        "unknown_rate": unknown_rate(hy),
        "insufficient_pose_rate": insufficient_rate(hy),
        "false_safe_rate": false_safe_rate(yt, hy),
        "false_alarm_rate": false_alarm_rate(yt, hy),
        "pose_quality": pose_block,
        "subject_metrics": _group(by_subject),
        "per_subject": _group(by_subject),
        "camera_metrics": _group(by_camera),
        "per_condition": {
            "distance": _group(by_camera),
            "speed": _group(by_speed),
        },
        "per_sequence": rows,
        "validation": {k: validation[k] for k in validation if k != "sequences"},
        "note": "Synthetic-trained model evaluated zero-shot. Not real-world CCTV performance.",
    }
