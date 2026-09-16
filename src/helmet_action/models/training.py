from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from helmet_action.config import repo_root
from helmet_action.features.temporal_features import extract_feature_vector
from helmet_action.features.v2 import extract_feature_vector_v2
from helmet_action.models.labels import ActionClass
from helmet_action.pose.confidence import prepare_sequence
from helmet_action.pose.normalizer import normalize_keypoints

REMOVE = ActionClass.HELMET_REMOVE.value


def vectorize_dataset(
    keypoints: np.ndarray,
    confidences: np.ndarray | None = None,
    feature_version: str = "v1",
) -> np.ndarray:
    rows = []
    extract = extract_feature_vector_v2 if feature_version == "v2" else extract_feature_vector
    for i in range(len(keypoints)):
        conf = None if confidences is None else confidences[i]
        repaired, _ = prepare_sequence(keypoints[i], conf)
        seq_norm, _ = normalize_keypoints(repaired)
        rows.append(extract(seq_norm))
    return np.stack(rows, axis=0)


def _percentile_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def binary_remove_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    yt = np.asarray(y_true) == REMOVE
    yp = np.asarray(y_pred) == REMOVE
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    tn = int(np.sum(~yt & ~yp))
    fn = int(np.sum(yt & ~yp))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fnr = fn / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    out = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fnr": float(fnr),
        "fpr": float(fpr),
    }
    if y_score is not None and yt.any() and (~yt).any():
        out["pr_auc"] = float(average_precision_score(yt.astype(int), y_score))
        out["brier"] = float(brier_score_loss(yt.astype(int), y_score))
    return out


def hard_negative_confusions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    keys = ("HELMET_ADJUST", "HEAD_TOUCH", "HEAD_SCRATCH")
    return {k: int(np.sum((np.asarray(y_true) == k) & (np.asarray(y_pred) == REMOVE))) for k in keys}


def remove_probability_by_class(y_true: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for cls in sorted(set(map(str, y_true))):
        out[cls] = _percentile_stats(scores[np.asarray(y_true) == cls])
    return out


def calibration_report(y_true_bin: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> dict:
    from sklearn.calibration import calibration_curve

    y = np.asarray(y_true_bin).astype(int)
    p = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or y.min() == y.max():
        return {"brier": None, "fraction_of_positives": [], "mean_predicted_value": []}
    frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    return {
        "brier": float(brier_score_loss(y, p)),
        "fraction_of_positives": [float(v) for v in frac],
        "mean_predicted_value": [float(v) for v in mean_pred],
        "note": "Synthetic probabilities are not field-calibrated.",
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    y_score_remove: np.ndarray | None = None,
) -> dict:
    labels = labels or [c.value for c in ActionClass if c is not ActionClass.INSUFFICIENT_POSE]
    present = [c for c in labels if c in set(y_true) or c in set(y_pred)]
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=present, zero_division=0)
    per_class = {}
    for i, c in enumerate(present):
        per_class[c] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=present)
    binary = binary_remove_metrics(y_true, y_pred, y_score_remove)
    fp_cases = []
    fn_cases = []
    for yt, yp in zip(y_true, y_pred):
        if yt == REMOVE and yp != REMOVE:
            fn_cases.append({"true": str(yt), "pred": str(yp)})
        if yt != REMOVE and yp == REMOVE:
            fp_cases.append({"true": str(yt), "pred": str(yp)})
    return {
        "accuracy": acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f),
        "per_class": per_class,
        "labels": present,
        "confusion_matrix": cm.tolist(),
        "binary_helmet_remove": binary,
        "helmet_remove_fnr": binary["fnr"],
        "hard_negative_to_remove": hard_negative_confusions(y_true, y_pred),
        "false_negatives": fn_cases[:40],
        "false_positives": fp_cases[:40],
        "n_false_negatives": len(fn_cases),
        "n_false_positives": len(fp_cases),
        "report": classification_report(y_true, y_pred, labels=present, zero_division=0),
    }


def default_model_path() -> Path:
    return repo_root() / "models" / "action_classifier.joblib"
