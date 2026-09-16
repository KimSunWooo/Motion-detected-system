from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from helmet_action.config import repo_root
from helmet_action.features.temporal_features import extract_feature_vector, feature_names
from helmet_action.models.labels import ActionClass
from helmet_action.pose.normalizer import normalize_keypoints


def vectorize_dataset(keypoints: np.ndarray, confidences: np.ndarray | None = None) -> np.ndarray:
    rows = []
    for i in range(len(keypoints)):
        seq = keypoints[i]
        conf = None if confidences is None else confidences[i]
        if conf is not None:
            seq = np.where(conf[..., None] < 0.10, np.nan, seq)
        seq_norm, _ = normalize_keypoints(np.nan_to_num(seq, nan=0.0))
        # restore nans after dummy fill for missing joints that were 0,0? skip — conf already applied in prepare
        from helmet_action.pose.confidence import prepare_sequence

        repaired, _ = prepare_sequence(keypoints[i], conf)
        seq_norm, _ = normalize_keypoints(repaired)
        rows.append(extract_feature_vector(seq_norm))
    return np.stack(rows, axis=0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None) -> dict:
    labels = labels or [c.value for c in ActionClass if c is not ActionClass.INSUFFICIENT_POSE]
    present = [c for c in labels if c in set(y_true) or c in set(y_pred)]
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present, zero_division=0
    )
    per_class = {}
    for i, c in enumerate(present):
        per_class[c] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    cm = confusion_matrix(y_true, y_pred, labels=present)
    remove = ActionClass.HELMET_REMOVE.value
    fnr = None
    fp_cases = []
    fn_cases = []
    if remove in present:
        idx = present.index(remove)
        fn = int(cm[:, idx].sum() and 0)
        # FNR = FN / (TP+FN) = 1 - recall
        rec_rm = per_class[remove]["recall"]
        fnr = float(1.0 - rec_rm)
        for yt, yp in zip(y_true, y_pred):
            if yt == remove and yp != remove:
                fn_cases.append({"true": str(yt), "pred": str(yp)})
            if yt != remove and yp == remove:
                fp_cases.append({"true": str(yt), "pred": str(yp)})
    return {
        "accuracy": acc,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "labels": present,
        "confusion_matrix": cm.tolist(),
        "helmet_remove_fnr": fnr,
        "false_negatives": fn_cases[:40],
        "false_positives": fp_cases[:40],
        "n_false_negatives": len(fn_cases),
        "n_false_positives": len(fp_cases),
        "report": classification_report(y_true, y_pred, labels=present, zero_division=0),
    }


def default_model_path() -> Path:
    return repo_root() / "models" / "action_classifier.joblib"
