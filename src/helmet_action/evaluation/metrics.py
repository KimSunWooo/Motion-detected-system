"""Evaluation helpers: aggregation, bootstrap CI, permutation importance."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from helmet_action.models.training import REMOVE, binary_remove_metrics


def bootstrap_binary_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Bootstrap CI for HELMET_REMOVE recall / FNR on one test set."""
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    recs, fnrs = [], []
    if n == 0:
        return {
            "recall": {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0},
            "fnr": {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        }
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b = binary_remove_metrics(yt[idx], yp[idx])
        recs.append(b["recall"])
        fnrs.append(b["fnr"])
    recs = np.asarray(recs, dtype=np.float64)
    fnrs = np.asarray(fnrs, dtype=np.float64)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "recall": {
            "mean": float(recs.mean()),
            "std": float(recs.std(ddof=1) if recs.size > 1 else 0.0),
            "ci_low": float(np.percentile(recs, lo)),
            "ci_high": float(np.percentile(recs, hi)),
        },
        "fnr": {
            "mean": float(fnrs.mean()),
            "std": float(fnrs.std(ddof=1) if fnrs.size > 1 else 0.0),
            "ci_low": float(np.percentile(fnrs, lo)),
            "ci_high": float(np.percentile(fnrs, hi)),
        },
        "n_boot": int(n_boot),
        "alpha": float(alpha),
    }


def summarize_numeric(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def aggregate_seed_results(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "helmet_remove_precision",
        "helmet_remove_recall",
        "helmet_remove_f1",
        "helmet_remove_fnr",
        "helmet_remove_fpr",
        "helmet_remove_pr_auc",
    ]
    out: dict[str, Any] = {"n_seeds": len(per_seed), "seeds": [r.get("seed") for r in per_seed]}
    for k in keys:
        out[k] = summarize_numeric([float(r.get(k, 0.0)) for r in per_seed])
    rec = np.array([float(r.get("helmet_remove_recall", 0.0)) for r in per_seed], dtype=np.float64)
    fnr = np.array([float(r.get("helmet_remove_fnr", 0.0)) for r in per_seed], dtype=np.float64)
    if rec.size:
        out["helmet_remove_recall_ci95"] = {
            "mean": float(rec.mean()),
            "std": float(rec.std(ddof=1) if rec.size > 1 else 0.0),
            "ci_low": float(np.percentile(rec, 2.5)) if rec.size > 1 else float(rec[0]),
            "ci_high": float(np.percentile(rec, 97.5)) if rec.size > 1 else float(rec[0]),
        }
        out["helmet_remove_fnr_ci95"] = {
            "mean": float(fnr.mean()),
            "std": float(fnr.std(ddof=1) if fnr.size > 1 else 0.0),
            "ci_low": float(np.percentile(fnr, 2.5)) if fnr.size > 1 else float(fnr[0]),
            "ci_high": float(np.percentile(fnr, 97.5)) if fnr.size > 1 else float(fnr[0]),
        }
    fn_dest = Counter()
    for r in per_seed:
        for row in r.get("fn_destinations", []):
            fn_dest[str(row)] += 1
    out["fn_destinations"] = dict(fn_dest)
    out["per_seed"] = per_seed
    return out


def flatten_eval(
    report: dict[str, Any],
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    b = report.get("binary_helmet_remove", {})
    dest: list[str] = []
    if y_true is not None and y_pred is not None:
        for yt, yp in zip(y_true, y_pred):
            if yt == REMOVE and yp != REMOVE:
                dest.append(str(yp))
    return {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_precision": float(report.get("macro_precision", 0.0)),
        "macro_recall": float(report.get("macro_recall", 0.0)),
        "macro_f1": float(report.get("macro_f1", 0.0)),
        "helmet_remove_precision": float(b.get("precision", 0.0)),
        "helmet_remove_recall": float(b.get("recall", 0.0)),
        "helmet_remove_f1": float(b.get("f1", 0.0)),
        "helmet_remove_fnr": float(b.get("fnr", 0.0)),
        "helmet_remove_fpr": float(b.get("fpr", 0.0)),
        "helmet_remove_pr_auc": float(b.get("pr_auc", 0.0) or 0.0),
        "helmet_remove_tp": int(b.get("tp", 0)),
        "helmet_remove_fp": int(b.get("fp", 0)),
        "helmet_remove_fn": int(b.get("fn", 0)),
        "fn_destinations": dest,
    }


def permutation_importance_report(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    encoder=None,
    n_repeats: int = 8,
    seed: int = 17,
) -> list[dict[str, float]]:
    from sklearn.inspection import permutation_importance

    y_idx = encoder.transform(y) if encoder is not None else y

    def _remove_ap(estimator, X, y_true):
        proba = estimator.predict_proba(X)
        classes = list(estimator.classes_)
        if encoder is not None:
            labels = [str(x) for x in encoder.inverse_transform(np.asarray(classes).astype(int))]
        else:
            labels = [str(c) for c in classes]
        if REMOVE not in labels:
            return 0.0
        col = labels.index(REMOVE)
        y_bin = encoder.inverse_transform(np.asarray(y_true).astype(int)) if encoder is not None else np.asarray(y_true)
        y_bin = (np.asarray(y_bin).astype(str) == REMOVE).astype(int)
        if y_bin.min() == y_bin.max():
            return 0.0
        return float(average_precision_score(y_bin, proba[:, col]))

    r = permutation_importance(
        estimator,
        X,
        y_idx,
        n_repeats=n_repeats,
        random_state=seed,
        scoring=_remove_ap,
    )
    rows = []
    for i, name in enumerate(feature_names):
        rows.append(
            {
                "feature": name,
                "importance_mean": float(r.importances_mean[i]),
                "importance_std": float(r.importances_std[i]),
            }
        )
    rows.sort(key=lambda d: d["importance_mean"], reverse=True)
    return rows


def reverse_sequence_report(clf, keypoints: np.ndarray, confidence: np.ndarray | None = None) -> dict[str, Any]:
    fwd = clf.predict_proba(keypoints, confidence)
    rev_k = np.asarray(keypoints)[::-1]
    rev_c = None if confidence is None else np.asarray(confidence)[::-1]
    rev = clf.predict_proba(rev_k, rev_c)
    return {
        "forward_pred": max(fwd, key=fwd.get),
        "reverse_pred": max(rev, key=rev.get),
        "forward_p_remove": float(fwd.get(REMOVE, 0.0)),
        "reverse_p_remove": float(rev.get(REMOVE, 0.0)),
        "same_remove_decision": (max(fwd, key=fwd.get) == REMOVE) and (max(rev, key=rev.get) == REMOVE),
    }
