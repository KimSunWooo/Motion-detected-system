"""False-safe rate: REMOVE predicted as a confirmed SAFE action (not UNKNOWN)."""

from __future__ import annotations

import numpy as np

from helmet_action.models.labels import ActionClass, SAFE_CONFIRMED_ACTIONS

REMOVE = ActionClass.HELMET_REMOVE.value
UNKNOWN = ActionClass.UNKNOWN.value
INSUFFICIENT = ActionClass.INSUFFICIENT_POSE.value
ABSTAIN = frozenset({UNKNOWN, INSUFFICIENT})


def false_safe_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    return (yt == REMOVE) & np.isin(yp, list(SAFE_CONFIRMED_ACTIONS))


def false_safe_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true).astype(str)
    pos = yt == REMOVE
    if not np.any(pos):
        return 0.0
    return float(false_safe_mask(y_true, y_pred)[pos].mean())


def unknown_rate(y_pred: np.ndarray) -> float:
    yp = np.asarray(y_pred).astype(str)
    if yp.size == 0:
        return 0.0
    return float((yp == UNKNOWN).mean())


def insufficient_rate(y_pred: np.ndarray) -> float:
    yp = np.asarray(y_pred).astype(str)
    if yp.size == 0:
        return 0.0
    return float((yp == INSUFFICIENT).mean())


def abstain_rate(y_pred: np.ndarray) -> float:
    yp = np.asarray(y_pred).astype(str)
    if yp.size == 0:
        return 0.0
    return float(np.isin(yp, list(ABSTAIN)).mean())


def confirmed_remove_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    pos = yt == REMOVE
    if not np.any(pos):
        return 0.0
    return float((yp[pos] == REMOVE).mean())


def unknown_on_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    pos = yt == REMOVE
    if not np.any(pos):
        return 0.0
    return float(np.isin(yp[pos], list(ABSTAIN)).mean())


def false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normal action predicted as a HELMET_REMOVE alert."""
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    neg = yt != REMOVE
    if not np.any(neg):
        return 0.0
    return float((yp[neg] == REMOVE).mean())


def safety_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "confirmed_remove_recall": confirmed_remove_recall(y_true, y_pred),
        "unknown_rate": unknown_rate(y_pred),
        "unknown_on_positive_rate": unknown_on_positive_rate(y_true, y_pred),
        "false_safe_rate": false_safe_rate(y_true, y_pred),
        "false_alarm_rate": false_alarm_rate(y_true, y_pred),
        "insufficient_pose_rate": insufficient_rate(y_pred),
        "abstain_rate": abstain_rate(y_pred),
    }
