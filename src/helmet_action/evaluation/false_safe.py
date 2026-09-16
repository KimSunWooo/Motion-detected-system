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
