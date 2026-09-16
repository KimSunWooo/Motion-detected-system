"""Validation-only fusion weight search. Never peek at test / REMOVE_C / OOD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from helmet_action.evaluation.false_safe import (
    confirmed_remove_recall,
    false_alarm_rate,
    false_safe_rate,
    unknown_rate,
)
from helmet_action.models.hybrid import HybridActionClassifier


@dataclass(frozen=True)
class FusionWeights:
    w_ml: float
    w_rule: float
    w_phase: float
    w_motion: float
    alert_threshold: float
    watch_threshold: float

    def as_dict(self) -> dict[str, float]:
        return {
            "w_ml": float(self.w_ml),
            "w_rule": float(self.w_rule),
            "w_phase": float(self.w_phase),
            "w_motion": float(self.w_motion),
            "alert_threshold": float(self.alert_threshold),
            "watch_threshold": float(self.watch_threshold),
        }


DEFAULT_FUSION = FusionWeights(0.40, 0.20, 0.30, 0.10, 0.62, 0.48)

# Small simplex. Sum of the four weights is 1.0.
WEIGHT_GRID = (
    (0.40, 0.20, 0.30, 0.10),
    (0.45, 0.15, 0.30, 0.10),
    (0.35, 0.20, 0.35, 0.10),
    (0.40, 0.15, 0.35, 0.10),
    (0.50, 0.15, 0.25, 0.10),
    (0.35, 0.25, 0.30, 0.10),
    (0.30, 0.20, 0.40, 0.10),
    (0.45, 0.20, 0.25, 0.10),
)

ALERT_GRID = (0.58, 0.62, 0.66)
WATCH_GRID = (0.44, 0.48, 0.52)


def iter_fusion_candidates() -> Iterable[FusionWeights]:
    for w in WEIGHT_GRID:
        s = sum(w)
        if abs(s - 1.0) > 1e-9:
            continue
        for alert in ALERT_GRID:
            for watch in WATCH_GRID:
                if watch >= alert:
                    continue
                yield FusionWeights(w[0], w[1], w[2], w[3], alert, watch)


def _apply_weights(cfg_mutate: dict[str, Any], weights: FusionWeights) -> None:
    fusion = cfg_mutate.setdefault("decision", {}).setdefault("fusion", {})
    fusion.update(weights.as_dict())


def score_fusion_on_validation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    max_false_safe: float = 0.03,
    max_false_alarm: float = 0.08,
) -> dict[str, float]:
    rec = confirmed_remove_recall(y_true, y_pred)
    fs = false_safe_rate(y_true, y_pred)
    fa = false_alarm_rate(y_true, y_pred)
    unk = unknown_rate(y_pred)
    feasible = fs <= max_false_safe and fa <= max_false_alarm
    # Higher confirmed recall, lower unknown; hard-penalize safety violations.
    objective = rec - 0.20 * unk - 4.0 * max(0.0, fs - max_false_safe) - 3.0 * max(0.0, fa - max_false_alarm)
    return {
        "confirmed_remove_recall": rec,
        "unknown_rate": unk,
        "false_safe_rate": fs,
        "false_alarm_rate": fa,
        "feasible": float(feasible),
        "objective": float(objective),
    }


def select_fusion_weights_on_validation(
    hybrid: HybridActionClassifier,
    sequences: list[tuple[np.ndarray, np.ndarray | None, str]],
    *,
    apply_to_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Grid-search weights on the provided validation sequences only."""
    from helmet_action.config import load_config

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    try:
        cfg = load_config()
        fusion_live = cfg._data.setdefault("decision", {}).setdefault("fusion", {})
        for cand in iter_fusion_candidates():
            fusion_live.update(cand.as_dict())
            y_true, y_pred = [], []
            for k, c, lab in sequences:
                dec = hybrid.predict_v2(k, c)
                y_true.append(lab)
                y_pred.append(dec.action.value)
            metrics = score_fusion_on_validation(np.array(y_true), np.array(y_pred))
            row = {**cand.as_dict(), **metrics}
            rows.append(row)
            if best is None or (
                (metrics["feasible"] > best["feasible"])
                or (metrics["feasible"] == best["feasible"] and metrics["objective"] > best["objective"])
            ):
                best = row
        selected = {k: best[k] for k in DEFAULT_FUSION.as_dict()} if best else DEFAULT_FUSION.as_dict()
        fusion_live.update(selected)
        if apply_to_config is not None:
            apply_to_config.setdefault("decision", {}).setdefault("fusion", {}).update(selected)
    except Exception:
        load_config.cache_clear()
        raise

    if best is None:
        best = {**DEFAULT_FUSION.as_dict(), "objective": 0.0, "feasible": 0.0}
    return {
        "selected": {k: best[k] for k in DEFAULT_FUSION.as_dict()},
        "selected_metrics": {
            k: best[k]
            for k in (
                "confirmed_remove_recall",
                "unknown_rate",
                "false_safe_rate",
                "false_alarm_rate",
                "objective",
                "feasible",
            )
        },
        "n_candidates": len(rows),
        "n_validation": len(sequences),
        "note": "Weights chosen on validation only. Test/OOD must be evaluated once after freeze.",
        "candidates": rows,
    }
