"""UNKNOWN rejection-reason statistics: why Hybrid abstained despite ML."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from helmet_action.models.evidence import REJECTION_REASONS


def summarize_unknown_reasons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Each row is one UNKNOWN (or INSUFFICIENT) decision with evidence fields."""
    n = max(len(rows), 1)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        reasons = row.get("rejection_reasons") or row.get("reasons") or []
        if not reasons:
            reasons = ["UNKNOWN"]
        for reason in reasons:
            buckets[str(reason)].append(row)

    ranked = []
    for reason, items in buckets.items():
        p_ml = [float(x.get("ml_remove_probability", x.get("ml_p_remove", 0.0)) or 0.0) for x in items]
        p_phase = [float(x.get("phase_confidence", 0.0) or 0.0) for x in items]
        p_pose = [float(x.get("pose_quality", 0.0) or 0.0) for x in items]
        ranked.append(
            {
                "reason": reason,
                "count": len(items),
                "ratio": float(len(items) / n),
                "ml_remove_probability_mean": float(np.mean(p_ml)) if p_ml else 0.0,
                "phase_confidence_mean": float(np.mean(p_phase)) if p_phase else 0.0,
                "pose_quality_mean": float(np.mean(p_pose)) if p_pose else 0.0,
            }
        )
    ranked.sort(key=lambda d: d["count"], reverse=True)
    return {
        "n_unknown": len(rows),
        "reason_catalog": list(REJECTION_REASONS),
        "ranked": ranked,
        "reason_counts": {d["reason"]: d["count"] for d in ranked},
        "reason_ratio": {d["reason"]: d["ratio"] for d in ranked},
    }
