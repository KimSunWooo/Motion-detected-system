"""Serialize HELMET_REMOVE false-negative artifacts for human analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from helmet_action.features.temporal_features import per_frame_features
from helmet_action.models.hybrid import HybridActionClassifier
from helmet_action.models.labels import ActionClass
from helmet_action.models.rule_based import RuleBasedActionClassifier
from helmet_action.pose.confidence import prepare_sequence
from helmet_action.pose.constants import L_EAR, L_WRIST, R_EAR, R_WRIST
from helmet_action.pose.geometry import head_center_norm
from helmet_action.pose.normalizer import normalize_keypoints
from helmet_action.state.action_state_machine import infer_phases


def _series(seq_norm: np.ndarray) -> dict[str, list[float]]:
    ff = per_frame_features(seq_norm)
    t = seq_norm.shape[0]
    heads = np.stack([head_center_norm(seq_norm[i]) for i in range(t)])
    lw, rw = seq_norm[:, L_WRIST], seq_norm[:, R_WRIST]
    min_wh = np.minimum(np.linalg.norm(lw - heads, axis=1), np.linalg.norm(rw - heads, axis=1))
    sep = np.linalg.norm(lw - rw, axis=1)
    vert_v = np.gradient(0.5 * (lw[:, 1] + rw[:, 1]))
    sep_v = np.gradient(sep)
    radial_v = np.gradient(min_wh)
    speed = np.linalg.norm(0.5 * (np.gradient(lw, axis=0) + np.gradient(rw, axis=0)), axis=1)
    return {
        "lw_head": ff.lw_head.tolist(),
        "rw_head": ff.rw_head.tolist(),
        "lw_lear": ff.lw_lear.tolist(),
        "rw_rear": ff.rw_rear.tolist(),
        "wrist_sep": sep.tolist(),
        "wrist_sep_v": sep_v.tolist(),
        "vertical_wrist_v": vert_v.tolist(),
        "radial_v": radial_v.tolist(),
        "motion_energy": (speed**2).tolist(),
        "left_elbow": ff.left_elbow.tolist(),
        "right_elbow": ff.right_elbow.tolist(),
    }


def dump_failure(
    out_dir: Path,
    failure_id: str,
    keypoints: np.ndarray,
    confidence: np.ndarray,
    meta: dict[str, Any],
    clf,
    feature_vector: np.ndarray | None = None,
    y_true: str = "HELMET_REMOVE",
    y_pred: str = "",
    ml_proba: dict[str, float] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq, quality = prepare_sequence(keypoints, confidence)
    seq_norm, _ = normalize_keypoints(seq)
    rule = RuleBasedActionClassifier().predict(seq)
    phase = infer_phases(seq)
    hybrid = HybridActionClassifier(ml=clf if hasattr(clf, "predict_proba") else None)
    decision = hybrid.predict(keypoints, confidence)
    proba = ml_proba or (clf.predict_proba(keypoints, confidence) if clf is not None else {})
    series = _series(seq_norm)
    series["phase"] = phase.history
    series["p_helmet_remove"] = [float(proba.get(ActionClass.HELMET_REMOVE.value, 0.0))] * seq_norm.shape[0]
    if clf is not None and hasattr(clf, "predict_proba") and seq.shape[0] >= 8:
        p_t = []
        win = max(12, seq.shape[0] // 3)
        for i in range(seq.shape[0]):
            a = max(0, i - win + 1)
            try:
                p_t.append(float(clf.predict_proba(keypoints[a : i + 1], confidence[a : i + 1]).get(ActionClass.HELMET_REMOVE.value, 0.0)))
            except Exception:
                p_t.append(float(proba.get(ActionClass.HELMET_REMOVE.value, 0.0)))
        series["p_helmet_remove"] = p_t
    if confidence is not None:
        series["lw_conf"] = np.asarray(confidence)[:, L_WRIST].tolist()
        series["rw_conf"] = np.asarray(confidence)[:, R_WRIST].tolist()
        series["lear_conf"] = np.asarray(confidence)[:, L_EAR].tolist()
        series["rear_conf"] = np.asarray(confidence)[:, R_EAR].tolist()
    payload = {
        "failure_id": failure_id,
        "y_true": y_true,
        "y_pred": y_pred or decision.action.value,
        "camera": meta.get("camera", {}),
        "body": meta.get("body", {}),
        "action": meta.get("action", {}),
        "noise": meta.get("noise", {}),
        "fps": meta.get("fps", 20.0),
        "family": meta.get("family", ""),
        "variant": meta.get("variant", ""),
        "occlusion": meta.get("occlusion", {}),
        "ml_proba": {k: float(v) for k, v in proba.items()},
        "rule_prediction": rule.label.value,
        "rule_confidence": float(rule.confidence),
        "phase": phase.phase.value,
        "phase_history": phase.history,
        "phase_ordered": bool(phase.ordered),
        "final_prediction": decision.action.value,
        "quality": quality.quality.value,
        "feature_vector": None if feature_vector is None else np.asarray(feature_vector).astype(float).tolist(),
        "series": series,
        "n_frames": int(seq.shape[0]),
        "seed": meta.get("seed"),
        "scenario": meta.get("scenario"),
    }
    np.savez_compressed(
        out_dir / f"{failure_id}.npz",
        keypoints=np.asarray(keypoints),
        confidence=np.asarray(confidence),
        feature_vector=np.asarray(feature_vector if feature_vector is not None else []),
    )
    (out_dir / f"{failure_id}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        from helmet_action.evaluation.plots import plot_failure

        plot_failure(out_dir / failure_id, seq_norm, series, phase.history, payload)
    except Exception:
        pass
    return payload
