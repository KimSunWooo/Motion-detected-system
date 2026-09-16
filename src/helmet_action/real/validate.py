"""Validate a processed real-pose dataset without inventing metrics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from helmet_action.real.dataset import RealSequence, list_processed_sequences, load_sequence


def _duration(seq: RealSequence) -> float:
    if seq.timestamps.size >= 2:
        return float(seq.timestamps[-1] - seq.timestamps[0])
    return float(seq.frame_count / max(seq.fps, 1e-6))


def _missing_ratio(seq: RealSequence) -> float:
    conf = seq.confidences
    if conf.size == 0:
        return 1.0
    return float((conf < 0.10).mean())


def _fragmentation(seq: RealSequence) -> int:
    if seq.timestamps.size < 2:
        return 0
    dt = np.diff(seq.timestamps)
    expected = 1.0 / max(seq.fps, 1e-6)
    return int(np.sum(dt > expected * 3.5))


def validate_sequence(seq: RealSequence, path: Path | None = None) -> dict[str, Any]:
    issues: list[str] = []
    k = seq.keypoints
    if k.ndim != 3 or k.shape[1:] != (17, 2):
        issues.append(f"keypoint shape {k.shape} is not (T,17,2)")
    if seq.confidences.shape[:2] != k.shape[:2] and seq.confidences.shape != (k.shape[0], 17):
        issues.append(f"confidence shape {seq.confidences.shape} does not match keypoints")
    if np.isnan(k).all():
        issues.append("all keypoints NaN")
    if seq.frame_count < 8:
        issues.append("sequence shorter than 8 frames")
    if seq.fps <= 1e-3:
        issues.append("invalid fps")
    if not seq.subject_id:
        issues.append("missing subject_id")
    if not seq.label:
        issues.append("missing label")
    if _missing_ratio(seq) > 0.85:
        issues.append("missing-keypoint ratio > 0.85 (pose detection likely failed)")
    return {
        "path": None if path is None else str(path),
        "ok": not issues,
        "issues": issues,
        "subject_id": seq.subject_id,
        "label": seq.label,
        "ml_label": seq.ml_label,
        "source_video": seq.source_video,
        "camera_id": seq.camera_id,
        "fps": float(seq.fps),
        "frame_count": int(seq.frame_count),
        "duration_s": _duration(seq),
        "missing_keypoint_ratio": _missing_ratio(seq),
        "track_fragmentation_gaps": _fragmentation(seq),
        "n_tracks": int(len(set(map(int, seq.track_ids.tolist())))) if seq.track_ids.size else 0,
        "mean_confidence": float(np.nanmean(seq.confidences)) if seq.confidences.size else 0.0,
        "nan_keypoint_ratio": float(np.isnan(k).any(axis=-1).mean()) if k.size else 1.0,
    }


def validate_dataset(processed_root: Path) -> dict[str, Any]:
    processed_root = Path(processed_root)
    seq_paths = list_processed_sequences(processed_root)
    meta_path = processed_root / "metadata.json"
    reports = []
    load_errors = []
    for path in seq_paths:
        try:
            seq = load_sequence(path)
            reports.append(validate_sequence(seq, path))
        except Exception as exc:
            load_errors.append({"path": str(path), "error": str(exc)})
    labels = Counter(r["label"] for r in reports)
    subjects = Counter(r["subject_id"] for r in reports)
    cameras = Counter(r["camera_id"] for r in reports)
    videos = Counter(r["source_video"] for r in reports)
    n_ok = sum(1 for r in reports if r["ok"])
    short = [r for r in reports if r["frame_count"] < 12]
    failed_pose = [r for r in reports if r["missing_keypoint_ratio"] > 0.85]
    return {
        "processed_root": str(processed_root),
        "metadata_exists": meta_path.exists(),
        "n_sequences": len(reports),
        "n_valid": n_ok,
        "n_invalid": len(reports) - n_ok + len(load_errors),
        "load_errors": load_errors,
        "label_distribution": dict(labels),
        "subject_distribution": dict(subjects),
        "camera_distribution": dict(cameras),
        "videos_with_sequences": dict(videos),
        "n_short_sequences": len(short),
        "n_pose_detection_failures": len(failed_pose),
        "pose_detection_failure_rate": float(len(failed_pose) / max(len(reports), 1)),
        "mean_fps": float(np.mean([r["fps"] for r in reports])) if reports else 0.0,
        "mean_duration_s": float(np.mean([r["duration_s"] for r in reports])) if reports else 0.0,
        "mean_missing_keypoint_ratio": float(np.mean([r["missing_keypoint_ratio"] for r in reports])) if reports else 0.0,
        "mean_confidence": float(np.mean([r["mean_confidence"] for r in reports])) if reports else 0.0,
        "subject_ids_preserved": all(bool(r["subject_id"]) for r in reports) if reports else True,
        "sequences": reports,
        "available": len(reports) > 0,
    }
