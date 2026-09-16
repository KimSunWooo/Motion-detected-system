from __future__ import annotations

import numpy as np

from helmet_action.config import load_config
from helmet_action.pose.geometry import compute_neck, compute_shoulder_width
from helmet_action.pose.types import NormalizeInfo


def as_sequence(keypoints: np.ndarray) -> np.ndarray:
    arr = np.asarray(keypoints, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    arr = arr[..., :2]
    if arr.shape[-2] != 17:
        raise ValueError(f"COCO 17 keypoints required, got shape={arr.shape}")
    return arr


def normalize_keypoints(keypoints: np.ndarray) -> tuple[np.ndarray, list[NormalizeInfo]]:
    """Pixel coords → neck origin / shoulder-width units.

    High-angle CCTV foreshortens torso length, so it is never used as scale:

        p̂ = (p − neck) / max(||Ls − Rs||, ε)
    """
    seq = as_sequence(keypoints)
    neck = compute_neck(seq)
    width = compute_shoulder_width(seq)
    eps = float(load_config().get("geometry.shoulder_width_eps", 1e-6))
    scale = np.maximum(width, eps)
    valid = np.isfinite(neck).all(axis=-1) & np.isfinite(scale)
    normalized = np.full_like(seq, np.nan)
    if np.any(valid):
        normalized[valid] = (seq[valid] - neck[valid, None, :]) / scale[valid, None, None]
    infos = [
        NormalizeInfo(
            origin=np.array(neck[i], dtype=np.float64),
            scale=float(scale[i]) if np.isfinite(scale[i]) else float("nan"),
        )
        for i in range(seq.shape[0])
    ]
    return normalized, infos
