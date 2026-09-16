"""Phase-aware FeatureExtractorV2: V1 (139) + quartile / timing features.

V1 names are never deleted. V2 is a strict superset with a frozen extra list.
"""

from __future__ import annotations

import numpy as np

from helmet_action.features.temporal_features import (
    _delta,
    _nanmax,
    _nanmean,
    extract_feature_dict,
    per_frame_features,
)
from helmet_action.features.v1_names import FEATURE_NAMES_V1
from helmet_action.pose.constants import L_WRIST, R_WRIST

QUARTILE_SERIES = (
    "wrist_head",
    "wrist_sep",
    "vert_v",
    "radial_v",
    "motion_energy",
    "head_dwell",
    "ear_dwell",
)

FEATURE_NAMES_V2_EXTRA: list[str] = []
for q in ("q1", "q2", "q3", "q4"):
    for name in QUARTILE_SERIES:
        FEATURE_NAMES_V2_EXTRA.append(f"{q}_{name}_mean")
    FEATURE_NAMES_V2_EXTRA.append(f"{q}_wrist_head_min")
    FEATURE_NAMES_V2_EXTRA.append(f"{q}_wrist_sep_max")

FEATURE_NAMES_V2_EXTRA.extend(
    [
        "el_wrist_head_delta",
        "el_wrist_sep_delta",
        "el_vert_v_delta",
        "el_radial_v_delta",
        "el_motion_energy_delta",
        "el_head_dwell_delta",
        "el_ear_dwell_delta",
        "peak_wrist_sep_t",
        "peak_radial_away_t",
        "peak_vert_lift_t",
        "first_head_contact_t",
        "grasp_start_t",
        "lift_start_t",
        "max_sep_t",
        "approach_to_lift_delay",
        "late_sep_mean",
        "late_radial_away_mean",
        "late_vert_lift_mean",
        "early_head_dwell_mean",
    ]
)

FEATURE_NAMES_V2: list[str] = list(FEATURE_NAMES_V1) + FEATURE_NAMES_V2_EXTRA
FEATURE_DIM_V2 = len(FEATURE_NAMES_V2)


def _norm_t(idx: int, n: int) -> float:
    return float(idx) / float(max(n - 1, 1))


def _peak_t(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(arr).any():
        return 0.0
    i = int(np.nanargmax(arr))
    return _norm_t(i, len(arr))


def _first_true_t(mask: np.ndarray) -> float:
    idx = np.where(np.asarray(mask, dtype=bool))[0]
    if idx.size == 0:
        return 1.0
    return _norm_t(int(idx[0]), len(mask))


def _quartile_slices(n: int) -> list[slice]:
    edges = [int(round(n * k / 4.0)) for k in range(5)]
    edges[0] = 0
    edges[-1] = n
    out = []
    for i in range(4):
        a, b = edges[i], max(edges[i + 1], edges[i] + 1)
        out.append(slice(a, min(b, n)))
    return out


def extract_feature_dict_v2(seq_norm: np.ndarray) -> dict[str, float]:
    base = extract_feature_dict(seq_norm)
    ff = per_frame_features(seq_norm)
    t = seq_norm.shape[0]
    lw = seq_norm[:, L_WRIST]
    rw = seq_norm[:, R_WRIST]
    wrist_head = np.minimum(ff.lw_head, ff.rw_head)
    radial_v = _delta(wrist_head)
    vert_v = _delta(0.5 * (lw[:, 1] + rw[:, 1]))
    speed = np.linalg.norm(0.5 * (np.gradient(lw, axis=0) + np.gradient(rw, axis=0)), axis=1)
    motion_energy = speed**2
    head_dwell = ((ff.in_bbox_l + ff.in_bbox_r) > 0).astype(np.float64)
    ear_dwell = (ff.in_ear_both > 0.5).astype(np.float64)
    radial_away = np.clip(radial_v, 0, None)
    vert_lift = np.clip(-vert_v, 0, None)

    series = {
        "wrist_head": wrist_head,
        "wrist_sep": ff.wrist_sep,
        "vert_v": vert_v,
        "radial_v": radial_v,
        "motion_energy": motion_energy,
        "head_dwell": head_dwell,
        "ear_dwell": ear_dwell,
    }
    extra: dict[str, float] = {}
    slices = _quartile_slices(t)
    q_means: dict[str, list[float]] = {k: [] for k in series}
    for qi, sl in enumerate(slices, start=1):
        prefix = f"q{qi}"
        for name, arr in series.items():
            extra[f"{prefix}_{name}_mean"] = _nanmean(arr[sl])
            q_means[name].append(extra[f"{prefix}_{name}_mean"])
        extra[f"{prefix}_wrist_head_min"] = float(np.nanmin(wrist_head[sl])) if np.isfinite(wrist_head[sl]).any() else 0.0
        extra[f"{prefix}_wrist_sep_max"] = _nanmax(ff.wrist_sep[sl])

    def _el(name: str) -> float:
        vals = q_means[name]
        return float(vals[-1] - vals[0]) if vals else 0.0

    extra["el_wrist_head_delta"] = _el("wrist_head")
    extra["el_wrist_sep_delta"] = _el("wrist_sep")
    extra["el_vert_v_delta"] = _el("vert_v")
    extra["el_radial_v_delta"] = _el("radial_v")
    extra["el_motion_energy_delta"] = _el("motion_energy")
    extra["el_head_dwell_delta"] = _el("head_dwell")
    extra["el_ear_dwell_delta"] = _el("ear_dwell")

    extra["peak_wrist_sep_t"] = _peak_t(ff.wrist_sep)
    extra["peak_radial_away_t"] = _peak_t(radial_away)
    extra["peak_vert_lift_t"] = _peak_t(vert_lift)
    extra["first_head_contact_t"] = _first_true_t(wrist_head < 0.38)
    extra["grasp_start_t"] = _first_true_t(ear_dwell > 0.5)
    extra["lift_start_t"] = _first_true_t((vert_lift > 0.008) & (np.arange(t) >= extra["grasp_start_t"] * max(t - 1, 1)))
    extra["max_sep_t"] = extra["peak_wrist_sep_t"]
    extra["approach_to_lift_delay"] = float(
        np.clip(extra["lift_start_t"] - extra["first_head_contact_t"], -1.0, 1.0)
    )
    late = slices[-1]
    extra["late_sep_mean"] = _nanmean(ff.wrist_sep[late])
    extra["late_radial_away_mean"] = _nanmean(radial_away[late])
    extra["late_vert_lift_mean"] = _nanmean(vert_lift[late])
    extra["early_head_dwell_mean"] = _nanmean(head_dwell[slices[0]])

    out = dict(base)
    for k in FEATURE_NAMES_V2_EXTRA:
        v = extra.get(k, 0.0)
        out[k] = 0.0 if not np.isfinite(v) else float(v)
    return out


def extract_feature_vector_v2(seq_norm: np.ndarray) -> np.ndarray:
    d = extract_feature_dict_v2(seq_norm)
    return np.array([d.get(n, 0.0) for n in FEATURE_NAMES_V2], dtype=np.float64)


class FeatureExtractorV2:
    """Explicit V2 extractor used by the sklearn classifier when feature_version=v2."""

    names = FEATURE_NAMES_V2
    dim = FEATURE_DIM_V2

    def extract_dict(self, seq_norm: np.ndarray) -> dict[str, float]:
        return extract_feature_dict_v2(seq_norm)

    def extract(self, seq_norm: np.ndarray) -> np.ndarray:
        return extract_feature_vector_v2(seq_norm)
