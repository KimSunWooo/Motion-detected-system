from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from helmet_action.config import load_config
from helmet_action.features.v1_names import FEATURE_DIM_V1, FEATURE_NAMES_V1
from helmet_action.pose.constants import L_EAR, L_ELBOW, L_SHOULDER, L_WRIST, R_EAR, R_ELBOW, R_SHOULDER, R_WRIST
from helmet_action.pose.geometry import (
    compute_head_regions,
    elbow_angles,
    forearm_angles,
    head_center_norm,
    head_scale_norm,
    shoulder_orientation,
    upper_arm_angles,
    wrist_head_direction_angles,
)


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else 0.0


def _nanstd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanstd(x)) if np.isfinite(x).sum() > 1 else 0.0


def _nanmax(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmax(x)) if np.isfinite(x).any() else 0.0


def _nanmin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmin(x)) if np.isfinite(x).any() else 0.0


def _delta(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan)
    if x.size:
        out[0] = 0.0
        out[1:] = x[1:] - x[:-1]
    return out


def _reversals(x: np.ndarray, deadband: float = 0.008) -> int:
    d = _delta(x)
    d = d[np.isfinite(d)]
    d = d[np.abs(d) > deadband]
    if d.size < 2:
        return 0
    s = np.sign(d)
    return int(np.sum(s[1:] * s[:-1] < 0))


@dataclass
class FrameFeatures:
    lw_head: np.ndarray
    rw_head: np.ndarray
    lw_lear: np.ndarray
    rw_rear: np.ndarray
    wrist_sep: np.ndarray
    lw_height: np.ndarray
    rw_height: np.ndarray
    left_elbow: np.ndarray
    right_elbow: np.ndarray
    left_upper: np.ndarray
    right_upper: np.ndarray
    left_fore: np.ndarray
    right_fore: np.ndarray
    lw_head_dir: np.ndarray
    rw_head_dir: np.ndarray
    head_scale: np.ndarray
    shoulder_ori: np.ndarray
    in_bbox_l: np.ndarray
    in_bbox_r: np.ndarray
    in_ear_both: np.ndarray
    in_center_any: np.ndarray


def per_frame_features(seq_norm: np.ndarray) -> FrameFeatures:
    t = seq_norm.shape[0]
    lw = seq_norm[:, L_WRIST]
    rw = seq_norm[:, R_WRIST]
    heads = np.stack([head_center_norm(seq_norm[i]) for i in range(t)])
    regions = [compute_head_regions(seq_norm[i]) for i in range(t)]
    elbows = np.array([elbow_angles(seq_norm[i]) for i in range(t)])
    upper = np.array([upper_arm_angles(seq_norm[i]) for i in range(t)])
    fore = np.array([forearm_angles(seq_norm[i]) for i in range(t)])
    dirs = np.array([wrist_head_direction_angles(seq_norm[i]) for i in range(t)])
    return FrameFeatures(
        lw_head=np.linalg.norm(lw - heads, axis=1),
        rw_head=np.linalg.norm(rw - heads, axis=1),
        lw_lear=np.linalg.norm(lw - seq_norm[:, L_EAR], axis=1),
        rw_rear=np.linalg.norm(rw - seq_norm[:, R_EAR], axis=1),
        wrist_sep=np.linalg.norm(lw - rw, axis=1),
        lw_height=lw[:, 1] - seq_norm[:, L_SHOULDER, 1],
        rw_height=rw[:, 1] - seq_norm[:, R_SHOULDER, 1],
        left_elbow=elbows[:, 0],
        right_elbow=elbows[:, 1],
        left_upper=upper[:, 0],
        right_upper=upper[:, 1],
        left_fore=fore[:, 0],
        right_fore=fore[:, 1],
        lw_head_dir=dirs[:, 0],
        rw_head_dir=dirs[:, 1],
        head_scale=np.array([head_scale_norm(seq_norm[i]) for i in range(t)]),
        shoulder_ori=np.array([shoulder_orientation(seq_norm[i]) for i in range(t)]),
        in_bbox_l=np.array([regions[i].in_bbox(lw[i]) for i in range(t)], dtype=np.float64),
        in_bbox_r=np.array([regions[i].in_bbox(rw[i]) for i in range(t)], dtype=np.float64),
        in_ear_both=np.array(
            [regions[i].in_left_ear(lw[i]) and regions[i].in_right_ear(rw[i]) for i in range(t)],
            dtype=np.float64,
        ),
        in_center_any=np.array(
            [regions[i].in_center(lw[i]) or regions[i].in_center(rw[i]) for i in range(t)],
            dtype=np.float64,
        ),
    )


def temporal_stats(ff: FrameFeatures, seq_norm: np.ndarray) -> dict[str, float]:
    cfg = load_config()
    pause_tau = float(cfg.get("state.grasp_std", 0.020))
    lw = seq_norm[:, L_WRIST]
    rw = seq_norm[:, R_WRIST]
    vl = np.gradient(lw, axis=0)
    vr = np.gradient(rw, axis=0)
    v_mean = 0.5 * (vl + vr)
    speed = np.linalg.norm(v_mean, axis=1)
    al = np.gradient(vl, axis=0)
    ar = np.gradient(vr, axis=0)

    min_wh = np.minimum(ff.lw_head, ff.rw_head)
    radial_v = _delta(min_wh)
    toward = np.clip(-radial_v, 0, None)
    away = np.clip(radial_v, 0, None)
    vert_v = _delta(0.5 * (lw[:, 1] + rw[:, 1]))
    sep_v = _delta(ff.wrist_sep)

    num = np.sum(vl * vr, axis=1)
    den = np.linalg.norm(vl, axis=1) * np.linalg.norm(vr, axis=1) + 1e-8
    cos_sim = num / den
    bilateral = 1.0 - np.abs(np.linalg.norm(vl, axis=1) - np.linalg.norm(vr, axis=1)) / (
        np.linalg.norm(vl, axis=1) + np.linalg.norm(vr, axis=1) + 1e-8
    )

    pause_mask = speed < pause_tau
    head_dwell = (ff.in_bbox_l + ff.in_bbox_r) > 0
    ear_dwell = ff.in_ear_both > 0.5
    approach = toward > 0.008
    # image y decrease = lift
    lift = (-vert_v) > 0.008

    series = {
        "lw_head": ff.lw_head,
        "rw_head": ff.rw_head,
        "lw_lear": ff.lw_lear,
        "rw_rear": ff.rw_rear,
        "wrist_sep": ff.wrist_sep,
        "lw_height": ff.lw_height,
        "rw_height": ff.rw_height,
        "left_elbow": ff.left_elbow,
        "right_elbow": ff.right_elbow,
        "head_scale": ff.head_scale,
        "shoulder_ori": ff.shoulder_ori,
        "radial_toward": toward,
        "radial_away": away,
        "vert_v": vert_v,
        "sep_v": sep_v,
        "cos_sim": cos_sim,
        "bilateral": bilateral,
        "speed": speed,
        "acc_l": np.linalg.norm(al, axis=1),
        "acc_r": np.linalg.norm(ar, axis=1),
        "in_bbox": (ff.in_bbox_l + ff.in_bbox_r) / 2.0,
        "in_ear_both": ff.in_ear_both,
        "in_center": ff.in_center_any,
    }
    out: dict[str, float] = {}
    for name, arr in series.items():
        out[f"{name}_mean"] = _nanmean(arr)
        out[f"{name}_std"] = _nanstd(arr)
        out[f"{name}_min"] = _nanmin(arr)
        out[f"{name}_max"] = _nanmax(arr)
        finite = arr[np.isfinite(arr)]
        out[f"{name}_delta"] = float(finite[-1] - finite[0]) if finite.size else 0.0

    t = max(len(speed), 1)
    out["pause_duration"] = float(pause_mask.mean())
    out["head_region_dwell"] = float(head_dwell.mean())
    out["ear_region_dwell"] = float(ear_dwell.mean())
    out["approach_duration"] = float(approach.mean())
    out["lift_duration"] = float(lift.mean())
    out["motion_energy"] = _nanmean(speed ** 2)
    out["traj_var_l"] = float(np.nanvar(lw[:, 0]) + np.nanvar(lw[:, 1]))
    out["traj_var_r"] = float(np.nanvar(rw[:, 0]) + np.nanvar(rw[:, 1]))
    out["reversal_l"] = float(_reversals(lw[:, 0]) + _reversals(lw[:, 1]))
    out["reversal_r"] = float(_reversals(rw[:, 0]) + _reversals(rw[:, 1]))
    out["wrist_vel_cos"] = _nanmean(cos_sim)
    out["n_frames"] = float(t)
    # drop raw pixel-like leftovers — all of the above are already neck/shoulder normalized
    return out


def extract_feature_dict(seq_norm: np.ndarray) -> dict[str, float]:
    from helmet_action.features.extractor import extract_features as baseline_extract

    ff = per_frame_features(seq_norm)
    stats = temporal_stats(ff, seq_norm)
    base = baseline_extract(seq_norm)
    stats["rule_bbox_frames"] = float(base.bbox_frames)
    stats["rule_center_frames"] = float(base.center_frames)
    stats["rule_both_ear_frames"] = float(base.both_ear_frames)
    stats["rule_scratch_radius"] = float(base.scratch_radius)
    stats["rule_scratch_std"] = float(base.scratch_std)
    stats["rule_n_oscillations"] = float(base.n_oscillations)
    stats["rule_pause"] = float(base.pause_detected)
    stats["rule_dx_spread"] = float(base.dx_spread)
    stats["rule_co_rise_y"] = float(base.co_rise_y)
    stats["rule_radial_expand"] = float(base.radial_expand)
    stats["rule_head_scale_up"] = float(base.head_scale_up)
    stats["rule_wrist_spread"] = float(base.wrist_spread)
    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in stats.items()}


FEATURE_ORDER: list[str] = list(FEATURE_NAMES_V1)
FEATURE_DIM_V1_PUBLIC = FEATURE_DIM_V1


def feature_names() -> list[str]:
    return list(FEATURE_ORDER)


def extract_feature_vector(seq_norm: np.ndarray) -> np.ndarray:
    d = extract_feature_dict(seq_norm)
    names = feature_names()
    return np.array([d.get(n, 0.0) for n in names], dtype=np.float64)
