from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from helmet_action.config import load_config
from helmet_action.models.labels import RemovalPhase
from helmet_action.pose.constants import L_WRIST, R_WRIST
from helmet_action.pose.geometry import compute_head_regions, head_center_norm
from helmet_action.pose.normalizer import normalize_keypoints
from helmet_action.pose.quality import phase_confidence as boolean_phase_confidence


class AlertLevel(str, Enum):
    CLEAR = "CLEAR"
    WATCH = "WATCH"
    HIGH = "HIGH"


@dataclass
class PhaseTrace:
    phase: RemovalPhase = RemovalPhase.IDLE
    history: list[str] = field(default_factory=list)
    saw_approach: bool = False
    saw_grasp: bool = False
    saw_partial_grasp: bool = False
    saw_lift: bool = False
    ordered: bool = False
    brim_grasp: bool = False
    one_then_two: bool = False
    lateral_lift: bool = False
    grasp_confidence: float = 0.0
    lift_confidence: float = 0.0
    separation_confidence: float = 0.0
    phase_confidence: float = 0.0


def _run_frac(mask: np.ndarray, min_frames: int) -> float:
    if mask.size == 0:
        return 0.0
    best = float(np.max(np.convolve(mask.astype(float), np.ones(min_frames), mode="same"))) if mask.size >= min_frames else float(mask.sum())
    return float(np.clip(best / max(min_frames, 1), 0.0, 1.0))


def infer_phases(keypoints: np.ndarray) -> PhaseTrace:
    """IDLE → APPROACH → (PARTIAL_GRASP) → GRASP → LIFT → CONFIRMED.

    Canonical both-ear grasp is preserved. One-hand, brim, and lateral lift are
    extra tolerated states that do not replace the canonical sequence.
    """
    cfg = load_config()
    seq, _ = normalize_keypoints(keypoints)
    t = seq.shape[0]
    if t < 4:
        return PhaseTrace()
    regions = [compute_head_regions(seq[i]) for i in range(t)]
    lw, rw = seq[:, L_WRIST], seq[:, R_WRIST]
    heads = np.stack([head_center_norm(seq[i]) for i in range(t)])
    min_d = np.minimum(np.linalg.norm(lw - heads, axis=1), np.linalg.norm(rw - heads, axis=1))
    radial_v = np.gradient(min_d)
    speed_l = np.linalg.norm(np.gradient(lw, axis=0), axis=1)
    speed_r = np.linalg.norm(np.gradient(rw, axis=0), axis=1)
    mean_xy = 0.5 * (lw + rw)
    speed = np.linalg.norm(np.gradient(mean_xy, axis=0), axis=1)
    dx = np.abs(rw[:, 0] - lw[:, 0])
    r_mean = 0.5 * (np.linalg.norm(lw - heads, axis=1) + np.linalg.norm(rw - heads, axis=1))
    sep = np.linalg.norm(lw - rw, axis=1)

    approach_tau = float(cfg.get("state.approach_radial_tau", -0.015))
    grasp_std = float(cfg.get("state.grasp_std", 0.020))
    grasp_min = int(cfg.get("state.grasp_min_frames", 6))
    lift_spread = float(cfg.get("state.lift_spread_tau", 0.06))
    lift_rise = float(cfg.get("state.lift_rise_tau", 0.04))
    lift_rad = float(cfg.get("state.lift_radial_tau", 0.06))

    left_ear = np.array([regions[i].in_left_ear(lw[i]) for i in range(t)])
    right_ear = np.array([regions[i].in_right_ear(rw[i]) for i in range(t)])
    left_brim = np.array([regions[i].in_brim(lw[i]) or regions[i].in_center(lw[i]) for i in range(t)])
    right_brim = np.array([regions[i].in_brim(rw[i]) or regions[i].in_center(rw[i]) for i in range(t)])

    approach = radial_v < approach_tau
    both_ear_grasp = left_ear & right_ear & (speed < grasp_std)
    brim_both = left_brim & right_brim & (speed < grasp_std * 1.25)
    # Two-hand helmet contact: canonical ears OR brim (front/top).
    full_grasp = both_ear_grasp | brim_both
    one_hand = ((left_ear & (speed_l < grasp_std * 1.15)) | (right_ear & (speed_r < grasp_std * 1.15))) & ~full_grasp
    one_brim = ((left_brim & (speed_l < grasp_std * 1.25)) | (right_brim & (speed_r < grasp_std * 1.25))) & ~full_grasp & ~one_hand
    partial = one_hand | one_brim

    grasp_like = full_grasp | partial
    g_idx = np.where(grasp_like)[0]
    lift = np.zeros(t, dtype=bool)
    lateral = np.zeros(t, dtype=bool)
    if g_idx.size:
        g0 = int(g_idx[0])
        spread = dx - dx[g0]
        co_rise = mean_xy[g0, 1] - mean_xy[:, 1]
        expand = r_mean - r_mean[g0]
        going_down = mean_xy[:, 1] > (mean_xy[g0, 1] + 0.02)
        left_rise = lw[g0, 1] - lw[:, 1]
        right_rise = rw[g0, 1] - rw[:, 1]
        candidate = (spread[g0:] > lift_spread) | (co_rise[g0:] > lift_rise) | (expand[g0:] > lift_rad)
        lateral_cand = (np.abs(left_rise[g0:] - right_rise[g0:]) > 0.035) & (
            np.maximum(left_rise[g0:], right_rise[g0:]) > lift_rise * 0.65
        )
        lift[g0:] = candidate & ~going_down[g0:]
        lateral[g0:] = lateral_cand & ~going_down[g0:]
        lift = lift | lateral

    history: list[str] = []
    for i in range(t):
        if lift[i]:
            history.append(RemovalPhase.LIFT_OR_SEPARATE.value)
        elif full_grasp[i]:
            history.append(RemovalPhase.HELMET_GRASP.value)
        elif partial[i]:
            history.append(RemovalPhase.PARTIAL_GRASP.value)
        elif approach[i]:
            history.append(RemovalPhase.HAND_APPROACH.value)
        else:
            history.append(RemovalPhase.IDLE.value)

    saw_approach = bool(np.any(approach))
    saw_grasp = bool(np.max(np.convolve(full_grasp.astype(float), np.ones(grasp_min), mode="same")) >= grasp_min)
    saw_partial = bool(np.max(np.convolve(partial.astype(float), np.ones(max(grasp_min - 2, 3)), mode="same")) >= max(grasp_min - 2, 3)) if t >= 3 else bool(np.any(partial))
    saw_lift = bool(np.sum(lift) >= 3)
    brim_grasp = bool(np.max(np.convolve(brim_both.astype(float), np.ones(max(grasp_min - 2, 3)), mode="same")) >= max(grasp_min - 2, 3)) if t >= 3 else bool(np.any(brim_both))
    first_partial = history.index(RemovalPhase.PARTIAL_GRASP.value) if RemovalPhase.PARTIAL_GRASP.value in history else 10**9
    first_full = history.index(RemovalPhase.HELMET_GRASP.value) if RemovalPhase.HELMET_GRASP.value in history else 10**9
    first_lift = history.index(RemovalPhase.LIFT_OR_SEPARATE.value) if RemovalPhase.LIFT_OR_SEPARATE.value in history else 10**9
    one_then_two = first_partial < first_full < 10**9
    # Order: some grasp-like contact before lift. Canonical still prefers full grasp before lift.
    grasp_like_first = min(first_partial, first_full)
    ordered = grasp_like_first < first_lift
    if saw_grasp and saw_lift and first_full < first_lift:
        ordered = True
    canonical_ordered = bool(ordered and saw_grasp and saw_lift)

    if saw_grasp and saw_lift and canonical_ordered:
        phase = RemovalPhase.REMOVAL_CONFIRMED
    elif saw_lift:
        phase = RemovalPhase.LIFT_OR_SEPARATE
    elif saw_grasp:
        phase = RemovalPhase.HELMET_GRASP
    elif saw_partial:
        phase = RemovalPhase.PARTIAL_GRASP
    elif saw_approach:
        phase = RemovalPhase.HAND_APPROACH
    else:
        phase = RemovalPhase.IDLE

    grasp_conf = float(
        np.clip(
            0.70 * _run_frac(full_grasp, grasp_min)
            + 0.20 * _run_frac(partial, max(grasp_min - 2, 3))
            + 0.10 * _run_frac(brim_both, max(grasp_min - 2, 3)),
            0.0,
            1.0,
        )
    )
    if saw_grasp:
        grasp_conf = max(grasp_conf, 0.72)
    elif saw_partial:
        grasp_conf = max(grasp_conf, 0.42)
    lift_conf = float(np.clip(np.sum(lift) / 8.0, 0.0, 1.0))
    if saw_lift:
        lift_conf = max(lift_conf, 0.55)
    early = float(np.nanmean(sep[: max(t // 4, 1)])) if np.isfinite(sep[: max(t // 4, 1)]).any() else 0.0
    late = float(np.nanmean(sep[-max(t // 4, 1) :])) if np.isfinite(sep[-max(t // 4, 1) :]).any() else 0.0
    sep_conf = float(np.clip((late - early) / 0.12, 0.0, 1.0))
    pconf = boolean_phase_confidence(saw_approach, saw_grasp, saw_lift, canonical_ordered)
    # Blend boolean (V1-compatible) with continuous evidence.
    pconf = float(np.clip(0.45 * pconf + 0.55 * (0.4 * grasp_conf + 0.4 * lift_conf + 0.2 * sep_conf), 0.0, 1.0))

    return PhaseTrace(
        phase=phase,
        history=history,
        saw_approach=saw_approach,
        saw_grasp=saw_grasp,
        saw_partial_grasp=saw_partial,
        saw_lift=saw_lift,
        ordered=canonical_ordered,
        brim_grasp=brim_grasp,
        one_then_two=one_then_two,
        lateral_lift=bool(np.sum(lateral) >= 3),
        grasp_confidence=grasp_conf,
        lift_confidence=lift_conf,
        separation_confidence=sep_conf,
        phase_confidence=pconf,
    )


def phase_debug_table(keypoints: np.ndarray, classifier=None) -> dict:
    seq, _ = normalize_keypoints(keypoints)
    trace = infer_phases(keypoints)
    heads = np.stack([head_center_norm(seq[i]) for i in range(len(seq))])
    lw, rw = seq[:, L_WRIST], seq[:, R_WRIST]
    d_l = np.linalg.norm(lw - heads, axis=1)
    d_r = np.linalg.norm(rw - heads, axis=1)
    sep = np.linalg.norm(lw - rw, axis=1)
    radial_v = np.gradient(np.minimum(d_l, d_r))
    p_rm = None
    if classifier is not None:
        p_rm = float(classifier.predict_proba(keypoints).get("HELMET_REMOVE", 0.0))
    rows = [
        {
            "frame": i,
            "phase": trace.history[i],
            "lw_head": float(d_l[i]),
            "rw_head": float(d_r[i]),
            "wrist_sep": float(sep[i]),
            "radial_v": float(radial_v[i]),
        }
        for i in range(len(seq))
    ]
    return {
        "final_phase": trace.phase.value,
        "p_remove": p_rm,
        "ordered": trace.ordered,
        "grasp_confidence": trace.grasp_confidence,
        "lift_confidence": trace.lift_confidence,
        "separation_confidence": trace.separation_confidence,
        "rows": rows,
    }


class ActionPhaseMachine:
    def __init__(self) -> None:
        self.trace = PhaseTrace()

    def update(self, keypoints: np.ndarray) -> PhaseTrace:
        self.trace = infer_phases(keypoints)
        return self.trace
