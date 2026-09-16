from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from helmet_action.config import load_config
from helmet_action.models.labels import RemovalPhase
from helmet_action.pose.constants import L_WRIST, R_WRIST
from helmet_action.pose.geometry import compute_head_regions, head_center_norm
from helmet_action.pose.normalizer import normalize_keypoints


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
    saw_lift: bool = False
    ordered: bool = False


def infer_phases(keypoints: np.ndarray) -> PhaseTrace:
    """Map a sequence onto IDLE → APPROACH → GRASP → LIFT → CONFIRMED."""
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
    both = np.array([regions[i].in_left_ear(lw[i]) and regions[i].in_right_ear(rw[i]) for i in range(t)])
    mean_xy = 0.5 * (lw + rw)
    speed = np.linalg.norm(np.gradient(mean_xy, axis=0), axis=1)
    dx = np.abs(rw[:, 0] - lw[:, 0])
    rise = -np.gradient(mean_xy[:, 1])
    r_mean = 0.5 * (np.linalg.norm(lw - heads, axis=1) + np.linalg.norm(rw - heads, axis=1))

    approach_tau = float(cfg.get("state.approach_radial_tau", -0.015))
    grasp_std = float(cfg.get("state.grasp_std", 0.020))
    grasp_min = int(cfg.get("state.grasp_min_frames", 6))
    lift_spread = float(cfg.get("state.lift_spread_tau", 0.06))
    lift_rise = float(cfg.get("state.lift_rise_tau", 0.04))
    lift_rad = float(cfg.get("state.lift_radial_tau", 0.06))

    approach = radial_v < approach_tau
    grasp = both & (speed < grasp_std)
    # lift signals after first grasp
    g_idx = np.where(grasp)[0]
    lift = np.zeros(t, dtype=bool)
    if g_idx.size:
        g0 = int(g_idx[0])
        spread = dx - dx[g0]
        co_rise = mean_xy[g0, 1] - mean_xy[:, 1]
        expand = r_mean - r_mean[g0]
        going_down = mean_xy[:, 1] > (mean_xy[g0, 1] + 0.02)
        candidate = (spread[g0:] > lift_spread) | (co_rise[g0:] > lift_rise) | (expand[g0:] > lift_rad)
        lift[g0:] = candidate & ~going_down[g0:]

    history: list[str] = []
    for i in range(t):
        if lift[i]:
            history.append(RemovalPhase.LIFT_OR_SEPARATE.value)
        elif grasp[i]:
            history.append(RemovalPhase.HELMET_GRASP.value)
        elif approach[i]:
            history.append(RemovalPhase.HAND_APPROACH.value)
        else:
            history.append(RemovalPhase.IDLE.value)

    saw_approach = bool(np.any(approach))
    saw_grasp = bool(np.max(np.convolve(grasp.astype(float), np.ones(grasp_min), mode="same")) >= grasp_min)
    saw_lift = bool(np.any(lift))
    # order: first approach or grasp, then grasp, then lift
    first = {p: (history.index(p) if p in history else 10**9) for p in (
        RemovalPhase.HAND_APPROACH.value,
        RemovalPhase.HELMET_GRASP.value,
        RemovalPhase.LIFT_OR_SEPARATE.value,
    )}
    ordered = first[RemovalPhase.HELMET_GRASP.value] < first[RemovalPhase.LIFT_OR_SEPARATE.value]
    if saw_grasp and saw_lift and ordered:
        phase = RemovalPhase.REMOVAL_CONFIRMED
    elif saw_lift:
        phase = RemovalPhase.LIFT_OR_SEPARATE
    elif saw_grasp:
        phase = RemovalPhase.HELMET_GRASP
    elif saw_approach:
        phase = RemovalPhase.HAND_APPROACH
    else:
        phase = RemovalPhase.IDLE
    return PhaseTrace(
        phase=phase,
        history=history,
        saw_approach=saw_approach,
        saw_grasp=saw_grasp,
        saw_lift=saw_lift,
        ordered=bool(ordered and saw_grasp and saw_lift),
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
    return {"final_phase": trace.phase.value, "p_remove": p_rm, "ordered": trace.ordered, "rows": rows}


class ActionPhaseMachine:
    def __init__(self) -> None:
        self.trace = PhaseTrace()

    def update(self, keypoints: np.ndarray) -> PhaseTrace:
        self.trace = infer_phases(keypoints)
        return self.trace
