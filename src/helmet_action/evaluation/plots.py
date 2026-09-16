"""Failure and stress plots. Import errors must never fail core evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")


def plot_failure(out_dir: Path, seq_norm: np.ndarray, series: dict, phases: list[str], payload: dict) -> None:
    import matplotlib.pyplot as plt

    from helmet_action.pose.constants import L_WRIST, R_WRIST, SKELETON_BONES

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(seq_norm.shape[0])

    fig, ax = plt.subplots(figsize=(6.2, 6.2), facecolor="#0b1220")
    ax.set_facecolor("#10182a")
    step = max(1, len(seq_norm) // 8)
    for fi in range(0, len(seq_norm), step):
        pose = seq_norm[fi]
        alpha = 0.25 + 0.75 * (fi / max(len(seq_norm) - 1, 1))
        for a, b in SKELETON_BONES:
            ax.plot([pose[a, 0], pose[b, 0]], [pose[a, 1], pose[b, 1]], color="#5ee0ff", lw=1.2, alpha=alpha)
        ax.scatter(pose[[L_WRIST, R_WRIST], 0], pose[[L_WRIST, R_WRIST], 1], c="#ffd166", s=18, alpha=alpha)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("skeleton trajectory", color="#e8eefc")
    fig.savefig(out_dir / "skeleton_trajectory.png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)

    def _line(path: str, ys, title: str, color: str) -> None:
        fig, ax = plt.subplots(figsize=(8.0, 3.2), facecolor="#0b1220")
        ax.set_facecolor("#10182a")
        ax.plot(t, ys, color=color, lw=2.0)
        ax.set_title(title, color="#e8eefc")
        ax.tick_params(colors="#8aa0c4")
        fig.tight_layout()
        fig.savefig(out_dir / path, dpi=120, facecolor=fig.get_facecolor())
        plt.close(fig)

    _line("wrist_head.png", np.minimum(series["lw_head"], series["rw_head"]), "wrist-head distance", "#ffd166")
    _line("wrist_sep.png", series["wrist_sep"], "wrist separation", "#ff8fab")
    _line("radial_v.png", series["radial_v"], "radial velocity", "#5ee0ff")
    _line("p_remove.png", series.get("p_helmet_remove", [payload.get("ml_proba", {}).get("HELMET_REMOVE", 0.0)] * len(t)), "P(HELMET_REMOVE)", "#ff5c5c")

    fig, ax = plt.subplots(figsize=(8.0, 2.6), facecolor="#0b1220")
    ax.set_facecolor("#10182a")
    order = ["IDLE", "HAND_APPROACH", "HELMET_GRASP", "LIFT_OR_SEPARATE", "REMOVAL_CONFIRMED"]
    ymap = {p: i for i, p in enumerate(order)}
    ys = [ymap.get(p, 0) for p in phases]
    ax.step(t, ys, where="post", color="#3dd68c", lw=2.0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, color="#c9d6f0", fontsize=8)
    ax.set_title("phase transition", color="#e8eefc")
    fig.tight_layout()
    fig.savefig(out_dir / "phase.png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_stress_curves(out_dir: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _curve(name: str, rows: list[dict], xkey: str, title: str) -> None:
        if not rows:
            return
        xs = [r[xkey] for r in rows]
        ys = [r.get("helmet_remove_fnr", r.get("fnr", 0.0)) for r in rows]
        fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor="#0b1220")
        ax.set_facecolor("#10182a")
        ax.plot(xs, ys, marker="o", color="#ff5c5c", lw=2.0)
        ax.set_xlabel(xkey, color="#9db0d0")
        ax.set_ylabel("HELMET_REMOVE FNR", color="#9db0d0")
        ax.set_title(title, color="#e8eefc")
        ax.tick_params(colors="#8aa0c4")
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=120, facecolor=fig.get_facecolor())
        plt.close(fig)

    _curve("fnr_vs_fps.png", summary.get("fps", []), "fps", "FNR vs FPS")
    _curve("fnr_vs_dropout.png", summary.get("occlusion", []), "dropout", "FNR vs dropout")
    _curve("fnr_vs_pitch.png", summary.get("camera_pitch", []), "pitch_deg", "FNR vs camera pitch")
    _curve("fnr_vs_yaw.png", summary.get("camera_yaw", []), "yaw_deg", "FNR vs yaw")
