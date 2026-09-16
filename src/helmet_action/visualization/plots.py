from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib

if os.environ.get("DISPLAY") in (None, "") or os.environ.get("MPLBACKEND") == "Agg":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from helmet_action.models.labels import LABEL_KO, ActionLabel, ClassificationResult
from helmet_action.pose.constants import (
    CENTER_RADIUS,
    FACE_IDX,
    L_ELBOW,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_SHOULDER,
    R_WRIST,
    SKELETON_BONES,
    UPPER_BONES,
    UPPER_JOINTS,
)
from helmet_action.pose.geometry import compute_head_regions
from helmet_action.pose.normalizer import as_sequence, normalize_keypoints
from helmet_action.synthetic.camera import HighAngleCamera
from helmet_action.synthetic.skeleton import canonical_pose_3d

_KOREAN_FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"


def configure_korean_font() -> None:
    if Path(_KOREAN_FONT).exists():
        font_manager.fontManager.addfont(_KOREAN_FONT)
        name = font_manager.FontProperties(fname=_KOREAN_FONT).get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


configure_korean_font()


def project_canonical(camera: HighAngleCamera | None = None) -> np.ndarray:
    cam = camera or HighAngleCamera.factory_ceiling()
    return cam.project(canonical_pose_3d())


def plot_static_skeleton(
    keypoints: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str = "하이 앵글 CCTV 상체 스켈레톤 (어깨너비 = 1.0)",
) -> plt.Axes:
    configure_korean_font()
    if keypoints is None:
        pix = project_canonical()
        pose, _ = normalize_keypoints(pix)
        pose = pose[0]
    else:
        pose = as_sequence(keypoints)[0]
        if np.linalg.norm(pose[L_SHOULDER] - pose[R_SHOULDER]) > 2.5:
            pose, _ = normalize_keypoints(pose)
            pose = pose[0]

    created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.4, 7.2), facecolor="#0b1220")
        fig.patch.set_facecolor("#0b1220")

    ax.set_facecolor("#10182a")
    rg = compute_head_regions(pose)
    x, y, w, h = rg.as_xywh()
    ax.add_patch(
        Rectangle(
            (x, y), w, h, fill=True, facecolor="#f5c518", alpha=0.08,
            edgecolor="#f5c518", linewidth=1.6, linestyle="--",
            label="머리 Bounding Box (목·어깨너비 비례)", zorder=1,
        )
    )
    ax.add_patch(
        Circle(
            rg.center, rg.center_r, fill=True, facecolor="#3dd68c", alpha=0.16,
            edgecolor="#3dd68c", linewidth=1.4, linestyle=":",
            label="긁기 중심부", zorder=1,
        )
    )
    ax.add_patch(
        Circle(
            rg.left_ear, rg.ear_r, fill=False, edgecolor="#ff8fab",
            linewidth=1.4, linestyle=":", label="귀 모서리 (양손 파지)", zorder=1,
        )
    )
    ax.add_patch(Circle(rg.right_ear, rg.ear_r, fill=False, edgecolor="#ff8fab", linewidth=1.4, linestyle=":"))

    neck = np.array([0.0, 0.0])
    for a, b in UPPER_BONES:
        ax.plot([pose[a, 0], pose[b, 0]], [pose[a, 1], pose[b, 1]], color="#5ee0ff", lw=2.6, solid_capstyle="round", zorder=2)
    ax.plot(
        [pose[L_SHOULDER, 0], neck[0], pose[R_SHOULDER, 0]],
        [pose[L_SHOULDER, 1], neck[1], pose[R_SHOULDER, 1]],
        color="#5ee0ff", lw=2.6, zorder=2,
    )
    ax.plot([neck[0], pose[NOSE, 0]], [neck[1], pose[NOSE, 1]], color="#5ee0ff", lw=2.6, zorder=2)

    js = list(UPPER_JOINTS)
    ax.scatter(pose[js, 0], pose[js, 1], c="#d7ecff", s=52, zorder=4, edgecolors="#0b1220", linewidths=0.6)
    ax.scatter(pose[[L_SHOULDER, R_SHOULDER], 0], pose[[L_SHOULDER, R_SHOULDER], 1], c="#7cffb2", s=64, zorder=5)
    ax.scatter(pose[[L_WRIST, R_WRIST], 0], pose[[L_WRIST, R_WRIST], 1], c="#ffd166", s=64, zorder=5)
    ax.scatter(pose[list(FACE_IDX), 0], pose[list(FACE_IDX), 1], c="#ff8fab", s=42, zorder=5)
    ax.scatter([0], [0], c="#7cffb2", s=80, zorder=6, edgecolors="#0b1220", label="목 (원점)")

    ax.annotate(
        "",
        xy=(pose[R_SHOULDER, 0], pose[R_SHOULDER, 1] - 0.05),
        xytext=(pose[L_SHOULDER, 0], pose[L_SHOULDER, 1] - 0.05),
        arrowprops=dict(arrowstyle="<->", color="#7cffb2", lw=1.6),
    )
    ax.text(0.0, float(pose[L_SHOULDER, 1] - 0.12), "Shoulder Width = 1.0  (유일한 정규화 단위)", ha="center", color="#7cffb2", fontsize=9)
    ax.annotate(
        "원근 단축: 팔이 어깨에 가깝게 겹침",
        xy=(pose[L_ELBOW, 0], pose[L_ELBOW, 1]),
        xytext=(-1.55, 0.72),
        color="#c9d6f0",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#8aa0c4"),
    )
    ax.text(-1.55, float(rg.y_min) - 0.05, "CCTV 하이 앵글\n(대각선 하향)", color="#f5c518", fontsize=9, ha="left", va="top")
    ax.add_patch(
        FancyArrowPatch(
            (-1.35, float(rg.y_min) + 0.15),
            (float(rg.center[0]) - 0.15, float(rg.center[1])),
            arrowstyle="-|>", mutation_scale=12, color="#f5c518", lw=1.3,
        )
    )
    labels = {NOSE: "코", L_SHOULDER: "왼어깨", R_SHOULDER: "오른어깨", L_ELBOW: "팔꿈치", L_WRIST: "왼손목", R_WRIST: "오른손목"}
    for idx, name in labels.items():
        ax.text(pose[idx, 0] + 0.05, pose[idx, 1], name, color="#c9d6f0", fontsize=8, va="center")

    ax.set_title(title, color="#f2f6ff", fontsize=13, pad=12)
    ax.set_xlabel("X (어깨너비 단위, 목=0)", color="#9db0d0")
    ax.set_ylabel("Y (이미지 아래가 +)", color="#9db0d0")
    ax.tick_params(colors="#8aa0c4")
    for spine in ax.spines.values():
        spine.set_color("#2a3a58")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlim(-1.7, 1.7)
    y_bottom = max(float(pose[L_WRIST, 1]), float(pose[R_WRIST, 1])) + 0.35
    ax.set_ylim(y_bottom, float(rg.y_min) - 0.28)
    ax.grid(True, color="#1c2a44", linestyle=":", linewidth=0.8)
    ax.legend(loc="lower left", facecolor="#152038", edgecolor="#2a3a58", labelcolor="#e8eefc", fontsize=8)
    if created:
        ax.figure.tight_layout()
    return ax


def plot_sequence_strip(
    seq_pixels: np.ndarray,
    frame_indices: Iterable[int],
    title: str,
    result: ClassificationResult,
    out_path: Path | None = None,
) -> plt.Figure:
    configure_korean_font()
    idx = list(frame_indices)
    fig, axes = plt.subplots(1, len(idx), figsize=(3.15 * len(idx), 5.8), facecolor="#0b1220")
    if len(idx) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0b1220")
    seq_norm, _ = normalize_keypoints(seq_pixels)
    accent = {
        ActionLabel.SCRATCH: "#3dd68c",
        ActionLabel.HELMET_OFF: "#ff5c5c",
        ActionLabel.NO_CONTACT: "#8aa0c4",
        ActionLabel.UNKNOWN_CONTACT: "#f5c518",
    }[result.label]

    for ax, fi in zip(axes, idx):
        ax.set_facecolor("#10182a")
        pose = seq_norm[fi]
        rg = compute_head_regions(pose)
        x, y, w, h = rg.as_xywh()
        ax.add_patch(Rectangle((x, y), w, h, fill=True, facecolor="#f5c518", alpha=0.10, edgecolor="#f5c518", lw=1.0, ls="--"))
        ax.add_patch(Circle(rg.center, rg.center_r, fill=False, edgecolor="#3dd68c", lw=1.0, ls=":"))
        ax.add_patch(Circle(rg.left_ear, rg.ear_r, fill=False, edgecolor="#ff8fab", lw=1.0, ls=":"))
        ax.add_patch(Circle(rg.right_ear, rg.ear_r, fill=False, edgecolor="#ff8fab", lw=1.0, ls=":"))
        for a, b in SKELETON_BONES:
            ax.plot([pose[a, 0], pose[b, 0]], [pose[a, 1], pose[b, 1]], color="#5ee0ff", lw=1.8)
        ax.scatter(pose[list(UPPER_JOINTS), 0], pose[list(UPPER_JOINTS), 1], c="#d7ecff", s=18, zorder=3)
        ax.scatter(pose[[L_WRIST, R_WRIST], 0], pose[[L_WRIST, R_WRIST], 1], c=accent, s=42, zorder=4)
        ax.set_title(f"t = {fi}", color="#e8eefc", fontsize=10)
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(1.05, -0.95)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(f"{title}\n판정: {LABEL_KO[result.label]}", color=accent, fontsize=13, y=0.98)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    return fig


def plot_feature_timeline(
    seq_pixels: np.ndarray,
    title: str,
    out_path: Path | None = None,
) -> plt.Figure:
    configure_korean_font()
    from helmet_action.pose.constants import L_WRIST, R_WRIST
    from helmet_action.pose.geometry import head_scale_norm

    seq_norm, _ = normalize_keypoints(seq_pixels)
    t = np.arange(seq_norm.shape[0])
    d_wrist = np.linalg.norm(seq_norm[:, L_WRIST] - seq_norm[:, R_WRIST], axis=1)
    h_scale = np.array([head_scale_norm(seq_norm[i]) for i in range(len(t))])
    regions = [compute_head_regions(seq_norm[i]) for i in range(len(t))]
    rw = seq_norm[:, R_WRIST]
    dist_center = np.array([np.linalg.norm(rw[i] - regions[i].center) for i in range(len(t))])

    fig, axes = plt.subplots(2, 1, figsize=(10.6, 6.3), sharex=True, facecolor="#0b1220")
    fig.patch.set_facecolor("#0b1220")
    for ax in axes:
        ax.set_facecolor("#10182a")
        ax.tick_params(colors="#8aa0c4")
        for spine in ax.spines.values():
            spine.set_color("#2a3a58")
        ax.grid(True, color="#1c2a44", linestyle=":", lw=0.8)

    axes[0].plot(t, dist_center, color="#ffd166", lw=2.0, label="오른손목 → 두상 중심")
    axes[0].axhline(CENTER_RADIUS, color="#3dd68c", ls="--", lw=1.2, label=f"중심원 r={CENTER_RADIUS}")
    axes[0].set_ylabel("거리 (어깨너비 단위)", color="#9db0d0")
    axes[0].legend(facecolor="#152038", edgecolor="#2a3a58", labelcolor="#e8eefc")
    axes[0].set_title(title, color="#f2f6ff")

    axes[1].plot(t, d_wrist, color="#ff8fab", lw=2.0, label="양손목 사이 거리")
    axes[1].plot(t, h_scale, color="#f5c518", lw=2.0, label="귀 간격 (머리 겉보기 크기)")
    axes[1].set_xlabel("프레임", color="#9db0d0")
    axes[1].set_ylabel("정규화 길이", color="#9db0d0")
    axes[1].legend(facecolor="#152038", edgecolor="#2a3a58", labelcolor="#e8eefc")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    return fig
