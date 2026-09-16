from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helmet_action.models.hybrid import HybridActionClassifier
from helmet_action.models.labels import ACTION_KO, ActionLabel, ClassificationResult, FeatureReport, LABEL_KO
from helmet_action.models.rule_based import classify_pose_sequence
from helmet_action.pose.constants import SKELETON_BONES
from helmet_action.pose.geometry import compute_head_regions
from helmet_action.pose.normalizer import normalize_keypoints
from helmet_action.synthetic.scenarios import (
    generate_helmet_off_sequence,
    generate_idle_sequence,
    generate_scratch_sequence,
)
from helmet_action.visualization.plots import plot_feature_timeline, plot_sequence_strip, plot_static_skeleton


def sequence_payload(name: str, seq_pixels: np.ndarray, expected: ActionLabel, scenario: str | None = None) -> dict:
    result = classify_pose_sequence(seq_pixels)
    hybrid = HybridActionClassifier()
    hybrid.gate._buf.clear()
    hybrid.gate.alert = False
    hdec = hybrid.predict(seq_pixels)
    seq_norm, infos = normalize_keypoints(seq_pixels)
    frames = []
    for i, pose in enumerate(seq_norm):
        rg = compute_head_regions(pose)
        frames.append(
            {
                "keypoints": pose.tolist(),
                "bbox": [rg.x_min, rg.y_min, rg.x_max, rg.y_max],
                "center": rg.center.tolist(),
                "center_r": rg.center_r,
                "left_ear": rg.left_ear.tolist(),
                "right_ear": rg.right_ear.tolist(),
                "ear_r": rg.ear_r,
                "scale_px": infos[i].scale,
                "origin_px": infos[i].origin.tolist(),
                "state": result.frame_labels[i] if i < len(result.frame_labels) else "idle",
            }
        )
    return {
        "name": name,
        "synthetic_scenario": scenario or expected.value,
        "expected": expected.value,
        "expected_label": expected.value,
        "expected_ko": LABEL_KO[expected],
        "result": result.to_dict(),
        "rule_prediction": result.label.value,
        "ml_prediction": hdec.ml_label,
        "ml_proba": hdec.ml_proba,
        "final_prediction": hdec.action.value,
        "confidence": hdec.confidence,
        "feature_values": hdec.features,
        "state_machine_phase": hdec.phase.value,
        "helmet_state": hdec.helmet_state.value,
        "risk": hdec.risk,
        "match": result.label == expected,
        "n_frames": int(seq_pixels.shape[0]),
        "bones": SKELETON_BONES,
        "frames": frames,
        "hybrid": hdec.to_dict(),
    }


def build_demo_scenarios() -> list[dict]:
    scratch = generate_scratch_sequence()
    helmet = generate_helmet_off_sequence()
    scratch_far = generate_scratch_sequence(pixel_scale=0.55, pixel_shift=(80.0, 40.0), seed=7)
    helmet_near = generate_helmet_off_sequence(pixel_scale=1.45, pixel_shift=(-60.0, 30.0), seed=11)
    return [
        sequence_payload("하이 앵글 · 단순 머리 긁기", scratch, ActionLabel.SCRATCH, "HEAD_SCRATCH"),
        sequence_payload("하이 앵글 · 안전모 벗기", helmet, ActionLabel.HELMET_OFF, "HELMET_REMOVE"),
        sequence_payload("원거리 축소 카메라 · 긁기", scratch_far, ActionLabel.SCRATCH, "HEAD_SCRATCH"),
        sequence_payload("근거리 확대 카메라 · 벗기", helmet_near, ActionLabel.HELMET_OFF, "HELMET_REMOVE"),
        sequence_payload("팔만 흔들기 · 머리 비접촉", generate_idle_sequence(), ActionLabel.NO_CONTACT, "IDLE"),
    ]


def render_all_figures(out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7.4, 8.4), facecolor="#0b1220")
    fig.patch.set_facecolor("#0b1220")
    plot_static_skeleton(ax=ax)
    fig.tight_layout()
    p = out_dir / "static_skeleton.png"
    fig.savefig(p, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths["static_skeleton"] = str(p)

    scratch = generate_scratch_sequence()
    helmet = generate_helmet_off_sequence()
    r_s = classify_pose_sequence(scratch)
    r_h = classify_pose_sequence(helmet)

    p = out_dir / "scratch_strip.png"
    fig = plot_sequence_strip(scratch, [0, 12, 24, 36, 50], "하이 앵글 · 머리 긁기", r_s, p)
    plt.close(fig)
    paths["scratch_strip"] = str(p)

    p = out_dir / "helmet_strip.png"
    fig = plot_sequence_strip(helmet, [0, 11, 20, 36, 52], "하이 앵글 · 안전모 벗기", r_h, p)
    plt.close(fig)
    paths["helmet_strip"] = str(p)

    p = out_dir / "scratch_timeline.png"
    fig = plot_feature_timeline(scratch, "긁기: 한쪽 손목이 머리 bbox에서 고주파 진동", p)
    plt.close(fig)
    paths["scratch_timeline"] = str(p)

    p = out_dir / "helmet_timeline.png"
    fig = plot_feature_timeline(helmet, "벗기: X축 벌어짐 + Y 동반 상승 + 방사형 팽창", p)
    plt.close(fig)
    paths["helmet_timeline"] = str(p)
    return paths
