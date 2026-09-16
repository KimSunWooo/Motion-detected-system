"""Public names matching the original pose_action_classifier.py module."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from helmet_action.config import load_config
from helmet_action.models.labels import (
    LABEL_KO,
    ActionLabel,
    ClassificationResult,
    FeatureReport,
)
from helmet_action.models.rule_based import (
    RuleBasedActionClassifier,
    classify_pose_sequence,
    extract_features,
)
from helmet_action.pose.constants import (
    CENTER_RADIUS,
    EAR_RADIUS,
    FACE_IDX,
    HEAD_CENTER_OFFSET_Y,
    HEAD_HALF_HEIGHT,
    HEAD_HALF_WIDTH,
    L_ANKLE,
    L_EAR,
    L_ELBOW,
    L_EYE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_ANKLE,
    R_EAR,
    R_ELBOW,
    R_EYE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    SKELETON_BONES,
    UPPER_BONES,
    UPPER_JOINTS,
)
from helmet_action.pose.geometry import (
    HeadRegions,
    compute_head_regions,
    compute_neck,
    compute_shoulder_width,
    head_center_norm,
    head_scale_norm,
)
from helmet_action.pose.normalizer import as_sequence as _as_sequence
from helmet_action.pose.normalizer import normalize_keypoints
from helmet_action.pose.types import NormalizeInfo
from helmet_action.synthetic.camera import HighAngleCamera
from helmet_action.synthetic.scenarios import (
    generate_helmet_off_sequence,
    generate_idle_sequence,
    generate_scratch_sequence,
)
from helmet_action.synthetic.skeleton import canonical_pose_3d
from helmet_action.visualization.demo import build_demo_scenarios, render_all_figures, sequence_payload
from helmet_action.visualization.plots import (
    configure_korean_font as _configure_korean_font,
    plot_feature_timeline,
    plot_sequence_strip,
    plot_static_skeleton,
    project_canonical,
)

_cfg = load_config()
MIN_CENTER_FRAMES = int(_cfg.get("rules.min_center_frames", 8))
MIN_EAR_FRAMES = int(_cfg.get("rules.min_ear_frames", 8))
PAUSE_FRAMES = int(_cfg.get("rules.pause_frames", 8))
TAU_SCRATCH_RADIUS = float(_cfg.get("rules.scratch.radius", 0.11))
TAU_SCRATCH_STD = float(_cfg.get("rules.scratch.std", 0.028))
MIN_OSCILLATIONS = int(_cfg.get("rules.scratch.min_oscillations", 4))
TAU_PAUSE_STD = float(_cfg.get("rules.helmet_remove.pause_std", 0.016))
TAU_DX_SPREAD = float(_cfg.get("rules.helmet_remove.dx_spread", 0.08))
TAU_CO_RISE_Y = float(_cfg.get("rules.helmet_remove.co_rise_y", 0.05))
TAU_RADIAL = float(_cfg.get("rules.helmet_remove.radial", 0.08))
TAU_SCALE_UP = float(_cfg.get("rules.helmet_remove.scale_up", 0.07))


def print_classification(name: str, result: ClassificationResult) -> None:
    print("=" * 72)
    print(f"[시퀀스] {name}")
    print(f"  판정     : {LABEL_KO[result.label]}  ({result.label.value})")
    print(f"  신뢰도   : {result.confidence:.2f}")
    print("  판단 근거:")
    for line in result.explanation:
        print(f"    - {line}")
    print("=" * 72)


def run_self_test(verbose: bool = True) -> bool:
    print(
        """
[하이브리드 알고리즘 개요]
  카메라   : 공장 상단 CCTV, 대각선 하향
  정규화   : p̂ = (p − neck) / ||L_shoulder − R_shoulder||
             하체 길이는 투시 단축 때문에 사용하지 않음
  머리 bbox: center=(nose_x, -0.30),  half=(0.52, 0.42)  [어깨너비 단위]

  긁기     : 한쪽 손목 ∈ 머리 bbox
             국소 반경 ≤ {:.2f}  이면서  σ_xy ≥ {:.3f},  영점교차 ≥ {}

  안전모   : 양손목 ∈ 귀 모서리 → 일시정지 후 하이브리드 OR
             Δ|x_R−x_L| > {:.2f}   (가로 벌어짐)
             −Δȳ_wrist  > {:.2f}   (함께 위쪽, y 감소)
             Δr 또는 Δd_ear > {:.2f} / {:.2f}  (방사형·겉보기 팽창)
""".format(
            TAU_SCRATCH_RADIUS,
            TAU_SCRATCH_STD,
            MIN_OSCILLATIONS,
            TAU_DX_SPREAD,
            TAU_CO_RISE_Y,
            TAU_RADIAL,
            TAU_SCALE_UP,
        )
    )
    ok = True
    for sc in build_demo_scenarios():
        result = ClassificationResult(
            label=ActionLabel(sc["result"]["label"]),
            confidence=sc["result"]["confidence"],
            features=FeatureReport(**sc["result"]["features"]),
            explanation=sc["result"]["explanation"],
            frame_labels=sc["result"]["frame_labels"],
        )
        if verbose:
            print_classification(sc["name"], result)
            print(f"  기대 라벨 : {sc['expected_ko']}  →  일치={sc['match']}\n")
        if not sc["match"]:
            ok = False
            print(f"[FAIL] {sc['name']}: expected {sc['expected']} got {result.label.value}")
    print("[SELF-TEST]", "PASS" if ok else "FAIL")
    return ok


def maybe_show(fig: plt.Figure | None = None) -> None:
    if os.environ.get("DISPLAY") and matplotlib.get_backend().lower() != "agg":
        plt.show()
    elif fig is not None:
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="하이 앵글 CCTV 안전모/긁기 분류")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)

    passed = run_self_test(verbose=True)
    out_dir = Path(args.out)
    if not args.skip_plots:
        paths = render_all_figures(out_dir)
        print("\n[시각화 저장]")
        for k, v in paths.items():
            print(f"  {k}: {v}")
        fig, ax = plt.subplots(figsize=(7.4, 8.4), facecolor="#0b1220")
        fig.patch.set_facecolor("#0b1220")
        plot_static_skeleton(ax=ax)
        fig.tight_layout()
        maybe_show(fig)

    if args.serve:
        from dashboard_server import serve

        serve(host=args.host, port=args.port, out_dir=out_dir)
    return 0 if passed else 1
