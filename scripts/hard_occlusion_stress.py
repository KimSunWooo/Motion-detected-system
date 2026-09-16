#!/usr/bin/env python3
"""SOFT vs HARD occlusion stress. Hard gaps are not interpolated.

PYTHONPATH=src python scripts/hard_occlusion_stress.py --model models/action_classifier.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.evaluation.false_safe import (  # noqa: E402
    false_safe_rate,
    insufficient_rate,
    unknown_rate,
)
from helmet_action.models.hybrid import HybridActionClassifier  # noqa: E402
from helmet_action.models.labels import ActionClass  # noqa: E402
from helmet_action.models.temporal_classifier import SklearnActionClassifier  # noqa: E402
from helmet_action.models.training import REMOVE, binary_remove_metrics  # noqa: E402
from helmet_action.pose.confidence import interpolate_short_gaps, interpolation_max_gap, mask_invalid  # noqa: E402
from helmet_action.pose.constants import L_EAR, L_ELBOW, L_WRIST, R_EAR, R_ELBOW, R_WRIST  # noqa: E402
from helmet_action.synthetic.augmentation import apply_consecutive_occlusion  # noqa: E402
from helmet_action.synthetic.generator import generate_one  # noqa: E402

JOINT_GROUPS = {
    "LEFT_WRIST": [L_WRIST],
    "RIGHT_WRIST": [R_WRIST],
    "BOTH_WRISTS": [L_WRIST, R_WRIST],
    "LEFT_EAR": [L_EAR],
    "RIGHT_EAR": [R_EAR],
    "BOTH_EARS": [L_EAR, R_EAR],
    "LEFT_ELBOW": [L_ELBOW],
    "RIGHT_ELBOW": [R_ELBOW],
    "HEAD_KEYPOINTS": [0, 1, 2, L_EAR, R_EAR],
}

HARD_GAPS = (5, 10, 15, 20, 30)
NEG_SCENARIOS = ("HELMET_ADJUST", "TWO_HAND_HEAD_TOUCH", "IDLE")


def _pairs(n: int, seed0: int):
    pos, neg = [], []
    for i in range(n):
        seq, conf, meta = generate_one(seed=seed0 + i, scenario="HELMET_REMOVE", split="test", apply_noise=False)
        pos.append((seq, conf, meta.label))
    for i, scen in enumerate(NEG_SCENARIOS * ((n + 2) // 3)):
        if i >= n:
            break
        seq, conf, meta = generate_one(seed=seed0 + 8000 + i, scenario=scen, split="test", apply_noise=False)
        neg.append((seq, conf, meta.label))
    return pos, neg


def _eval(hybrid: HybridActionClassifier, items: list[tuple[np.ndarray, np.ndarray, str]]) -> dict:
    y_true, y_pred, scores = [], [], []
    for k, c, lab in items:
        dec = hybrid.predict(k, c)
        y_true.append(lab)
        y_pred.append(dec.action.value)
        scores.append(float(dec.ml_proba.get(REMOVE, 0.0)) if dec.ml_proba else 0.0)
    yt, yp = np.array(y_true), np.array(y_pred)
    b = binary_remove_metrics(yt, yp, np.array(scores))
    return {
        "remove_recall": float(b["recall"]),
        "fnr": float(b["fnr"]),
        "fpr": float(b["fpr"]),
        "unknown_rate": unknown_rate(yp),
        "insufficient_pose_rate": insufficient_rate(yp),
        "false_safe_rate": false_safe_rate(yt, yp),
        "pred_counts": {str(k): int(v) for k, v in zip(*np.unique(yp, return_counts=True))},
        "n": int(len(yt)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "action_classifier.joblib")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "evaluation" / "hard_occlusion.json")
    args = parser.parse_args(argv)

    clf = SklearnActionClassifier.load(args.model)
    hybrid = HybridActionClassifier(ml=clf)
    max_gap = interpolation_max_gap()
    pos, neg = _pairs(args.n, 70_000)
    rows = []
    policy = []

    # SOFT: 1-3 frame gaps (interpolated)
    for gap in (1, 2, 3):
        for name, joints in (("BOTH_WRISTS", JOINT_GROUPS["BOTH_WRISTS"]), ("LEFT_WRIST", JOINT_GROUPS["LEFT_WRIST"])):
            items = []
            rng = np.random.default_rng(gap * 13 + len(name))
            for seq, conf, lab in pos + neg:
                s2, c2 = apply_consecutive_occlusion(seq, conf, joints, gap, start=seq.shape[0] // 3, rng=rng)
                items.append((s2, c2, lab))
            flat = _eval(hybrid, items)
            flat.update({"kind": "SOFT_OCCLUSION", "gap_frames": gap, "joint_group": name})
            rows.append(flat)
            print(f"SOFT {name} gap={gap} R={flat['remove_recall']:.3f} UNK={flat['unknown_rate']:.3f}", flush=True)

    by_gap: dict[int, list[dict]] = {g: [] for g in HARD_GAPS}
    for gap in HARD_GAPS:
        for name, joints in JOINT_GROUPS.items():
            items = []
            rng = np.random.default_rng(gap * 97 + sum(map(ord, name)))
            for seq, conf, lab in pos + neg:
                start = max(4, seq.shape[0] // 4)
                s2, c2 = apply_consecutive_occlusion(seq, conf, joints, gap, start=start, rng=rng)
                items.append((s2, c2, lab))
            flat = _eval(hybrid, items)
            flat.update({"kind": "HARD_OCCLUSION", "gap_frames": gap, "joint_group": name})
            rows.append(flat)
            by_gap[gap].append(flat)
            print(
                f"HARD {name} gap={gap} R={flat['remove_recall']:.3f} FNR={flat['fnr']:.3f} "
                f"UNK={flat['unknown_rate']:.3f} FS={flat['false_safe_rate']:.3f}",
                flush=True,
            )

    # Policy checks
    seq, conf, _ = pos[0]
    both20, c20 = apply_consecutive_occlusion(seq, conf, [L_WRIST, R_WRIST], 20, start=8)
    dec20 = hybrid.predict(both20, c20)
    policy.append(
        {
            "case": "BOTH_WRISTS_20",
            "action": dec20.action.value,
            "decision_status": dec20.decision_status,
            "ok": dec20.action in (ActionClass.UNKNOWN, ActionClass.INSUFFICIENT_POSE),
        }
    )
    one2, c2 = apply_consecutive_occlusion(seq, conf, [L_WRIST], 2, start=8)
    masked = mask_invalid(one2, c2)
    repaired = interpolate_short_gaps(masked)
    filled = (~np.isfinite(masked[:, L_WRIST]).all(axis=-1)) & np.isfinite(repaired[:, L_WRIST]).all(axis=-1)
    policy.append(
        {
            "case": "LEFT_WRIST_2",
            "interpolated": bool(filled.any()),
            "max_gap": max_gap,
            "ok": bool(filled.any()),
        }
    )
    ears, ce = apply_consecutive_occlusion(seq, conf, [L_EAR, R_EAR], 20, start=8)
    de = hybrid.predict(ears, ce)
    policy.append(
        {
            "case": "BOTH_EARS_20",
            "action": de.action.value,
            "phase_confidence": de.phase_confidence,
            "decision_status": de.decision_status,
        }
    )

    gap_summary = {}
    for gap, group in by_gap.items():
        rec = [r["remove_recall"] for r in group]
        fs = [r["false_safe_rate"] for r in group]
        unk = [r["unknown_rate"] for r in group]
        gap_summary[str(gap)] = {
            "remove_recall_mean": float(np.mean(rec)),
            "fnr_mean": float(np.mean([r["fnr"] for r in group])),
            "false_safe_rate_mean": float(np.mean(fs)),
            "unknown_rate_mean": float(np.mean(unk)),
            "insufficient_pose_rate_mean": float(np.mean([r["insufficient_pose_rate"] for r in group])),
        }

    hard_rows = [r for r in rows if r.get("kind") == "HARD_OCCLUSION"]
    worst = min(hard_rows, key=lambda r: r["remove_recall"]) if hard_rows else {}
    payload = {
        "max_gap_frames": max_gap,
        "n_pos": args.n,
        "rows": rows,
        "by_gap": gap_summary,
        "worst_keypoint": worst.get("joint_group"),
        "worst_gap": worst.get("gap_frames"),
        "worst_recall": worst.get("remove_recall"),
        "policy_checks": policy,
        "disclaimer": "Synthetic hard occlusion. Not CCTV performance.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
