#!/usr/bin/env python3
"""Evaluate the action classifier on a synthetic split.

Results are synthetic-only. They are not field accuracy.

python scripts/evaluate_action_model.py
python scripts/evaluate_action_model.py --data /tmp/helmet_synthetic_validation --model /tmp/action_classifier.joblib
python scripts/evaluate_action_model.py --data /tmp/helmet_synthetic_validation/test.npz --model /tmp/action_classifier.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.models.temporal_classifier import SklearnActionClassifier
from helmet_action.models.training import (
    REMOVE,
    calibration_report,
    evaluate_predictions,
    remove_probability_by_class,
)


def _resolve_npz(data: Path, split: str) -> Path:
    if data.is_file() and data.suffix == ".npz":
        return data
    return data / f"{split}.npz"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "action_classifier.joblib")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", "--output", dest="out", type=Path, default=ROOT / "outputs" / "eval_report.json")
    parser.add_argument("--calibration-split", default="validation")
    args = parser.parse_args(argv)

    if not args.model.exists():
        print(f"missing model {args.model}. Train first: python scripts/train_action_model.py")
        return 1
    npz_path = _resolve_npz(args.data, args.split)
    if not npz_path.exists():
        print(f"missing {npz_path}. Generate first: python scripts/generate_dataset.py")
        return 1

    clf = SklearnActionClassifier.load(args.model)
    npz = np.load(npz_path, allow_pickle=True)
    kpts, conf, labels, lengths = npz["keypoints"], npz["confidences"], npz["labels"], npz["lengths"]
    y_pred = []
    y_true = []
    scores = []
    for i in range(len(labels)):
        t = int(lengths[i])
        proba = clf.predict_proba(kpts[i, :t], conf[i, :t])
        pred = max(proba, key=proba.get)
        y_pred.append(pred)
        y_true.append(str(labels[i]))
        scores.append(float(proba.get(REMOVE, 0.0)))
    y_true_a = np.array(y_true)
    y_pred_a = np.array(y_pred)
    scores_a = np.array(scores, dtype=np.float64)
    report = evaluate_predictions(y_true_a, y_pred_a, y_score_remove=scores_a)
    report["remove_probability_by_class"] = remove_probability_by_class(y_true_a, scores_a)
    report["feature_count"] = len(clf.feature_names)
    report["classes"] = clf.classes_
    report["split"] = args.split
    report["n"] = len(y_true)
    report["disclaimer"] = (
        "Synthetic OOD split only. This is NOT construction-site accuracy. "
        "Pose-only models cannot observe helmet presence at rest. "
        "predict_proba is not a field-calibrated confidence."
    )

    cal_path = _resolve_npz(args.data if args.data.is_dir() else args.data.parent, args.calibration_split)
    if cal_path.exists():
        cnpz = np.load(cal_path, allow_pickle=True)
        cal_scores, cal_bin = [], []
        for i in range(len(cnpz["labels"])):
            t = int(cnpz["lengths"][i])
            proba = clf.predict_proba(cnpz["keypoints"][i, :t], cnpz["confidences"][i, :t])
            cal_scores.append(float(proba.get(REMOVE, 0.0)))
            cal_bin.append(int(str(cnpz["labels"][i]) == REMOVE))
        report["calibration_validation"] = calibration_report(np.array(cal_bin), np.array(cal_scores))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report["report"])
    print(f"accuracy={report['accuracy']:.3f}  macro_p={report['macro_precision']:.3f}  "
          f"macro_r={report['macro_recall']:.3f}  macro_f1={report['macro_f1']:.3f}")
    b = report["binary_helmet_remove"]
    print(
        "HELMET_REMOVE binary "
        f"TP={b['tp']} FP={b['fp']} TN={b['tn']} FN={b['fn']} "
        f"P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f} "
        f"FNR={b['fnr']:.3f} FPR={b['fpr']:.3f} PR-AUC={b.get('pr_auc')}"
    )
    print("hard negatives → REMOVE", report["hard_negative_to_remove"])
    print("P(remove) by class")
    for cls, stats in report["remove_probability_by_class"].items():
        print(f"  {cls}: {stats}")
    if "calibration_validation" in report:
        print("validation calibration", {k: report["calibration_validation"][k] for k in ("brier", "note")})
    print(f"wrote {args.out}")
    print("WARNING: Synthetic evaluation results are NOT real-world CCTV performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
