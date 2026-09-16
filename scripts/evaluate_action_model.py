#!/usr/bin/env python3
"""Evaluate the action classifier on the synthetic OOD test split.

Results are synthetic-only. They are not field accuracy.

python scripts/evaluate_action_model.py
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
from helmet_action.models.training import evaluate_predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "action_classifier.joblib")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval_report.json")
    args = parser.parse_args(argv)

    if not args.model.exists():
        print(f"missing model {args.model}. Train first: python scripts/train_action_model.py")
        return 1
    npz_path = args.data / f"{args.split}.npz"
    if not npz_path.exists():
        print(f"missing {npz_path}. Generate first: python scripts/generate_dataset.py")
        return 1

    clf = SklearnActionClassifier.load(args.model)
    npz = np.load(npz_path, allow_pickle=True)
    kpts, conf, labels, lengths = npz["keypoints"], npz["confidences"], npz["labels"], npz["lengths"]
    y_pred = []
    y_true = []
    for i in range(len(labels)):
        t = int(lengths[i])
        y_pred.append(clf.predict(kpts[i, :t], conf[i, :t]))
        y_true.append(str(labels[i]))
    report = evaluate_predictions(np.array(y_true), np.array(y_pred))
    report["split"] = args.split
    report["n"] = len(y_true)
    report["disclaimer"] = (
        "Synthetic OOD split only. This is NOT construction-site accuracy. "
        "Pose-only models cannot observe helmet presence at rest."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report["report"])
    print(f"accuracy={report['accuracy']:.3f}  macro_f1={report['macro_f1']:.3f}")
    print(f"HELMET_REMOVE false-negative rate={report['helmet_remove_fnr']}")
    print(f"FN (remove→other)={report['n_false_negatives']}  FP (other→remove)={report['n_false_positives']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
