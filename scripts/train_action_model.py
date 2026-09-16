#!/usr/bin/env python3
"""Train the lightweight temporal action classifier.

python scripts/train_action_model.py
python scripts/train_action_model.py --data data/synthetic --output models/action_classifier.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.config import load_config
from helmet_action.models.temporal_classifier import train_sklearn_classifier
from helmet_action.models.training import vectorize_dataset


def _load_split(path: Path, split: str):
    npz = np.load(path / f"{split}.npz", allow_pickle=True)
    return npz["keypoints"], npz["confidences"], npz["labels"], npz["lengths"]


def _trim(kpts, conf, lengths):
    out_k, out_c = [], []
    for i, t in enumerate(lengths):
        t = int(t)
        out_k.append(kpts[i, :t])
        out_c.append(conf[i, :t])
    return out_k, out_c


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--out", "--output", dest="out", type=Path, default=ROOT / "models" / "action_classifier.joblib")
    parser.add_argument("--generate", type=int, default=0, help="generate this many samples first")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--estimator", default=None)
    parser.add_argument("--calibrate", action="store_true", help="fit sigmoid calibration on the validation split")
    parser.add_argument("--feature-version", default=None, help="v1 or v2")
    args = parser.parse_args(argv)

    if not (args.data / "train.npz").exists():
        sys.path.insert(0, str(ROOT / "scripts"))
        import generate_dataset

        n = args.generate or (500 if args.quick else 2000)
        generate_dataset.main(["--samples", str(n), "--output", str(args.data)] + (["--quick"] if args.quick else []))

    k, c, y, lengths = _load_split(args.data, "train")
    kv, cv, yv, lv = _load_split(args.data, "validation")
    k_list, c_list = _trim(k, c, lengths)
    print(f"vectorizing {len(k_list)} train sequences…")
    version = args.feature_version or str(load_config().get("ml.feature_version", "v1"))
    X = vectorize_dataset(k_list, c_list, feature_version=version)
    Xv = yc = None
    kvl, cvl = _trim(kv, cv, lv)
    if args.calibrate:
        print(f"vectorizing {len(kvl)} validation sequences for calibration…")
        Xv = vectorize_dataset(kvl, cvl, feature_version=version)
        yc = yv
    clf = train_sklearn_classifier(
        X, y, estimator_name=args.estimator, X_calibrate=Xv, y_calibrate=yc, feature_version=version
    )
    dest = clf.save(args.out)
    print(f"saved {dest}")
    print("classes:", clf.classes_)
    print("feature_count:", len(clf.feature_names))

    pred = np.array([clf.predict(kvl[i], cvl[i]) for i in range(len(kvl))])
    acc = float((pred == yv).mean()) if len(yv) else 0.0
    print(f"validation accuracy (synthetic, not field): {acc:.3f}")
    meta = {
        "model": str(dest),
        "train_n": int(len(y)),
        "val_acc_synthetic": acc,
        "classes": clf.classes_,
        "feature_count": len(clf.feature_names),
        "calibrated": bool(args.calibrate),
        "disclaimer": "Synthetic validation accuracy is not field CCTV performance.",
    }
    dest.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
