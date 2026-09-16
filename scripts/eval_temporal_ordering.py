#!/usr/bin/env python3
"""V1 vs V2 P(HELMET_REMOVE) on NORMAL/REVERSED/SHUFFLED/PARTIAL sequences.

PYTHONPATH=src python scripts/eval_temporal_ordering.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from helmet_action.evaluation.temporal_order import temporal_variants  # noqa: E402
from helmet_action.features.v1_names import FEATURE_DIM_V1  # noqa: E402
from helmet_action.features.v2 import FEATURE_DIM_V2  # noqa: E402
from helmet_action.models.temporal_classifier import SklearnActionClassifier, train_sklearn_classifier  # noqa: E402
from helmet_action.models.training import REMOVE, vectorize_dataset  # noqa: E402
from helmet_action.synthetic.generator import generate_one  # noqa: E402
import generate_dataset  # noqa: E402


def _load_or_train(data_dir: Path, version: str, out_model: Path) -> SklearnActionClassifier:
    if out_model.exists():
        clf = SklearnActionClassifier.load(out_model)
        if clf.feature_version == version:
            return clf
    npz = np.load(data_dir / "train.npz", allow_pickle=True)
    k, c = [], []
    for i, t in enumerate(npz["lengths"]):
        k.append(npz["keypoints"][i, : int(t)])
        c.append(npz["confidences"][i, : int(t)])
    X = vectorize_dataset(k, c, feature_version=version)
    clf = train_sklearn_classifier(X, npz["labels"], feature_version=version)
    clf.save(out_model)
    return clf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs" / "evaluation" / "runs" / "data_seed_42")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "evaluation" / "temporal_ordering.json")
    args = parser.parse_args(argv)

    if not (args.data / "train.npz").exists():
        generate_dataset.main(["--samples", "500", "--output", str(args.data), "--seed", "42"])

    v1 = _load_or_train(args.data, "v1", args.data.parent / "model_seed_42_v1.joblib")
    v2 = _load_or_train(args.data, "v2", args.data.parent / "model_seed_42_v2.joblib")

    variants = ("NORMAL", "REVERSED", "SHUFFLED", "PARTIAL_START", "PARTIAL_END")
    bucket = {ver: {v: [] for v in variants} for ver in ("v1", "v2")}
    still_remove = {ver: {v: 0 for v in variants} for ver in ("v1", "v2")}

    for i in range(args.n):
        seq, conf, _ = generate_one(seed=80_000 + i * 11, scenario="HELMET_REMOVE", split="test", apply_noise=False)
        pack = temporal_variants(seq, conf, seed=i)
        for name, (k, c) in pack.items():
            for ver, clf in (("v1", v1), ("v2", v2)):
                proba = clf.predict_proba(k, c)
                p = float(proba.get(REMOVE, 0.0))
                bucket[ver][name].append(p)
                if max(proba, key=proba.get) == REMOVE:
                    still_remove[ver][name] += 1

    def _summ(vals: list[float]) -> dict:
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0)}

    payload = {
        "n": args.n,
        "feature_dim_v1": FEATURE_DIM_V1,
        "feature_dim_v2": FEATURE_DIM_V2,
        "p_helmet_remove": {ver: {name: _summ(bucket[ver][name]) for name in variants} for ver in ("v1", "v2")},
        "fraction_predicted_remove": {
            ver: {name: still_remove[ver][name] / args.n for name in variants} for ver in ("v1", "v2")
        },
        "ordering_gap_v1": float(
            _summ(bucket["v1"]["NORMAL"])["mean"] - _summ(bucket["v1"]["REVERSED"])["mean"]
        ),
        "ordering_gap_v2": float(
            _summ(bucket["v2"]["NORMAL"])["mean"] - _summ(bucket["v2"]["REVERSED"])["mean"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "p_helmet_remove"}, indent=2))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
