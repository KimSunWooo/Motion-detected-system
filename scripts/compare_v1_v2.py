#!/usr/bin/env python3
"""Fair V1 vs V2 A/B on identical splits/seeds.

Reuses existing 10000-sample datasets when present so both models see the same data.

PYTHONPATH=src python scripts/compare_v1_v2.py --samples 10000 --seeds 42 101 202 303 404
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_dataset  # noqa: E402
from helmet_action.evaluation.metrics import paired_delta_report, summarize_numeric  # noqa: E402
from helmet_action.features.v1_names import FEATURE_DIM_V1  # noqa: E402
from helmet_action.features.v2 import FEATURE_DIM_V2  # noqa: E402
from helmet_action.models.temporal_classifier import train_sklearn_classifier  # noqa: E402
from helmet_action.models.training import REMOVE, evaluate_predictions, vectorize_dataset  # noqa: E402

SEEDS = [42, 101, 202, 303, 404]
METRIC_KEYS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "helmet_remove_precision",
    "helmet_remove_recall",
    "helmet_remove_f1",
    "helmet_remove_fnr",
    "helmet_remove_fpr",
    "helmet_remove_pr_auc",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _trim(kpts, conf, lengths):
    out_k, out_c = [], []
    for i, t in enumerate(lengths):
        t = int(t)
        out_k.append(kpts[i, :t])
        out_c.append(conf[i, :t])
    return out_k, out_c


def _ensure_dataset(data_dir: Path, samples: int, seed: int) -> Path:
    data_dir = Path(data_dir)
    if (data_dir / "train.npz").exists() and (data_dir / "test.npz").exists():
        print(f"  reuse dataset {data_dir}", flush=True)
        return data_dir
    print(f"  generate dataset seed={seed} n={samples}", flush=True)
    generate_dataset.main(["--samples", str(samples), "--output", str(data_dir), "--seed", str(seed)])
    return data_dir


def _predict(clf, npz):
    labels = npz["labels"]
    lengths = npz["lengths"]
    y_pred, y_true, scores = [], [], []
    for i in range(len(labels)):
        t = int(lengths[i])
        proba = clf.predict_proba(npz["keypoints"][i, :t], npz["confidences"][i, :t])
        y_pred.append(max(proba, key=proba.get))
        y_true.append(str(labels[i]))
        scores.append(float(proba.get(REMOVE, 0.0)))
    return np.array(y_true), np.array(y_pred), np.array(scores, dtype=np.float64)


def _flat(report, y_true, y_pred) -> dict:
    b = report["binary_helmet_remove"]
    return {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro_precision"]),
        "macro_recall": float(report["macro_recall"]),
        "macro_f1": float(report["macro_f1"]),
        "helmet_remove_precision": float(b["precision"]),
        "helmet_remove_recall": float(b["recall"]),
        "helmet_remove_f1": float(b["f1"]),
        "helmet_remove_fnr": float(b["fnr"]),
        "helmet_remove_fpr": float(b["fpr"]),
        "helmet_remove_pr_auc": float(b.get("pr_auc") or 0.0),
        "helmet_remove_tp": int(b.get("tp", 0)),
        "helmet_remove_fp": int(b.get("fp", 0)),
        "helmet_remove_fn": int(b.get("fn", 0)),
    }


def train_eval_version(npz_tr, npz_te, version: str, seed: int, model_path: Path) -> dict:
    ktr, ctr = _trim(npz_tr["keypoints"], npz_tr["confidences"], npz_tr["lengths"])
    print(f"    vectorize {version} train n={len(ktr)}", flush=True)
    Xtr = vectorize_dataset(ktr, ctr, feature_version=version)
    clf = train_sklearn_classifier(Xtr, npz_tr["labels"], feature_version=version)
    clf.save(model_path)
    yt, yp, sc = _predict(clf, npz_te)
    report = evaluate_predictions(yt, yp, y_score_remove=sc)
    flat = _flat(report, yt, yp)
    flat.update(
        {
            "seed": seed,
            "feature_version": version,
            "feature_dimension": int(Xtr.shape[1]),
            "n_train": int(len(npz_tr["labels"])),
            "n_test": int(len(yt)),
            "model_path": str(model_path),
        }
    )
    return flat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--data-root", type=Path, default=ROOT / "outputs" / "evaluation" / "runs")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "evaluation")
    args = parser.parse_args(argv)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    per_seed = []
    v1_rows, v2_rows = [], []

    for seed in args.seeds:
        print(f"\n=== seed {seed} identical dataset, V1 then V2 ===", flush=True)
        data = _ensure_dataset(args.data_root / f"data_seed_{seed}", args.samples, seed)
        npz_tr = np.load(data / "train.npz", allow_pickle=True)
        npz_te = np.load(data / "test.npz", allow_pickle=True)
        v1 = train_eval_version(npz_tr, npz_te, "v1", seed, args.data_root / f"model_seed_{seed}_v1.joblib")
        v2 = train_eval_version(npz_tr, npz_te, "v2", seed, args.data_root / f"model_seed_{seed}_v2.joblib")
        print(
            f"  V1 F1={v1['macro_f1']:.3f} R={v1['helmet_remove_recall']:.3f} FNR={v1['helmet_remove_fnr']:.3f}",
            flush=True,
        )
        print(
            f"  V2 F1={v2['macro_f1']:.3f} R={v2['helmet_remove_recall']:.3f} FNR={v2['helmet_remove_fnr']:.3f}",
            flush=True,
        )
        v1_rows.append(v1)
        v2_rows.append(v2)
        per_seed.append(
            {
                "seed": seed,
                "n_train": v1["n_train"],
                "n_test": v1["n_test"],
                "data_path": str(data),
                "v1": v1,
                "v2": v2,
                "delta_macro_f1": v2["macro_f1"] - v1["macro_f1"],
                "delta_remove_recall": v2["helmet_remove_recall"] - v1["helmet_remove_recall"],
                "delta_remove_fnr": v2["helmet_remove_fnr"] - v1["helmet_remove_fnr"],
            }
        )

    def _agg(rows: list[dict]) -> dict:
        return {k: summarize_numeric([float(r[k]) for r in rows]) for k in METRIC_KEYS}

    summary = {
        "git_commit": _git_commit(),
        "samples": args.samples,
        "seeds": list(args.seeds),
        "identical_splits": True,
        "classifier": "HistGradientBoostingClassifier",
        "feature_dim_v1": FEATURE_DIM_V1,
        "feature_dim_v2": FEATURE_DIM_V2,
        "v1": _agg(v1_rows),
        "v2": _agg(v2_rows),
        "delta_v2_minus_v1": {
            "macro_f1": paired_delta_report([r["macro_f1"] for r in v2_rows], [r["macro_f1"] for r in v1_rows]),
            "helmet_remove_recall": paired_delta_report(
                [r["helmet_remove_recall"] for r in v2_rows],
                [r["helmet_remove_recall"] for r in v1_rows],
            ),
            "helmet_remove_fnr": paired_delta_report(
                [r["helmet_remove_fnr"] for r in v2_rows],
                [r["helmet_remove_fnr"] for r in v1_rows],
            ),
        },
        "per_seed": per_seed,
        "disclaimer": "Synthetic OOD only. Not construction-site CCTV performance.",
    }
    json_path = out / "v1_v2_comparison.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out / "v1_v2_comparison.csv"
    csv_rows = []
    for row in per_seed:
        for ver in ("v1", "v2"):
            rec = dict(row[ver])
            rec["seed"] = row["seed"]
            csv_rows.append(rec)
    keys = sorted({k for r in csv_rows for k in r})
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"wrote {json_path}", flush=True)
    d = summary["delta_v2_minus_v1"]
    print(
        f"Δ Macro F1={d['macro_f1']['mean']:+.4f}  Δ Recall={d['helmet_remove_recall']['mean']:+.4f}  "
        f"Δ FNR={d['helmet_remove_fnr']['mean']:+.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
