#!/usr/bin/env python3
"""REMOVE_C subtype evaluation + FN reason tagging.

PYTHONPATH=src python scripts/analyze_remove_c.py --model models/action_classifier.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.evaluation.failure_reasons import summarize_reasons, tag_failure_reasons  # noqa: E402
from helmet_action.evaluation.false_safe import false_safe_rate, unknown_rate  # noqa: E402
from helmet_action.models.hybrid import HybridActionClassifier  # noqa: E402
from helmet_action.models.labels import ActionClass  # noqa: E402
from helmet_action.models.temporal_classifier import SklearnActionClassifier  # noqa: E402
from helmet_action.synthetic.families import REMOVE_C_SUBTYPES  # noqa: E402
from helmet_action.synthetic.generator import generate_remove_c_subtype  # noqa: E402

REMOVE = ActionClass.HELMET_REMOVE.value


def _eval_family(clf, hybrid, subtype: str, n: int, seed0: int) -> dict:
    y_pred_ml, y_pred_hy, reasons = [], [], []
    n_unknown = n_touch = n_adjust = n_fn = 0
    for i in range(n):
        seq, conf, meta = generate_remove_c_subtype(seed0 + i * 17, subtype, split="test", apply_noise=True)
        proba = clf.predict_proba(seq, conf)
        ml = max(proba, key=proba.get)
        dec = hybrid.predict(seq, conf)
        y_pred_ml.append(ml)
        y_pred_hy.append(dec.action.value)
        if ml != REMOVE:
            n_fn += 1
            tags = tag_failure_reasons(
                seq,
                conf,
                y_pred=ml,
                ml_proba=proba,
                rule_label=dec.rule_label,
                meta=meta.to_dict(),
            )
            reasons.append(
                {
                    "subtype": subtype,
                    "seed": meta.seed,
                    "ml_pred": ml,
                    "hybrid_pred": dec.action.value,
                    "reasons": tags,
                    "family": meta.family,
                    "variant": meta.variant,
                }
            )
        if dec.action is ActionClass.UNKNOWN or dec.action is ActionClass.INSUFFICIENT_POSE:
            n_unknown += 1
        if ml in ("HEAD_TOUCH",):
            n_touch += 1
        if ml in ("HELMET_ADJUST",):
            n_adjust += 1
    yt = np.array([REMOVE] * n)
    yp = np.array(y_pred_ml)
    tp = int(np.sum(yp == REMOVE))
    fn = n - tp
    return {
        "subtype": subtype,
        "sample_count": n,
        "accuracy": float(tp / n),
        "remove_recall": float(tp / n),
        "fnr": float(fn / n),
        "unknown_rate": float(n_unknown / n),
        "head_touch_confusion": float(n_touch / n),
        "helmet_adjust_confusion": float(n_adjust / n),
        "hybrid_false_safe_rate": false_safe_rate(yt, np.array(y_pred_hy)),
        "false_negatives": reasons,
        "pred_counts": dict(Counter(y_pred_ml)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "action_classifier.joblib")
    parser.add_argument("--n-per-family", type=int, default=40)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "evaluation")
    args = parser.parse_args(argv)

    clf = SklearnActionClassifier.load(args.model)
    hybrid = HybridActionClassifier(ml=clf)
    families = []
    all_fn = []
    for i, subtype in enumerate(REMOVE_C_SUBTYPES):
        print(f"=== {subtype} ===", flush=True)
        row = _eval_family(clf, hybrid, subtype, args.n_per_family, seed0=50_000 + i * 1000)
        print(
            f"  n={row['sample_count']} recall={row['remove_recall']:.3f} fnr={row['fnr']:.3f} "
            f"UNK={row['unknown_rate']:.3f} TOUCH={row['head_touch_confusion']:.3f}",
            flush=True,
        )
        families.append({k: v for k, v in row.items() if k != "false_negatives"})
        all_fn.extend(row["false_negatives"])

    worst = max(families, key=lambda r: r["fnr"])
    reason_summary = summarize_reasons(all_fn)
    payload = {
        "model": str(args.model),
        "feature_version": clf.feature_version,
        "n_per_family": args.n_per_family,
        "families": families,
        "worst_subtype": worst["subtype"],
        "worst_recall": worst["remove_recall"],
        "worst_fnr": worst["fnr"],
        "n_false_negatives": len(all_fn),
        "disclaimer": "Synthetic REMOVE_C subtypes only.",
    }
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "remove_c_analysis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fail_dir = ROOT / "outputs" / "failures"
    fail_dir.mkdir(parents=True, exist_ok=True)
    reason_summary["worst_subtype"] = worst["subtype"]
    (fail_dir / "failure_reason_summary.json").write_text(
        json.dumps(reason_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"worst={worst['subtype']} FNR={worst['fnr']:.3f}", flush=True)
    print(f"wrote {out / 'remove_c_analysis.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
