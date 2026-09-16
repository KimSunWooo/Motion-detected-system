#!/usr/bin/env python3
"""Aggregate already-trained 10000-sample seed runs without regenerating data."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_benchmark
from helmet_action.evaluation.failures import dump_failure
from helmet_action.evaluation.metrics import (
    aggregate_seed_results,
    bootstrap_binary_ci,
    flatten_eval,
    permutation_importance_report,
    reverse_sequence_report,
)
from helmet_action.features.v1_names import FEATURE_DIM_V1
from helmet_action.features.v2 import FEATURE_DIM_V2
from helmet_action.models.temporal_classifier import SklearnActionClassifier
from helmet_action.models.training import REMOVE, evaluate_predictions, vectorize_dataset
from helmet_action.synthetic.generator import GENERATOR_VERSION

_load_split = run_benchmark._load_split
_predict_split = run_benchmark._predict_split
_trim = run_benchmark._trim
_write_csv = run_benchmark._write_csv
compare_classical_models = run_benchmark.compare_classical_models
compare_feature_versions = run_benchmark.compare_feature_versions
evaluate_counterfactuals = run_benchmark.evaluate_counterfactuals
write_baseline_v2 = run_benchmark.write_baseline_v2


def main() -> int:
    work = ROOT / "outputs" / "evaluation" / "runs"
    out = ROOT / "outputs" / "evaluation"
    seeds = [42, 101, 202, 303, 404]
    per_seed = []
    last = None
    for seed in seeds:
        print(f"eval seed {seed}", flush=True)
        data = work / f"data_seed_{seed}"
        model_path = work / f"model_seed_{seed}.joblib"
        clf = SklearnActionClassifier.load(model_path)
        npz_te, rec_te = _load_split(data, "test")
        npz_va, _ = _load_split(data, "validation")
        y_true, y_pred, scores = _predict_split(clf, npz_te)
        test_report = evaluate_predictions(y_true, y_pred, y_score_remove=scores)
        ytv, ypv, sv = _predict_split(clf, npz_va)
        val_report = evaluate_predictions(ytv, ypv, y_score_remove=sv)
        flat = flatten_eval(test_report, y_true, y_pred)
        dest = Counter(flat.get("fn_destinations", []))
        flat.update(
            {
                "seed": seed,
                "samples": 10000,
                "feature_version": clf.feature_version,
                "feature_count": len(clf.feature_names),
                "classifier": "HistGradientBoostingClassifier",
                "validation_macro_f1": float(val_report["macro_f1"]),
                "validation_remove_recall": float(val_report["binary_helmet_remove"]["recall"]),
                "bootstrap": bootstrap_binary_ci(y_true, y_pred, n_boot=400, seed=seed),
                "families_test": sorted({str(x) for x in npz_te["families"]}) if "families" in npz_te.files else [],
                "n_test": int(len(y_true)),
                "model_path": str(model_path),
                "data_path": str(data),
                "fn_to_head_touch": int(dest.get("HEAD_TOUCH", 0)),
                "fn_to_helmet_adjust": int(dest.get("HELMET_ADJUST", 0)),
                "fn_to_other": int(sum(v for k, v in dest.items() if k not in ("HEAD_TOUCH", "HELMET_ADJUST"))),
            }
        )
        print(
            f"  acc={flat['accuracy']:.3f} macroF1={flat['macro_f1']:.3f} "
            f"REMOVE R={flat['helmet_remove_recall']:.3f} FNR={flat['helmet_remove_fnr']:.3f} "
            f"FN dest {dict(dest)}",
            flush=True,
        )
        per_seed.append(flat)
        last = {
            "clf": clf,
            "npz_te": npz_te,
            "npz_va": npz_va,
            "records": rec_te,
            "flat": flat,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    # FN dump for seed 42 only, skip matplotlib-heavy if many
    print("dump FNs seed 42", flush=True)
    clf42 = SklearnActionClassifier.load(work / "model_seed_42.joblib")
    npz42, rec42 = _load_split(work / "data_seed_42", "test")
    n_fn = 0
    fail_dir = work / "failures"
    for i in range(len(npz42["labels"])):
        if str(npz42["labels"][i]) != REMOVE:
            continue
        t = int(npz42["lengths"][i])
        k = npz42["keypoints"][i, :t]
        c = npz42["confidences"][i, :t]
        proba = clf42.predict_proba(k, c)
        pred = max(proba, key=proba.get)
        if pred == REMOVE:
            continue
        n_fn += 1
        if n_fn > 12:
            break
        meta = rec42[i] if i < len(rec42) else {"seed": int(npz42["seeds"][i])}
        dump_failure(fail_dir, f"seed_42_remove_fn_{n_fn:03d}", k, c, meta, clf42, y_pred=pred, ml_proba=proba)

    summary = aggregate_seed_results(per_seed)
    summary["git_commit"] = "ceb4039bc80640e7e5e4f7ad913224fa7cc7eac4"
    summary["generator_version"] = GENERATOR_VERSION
    summary["feature_version"] = "v1"
    summary["feature_dim_v1"] = FEATURE_DIM_V1
    summary["feature_dim_v2"] = FEATURE_DIM_V2
    summary["samples"] = 10000
    summary["disclaimer"] = "Synthetic OOD only. Not construction-site CCTV performance."

    clf = last["clf"]
    npz_te = last["npz_te"]
    rev_rows = []
    n_seen = 0
    for i in range(len(npz_te["labels"])):
        if str(npz_te["labels"][i]) != REMOVE:
            continue
        t = int(npz_te["lengths"][i])
        rev_rows.append(reverse_sequence_report(clf, npz_te["keypoints"][i, :t], npz_te["confidences"][i, :t]))
        n_seen += 1
        if n_seen >= 16:
            break
    summary["temporal_reverse"] = {
        "n": len(rev_rows),
        "fraction_still_remove": float(np.mean([r["same_remove_decision"] for r in rev_rows])) if rev_rows else None,
        "forward_p_mean": float(np.mean([r["forward_p_remove"] for r in rev_rows])) if rev_rows else None,
        "reverse_p_mean": float(np.mean([r["reverse_p_remove"] for r in rev_rows])) if rev_rows else None,
        "examples": rev_rows[:8],
    }
    print("permutation importance (n_repeats=4, val subsample 400)", flush=True)
    npz_va = last["npz_va"]
    kva, cva = _trim(npz_va["keypoints"], npz_va["confidences"], npz_va["lengths"])
    # subsample for speed
    rng = np.random.default_rng(0)
    idx = rng.choice(len(kva), size=min(400, len(kva)), replace=False)
    kva_s = [kva[i] for i in idx]
    cva_s = [cva[i] for i in idx]
    yva_s = np.asarray(npz_va["labels"])[idx]
    Xva = vectorize_dataset(kva_s, cva_s, feature_version=clf.feature_version)
    try:
        summary["permutation_importance_top"] = permutation_importance_report(
            clf.estimator, Xva, yva_s, clf.feature_names, encoder=clf.encoder, n_repeats=4, seed=0
        )[:30]
    except Exception as exc:
        summary["permutation_importance_error"] = str(exc)
    print("counterfactual", flush=True)
    summary["counterfactual"] = evaluate_counterfactuals(clf, n_pairs=16, seed0=7)
    print("feature v1 vs v2 (last seed)", flush=True)
    npz_tr, _ = _load_split(Path(last["flat"]["data_path"]), "train")
    summary["feature_v1_v2"] = compare_feature_versions(npz_tr, last["npz_va"], last["npz_te"])
    print("classical models (last seed)", flush=True)
    summary["classical_models"] = compare_classical_models(npz_tr, last["npz_va"], last["npz_te"], "v1")

    write_baseline_v2(out / "baseline_v2.json")
    (out / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out / "benchmark_summary.csv", per_seed)
    print("=== aggregate ===", flush=True)
    for k in ("accuracy", "macro_f1", "helmet_remove_recall", "helmet_remove_fnr"):
        s = summary.get(k, {})
        print(f"  {k}: {s.get('mean', 0):.3f} ± {s.get('std', 0):.3f}  [{s.get('min', 0):.3f}, {s.get('max', 0):.3f}]", flush=True)
    print("wrote", out / "benchmark_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
