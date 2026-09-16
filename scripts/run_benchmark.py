#!/usr/bin/env python3
"""Multi-seed synthetic benchmark. Trains a fresh model per seed.

PYTHONPATH=src python scripts/run_benchmark.py --samples 10000 --seeds 42 101 202 303 404
PYTHONPATH=src python scripts/run_benchmark.py --samples 500 --seeds 42 --regression
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_dataset  # noqa: E402
from helmet_action.config import as_dict, load_config, repo_root  # noqa: E402
from helmet_action.evaluation.failures import dump_failure  # noqa: E402
from helmet_action.evaluation.metrics import (  # noqa: E402
    aggregate_seed_results,
    bootstrap_binary_ci,
    flatten_eval,
    permutation_importance_report,
    reverse_sequence_report,
)
from helmet_action.features.v1_names import FEATURE_DIM_V1  # noqa: E402
from helmet_action.features.v2 import FEATURE_DIM_V2  # noqa: E402
from helmet_action.models.labels import ActionClass  # noqa: E402
from helmet_action.models.temporal_classifier import SklearnActionClassifier, train_sklearn_classifier  # noqa: E402
from helmet_action.models.training import REMOVE, evaluate_predictions, vectorize_dataset  # noqa: E402
from helmet_action.synthetic.generator import GENERATOR_VERSION, generate_counterfactual_pair, generate_one  # noqa: E402


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


def _load_split(path: Path, split: str):
    npz = np.load(path / f"{split}.npz", allow_pickle=True)
    meta_path = path / "metadata.json"
    records = []
    if meta_path.exists():
        records = json.loads(meta_path.read_text(encoding="utf-8")).get("records", {}).get(split, [])
    return npz, records


def _predict_split(clf: SklearnActionClassifier, npz) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = npz["labels"]
    lengths = npz["lengths"]
    y_pred, y_true, scores = [], [], []
    for i in range(len(labels)):
        t = int(lengths[i])
        proba = clf.predict_proba(npz["keypoints"][i, :t], npz["confidences"][i, :t])
        pred = max(proba, key=proba.get)
        y_pred.append(pred)
        y_true.append(str(labels[i]))
        scores.append(float(proba.get(REMOVE, 0.0)))
    return np.array(y_true), np.array(y_pred), np.array(scores, dtype=np.float64)


def _dump_remove_fns(clf, npz, records, out_dir: Path, seed: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dumped = []
    n_fn = 0
    for i in range(len(npz["labels"])):
        if str(npz["labels"][i]) != REMOVE:
            continue
        t = int(npz["lengths"][i])
        k = npz["keypoints"][i, :t]
        c = npz["confidences"][i, :t]
        proba = clf.predict_proba(k, c)
        pred = max(proba, key=proba.get)
        if pred == REMOVE:
            continue
        n_fn += 1
        fid = f"seed_{seed}_remove_fn_{n_fn:03d}"
        meta = records[i] if i < len(records) else {"seed": int(npz["seeds"][i]), "scenario": str(npz["scenarios"][i])}
        payload = dump_failure(
            out_dir,
            fid,
            k,
            c,
            meta,
            clf,
            y_true=REMOVE,
            y_pred=pred,
            ml_proba=proba,
        )
        dumped.append({"id": fid, "pred": pred, "true": REMOVE, "family": meta.get("family")})
    return dumped


def evaluate_counterfactuals(clf: SklearnActionClassifier, n_pairs: int = 24, seed0: int = 7) -> dict:
    p_rm, p_adj, p_touch, deltas = [], [], [], []
    for i in range(n_pairs):
        (neg, cneg, mneg), (rm, crm, _) = generate_counterfactual_pair(seed0 + i * 13, kind="adjust_vs_remove")
        pr = float(clf.predict_proba(rm, crm).get(REMOVE, 0.0))
        pa = float(clf.predict_proba(neg, cneg).get(REMOVE, 0.0))
        p_rm.append(pr)
        p_adj.append(pa)
        deltas.append(pr - pa)
        (neg2, c2, _), (rm2, cr2, _) = generate_counterfactual_pair(seed0 + 1000 + i * 13, kind="touch_vs_remove")
        p_touch.append(float(clf.predict_proba(neg2, c2).get(REMOVE, 0.0)))
        p_rm.append(float(clf.predict_proba(rm2, cr2).get(REMOVE, 0.0)))
    return {
        "remove_mean_p": float(np.mean(p_rm)),
        "adjust_mean_p": float(np.mean(p_adj)),
        "head_touch_mean_p": float(np.mean(p_touch)),
        "delta_p_remove_mean": float(np.mean(deltas)),
        "n_pairs": n_pairs,
    }


def _run_one_seed(
    seed: int,
    samples: int,
    work: Path,
    feature_version: str,
    dump_failures: bool,
) -> dict:
    data = work / f"data_seed_{seed}"
    model_path = work / f"model_seed_{seed}.joblib"
    generate_dataset.main(["--samples", str(samples), "--output", str(data), "--seed", str(seed)])
    npz_tr, _ = _load_split(data, "train")
    npz_va, _ = _load_split(data, "validation")
    npz_te, rec_te = _load_split(data, "test")
    ktr, ctr = _trim(npz_tr["keypoints"], npz_tr["confidences"], npz_tr["lengths"])
    kva, cva = _trim(npz_va["keypoints"], npz_va["confidences"], npz_va["lengths"])
    Xtr = vectorize_dataset(ktr, ctr, feature_version=feature_version)
    Xva = vectorize_dataset(kva, cva, feature_version=feature_version)
    clf = train_sklearn_classifier(Xtr, npz_tr["labels"], feature_version=feature_version)
    clf.save(model_path)
    y_true_va, y_pred_va, scores_va = _predict_split(clf, npz_va)
    val_report = evaluate_predictions(y_true_va, y_pred_va, y_score_remove=scores_va)
    y_true, y_pred, scores = _predict_split(clf, npz_te)
    test_report = evaluate_predictions(y_true, y_pred, y_score_remove=scores)
    flat = flatten_eval(test_report, y_true, y_pred)
    flat.update(
        {
            "seed": seed,
            "samples": samples,
            "feature_version": feature_version,
            "feature_count": len(clf.feature_names),
            "classifier": "HistGradientBoostingClassifier",
            "validation_macro_f1": float(val_report["macro_f1"]),
            "validation_remove_recall": float(val_report["binary_helmet_remove"]["recall"]),
            "bootstrap": bootstrap_binary_ci(y_true, y_pred, n_boot=400, seed=seed),
            "families_test": sorted({str(x) for x in npz_te["families"]}) if "families" in npz_te.files else [],
            "n_test": int(len(y_true)),
            "model_path": str(model_path),
            "data_path": str(data),
        }
    )
    if dump_failures:
        flat["failures"] = _dump_remove_fns(clf, npz_te, rec_te, work / "failures", seed)
    dest = Counter(flat.get("fn_destinations", []))
    flat["fn_to_head_touch"] = int(dest.get("HEAD_TOUCH", 0))
    flat["fn_to_helmet_adjust"] = int(dest.get("HELMET_ADJUST", 0))
    flat["fn_to_other"] = int(sum(v for k, v in dest.items() if k not in ("HEAD_TOUCH", "HELMET_ADJUST")))
    return {"flat": flat, "clf": clf, "npz_va": npz_va, "npz_te": npz_te, "Xva": Xva, "yva": npz_va["labels"]}


def compare_feature_versions(npz_tr, npz_va, npz_te) -> dict:
    ktr, ctr = _trim(npz_tr["keypoints"], npz_tr["confidences"], npz_tr["lengths"])
    kva, cva = _trim(npz_va["keypoints"], npz_va["confidences"], npz_va["lengths"])
    results = {}
    chosen = None
    chosen_val = -1.0
    for version in ("v1", "v2"):
        Xtr = vectorize_dataset(ktr, ctr, feature_version=version)
        Xva = vectorize_dataset(kva, cva, feature_version=version)
        clf = train_sklearn_classifier(Xtr, npz_tr["labels"], feature_version=version)
        ytv, ypv, sv = _predict_split(clf, npz_va)
        val = evaluate_predictions(ytv, ypv, y_score_remove=sv)
        score = float(val["macro_f1"]) + 0.15 * float(val["binary_helmet_remove"]["recall"])
        results[version] = {
            "dimension": int(Xtr.shape[1]),
            "validation_macro_f1": float(val["macro_f1"]),
            "validation_remove_recall": float(val["binary_helmet_remove"]["recall"]),
            "selection_score": score,
        }
        if score > chosen_val:
            chosen_val = score
            chosen = (version, clf)
    version, clf = chosen
    yt, yp, s = _predict_split(clf, npz_te)
    test = evaluate_predictions(yt, yp, y_score_remove=s)
    b = test["binary_helmet_remove"]
    results["selected_on_validation"] = version
    results["ood_test"] = {
        version: {
            "macro_f1": float(test["macro_f1"]),
            "accuracy": float(test["accuracy"]),
            "remove_recall": float(b["recall"]),
            "remove_fnr": float(b["fnr"]),
        }
    }
    # Always also report the other version's test for honesty AFTER selection is recorded.
    other = "v2" if version == "v1" else "v1"
    Xtr_o = vectorize_dataset(ktr, ctr, feature_version=other)
    clf_o = train_sklearn_classifier(Xtr_o, npz_tr["labels"], feature_version=other)
    yt_o, yp_o, s_o = _predict_split(clf_o, npz_te)
    test_o = evaluate_predictions(yt_o, yp_o, y_score_remove=s_o)
    b_o = test_o["binary_helmet_remove"]
    results["ood_test"][other] = {
        "macro_f1": float(test_o["macro_f1"]),
        "accuracy": float(test_o["accuracy"]),
        "remove_recall": float(b_o["recall"]),
        "remove_fnr": float(b_o["fnr"]),
        "note": "reported after validation selection; not used to pick the model",
    }
    return results


def compare_classical_models(npz_tr, npz_va, npz_te, feature_version: str) -> dict:
    ktr, ctr = _trim(npz_tr["keypoints"], npz_tr["confidences"], npz_tr["lengths"])
    Xtr = vectorize_dataset(ktr, ctr, feature_version=feature_version)
    names = ["hist_gradient_boosting", "random_forest", "extra_trees", "logistic_regression"]
    val_rows = []
    best = None
    best_score = -1.0
    for name in names:
        clf = train_sklearn_classifier(Xtr, npz_tr["labels"], estimator_name=name, feature_version=feature_version)
        ytv, ypv, sv = _predict_split(clf, npz_va)
        val = evaluate_predictions(ytv, ypv, y_score_remove=sv)
        score = float(val["macro_f1"]) + 0.15 * float(val["binary_helmet_remove"]["recall"])
        row = {
            "estimator": name,
            "validation_macro_f1": float(val["macro_f1"]),
            "validation_remove_recall": float(val["binary_helmet_remove"]["recall"]),
            "selection_score": score,
        }
        val_rows.append(row)
        if score > best_score:
            best_score = score
            best = (name, clf)
    name, clf = best
    yt, yp, s = _predict_split(clf, npz_te)
    test = evaluate_predictions(yt, yp, y_score_remove=s)
    return {
        "validation_comparison": val_rows,
        "selected_on_validation": name,
        "ood_test_selected": {
            "macro_f1": float(test["macro_f1"]),
            "remove_recall": float(test["binary_helmet_remove"]["recall"]),
            "remove_fnr": float(test["binary_helmet_remove"]["fnr"]),
        },
    }


def write_baseline_v2(path: Path) -> None:
    cfg = load_config()
    payload = {
        "name": "baseline-v2",
        "git_commit": "f86f0954947fa658306ccecf84143e621038eb82",
        "generator_version": "1.0.0",
        "feature_version": "v1",
        "classifier": "HistGradientBoostingClassifier",
        "feature_count": FEATURE_DIM_V1,
        "config": as_dict(cfg),
        "dataset_seed": 42,
        "dataset_size": 500,
        "note": (
            "Frozen from the verified pre-robustness run (OOD test n=100, 14 HELMET_REMOVE). "
            "These numbers are the comparison baseline; they are synthetic-only."
        ),
        "accuracy": 0.890,
        "macro_f1": 0.890,
        "helmet_remove_precision": 1.0,
        "helmet_remove_recall": 0.857,
        "helmet_remove_fnr": 0.143,
        "helmet_remove_fpr": 0.0,
        "helmet_remove_tp": 12,
        "helmet_remove_fp": 0,
        "helmet_remove_fn": 2,
        "fn_pattern": "HELMET_REMOVE → HEAD_TOUCH (2/2)",
        "hard_negative_remove_fp": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r if not isinstance(r[k], (dict, list))})
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "evaluation")
    parser.add_argument("--feature-version", default="v1")
    parser.add_argument("--compare-features", action="store_true")
    parser.add_argument("--compare-models", action="store_true")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--skip-failures", action="store_true")
    parser.add_argument("--importance-top", type=int, default=30)
    args = parser.parse_args(argv)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_baseline_v2(out / "baseline_v2.json")
    work = out / ("regression" if args.regression else "runs")
    work.mkdir(parents=True, exist_ok=True)

    per_seed = []
    last = None
    for seed in args.seeds:
        print(f"\n=== seed {seed} samples={args.samples} feature={args.feature_version} ===")
        last = _run_one_seed(
            seed,
            args.samples,
            work,
            args.feature_version,
            dump_failures=(not args.skip_failures and seed == args.seeds[0]),
        )
        per_seed.append(last["flat"])
        b = last["flat"]
        print(
            f"  acc={b['accuracy']:.3f} macroF1={b['macro_f1']:.3f} "
            f"REMOVE R={b['helmet_remove_recall']:.3f} FNR={b['helmet_remove_fnr']:.3f}"
        )

    summary = aggregate_seed_results(per_seed)
    summary["git_commit"] = _git_commit()
    summary["generator_version"] = GENERATOR_VERSION
    summary["feature_version"] = args.feature_version
    summary["feature_dim_v1"] = FEATURE_DIM_V1
    summary["feature_dim_v2"] = FEATURE_DIM_V2
    summary["samples"] = args.samples
    summary["disclaimer"] = "Synthetic OOD only. Not construction-site CCTV performance."

    if last is not None:
        clf = last["clf"]
        npz_te = last["npz_te"]
        # reverse temporal test on first 12 REMOVE samples
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
        try:
            imp = permutation_importance_report(
                clf.estimator,
                last["Xva"],
                last["yva"],
                clf.feature_names,
                encoder=clf.encoder,
                n_repeats=6,
            )
            summary["permutation_importance_top"] = imp[: args.importance_top]
        except Exception as exc:
            summary["permutation_importance_error"] = str(exc)
        try:
            summary["counterfactual"] = evaluate_counterfactuals(clf)
        except Exception as exc:
            summary["counterfactual_error"] = str(exc)
        if args.compare_features:
            npz_tr, _ = _load_split(Path(last["flat"]["data_path"]), "train")
            summary["feature_v1_v2"] = compare_feature_versions(npz_tr, last["npz_va"], last["npz_te"])
        if args.compare_models:
            npz_tr, _ = _load_split(Path(last["flat"]["data_path"]), "train")
            summary["classical_models"] = compare_classical_models(
                npz_tr, last["npz_va"], last["npz_te"], args.feature_version
            )

    (out / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out / "benchmark_summary.csv", per_seed)
    print("\n=== aggregate ===")
    for k in ("accuracy", "macro_f1", "helmet_remove_recall", "helmet_remove_fnr"):
        s = summary.get(k, {})
        print(f"  {k}: {s.get('mean', 0):.3f} ± {s.get('std', 0):.3f}  [{s.get('min', 0):.3f}, {s.get('max', 0):.3f}]")
    print(f"wrote {out / 'benchmark_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
