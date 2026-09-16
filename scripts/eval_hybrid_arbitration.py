#!/usr/bin/env python3
"""Hybrid V1 vs V2 arbitration eval + occlusion dumps + UNKNOWN reasons.

Weights for V2 are selected on a train-family validation set only.
REMOVE_C / test / OOD are evaluated once after freeze.

PYTHONPATH=src python scripts/eval_hybrid_arbitration.py --model models/action_classifier.joblib
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

from helmet_action.evaluation.false_safe import safety_metrics  # noqa: E402
from helmet_action.evaluation.unknown_reasons import summarize_unknown_reasons  # noqa: E402
from helmet_action.models.fusion import DEFAULT_FUSION, select_fusion_weights_on_validation  # noqa: E402
from helmet_action.models.hybrid import HybridActionClassifier  # noqa: E402
from helmet_action.models.labels import ActionClass, RiskLevel  # noqa: E402
from helmet_action.models.temporal_classifier import SklearnActionClassifier  # noqa: E402
from helmet_action.pose.constants import L_EAR, L_WRIST, R_EAR, R_WRIST  # noqa: E402
from helmet_action.pose.quality import compute_pose_quality_score  # noqa: E402
from helmet_action.real.dataset import default_raw_root, smoke_raw_capture_layout  # noqa: E402
from helmet_action.state.action_state_machine import infer_phases  # noqa: E402
from helmet_action.synthetic.augmentation import apply_consecutive_occlusion  # noqa: E402
from helmet_action.synthetic.families import REMOVE_C_SUBTYPES  # noqa: E402
from helmet_action.synthetic.generator import generate_one, generate_remove_c_subtype  # noqa: E402

REMOVE = ActionClass.HELMET_REMOVE.value
COMPARE_SUBTYPES = [
    "REMOVE_C_LATERAL_LEFT",
    "REMOVE_C_LATERAL_RIGHT",
    "REMOVE_C_STUTTER",
    "REMOVE_C_SLOW",
    "REMOVE_C_BRIM",
    "REMOVE_C_ONE_THEN_TWO",
    "REMOVE_C_LOW_WRIST_CONF",
    "REMOVE_C_PARTIAL_OCCLUSION",
]


def _predict_row(hybrid: HybridActionClassifier, seq, conf, version: str) -> dict:
    ml_proba = hybrid.ml.predict_proba(seq, conf) if hybrid.ml is not None else {}
    ml_pred = max(ml_proba, key=ml_proba.get) if ml_proba else None
    dec = hybrid.predict(seq, conf, version=version)
    ev = dec.evidence
    return {
        "ml_pred": ml_pred,
        "ml_p_remove": float(ml_proba.get(REMOVE, 0.0)),
        "hybrid_pred": dec.action.value,
        "risk_level": dec.risk_level,
        "decision_status": dec.decision_status,
        "pose_quality": float(dec.pose_quality),
        "phase_confidence": float(dec.phase_confidence),
        "wrist_quality": float(ev.wrist_quality) if ev else 0.0,
        "ear_quality": float(ev.ear_quality) if ev else 0.0,
        "phase": dec.phase.value,
        "rule_label": dec.rule_label,
        "rejection_reasons": list(dec.rejection_reasons),
        "ml_remove_probability": float(ev.ml_remove_probability) if ev else float(ml_proba.get(REMOVE, 0.0)),
        "fusion_remove_score": float(ev.fusion_remove_score) if ev else 0.0,
        "grasp_confidence": float(ev.grasp_confidence) if ev else 0.0,
        "lift_confidence": float(ev.lift_confidence) if ev else 0.0,
        "separation_confidence": float(ev.separation_confidence) if ev else 0.0,
    }


def _eval_pack(rows: list[dict], y_true: list[str]) -> dict:
    yp = np.array([r["hybrid_pred"] for r in rows])
    ml = np.array([r["ml_pred"] or "UNKNOWN" for r in rows])
    yt = np.array(y_true)
    sm = safety_metrics(yt, yp)
    n = max(len(yt), 1)
    pos = yt == REMOVE
    ml_recall = float((ml[pos] == REMOVE).mean()) if np.any(pos) else 0.0
    risk = Counter(r["risk_level"] for r in rows)
    return {
        "ml_recall": ml_recall,
        "hybrid_confirmed_recall": sm["confirmed_remove_recall"],
        "unknown_rate": sm["unknown_rate"],
        "false_safe_rate": sm["false_safe_rate"],
        "false_alarm_rate": sm["false_alarm_rate"],
        "false_positive_rate": sm["false_alarm_rate"],
        "unknown_on_positive_rate": sm["unknown_on_positive_rate"],
        "insufficient_pose_rate": sm["insufficient_pose_rate"],
        "risk_counts": dict(risk),
        "risk_rates": {k: float(v) / n for k, v in risk.items()},
        "pred_counts": dict(Counter(yp.tolist())),
        "n": int(len(yt)),
    }


def _validation_sequences(n_pos: int, n_neg: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    out = []
    for i in range(n_pos):
        seq, conf, meta = generate_one(seed=8_000 + i, scenario="HELMET_REMOVE", split="train", apply_noise=True)
        out.append((seq, conf, meta.label))
    negs = ("HELMET_ADJUST", "HEAD_SCRATCH", "HEAD_TOUCH", "IDLE", "TWO_HAND_HEAD_TOUCH")
    for i in range(n_neg):
        scen = negs[i % len(negs)]
        seq, conf, meta = generate_one(seed=9_000 + i, scenario=scen, split="train", apply_noise=True)
        out.append((seq, conf, meta.label))
    return out


def _phase_tags(seq) -> dict:
    tr = infer_phases(seq)
    return {
        "NO_GRASP": not tr.saw_grasp and not tr.saw_partial_grasp,
        "PHASE_INCONSISTENCY": (tr.saw_grasp or tr.saw_lift or tr.saw_partial_grasp) and not tr.ordered,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "action_classifier.joblib")
    parser.add_argument("--n-per-family", type=int, default=24)
    parser.add_argument("--n-occlusion", type=int, default=12)
    parser.add_argument("--n-val", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "evaluation")
    args = parser.parse_args(argv)

    clf = SklearnActionClassifier.load(args.model)
    hybrid = HybridActionClassifier(ml=clf)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    print("=== validation-only fusion search ===", flush=True)
    val_seq = _validation_sequences(args.n_val, args.n_val)
    fusion_search = select_fusion_weights_on_validation(hybrid, val_seq)
    print("selected", fusion_search["selected"], fusion_search["selected_metrics"], flush=True)
    (out / "fusion_val_search.json").write_text(json.dumps(fusion_search, indent=2), encoding="utf-8")

    print("=== REMOVE_C Hybrid V1 vs V2 ===", flush=True)
    families = []
    unknown_v1, unknown_v2 = [], []
    phase_v1 = {"NO_GRASP": 0, "PHASE_INCONSISTENCY": 0}
    phase_v2 = {"NO_GRASP": 0, "PHASE_INCONSISTENCY": 0}
    n_phase = 0
    overall_v1_true, overall_v1_pred, overall_v1_ml = [], [], []
    overall_v2_true, overall_v2_pred, overall_v2_ml = [], [], []
    overall_v1_risk, overall_v2_risk = [], []

    for i, subtype in enumerate(COMPARE_SUBTYPES):
        print(f"--- {subtype} ---", flush=True)
        rows_v1, rows_v2, y_true = [], [], []
        for j in range(args.n_per_family):
            seq, conf, meta = generate_remove_c_subtype(60_000 + i * 1000 + j * 17, subtype, split="test", apply_noise=True)
            y_true.append(REMOVE)
            r1 = _predict_row(hybrid, seq, conf, "v1")
            r2 = _predict_row(hybrid, seq, conf, "v2")
            rows_v1.append(r1)
            rows_v2.append(r2)
            tags = _phase_tags(seq)
            n_phase += 1
            for k, flag in tags.items():
                if flag:
                    phase_v1[k] += 1
                    phase_v2[k] += 1
            if r1["hybrid_pred"] in (ActionClass.UNKNOWN.value, ActionClass.INSUFFICIENT_POSE.value):
                unknown_v1.append(r1)
                if "NO_GRASP" in r1["rejection_reasons"]:
                    pass
            if r2["hybrid_pred"] in (ActionClass.UNKNOWN.value, ActionClass.INSUFFICIENT_POSE.value):
                unknown_v2.append(r2)
            overall_v1_true.append(REMOVE)
            overall_v1_pred.append(r1["hybrid_pred"])
            overall_v1_ml.append(r1["ml_pred"])
            overall_v1_risk.append(r1["risk_level"])
            overall_v2_true.append(REMOVE)
            overall_v2_pred.append(r2["hybrid_pred"])
            overall_v2_ml.append(r2["ml_pred"])
            overall_v2_risk.append(r2["risk_level"])
        pack_v1 = _eval_pack(rows_v1, y_true)
        pack_v2 = _eval_pack(rows_v2, y_true)
        print(
            f"  V1 hyR={pack_v1['hybrid_confirmed_recall']:.3f} UNK={pack_v1['unknown_rate']:.3f} "
            f"FS={pack_v1['false_safe_rate']:.3f} FA={pack_v1['false_alarm_rate']:.3f} "
            f"| V2 hyR={pack_v2['hybrid_confirmed_recall']:.3f} UNK={pack_v2['unknown_rate']:.3f} "
            f"FS={pack_v2['false_safe_rate']:.3f} FA={pack_v2['false_alarm_rate']:.3f}",
            flush=True,
        )
        families.append({"subtype": subtype, "v1": pack_v1, "v2": pack_v2})

    # Negatives for false alarm / risk mix
    neg_rows_v1, neg_rows_v2, neg_true = [], [], []
    for i, scen in enumerate(("HELMET_ADJUST", "HEAD_SCRATCH", "IDLE", "TWO_HAND_HEAD_TOUCH") * 6):
        seq, conf, meta = generate_one(seed=70_000 + i, scenario=scen, split="test", apply_noise=True)
        neg_true.append(meta.label)
        n1 = _predict_row(hybrid, seq, conf, "v1")
        n2 = _predict_row(hybrid, seq, conf, "v2")
        neg_rows_v1.append(n1)
        neg_rows_v2.append(n2)
        overall_v1_true.append(meta.label)
        overall_v1_pred.append(n1["hybrid_pred"])
        overall_v1_risk.append(n1["risk_level"])
        overall_v2_true.append(meta.label)
        overall_v2_pred.append(n2["hybrid_pred"])
        overall_v2_risk.append(n2["risk_level"])

    overall_v1 = _eval_pack(
        [{"hybrid_pred": p, "ml_pred": "HELMET_REMOVE", "risk_level": r} for p, r in zip(overall_v1_pred, overall_v1_risk)],
        overall_v1_true,
    )
    overall_v2 = _eval_pack(
        [{"hybrid_pred": p, "ml_pred": "HELMET_REMOVE", "risk_level": r} for p, r in zip(overall_v2_pred, overall_v2_risk)],
        overall_v2_true,
    )
    # Fix ML recall on mixed set: recompute from stored ml on REMOVE_C only
    pos_n = len(COMPARE_SUBTYPES) * args.n_per_family
    overall_v1["ml_recall"] = float(sum(1 for x in overall_v1_ml[:pos_n] if x == REMOVE) / max(pos_n, 1))
    overall_v2["ml_recall"] = float(sum(1 for x in overall_v2_ml[:pos_n] if x == REMOVE) / max(pos_n, 1))

    unk_stats_v1 = summarize_unknown_reasons(unknown_v1)
    unk_stats_v2 = summarize_unknown_reasons(unknown_v2)
    (out / "unknown_reason_stats.json").write_text(
        json.dumps({"v1": unk_stats_v1, "v2": unk_stats_v2}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    nograsp_v1 = sum(1 for r in unknown_v1 if "NO_GRASP" in r.get("rejection_reasons", []))
    nograsp_v2 = sum(1 for r in unknown_v2 if "NO_GRASP" in r.get("rejection_reasons", []))
    phaseinc_v1 = sum(1 for r in unknown_v1 if "PHASE_INCONSISTENCY" in r.get("rejection_reasons", []))
    phaseinc_v2 = sum(1 for r in unknown_v2 if "PHASE_INCONSISTENCY" in r.get("rejection_reasons", []))
    phase_report = {
        "n_remove_c": n_phase,
        "unknown_n_v1": len(unknown_v1),
        "unknown_n_v2": len(unknown_v2),
        "NO_GRASP_unknown_v1": nograsp_v1,
        "NO_GRASP_unknown_v2": nograsp_v2,
        "NO_GRASP_reduction": float(nograsp_v1 - nograsp_v2),
        "PHASE_INCONSISTENCY_unknown_v1": phaseinc_v1,
        "PHASE_INCONSISTENCY_unknown_v2": phaseinc_v2,
        "PHASE_INCONSISTENCY_reduction": float(phaseinc_v1 - phaseinc_v2),
    }

    worst_v1 = min(families, key=lambda r: (r["v1"]["hybrid_confirmed_recall"], -r["v1"]["unknown_rate"]))
    worst_v2 = min(families, key=lambda r: (r["v2"]["hybrid_confirmed_recall"], -r["v2"]["unknown_rate"]))

    print("=== occlusion ===", flush=True)
    pos, neg = [], []
    for i in range(args.n_occlusion):
        seq, conf, meta = generate_one(seed=71_000 + i, scenario="HELMET_REMOVE", split="test", apply_noise=False)
        pos.append((seq, conf, meta.label))
    for i, scen in enumerate(("HELMET_ADJUST", "TWO_HAND_HEAD_TOUCH", "IDLE") * ((args.n_occlusion + 2) // 3)):
        if i >= args.n_occlusion:
            break
        seq, conf, meta = generate_one(seed=72_000 + i, scenario=scen, split="test", apply_noise=False)
        neg.append((seq, conf, meta.label))

    def _occ(joints, gaps, version: str) -> dict:
        by_gap = {}
        dumps = []
        for gap in gaps:
            items_true, items_pred, rows = [], [], []
            rng = np.random.default_rng(gap * 97 + sum(joints) + (0 if version == "v1" else 13))
            for seq, conf, lab in pos + neg:
                start = max(4, seq.shape[0] // 4)
                s2, c2 = apply_consecutive_occlusion(seq, conf, joints, gap, start=start, rng=rng)
                row = _predict_row(hybrid, s2, c2, version)
                pq = compute_pose_quality_score(s2, c2)
                row.update(
                    {
                        "true_label": lab,
                        "gap_frames": gap,
                        "pose_quality_status": pq.decision_status,
                        "longest_ear_streak": int(pq.longest_ear_streak),
                        "both_wrist_streak": int(pq.both_wrist_streak),
                    }
                )
                items_true.append(lab)
                items_pred.append(row["hybrid_pred"])
                rows.append(row)
                if gap == 30 and joints == [L_EAR, R_EAR] and lab == REMOVE:
                    dumps.append(row)
            pack = safety_metrics(np.array(items_true), np.array(items_pred))
            pack["n"] = len(items_true)
            by_gap[str(gap)] = pack
            print(
                f"  {version} joints={joints} gap={gap} R={pack['confirmed_remove_recall']:.3f} "
                f"UNK={pack['unknown_rate']:.3f} FS={pack['false_safe_rate']:.3f}",
                flush=True,
            )
        return {"by_gap": by_gap, "dump_30": dumps}

    ear_v1 = _occ([L_EAR, R_EAR], (5, 10, 20, 30), "v1")
    ear_v2 = _occ([L_EAR, R_EAR], (5, 10, 20, 30), "v2")
    wrist_v1 = _occ([L_WRIST, R_WRIST], (5, 10, 20, 30), "v1")
    wrist_v2 = _occ([L_WRIST, R_WRIST], (5, 10, 20, 30), "v2")
    (out / "ear_occlusion_dump.json").write_text(
        json.dumps({"v1": ear_v1["dump_30"], "v2": ear_v2["dump_30"]}, indent=2, default=str),
        encoding="utf-8",
    )

    raw_smoke = smoke_raw_capture_layout(default_raw_root())
    (out / "real_raw_smoke.json").write_text(json.dumps(raw_smoke, indent=2), encoding="utf-8")

    feature_decision = {
        "default": "v1",
        "reason": (
            "Feature V2 wins Macro F1 on 5/5 identical seeds but REMOVE recall/FNR CIs include 0 "
            "(2/5 seeds slightly worse recall). Production model remains Feature V1 139-D. "
            "Hybrid V2 is an arbitration layer, not a new feature extractor."
        ),
        "v2_macro_f1_better_seeds": "5/5",
        "remove_recall_ci_includes_zero": True,
    }

    payload = {
        "fusion_selected": fusion_search["selected"],
        "fusion_val_metrics": fusion_search["selected_metrics"],
        "hybrid_v1_overall": overall_v1,
        "hybrid_v2_overall": overall_v2,
        "remove_c": families,
        "worst_subtype_v1": worst_v1["subtype"],
        "worst_subtype_v2": worst_v2["subtype"],
        "unknown_reasons": {"v1": unk_stats_v1, "v2": unk_stats_v2},
        "phase": phase_report,
        "ear_occlusion": {"v1": ear_v1["by_gap"], "v2": ear_v2["by_gap"]},
        "both_wrist_occlusion": {"v1": wrist_v1["by_gap"], "v2": wrist_v2["by_gap"]},
        "risk": {
            "v1": overall_v1["risk_rates"],
            "v2": overall_v2["risk_rates"],
        },
        "feature_default": feature_decision,
        "real_pose": {
            "pipeline_ready": True,
            "manifest_ready": True,
            "dataset_present": bool(raw_smoke.get("dataset_present")),
            "REAL_METRICS": "NOT AVAILABLE" if not raw_smoke.get("dataset_present") else "see real_zero_shot.json",
            "raw_smoke": raw_smoke,
        },
        "n_per_family": args.n_per_family,
        "disclaimer": "Synthetic REMOVE_C / occlusion only. Not real CCTV performance.",
    }
    (out / "hybrid_v1_v2_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out / 'hybrid_v1_v2_comparison.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
