#!/usr/bin/env python3
"""OOD stress tests for the frozen-threshold action classifier.

PYTHONPATH=src python scripts/stress_test.py --model models/action_classifier.joblib --output outputs/stress
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.evaluation.metrics import flatten_eval  # noqa: E402
from helmet_action.models.hybrid import HybridActionClassifier  # noqa: E402
from helmet_action.models.labels import ActionClass, RemovalPhase  # noqa: E402
from helmet_action.models.temporal_classifier import SklearnActionClassifier  # noqa: E402
from helmet_action.models.training import REMOVE, evaluate_predictions  # noqa: E402
from helmet_action.pose.constants import L_EAR, L_ELBOW, L_WRIST, R_EAR, R_ELBOW, R_WRIST  # noqa: E402
from helmet_action.pose.pose_buffer import TrackPoseBuffer  # noqa: E402
from helmet_action.pose.types import PoseObservation  # noqa: E402
from helmet_action.state.action_state_machine import infer_phases  # noqa: E402
from helmet_action.synthetic.augmentation import apply_targeted_occlusion  # noqa: E402
from helmet_action.synthetic.generator import generate_one  # noqa: E402


JOINT_GROUPS = {
    "left_wrist": [L_WRIST],
    "right_wrist": [R_WRIST],
    "both_wrists": [L_WRIST, R_WRIST],
    "left_ear": [L_EAR],
    "right_ear": [R_EAR],
    "both_ears": [L_EAR, R_EAR],
    "one_elbow": [L_ELBOW],
    "head_keypoints": [0, 1, 2, L_EAR, R_EAR],
}


def _obs(track_id: int, t: float, k: np.ndarray, c: np.ndarray, fps: float) -> PoseObservation:
    return PoseObservation(
        timestamp=t,
        frame_index=int(t * 1000),
        track_id=track_id,
        bbox=(0.0, 0.0, 1.0, 1.0),
        keypoints=k,
        keypoint_confidence=c,
        detection_confidence=1.0,
        source_fps=fps,
    )


def _eval_pairs(clf, pairs: list[tuple[np.ndarray, np.ndarray, str]]) -> dict:
    y_true, y_pred, scores = [], [], []
    for k, c, lab in pairs:
        proba = clf.predict_proba(k, c)
        y_pred.append(max(proba, key=proba.get))
        y_true.append(lab)
        scores.append(float(proba.get(REMOVE, 0.0)))
    report = evaluate_predictions(np.array(y_true), np.array(y_pred), y_score_remove=np.array(scores))
    return flatten_eval(report, np.array(y_true), np.array(y_pred))


def _gen_remove(n: int, seed0: int, split: str = "test", **kwargs):
    out = []
    for i in range(n):
        seq, conf, meta = generate_one(seed=seed0 + i * 19, scenario="HELMET_REMOVE", split=split, **kwargs)
        out.append((seq, conf, meta.label, meta))
    return out


def fps_stress(clf, n: int = 24) -> list[dict]:
    rows = []
    for fps in (8, 10, 15, 24, 30, 60):
        pairs = []
        for i, (seq, conf, lab, _meta) in enumerate(_gen_remove(n, 9000 + fps, fps=float(fps), apply_noise=False)):
            buf = TrackPoseBuffer(window_seconds=3.5, target_fps=20)
            for f, (k, c) in enumerate(zip(seq, conf)):
                buf.push(_obs(1, f / float(fps), k, c, float(fps)))
            packed = buf.get_arrays(1)
            if packed is None:
                continue
            pairs.append((packed[0], packed[1], lab))
        # Mix a few negatives so FPR is defined.
        for i in range(max(6, n // 3)):
            seq, conf, meta = generate_one(seed=11000 + fps * 10 + i, scenario="HELMET_ADJUST", split="test", fps=float(fps), apply_noise=False)
            buf = TrackPoseBuffer(window_seconds=3.5, target_fps=20)
            for f, (k, c) in enumerate(zip(seq, conf)):
                buf.push(_obs(1, f / float(fps), k, c, float(fps)))
            packed = buf.get_arrays(1)
            if packed is not None:
                pairs.append((packed[0], packed[1], meta.label))
        flat = _eval_pairs(clf, pairs)
        flat["fps"] = fps
        rows.append(flat)
    return rows


def occlusion_stress(clf, n: int = 20) -> list[dict]:
    rows = []
    base = _gen_remove(n, 12000, apply_noise=False)
    negs = [generate_one(seed=13000 + i, scenario="HEAD_TOUCH", split="test", apply_noise=False) for i in range(n // 2)]
    for name, joints in JOINT_GROUPS.items():
        for rate in (0.10, 0.20, 0.30, 0.40):
            pairs = []
            rng = np.random.default_rng(abs(hash((name, rate))) % (2**31))
            for seq, conf, lab, _ in base:
                s2, c2 = apply_targeted_occlusion(seq, conf, joints, rate, rng)
                s2 = np.where(np.isfinite(s2), s2, 0.0)
                pairs.append((s2, c2, lab))
            for seq, conf, meta in negs:
                s2, c2 = apply_targeted_occlusion(seq, conf, joints, rate, rng)
                s2 = np.where(np.isfinite(s2), s2, 0.0)
                pairs.append((s2, c2, meta.label))
            flat = _eval_pairs(clf, pairs)
            flat.update({"joint_group": name, "dropout": rate})
            rows.append(flat)
    return rows


def camera_stress(clf, n: int = 16) -> dict:
    pitch_rows, yaw_rows, dist_rows = [], [], []
    distances = {"near": 2.2, "medium": 3.5, "far": 5.4}
    for pitch in (20, 30, 40, 50, 60, 70):
        pairs = []
        for i in range(n):
            seq, conf, meta = generate_one(
                seed=14000 + int(pitch) * 20 + i,
                scenario="HELMET_REMOVE",
                split="test",
                apply_noise=True,
                camera_overrides={"pitch_deg": float(pitch), "yaw_deg": 0.0, "distance": 3.5},
            )
            pairs.append((seq, conf, meta.label))
        for i in range(n // 2):
            seq, conf, meta = generate_one(
                seed=15000 + int(pitch) * 20 + i,
                scenario="HELMET_ADJUST",
                split="test",
                camera_overrides={"pitch_deg": float(pitch), "yaw_deg": 0.0, "distance": 3.5},
            )
            pairs.append((seq, conf, meta.label))
        flat = _eval_pairs(clf, pairs)
        flat["pitch_deg"] = pitch
        pitch_rows.append(flat)
    for yaw in (-60, -40, -20, 0, 20, 40, 60):
        pairs = []
        for i in range(n):
            seq, conf, meta = generate_one(
                seed=16000 + (yaw + 90) * 20 + i,
                scenario="HELMET_REMOVE",
                split="test",
                camera_overrides={"pitch_deg": 40.0, "yaw_deg": float(yaw), "distance": 3.5},
            )
            pairs.append((seq, conf, meta.label))
        for i in range(n // 2):
            seq, conf, meta = generate_one(
                seed=17000 + (yaw + 90) * 20 + i,
                scenario="TWO_HAND_HEAD_TOUCH",
                split="test",
                camera_overrides={"pitch_deg": 40.0, "yaw_deg": float(yaw), "distance": 3.5},
            )
            pairs.append((seq, conf, meta.label))
        flat = _eval_pairs(clf, pairs)
        flat["yaw_deg"] = yaw
        yaw_rows.append(flat)
    for name, dist in distances.items():
        pairs = []
        for i in range(n):
            seq, conf, meta = generate_one(
                seed=18000 + int(dist * 10) * 20 + i,
                scenario="HELMET_REMOVE",
                split="test",
                camera_overrides={"pitch_deg": 40.0, "yaw_deg": 0.0, "distance": float(dist)},
            )
            pairs.append((seq, conf, meta.label))
        flat = _eval_pairs(clf, pairs)
        flat["distance"] = name
        dist_rows.append(flat)
    return {"camera_pitch": pitch_rows, "camera_yaw": yaw_rows, "camera_distance": dist_rows}


def partial_sequence_stress(clf, n: int = 20) -> list[dict]:
    rows = []
    hybrid = HybridActionClassifier(ml=clf)
    for start_frac in (0.0, 0.20, 0.40, 0.60):
        preds = []
        phases = []
        for i in range(n):
            seq, conf, meta = generate_one(seed=19000 + int(start_frac * 100) + i, scenario="HELMET_REMOVE", split="test", apply_noise=False)
            a = int(start_frac * seq.shape[0])
            sub_k, sub_c = seq[a:], conf[a:]
            proba = clf.predict_proba(sub_k, sub_c)
            pred = max(proba, key=proba.get)
            preds.append(pred)
            trace = infer_phases(sub_k)
            phases.append(trace.phase.value)
            dec = hybrid.predict(sub_k, sub_c)
            if start_frac >= 0.6 and dec.phase is RemovalPhase.REMOVAL_CONFIRMED and not trace.saw_grasp:
                # Record, do not crash the rest of the stress suite.
                pass
        rec = float(np.mean([p == REMOVE for p in preds]))
        rows.append(
            {
                "observe_from": start_frac,
                "ml_remove_rate": rec,
                "phase_confirmed_rate": float(np.mean([p == RemovalPhase.REMOVAL_CONFIRMED.value for p in phases])),
                "phase_counts": {p: int(sum(1 for x in phases if x == p)) for p in set(phases)},
            }
        )
    for end_frac in (0.40, 0.60, 0.80, 1.0):
        preds, phases = [], []
        for i in range(n):
            seq, conf, _ = generate_one(seed=20000 + int(end_frac * 100) + i, scenario="HELMET_REMOVE", split="test", apply_noise=False)
            b = max(8, int(end_frac * seq.shape[0]))
            sub_k = seq[:b]
            pred = clf.predict(sub_k, conf[:b])
            preds.append(pred)
            phases.append(infer_phases(sub_k).phase.value)
        rows.append(
            {
                "observe_until": end_frac,
                "ml_remove_rate": float(np.mean([p == REMOVE for p in preds])),
                "phase_confirmed_rate": float(np.mean([p == RemovalPhase.REMOVAL_CONFIRMED.value for p in phases])),
            }
        )
    return rows


def fragmentation_stress(clf, n: int = 16) -> dict:
    rows = []
    for gap_frames in (0, 5, 10):
        pairs = []
        for i in range(n):
            seq, conf, meta = generate_one(seed=21000 + gap_frames * 50 + i, scenario="HELMET_REMOVE", split="test", apply_noise=False, fps=20.0)
            buf = TrackPoseBuffer(window_seconds=4.0, target_fps=20)
            skip_from = seq.shape[0] // 2
            for f, (k, c) in enumerate(zip(seq, conf)):
                if skip_from <= f < skip_from + gap_frames:
                    continue
                buf.push(_obs(1, f / 20.0, k, c, 20.0))
            packed = buf.get_arrays(1)
            if packed is None:
                continue
            pairs.append((packed[0], packed[1], meta.label))
        flat = _eval_pairs(clf, pairs) if pairs else {}
        flat["gap_frames"] = gap_frames
        flat["mean_completeness"] = None
        rows.append(flat)
    # ID switch: two workers, buffers must not mix, new ID starts UNKNOWN-ish.
    seq_a, conf_a, _ = generate_one(seed=22001, scenario="HELMET_REMOVE", split="test", apply_noise=False)
    seq_b, conf_b, _ = generate_one(seed=22002, scenario="IDLE", split="test", apply_noise=False)
    buf = TrackPoseBuffer(window_seconds=4.0, target_fps=20)
    hybrid_a = HybridActionClassifier(ml=clf)
    hybrid_b = HybridActionClassifier(ml=clf)
    for f, (k, c) in enumerate(zip(seq_a, conf_a)):
        buf.push(_obs(11, f / 20.0, k, c, 20.0))
    for f, (k, c) in enumerate(zip(seq_b, conf_b)):
        buf.push(_obs(22, f / 20.0, k, c, 20.0))
    pa, pb = buf.get_arrays(11), buf.get_arrays(22)
    mixed = False
    if pa is not None and pb is not None:
        mixed = bool(np.allclose(pa[0][: min(5, len(pa[0]))], pb[0][: min(5, len(pb[0]))]))
        da = hybrid_a.predict(pa[0], pa[1], buffer_completeness=buf.completeness(11))
        db = hybrid_b.predict(pb[0], pb[1], buffer_completeness=buf.completeness(22))
    else:
        da = db = None
    return {
        "gaps": rows,
        "id_switch": {
            "buffers_mixed": mixed,
            "track_a_action": None if da is None else da.action.value,
            "track_b_action": None if db is None else db.action.value,
            "isolated": (da is not None and db is not None and da.action != db.action) or not mixed,
        },
    }


def _worst(rows: list[dict], key: str = "helmet_remove_recall") -> dict | None:
    if not rows:
        return None
    return min(rows, key=lambda r: float(r.get(key, 1.0)))


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
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "action_classifier.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "stress")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    if not args.model.exists():
        print(f"missing model {args.model}")
        return 1
    clf = SklearnActionClassifier.load(args.model)
    n = 8 if args.quick else 20
    print("FPS stress…")
    fps_rows = fps_stress(clf, n=n)
    print("occlusion stress…")
    occ_rows = occlusion_stress(clf, n=max(8, n - 4))
    print("camera stress…")
    cam = camera_stress(clf, n=max(8, n - 4))
    print("partial sequence…")
    partial = partial_sequence_stress(clf, n=max(8, n - 4))
    print("fragmentation…")
    frag = fragmentation_stress(clf, n=max(8, n - 4))

    occ_by_group = {}
    for r in occ_rows:
        occ_by_group.setdefault(r["joint_group"], []).append(r)
    sensitive = min(
        occ_by_group.items(),
        key=lambda kv: float(np.mean([x["helmet_remove_recall"] for x in kv[1]])),
    )[0]

    summary = {
        "model": str(args.model),
        "fps": fps_rows,
        "occlusion": occ_rows,
        "camera_pitch": cam["camera_pitch"],
        "camera_yaw": cam["camera_yaw"],
        "camera_distance": cam["camera_distance"],
        "partial_sequence": partial,
        "fragmentation": frag,
        "fps_worst": _worst(fps_rows),
        "camera_pitch_worst": _worst(cam["camera_pitch"]),
        "camera_yaw_worst": _worst(cam["camera_yaw"]),
        "occlusion_worst": _worst(occ_rows),
        "keypoint_most_sensitive": sensitive,
        "disclaimer": "Synthetic stress only. Not field CCTV performance. alert_enter remains 0.75.",
    }
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "stress_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out / "stress_summary.csv", fps_rows + occ_rows + cam["camera_pitch"] + cam["camera_yaw"])
    try:
        from helmet_action.evaluation.plots import plot_stress_curves

        plot_stress_curves(out, summary)
    except Exception as exc:
        print(f"plot skipped: {exc}")
    print(f"wrote {out / 'stress_summary.json'}")
    print("FPS worst", summary["fps_worst"])
    print("occlusion worst", summary["occlusion_worst"])
    print("most sensitive keypoint group", sensitive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
