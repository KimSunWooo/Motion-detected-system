from __future__ import annotations

from pathlib import Path

import numpy as np

from helmet_action.config import load_config
from helmet_action.inference.pose_provider import PoseProvider
from helmet_action.models.hybrid import HybridActionClassifier
from helmet_action.pose.constants import SKELETON_BONES, L_WRIST, R_WRIST, UPPER_JOINTS
from helmet_action.pose.geometry import compute_head_regions
from helmet_action.pose.normalizer import normalize_keypoints
from helmet_action.pose.pose_buffer import TrackPoseBuffer
from helmet_action.state.helmet_state import AlertGate, DummyHelmetPresenceDetector, HelmetStateMachine


def _draw(frame, obs, decision, seq_norm_last=None):
    import cv2

    x1, y1, x2, y2 = [int(v) for v in obs.bbox]
    color = (0, 80, 255) if decision.alert else (80, 200, 80)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    k = obs.keypoints
    for a, b in SKELETON_BONES:
        pa, pb = k[a], k[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 224, 94), 2)
    for j in UPPER_JOINTS:
        p = k[j]
        if np.isfinite(p).all():
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1)
    if seq_norm_last is not None:
        rg = compute_head_regions(seq_norm_last)
        # skip drawing normalized regions in pixel space — overlay text only
    risk = "HIGH" if decision.alert or decision.risk >= 0.75 else ("WATCH" if decision.risk >= 0.45 else "LOW")
    lines = [
        f"ID: {obs.track_id}",
        f"Action: {decision.action.value}",
        f"Action Confidence: {decision.confidence:.2f}",
        f"Helmet: {decision.helmet_state.value}",
        f"Risk: {risk}",
        f"Phase: {decision.phase.value}",
    ]
    y = max(20, y1 - 12)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x1, y + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return frame


def run_video(
    source: str | int,
    provider: PoseProvider,
    out_path: str | Path | None = None,
    show: bool = False,
    max_frames: int | None = None,
) -> Path | None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for run_video") from exc

    cfg = load_config()
    buffers = TrackPoseBuffer()
    engines: dict[int, HybridActionClassifier] = {}
    writer = None
    dest = Path(out_path) if out_path else None
    n = 0
    for frame, observations in provider.iter_frames(source):
        if writer is None and dest is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        vis = frame.copy()
        for obs in observations:
            buffers.push(obs)
            packed = buffers.get_arrays(obs.track_id)
            if packed is None:
                continue
            kpts, conf, _ = packed
            min_frames = int(cfg.get("window.min_frames", 12))
            if kpts.shape[0] < min_frames:
                continue
            eng = engines.get(obs.track_id)
            if eng is None:
                eng = HybridActionClassifier(
                    helmet_sm=HelmetStateMachine(),
                    alert_gate=AlertGate(
                        enter=float(cfg.get("decision.alert_enter", 0.75)),
                        exit=float(cfg.get("decision.alert_exit", 0.45)),
                    ),
                )
                engines[obs.track_id] = eng
            decision = eng.predict(kpts, conf)
            seq_norm, _ = normalize_keypoints(kpts)
            vis = _draw(vis, obs, decision, seq_norm[-1])
        if writer is not None:
            writer.write(vis)
        if show:
            cv2.imshow("helmet-action", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    return dest
