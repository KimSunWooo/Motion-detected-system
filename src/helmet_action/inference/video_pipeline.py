from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from helmet_action.config import load_config
from helmet_action.inference.capture import FrameCapture, sanitize_fps
from helmet_action.inference.diagnostics import (
    FpsMeter,
    PoseDiagState,
    TrackEventMonitor,
    classify_pose_diag,
    person_pose_diag,
)
from helmet_action.inference.pose_provider import PoseProvider, NumpySequenceProvider
from helmet_action.models.hybrid import HybridActionClassifier
from helmet_action.models.labels import ActionClass
from helmet_action.models.temporal_classifier import SklearnActionClassifier
from helmet_action.pose.constants import (
    L_EAR,
    L_ELBOW,
    L_SHOULDER,
    L_WRIST,
    R_EAR,
    R_ELBOW,
    R_SHOULDER,
    R_WRIST,
    SKELETON_BONES,
    UPPER_JOINTS,
)
from helmet_action.pose.normalizer import normalize_keypoints
from helmet_action.pose.pose_buffer import TrackPoseBuffer
from helmet_action.pose.quality import compute_pose_quality_score
from helmet_action.state.helmet_state import AlertGate, HelmetStateMachine


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        if isinstance(value, float):
            if not np.isfinite(value):
                return "N/A"
            return f"{value:.{digits}f}"
        return str(value)
    except Exception:
        return "N/A"


def _kp_color(conf: float) -> tuple[int, int, int]:
    # BGR
    if conf >= 0.50:
        return (255, 255, 255)
    if conf >= 0.25:
        return (0, 200, 255)  # low
    return (40, 40, 220)  # weak / missing-ish


def _draw(
    frame,
    obs,
    decision=None,
    *,
    debug: bool = False,
    hybrid_version: str = "v1",
    pose_quality: float | None = None,
    wrist_quality: float | None = None,
    ear_quality: float | None = None,
    infer_fps: float | None = None,
    source_fps: float | None = None,
    diag_state: str | None = None,
):
    import cv2

    x1, y1, x2, y2 = [int(v) for v in obs.bbox]
    alert = bool(getattr(decision, "alert", False)) if decision is not None else False
    risk_v = float(getattr(decision, "risk", 0.0) or 0.0) if decision is not None else 0.0
    color = (0, 80, 255) if alert else (80, 200, 80)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    k = obs.keypoints
    kconf = np.asarray(obs.keypoint_confidence, dtype=np.float64).reshape(-1)
    for a, b in SKELETON_BONES:
        pa, pb = k[a], k[b]
        ca = float(kconf[a]) if a < len(kconf) else 0.0
        cb = float(kconf[b]) if b < len(kconf) else 0.0
        if np.isfinite(pa).all() and np.isfinite(pb).all() and min(ca, cb) >= 0.25:
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 224, 94), 2)
    for j in UPPER_JOINTS:
        p = k[j]
        c = float(kconf[j]) if j < len(kconf) else 0.0
        if not np.isfinite(p).all():
            continue
        if debug and c < 0.25:
            # omit very weak points in debug, mark with tiny dark dot optionally
            cv2.circle(frame, (int(p[0]), int(p[1])), 2, _kp_color(c), -1)
            continue
        cv2.circle(frame, (int(p[0]), int(p[1])), 3 if not debug else 4, _kp_color(c) if debug else (255, 255, 255), -1)

    risk_level = getattr(decision, "risk_level", None) if decision is not None else None
    if risk_level is None:
        risk_level = "HIGH" if alert or risk_v >= 0.75 else ("WATCH" if risk_v >= 0.45 else "LOW")
    action = getattr(decision, "action", None)
    action_s = action.value if isinstance(action, ActionClass) else (str(action) if action else "N/A")
    phase = getattr(getattr(decision, "phase", None), "value", None) if decision is not None else None
    helmet = getattr(getattr(decision, "helmet_state", None), "value", None) if decision is not None else None
    conf = getattr(decision, "confidence", None) if decision is not None else None
    status = getattr(decision, "decision_status", None) if decision is not None else None
    ml_p = None
    if decision is not None and getattr(decision, "ml_proba", None):
        ml_p = float(decision.ml_proba.get(ActionClass.HELMET_REMOVE.value, 0.0))

    lw = float(kconf[L_WRIST]) if L_WRIST < len(kconf) else None
    rw = float(kconf[R_WRIST]) if R_WRIST < len(kconf) else None

    if debug:
        lines = [
            f"ID: {obs.track_id}",
            f"Action: {action_s}",
            f"Hybrid: {str(hybrid_version).upper()}",
            f"Risk Level: {_fmt(risk_level, 0)}",
            f"Phase: {_fmt(phase, 0)}",
            f"Det Conf: {_fmt(obs.detection_confidence)}",
            f"Pose Quality: {_fmt(pose_quality)}",
            f"Wrist Quality: {_fmt(wrist_quality)}",
            f"Ear Quality: {_fmt(ear_quality)}",
            f"Decision Status: {_fmt(status, 0)}",
            f"Diag: {_fmt(diag_state, 0)}",
            f"ML P(remove): {_fmt(ml_p)}",
            f"LW: {_fmt(lw)}  RW: {_fmt(rw)}",
            f"Inference FPS: {_fmt(infer_fps, 1)}",
            f"Source FPS: {_fmt(source_fps, 1)}",
        ]
    else:
        lines = [
            f"ID: {obs.track_id}",
            f"Action: {action_s}",
            f"Action Confidence: {_fmt(conf)}",
            f"Helmet: {_fmt(helmet, 0)}",
            f"Risk: {_fmt(risk_level, 0)}",
            f"Phase: {_fmt(phase, 0)}",
        ]
    y = max(20, y1 - 12)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x1, y + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return frame


def _draw_no_person(frame, *, diag_state: str, infer_fps: float | None, source_fps: float | None):
    import cv2

    lines = [
        f"Diag: {diag_state}",
        f"Inference FPS: {_fmt(infer_fps, 1)}",
        f"Source FPS: {_fmt(source_fps, 1)}",
        "NO_PERSON",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, 24 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1, cv2.LINE_AA)
    return frame


def print_runtime_banner(
    *,
    source: str | int,
    pose_model: str | None,
    pose_conf: float,
    imgsz: int | None,
    rotate: int,
    tracking: bool,
    latest_frame: bool,
    ml_path: str | None,
    ml_loaded: bool,
    ml_reason: str | None,
    feature_version: str | None,
    hybrid_version: str,
) -> None:
    from helmet_action.inference.source import redact_source

    print("=== Runtime Configuration ===")
    print()
    print("Source:")
    print(redact_source(source))
    print()
    print("Pose Model:")
    print(pose_model or "N/A")
    print()
    print("Pose Confidence:")
    print(pose_conf)
    print()
    print("Image Size:")
    print(imgsz if imgsz is not None else "default")
    print()
    print("Rotation:")
    print(rotate)
    print()
    print("Tracking:")
    print("ON" if tracking else "OFF")
    print()
    print("Latest-frame:")
    print("ON" if latest_frame else "OFF")
    print()
    print("ML Model:")
    print(ml_path or "N/A")
    print()
    print("ML Loaded:")
    print("YES" if ml_loaded else "NO")
    if not ml_loaded:
        print(f"Reason: {ml_reason or 'unknown'}")
        print("Running rule/phase only.")
    print()
    print("Feature Version:")
    print(feature_version or "N/A")
    print()
    print("Hybrid Version:")
    print(hybrid_version)
    print()


def load_ml_status(model_path: str | Path | None = None) -> tuple[SklearnActionClassifier | None, str | None, str | None, str | None]:
    """Return (clf, path, feature_version, error_reason)."""
    from helmet_action.config import repo_root

    path = Path(model_path) if model_path else (repo_root() / "models" / "action_classifier.joblib")
    path_s = str(path)
    if not path.exists():
        return None, path_s, None, f"model file not found: {path_s}"
    try:
        clf = SklearnActionClassifier.load(path)
        return clf, path_s, getattr(clf, "feature_version", None), None
    except Exception as exc:
        return None, path_s, None, str(exc)


def run_video(
    source: str | int,
    provider: PoseProvider,
    out_path: str | Path | None = None,
    show: bool = False,
    max_frames: int | None = None,
    *,
    rotate: int = 0,
    latest_frame: bool = False,
    hybrid_version: str | None = None,
    debug_overlay: bool = False,
    no_record: bool = False,
    record: bool = True,
    diag_interval_s: float = 1.0,
    ml: SklearnActionClassifier | None = None,
    print_banner: bool = False,
) -> Path | None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for run_video") from exc

    cfg = load_config()
    version = (hybrid_version or str(cfg.get("decision.hybrid_version", "v1"))).lower()
    do_record = bool(record) and (not no_record) and out_path is not None
    dest = Path(out_path) if (do_record and out_path) else None

    buffers = TrackPoseBuffer()
    engines: dict[int, HybridActionClassifier] = {}
    monitor = TrackEventMonitor()
    infer_meter = FpsMeter()
    writer = None
    writer_fps = 20.0
    source_fps = 20.0
    n = 0
    last_diag_t = 0.0

    pose_model = getattr(provider, "model_path", None)
    pose_conf = float(getattr(provider, "conf", cfg.get("pose.confidence_threshold", 0.35)))
    imgsz = getattr(provider, "imgsz", None)
    tracking = bool(getattr(provider, "track", True))

    if ml is None:
        ml, ml_path, feat_ver, ml_err = load_ml_status()
        ml_loaded = ml is not None
    else:
        ml_path = "provided"
        feat_ver = getattr(ml, "feature_version", None)
        ml_err = None
        ml_loaded = True

    if print_banner:
        print_runtime_banner(
            source=source,
            pose_model=str(pose_model) if pose_model else None,
            pose_conf=pose_conf,
            imgsz=int(imgsz) if imgsz is not None else None,
            rotate=int(rotate),
            tracking=tracking,
            latest_frame=latest_frame,
            ml_path=ml_path,
            ml_loaded=ml_loaded,
            ml_reason=ml_err,
            feature_version=feat_ver,
            hybrid_version=version,
        )

    capture: FrameCapture | None = None
    frame_iter = None
    try:
        # Prefer FrameCapture path for rotate / latest-frame / FPS logging.
        use_capture = not isinstance(provider, NumpySequenceProvider)
        try:
            if not use_capture:
                raise RuntimeError("synthetic provider")
            capture = FrameCapture(source, rotate=rotate, latest_frame=latest_frame)
            capture.start()
            source_fps = capture.source_fps
            writer_fps = source_fps
            print(f"Source FPS: {capture.raw_fps_metadata if np.isfinite(capture.raw_fps_metadata) else 'N/A'}")
            print(f"Writer FPS: {writer_fps}")
        except Exception:
            # Fall back to provider.iter_frames for synthetic NumpySequenceProvider etc.
            use_capture = False
            capture = None
            frame_iter = provider.iter_frames(source)

        def _engine(track_id: int) -> HybridActionClassifier:
            eng = engines.get(track_id)
            if eng is None:
                eng = HybridActionClassifier(
                    ml=ml,
                    helmet_sm=HelmetStateMachine(),
                    alert_gate=AlertGate(
                        enter=float(cfg.get("decision.alert_enter", 0.75)),
                        exit=float(cfg.get("decision.alert_exit", 0.45)),
                    ),
                    version=version,
                )
                engines[track_id] = eng
            return eng

        while True:
            if use_capture and capture is not None:
                item = capture.read()
                if item is None:
                    break
                frame = item.frame
                frame_index = int(item.index)
                observations = provider.infer_frame(frame, frame_index=frame_index, fps=source_fps)
                dropped = capture.dropped_frames()
                buf_depth = capture.buffer_depth()
            else:
                try:
                    frame, observations = next(frame_iter)  # type: ignore[arg-type]
                except StopIteration:
                    break
                if rotate:
                    from helmet_action.inference.capture import rotate_frame

                    frame = rotate_frame(frame, rotate)
                frame_index = n
                source_fps = float(observations[0].source_fps) if observations else 20.0
                writer_fps = sanitize_fps(source_fps)
                dropped = 0
                buf_depth = 0

            infer_fps = infer_meter.tick()
            vis = frame.copy()
            if writer is None and dest is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), float(writer_fps), (w, h))

            track_ids = [int(o.track_id) for o in observations]
            ts = frame_index / max(source_fps, 1e-6)
            events = monitor.update(track_ids, frame_index=frame_index, timestamp=ts)
            for ev in events:
                if debug_overlay:
                    print(
                        f"[TRACK] {ev.kind} id={ev.track_id} frame={ev.frame_index} "
                        f"age={ev.age} last={ev.last_seen_frame} gap={ev.gap_duration:.2f} {ev.detail}",
                        flush=True,
                    )

            if not observations:
                if debug_overlay:
                    vis = _draw_no_person(
                        vis,
                        diag_state=PoseDiagState.NO_PERSON.value,
                        infer_fps=infer_fps,
                        source_fps=source_fps,
                    )
            for obs in observations:
                buffers.push(obs)
                packed = buffers.get_arrays(obs.track_id)
                decision = None
                pq = None
                wq = None
                eq = None
                diag = PoseDiagState.POSE_VALID
                if packed is not None:
                    kpts, conf, _ = packed
                    min_frames = int(cfg.get("window.min_frames", 12))
                    if kpts.shape[0] >= min_frames:
                        if buffers.has_long_gap(obs.track_id, 0.40):
                            engines.pop(obs.track_id, None)
                        eng = _engine(obs.track_id)
                        decision = eng.predict(
                            kpts,
                            conf,
                            buffer_completeness=buffers.completeness(obs.track_id),
                            version=version,
                        )
                        pq = float(getattr(decision, "pose_quality", 0.0))
                        if decision.evidence is not None:
                            wq = float(decision.evidence.wrist_quality)
                            eq = float(decision.evidence.ear_quality)
                        else:
                            score = compute_pose_quality_score(kpts, conf)
                            wq = float(score.wrist_quality)
                            eq = float(score.ear_quality)
                        diag = classify_pose_diag(
                            1,
                            pose_quality=pq,
                            decision_status=decision.decision_status,
                            action=decision.action,
                        )
                    else:
                        score = compute_pose_quality_score(obs.keypoints[None, ...], obs.keypoint_confidence[None, ...])
                        pq, wq, eq = float(score.score), float(score.wrist_quality), float(score.ear_quality)
                        diag = PoseDiagState.POSE_LOW_CONFIDENCE if pq < 0.52 else PoseDiagState.POSE_VALID
                vis = _draw(
                    vis,
                    obs,
                    decision,
                    debug=debug_overlay,
                    hybrid_version=version,
                    pose_quality=pq,
                    wrist_quality=wq,
                    ear_quality=eq,
                    infer_fps=infer_fps,
                    source_fps=source_fps,
                    diag_state=diag.value,
                )

            now = infer_meter._times[-1] if infer_meter._times else 0.0
            if debug_overlay and (now - last_diag_t) >= float(diag_interval_s):
                last_diag_t = now
                if not observations:
                    print(
                        f"[POSE] people=0 tracks=[] det=N/A kp_mean=N/A lw=N/A rw=N/A "
                        f"infer_fps={_fmt(infer_fps, 1)} source_fps={_fmt(source_fps, 1)} "
                        f"dropped={dropped} buf={buf_depth}",
                        flush=True,
                    )
                else:
                    for obs in observations:
                        d = person_pose_diag(obs, frame.shape[:2])
                        print(
                            f"[POSE] people={len(observations)} tracks={track_ids} "
                            f"det={_fmt(d.detection_confidence)} kp_mean={_fmt(d.keypoint_mean)} "
                            f"lw={_fmt(d.wrist_l)} rw={_fmt(d.wrist_r)} "
                            f"sh=({_fmt(d.shoulder_l)},{_fmt(d.shoulder_r)}) "
                            f"bbox_area={_fmt(d.bbox_area_ratio, 4)} aspect={_fmt(d.bbox_aspect_ratio)} "
                            f"vis_kp={d.visible_keypoints} "
                            f"infer_fps={_fmt(infer_fps, 1)} source_fps={_fmt(source_fps, 1)} "
                            f"dropped={dropped} buf={buf_depth}",
                            flush=True,
                        )

            if writer is not None:
                writer.write(vis)
            if show:
                cv2.imshow("helmet-action", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            n += 1
            if max_frames is not None and n >= max_frames:
                break
    finally:
        if capture is not None:
            capture.close()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    return dest
