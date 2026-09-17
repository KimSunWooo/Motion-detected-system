"""v0.5 ROI inference pipeline: detect → track → ROI pose → gate → action."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from helmet_action.config import load_config
from helmet_action.inference.capture import FrameCapture, sanitize_fps
from helmet_action.inference.diagnostics import FpsMeter, TrackEventMonitor
from helmet_action.inference.diagnostics_v5 import (
    CallRateMeter,
    PipelineDecisionStatus,
    StageTimer,
    TimedStage,
    TrackRuntimeState,
)
from helmet_action.inference.human_detector import HumanDetection, HumanDetector
from helmet_action.inference.pose_estimator import PoseEstimator, crop_roi
from helmet_action.inference.tracker import HumanTrack, IoUPersonTracker, PersonTracker
from helmet_action.models.hybrid import HybridActionClassifier
from helmet_action.models.labels import ActionClass
from helmet_action.models.temporal_classifier import SklearnActionClassifier
from helmet_action.pose.constants import L_WRIST, R_WRIST, SKELETON_BONES, UPPER_JOINTS
from helmet_action.pose.pose_buffer import TrackPoseBuffer
from helmet_action.pose.quality import compute_pose_quality_score
from helmet_action.pose.types import PoseObservation
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
    if conf >= 0.50:
        return (255, 255, 255)
    if conf >= 0.25:
        return (0, 200, 255)
    return (40, 40, 220)


@dataclass
class PipelineStats:
    camera_fps: float = 0.0
    render_fps: float = 0.0
    detector_fps: float = 0.0
    pose_fps: float = 0.0
    action_fps: float = 0.0
    dropped_frames: int = 0
    active_tracks: float = 0.0
    detector_calls: int = 0
    pose_calls: int = 0
    action_calls: int = 0
    mean_detector_ms: float | None = None
    p95_detector_ms: float | None = None
    mean_pose_ms: float | None = None
    p95_pose_ms: float | None = None
    mean_action_ms: float | None = None
    mean_e2e_ms: float | None = None
    resolution: tuple[int, int] | None = None
    source_fps: float = 0.0
    stage_snapshot: dict[str, dict[str, float]] = field(default_factory=dict)


class CadenceController:
    """FPS-based cadence; falls back to interval=1 when fps invalid."""

    def __init__(self, target_fps: float, source_fps: float) -> None:
        self.target_fps = float(target_fps)
        self.source_fps = max(float(source_fps), 1.0)
        if not np.isfinite(self.target_fps) or self.target_fps <= 0:
            self.interval = 1
        else:
            self.interval = max(1, int(round(self.source_fps / self.target_fps)))
        self._last_run_frame = -10**9

    def should_run(self, frame_index: int) -> bool:
        if frame_index - self._last_run_frame >= self.interval:
            self._last_run_frame = frame_index
            return True
        return False


def is_live_source(source: str | int) -> bool:
    if isinstance(source, int):
        return True
    text = str(source).lower()
    if text.startswith(("http://", "https://", "rtsp://", "rtsps://")):
        return True
    if text.isdigit():
        return True
    return False


def resolve_latest_frame(
    source: str | int,
    *,
    latest_frame: bool | None,
    sequential: bool | None,
) -> bool:
    """LIVE → latest-frame; OFFLINE file → sequential. CLI overrides win."""
    if sequential is True:
        return False
    if latest_frame is True:
        return True
    if latest_frame is False:
        return False
    return is_live_source(source)


class RoiInferencePipeline:
    """Detect (cadenced) → track → ROI pose (cadenced) → quality gate → action."""

    def __init__(
        self,
        detector: HumanDetector,
        tracker: PersonTracker,
        pose_estimator: PoseEstimator,
        *,
        ml: SklearnActionClassifier | None = None,
        hybrid_version: str | None = None,
        detector_fps: float | None = None,
        pose_fps: float | None = None,
        action_fps: float | None = None,
        roi_margin: float | None = None,
        pose_quality_threshold: float | None = None,
        min_buffer_completeness: float | None = None,
        source_fps: float = 20.0,
        action_callable: Callable[..., Any] | None = None,
    ) -> None:
        cfg = load_config()
        self.detector = detector
        self.tracker = tracker
        self.pose_estimator = pose_estimator
        self.version = (hybrid_version or str(cfg.get("decision.hybrid_version", "v1"))).lower()
        self.roi_margin = float(
            roi_margin if roi_margin is not None else cfg.get("inference.pose.roi_margin", 0.15)
        )
        self.pose_quality_threshold = float(
            pose_quality_threshold
            if pose_quality_threshold is not None
            else cfg.get("inference.pose.quality_threshold", 0.45)
        )
        self.min_buffer_completeness = float(
            min_buffer_completeness
            if min_buffer_completeness is not None
            else cfg.get(
                "inference.action.min_buffer_completeness",
                cfg.get("decision.min_buffer_completeness", 0.45),
            )
        )
        self.min_frames = int(cfg.get("window.min_frames", 12))
        det_fps = float(
            detector_fps if detector_fps is not None else cfg.get("inference.detector.fps", 5)
        )
        p_fps = float(pose_fps if pose_fps is not None else cfg.get("inference.pose.fps", 10))
        a_fps = float(action_fps if action_fps is not None else cfg.get("inference.action.fps", 10))
        self.detector_cadence = CadenceController(det_fps, source_fps)
        self.pose_cadence = CadenceController(p_fps, source_fps)
        self.action_cadence = CadenceController(a_fps, source_fps)
        self.source_fps = float(source_fps)

        self.buffers = TrackPoseBuffer()
        self.engines: dict[int, HybridActionClassifier] = {}
        self.ml = ml
        self._action_callable = action_callable
        self._action_call_count = 0
        self._pose_call_count = 0
        self._detector_call_count = 0

        self.timer = StageTimer()
        self.det_rate = CallRateMeter()
        self.pose_rate = CallRateMeter()
        self.action_rate = CallRateMeter()
        self.render_meter = FpsMeter()
        self.camera_meter = FpsMeter()

        self.last_detections: list[HumanDetection] = []
        self.last_obs: dict[int, PoseObservation] = {}
        self.track_states: dict[int, TrackRuntimeState] = {}
        self.last_decisions: dict[int, Any] = {}
        self._active_track_sum = 0
        self._frames = 0
        self.stats = PipelineStats(source_fps=self.source_fps)

        if hasattr(pose_estimator, "source_fps"):
            try:
                pose_estimator.source_fps = self.source_fps  # type: ignore[attr-defined]
            except Exception:
                pass

    @property
    def action_call_count(self) -> int:
        return int(self._action_call_count)

    @property
    def pose_call_count(self) -> int:
        return int(self._pose_call_count)

    @property
    def detector_call_count(self) -> int:
        return int(self._detector_call_count)

    def _engine(self, track_id: int) -> HybridActionClassifier:
        eng = self.engines.get(track_id)
        if eng is None:
            cfg = load_config()
            eng = HybridActionClassifier(
                ml=self.ml,
                helmet_sm=HelmetStateMachine(),
                alert_gate=AlertGate(
                    enter=float(cfg.get("decision.alert_enter", 0.75)),
                    exit=float(cfg.get("decision.alert_exit", 0.45)),
                ),
                version=self.version,
            )
            self.engines[track_id] = eng
        return eng

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float | None = None,
    ) -> tuple[np.ndarray, list[HumanTrack], dict[int, PoseObservation], dict[int, Any]]:
        """Run one inference step. Returns (vis unused raw frame, tracks, obs, decisions)."""
        t_e2e = time.perf_counter()
        ts = float(timestamp if timestamp is not None else frame_index / max(self.source_fps, 1e-6))
        self.camera_meter.tick()
        self._frames += 1
        h, w = int(frame.shape[0]), int(frame.shape[1])
        self.stats.resolution = (w, h)

        # --- Detection (cadenced) ---
        detections: list[HumanDetection] | None
        if self.detector_cadence.should_run(frame_index):
            with TimedStage(self.timer, "detection"):
                detections = self.detector.detect(frame)
            self._detector_call_count += 1
            self.det_rate.tick()
            self.last_detections = detections
        else:
            detections = None  # tracker coasts

        # --- Tracking ---
        with TimedStage(self.timer, "tracking"):
            tracks = self.tracker.update(frame, detections)

        lost = getattr(self.tracker, "lost_track_ids", []) or []
        for tid in lost:
            self.buffers.clear(tid)
            self.engines.pop(tid, None)
            self.last_obs.pop(tid, None)
            self.last_decisions.pop(tid, None)
            self.track_states.pop(tid, None)

        run_pose = self.pose_cadence.should_run(frame_index)
        observations: dict[int, PoseObservation] = {}
        decisions: dict[int, Any] = {}

        for track in tracks:
            st = self.track_states.get(track.track_id)
            if st is None:
                st = TrackRuntimeState(track_id=track.track_id)
                self.track_states[track.track_id] = st
            st.age_frames = track.age_frames
            st.stable = track.stable
            st.detection_confidence = track.confidence

            if not track.stable:
                st.gate = PipelineDecisionStatus.TRACK_WARMUP
                # Pose may still run during warmup (per spec), but action must not.
            else:
                if st.gate == PipelineDecisionStatus.TRACK_WARMUP:
                    st.gate = PipelineDecisionStatus.POSE_MISSING

            obs: PoseObservation | None = None
            if run_pose:
                with TimedStage(self.timer, "roi_crop"):
                    roi, roi_xyxy = crop_roi(frame, track.bbox, margin=self.roi_margin)
                with TimedStage(self.timer, "pose"):
                    # PoseEstimator protocol uses estimate(roi, track, ...); pass roi_xyxy if supported
                    estimate_fn = self.pose_estimator.estimate
                    try:
                        obs = estimate_fn(
                            roi,
                            track,
                            frame_index,
                            ts,
                            roi_xyxy=roi_xyxy,  # type: ignore[call-arg]
                        )
                    except TypeError:
                        obs = estimate_fn(roi, track, frame_index, ts)
                self._pose_call_count += 1
                self.pose_rate.tick()
                if obs is not None:
                    self.last_obs[track.track_id] = obs
                    self.buffers.push(obs)
                    observations[track.track_id] = obs
            else:
                # Hold last skeleton for viz only — do not push fake pose into buffer.
                held = self.last_obs.get(track.track_id)
                if held is not None:
                    observations[track.track_id] = held

            held_obs = observations.get(track.track_id) or self.last_obs.get(track.track_id)
            if held_obs is None:
                if track.stable:
                    st.gate = PipelineDecisionStatus.POSE_MISSING
                continue

            kconf = np.asarray(held_obs.keypoint_confidence, dtype=np.float64).reshape(-1)
            st.last_wrist_l = float(kconf[L_WRIST]) if L_WRIST < len(kconf) else None
            st.last_wrist_r = float(kconf[R_WRIST]) if R_WRIST < len(kconf) else None

            packed = self.buffers.get_arrays(track.track_id)
            raw_len = len(self.buffers.raw(track.track_id))
            st.buffer_frames = raw_len
            completeness = self.buffers.completeness(track.track_id)
            st.buffer_completeness = float(completeness)

            pq = None
            if packed is not None:
                kpts, conf, _ = packed
                score = compute_pose_quality_score(kpts, conf)
                pq = float(score.score)
                st.last_pose_quality = pq
            else:
                score = compute_pose_quality_score(
                    held_obs.keypoints[None, ...],
                    held_obs.keypoint_confidence[None, ...],
                )
                pq = float(score.score)
                st.last_pose_quality = pq

            # --- Action gate ---
            if not track.stable:
                st.gate = PipelineDecisionStatus.TRACK_WARMUP
                continue
            if pq is None or pq < self.pose_quality_threshold:
                st.gate = (
                    PipelineDecisionStatus.POSE_INSUFFICIENT
                    if pq is not None and pq < 0.32
                    else PipelineDecisionStatus.POSE_LOW_CONFIDENCE
                )
                continue
            if raw_len < self.min_frames or completeness < self.min_buffer_completeness:
                st.gate = PipelineDecisionStatus.BUFFER_WARMUP
                continue

            st.gate = PipelineDecisionStatus.ACTION_READY

            # Only run action when a new valid pose entered the buffer this frame
            # (or action cadence allows and we have a fresh pose update).
            pose_updated = track.track_id in observations and run_pose
            if not pose_updated:
                prev = self.last_decisions.get(track.track_id)
                if prev is not None:
                    decisions[track.track_id] = prev
                continue
            if not self.action_cadence.should_run(frame_index):
                prev = self.last_decisions.get(track.track_id)
                if prev is not None:
                    decisions[track.track_id] = prev
                continue

            if packed is None:
                continue
            kpts, conf, _ = packed
            with TimedStage(self.timer, "action"):
                if self._action_callable is not None:
                    decision = self._action_callable(kpts, conf, completeness)
                else:
                    eng = self._engine(track.track_id)
                    if self.buffers.has_long_gap(track.track_id, 0.40):
                        self.engines.pop(track.track_id, None)
                        eng = self._engine(track.track_id)
                    decision = eng.predict(
                        kpts,
                        conf,
                        buffer_completeness=completeness,
                        version=self.version,
                    )
            self._action_call_count += 1
            self.action_rate.tick()
            self.last_decisions[track.track_id] = decision
            decisions[track.track_id] = decision
            st.last_action = (
                decision.action.value
                if isinstance(getattr(decision, "action", None), ActionClass)
                else str(getattr(decision, "action", None))
            )
            st.last_risk_level = str(getattr(decision, "risk_level", None))
            st.last_decision_status = str(getattr(decision, "decision_status", None))

        if not tracks:
            # Clear stale viz for NO_PERSON
            pass

        self._active_track_sum += len(tracks)
        e2e_ms = (time.perf_counter() - t_e2e) * 1000.0
        self.timer.record("e2e", e2e_ms)
        self._refresh_stats()
        return frame, tracks, observations, decisions

    def _refresh_stats(self) -> None:
        snap = self.timer.snapshot()
        self.stats.camera_fps = self._fps_from(self.camera_meter)
        self.stats.detector_fps = self.det_rate.tick(0)
        self.stats.pose_fps = self.pose_rate.tick(0)
        self.stats.action_fps = self.action_rate.tick(0)
        self.stats.detector_calls = self._detector_call_count
        self.stats.pose_calls = self._pose_call_count
        self.stats.action_calls = self._action_call_count
        self.stats.mean_detector_ms = self.timer.mean("detection")
        self.stats.p95_detector_ms = self.timer.p95("detection")
        self.stats.mean_pose_ms = self.timer.mean("pose")
        self.stats.p95_pose_ms = self.timer.p95("pose")
        self.stats.mean_action_ms = self.timer.mean("action")
        self.stats.mean_e2e_ms = self.timer.mean("e2e")
        self.stats.active_tracks = (
            self._active_track_sum / max(self._frames, 1)
        )
        self.stats.stage_snapshot = snap
        self.stats.source_fps = self.source_fps

    @staticmethod
    def _fps_from(meter: FpsMeter) -> float:
        times = meter._times
        if len(times) < 2:
            return 0.0
        dt = times[-1] - times[0]
        if dt <= 1e-6:
            return 0.0
        return float((len(times) - 1) / dt)


def draw_pipeline_overlay(
    frame: np.ndarray,
    tracks: list[HumanTrack],
    observations: dict[int, PoseObservation],
    decisions: dict[int, Any],
    pipeline: RoiInferencePipeline,
    *,
    debug: bool = False,
) -> np.ndarray:
    import cv2

    vis = frame.copy()
    stats = pipeline.stats
    render_fps = pipeline.render_meter.tick()
    stats.render_fps = render_fps

    if not tracks:
        lines = [
            "Diag: NO_PERSON",
            f"Camera FPS: {_fmt(stats.camera_fps, 1)}",
            f"Render FPS: {_fmt(render_fps, 1)}",
            f"Detector FPS: {_fmt(stats.detector_fps, 1)}",
            f"Pose FPS: {_fmt(stats.pose_fps, 1)}",
            f"Action FPS: {_fmt(stats.action_fps, 1)}",
            f"Dropped: {stats.dropped_frames}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (12, 24 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1, cv2.LINE_AA)
        return vis

    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        st = pipeline.track_states.get(track.track_id)
        decision = decisions.get(track.track_id)
        alert = bool(getattr(decision, "alert", False)) if decision is not None else False
        color = (0, 80, 255) if alert else ((80, 200, 80) if track.stable else (0, 200, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        obs = observations.get(track.track_id)
        if obs is not None:
            k = obs.keypoints
            kconf = np.asarray(obs.keypoint_confidence, dtype=np.float64).reshape(-1)
            for a, b in SKELETON_BONES:
                pa, pb = k[a], k[b]
                ca = float(kconf[a]) if a < len(kconf) else 0.0
                cb = float(kconf[b]) if b < len(kconf) else 0.0
                if np.isfinite(pa).all() and np.isfinite(pb).all() and min(ca, cb) >= 0.25:
                    cv2.line(vis, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 224, 94), 2)
            for j in UPPER_JOINTS:
                p = k[j]
                c = float(kconf[j]) if j < len(kconf) else 0.0
                if not np.isfinite(p).all():
                    continue
                cv2.circle(vis, (int(p[0]), int(p[1])), 3, _kp_color(c) if debug else (255, 255, 255), -1)

        gate = st.gate.value if st else "N/A"
        action = getattr(decision, "action", None) if decision is not None else None
        action_s = action.value if isinstance(action, ActionClass) else (str(action) if action else "N/A")
        risk = getattr(decision, "risk_level", None) if decision is not None else None
        if debug:
            lines = [
                f"ID: {track.track_id} age={track.age_frames} stable={track.stable}",
                f"DetConf: {_fmt(track.confidence)} Gate: {gate}",
                f"PoseQ: {_fmt(st.last_pose_quality if st else None)} "
                f"LW: {_fmt(st.last_wrist_l if st else None)} RW: {_fmt(st.last_wrist_r if st else None)}",
                f"Buf: {st.buffer_frames if st else 0} complete={_fmt(st.buffer_completeness if st else None)}",
                f"Action: {action_s} Risk: {_fmt(risk, 0)}",
                f"DecStatus: {_fmt(st.last_decision_status if st else None, 0)}",
            ]
        else:
            lines = [
                f"ID: {track.track_id}",
                f"Action: {action_s}",
                f"Gate: {gate}",
                f"Risk: {_fmt(risk, 0)}",
            ]
        y = max(20, y1 - 12)
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (x1, y + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if debug:
        hdr = [
            f"CamFPS {_fmt(stats.camera_fps, 1)} Render {_fmt(render_fps, 1)} "
            f"Det {_fmt(stats.detector_fps, 1)} Pose {_fmt(stats.pose_fps, 1)} Act {_fmt(stats.action_fps, 1)}",
            f"People {len(pipeline.last_detections)} Tracks {len(tracks)} "
            f"E2E {_fmt(stats.mean_e2e_ms, 1)}ms Dropped {stats.dropped_frames}",
            pipeline.timer.format_line()[:110],
        ]
        for i, line in enumerate(hdr):
            cv2.putText(vis, line, (12, 22 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return vis


def run_roi_video(
    source: str | int,
    *,
    human_model: str | None = None,
    pose_model: str | None = None,
    detector_conf: float | None = None,
    pose_conf: float | None = None,
    detector_imgsz: int | None = None,
    pose_imgsz: int | None = None,
    detector_fps: float | None = None,
    pose_fps: float | None = None,
    action_fps: float | None = None,
    roi_margin: float | None = None,
    hybrid_version: str | None = None,
    rotate: int = 0,
    latest_frame: bool | None = None,
    sequential: bool | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
    max_frames: int | None = None,
    no_record: bool = False,
    debug_overlay: bool = False,
    ml: SklearnActionClassifier | None = None,
    diag_interval_s: float = 1.0,
    print_banner: bool = False,
    detector: HumanDetector | None = None,
    pose_estimator: PoseEstimator | None = None,
    tracker: PersonTracker | None = None,
) -> tuple[Path | None, PipelineStats]:
    """Run the v0.5 ROI pipeline end-to-end."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required") from exc

    from helmet_action.inference.source import redact_source
    from helmet_action.inference.ultralytics_human_detector import UltralyticsHumanDetector
    from helmet_action.inference.ultralytics_roi_pose import UltralyticsRoiPoseEstimator
    from helmet_action.inference.video_pipeline import load_ml_status

    cfg = load_config()
    use_latest = resolve_latest_frame(source, latest_frame=latest_frame, sequential=sequential)
    version = (hybrid_version or str(cfg.get("decision.hybrid_version", "v1"))).lower()

    if ml is None:
        ml, ml_path, feat_ver, ml_err = load_ml_status()
    else:
        ml_path, feat_ver, ml_err = "provided", getattr(ml, "feature_version", None), None

    if print_banner:
        print("=== Runtime Configuration (ROI Pose) ===")
        print(f"Source: {redact_source(source)}")
        print(f"Human Model: {human_model or cfg.get('inference.detector.model')}")
        print(f"Pose Model: {pose_model or cfg.get('inference.pose.model')}")
        print(f"Detector conf/imgsz/fps: {detector_conf} / {detector_imgsz} / {detector_fps}")
        print(f"Pose conf/imgsz/fps: {pose_conf} / {pose_imgsz} / {pose_fps}")
        print(f"ROI margin: {roi_margin}")
        print(f"Latest-frame: {'ON' if use_latest else 'OFF'}")
        print(f"Hybrid: {version}")
        print(f"ML Loaded: {'YES' if ml is not None else 'NO'} ({ml_path})")
        if ml is None:
            print(f"Reason: {ml_err}")
        print(f"Feature Version: {feat_ver}")

    capture: FrameCapture | None = None
    writer = None
    dest = Path(out_path) if (out_path and not no_record) else None
    monitor = TrackEventMonitor()
    last_diag_t = 0.0
    n = 0

    try:
        capture = FrameCapture(source, rotate=rotate, latest_frame=use_latest)
        capture.start()
        source_fps = capture.source_fps
        writer_fps = source_fps
        print(f"Source FPS: {capture.raw_fps_metadata if np.isfinite(capture.raw_fps_metadata) else 'N/A'}")
        print(f"Writer FPS: {writer_fps}")

        if detector is None:
            detector = UltralyticsHumanDetector(
                model_path=human_model,
                conf=detector_conf,
                imgsz=detector_imgsz,
            )
        if pose_estimator is None:
            pose_estimator = UltralyticsRoiPoseEstimator(
                model_path=pose_model,
                conf=pose_conf,
                imgsz=pose_imgsz,
                source_fps=source_fps,
            )
        if tracker is None:
            tracker = IoUPersonTracker()

        pipe = RoiInferencePipeline(
            detector,
            tracker,
            pose_estimator,
            ml=ml,
            hybrid_version=version,
            detector_fps=detector_fps,
            pose_fps=pose_fps,
            action_fps=action_fps,
            roi_margin=roi_margin,
            source_fps=source_fps,
        )

        while True:
            item = capture.read()
            if item is None:
                break
            frame = item.frame
            frame_index = int(item.index)
            ts = frame_index / max(source_fps, 1e-6)

            with TimedStage(pipe.timer, "capture"):
                pass  # already captured

            _, tracks, observations, decisions = pipe.process_frame(frame, frame_index, ts)
            pipe.stats.dropped_frames = capture.dropped_frames()

            track_ids = [t.track_id for t in tracks]
            events = monitor.update(track_ids, frame_index=frame_index, timestamp=ts)
            for ev in events:
                if debug_overlay:
                    print(
                        f"[TRACK] {ev.kind} id={ev.track_id} frame={ev.frame_index} "
                        f"age={ev.age} last={ev.last_seen_frame} gap={ev.gap_duration:.2f} {ev.detail}",
                        flush=True,
                    )

            with TimedStage(pipe.timer, "render"):
                vis = draw_pipeline_overlay(
                    frame, tracks, observations, decisions, pipe, debug=debug_overlay
                )

            if writer is None and dest is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                hh, ww = vis.shape[:2]
                writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), float(writer_fps), (ww, hh))
            if writer is not None:
                writer.write(vis)
            if show:
                cv2.imshow("helmet-action-roi", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            now = time.monotonic()
            if debug_overlay and (now - last_diag_t) >= float(diag_interval_s):
                last_diag_t = now
                print(
                    f"[PIPE] tracks={track_ids} det_calls={pipe.detector_call_count} "
                    f"pose_calls={pipe.pose_call_count} action_calls={pipe.action_call_count} "
                    f"cam={_fmt(pipe.stats.camera_fps,1)} render={_fmt(pipe.stats.render_fps,1)} "
                    f"dropped={pipe.stats.dropped_frames} | {pipe.timer.format_line()}",
                    flush=True,
                )

            n += 1
            if max_frames is not None and n >= max_frames:
                break

        return dest, pipe.stats
    finally:
        if capture is not None:
            capture.close()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
