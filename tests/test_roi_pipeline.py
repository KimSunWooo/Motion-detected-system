"""v0.5 ROI inference architecture unit tests (no YOLO required for core gates)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from helmet_action.inference.capture import LatestFrameBuffer, LatestFrameCapture
from helmet_action.inference.diagnostics_v5 import PipelineDecisionStatus
from helmet_action.inference.human_detector import (
    HumanDetection,
    downscale_frame,
    rescale_bbox,
)
from helmet_action.inference.pipeline import (
    CadenceController,
    RoiInferencePipeline,
    draw_pipeline_overlay,
    is_live_source,
    resolve_latest_frame,
)
from helmet_action.inference.pose_estimator import (
    crop_roi,
    expand_bbox,
    local_to_global_keypoints,
)
from helmet_action.inference.source import parse_video_source
from helmet_action.inference.tracker import HumanTrack, IoUPersonTracker
from helmet_action.pose.types import PoseObservation


cv2 = pytest.importorskip("cv2")


class FakeDetector:
    def __init__(self, detections: list[HumanDetection] | None = None) -> None:
        self.detections = detections or []
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[HumanDetection]:
        self.calls += 1
        return list(self.detections)


class FakePose:
    def __init__(self, obs: PoseObservation | None = None, *, always: bool = True) -> None:
        self.obs = obs
        self.always = always
        self.calls = 0
        self.call_rois: list[np.ndarray] = []

    def estimate(self, roi, track, frame_index, timestamp, **kwargs):
        self.calls += 1
        self.call_rois.append(roi)
        if not self.always:
            return None
        if self.obs is not None:
            return PoseObservation(
                timestamp=timestamp,
                frame_index=frame_index,
                track_id=track.track_id,
                bbox=track.bbox,
                keypoints=self.obs.keypoints.copy(),
                keypoint_confidence=self.obs.keypoint_confidence.copy(),
                detection_confidence=track.confidence,
                source_fps=20.0,
            )
        roi_xyxy = kwargs.get("roi_xyxy", (0, 0, 10, 10))
        x1, y1, _, _ = roi_xyxy
        k = np.zeros((17, 2), dtype=np.float64)
        for i in range(17):
            k[i] = [x1 + 10 + i, y1 + 20 + i]
        conf = np.ones(17, dtype=np.float64) * 0.9
        return PoseObservation(
            timestamp=timestamp,
            frame_index=frame_index,
            track_id=track.track_id,
            bbox=track.bbox,
            keypoints=k,
            keypoint_confidence=conf,
            detection_confidence=track.confidence,
            source_fps=20.0,
        )


def _good_obs(track_id: int = 1) -> PoseObservation:
    k = np.zeros((17, 2), dtype=np.float64)
    k[5] = [40, 50]
    k[6] = [80, 50]
    k[9] = [30, 90]
    k[10] = [90, 90]
    k[0] = [60, 20]
    conf = np.ones(17, dtype=np.float64) * 0.95
    return PoseObservation(
        timestamp=0.0,
        frame_index=0,
        track_id=track_id,
        bbox=(20, 10, 100, 120),
        keypoints=k,
        keypoint_confidence=conf,
        detection_confidence=0.9,
        source_fps=20.0,
    )


def test_latest_frame_buffer_size_leq_one():
    buf = LatestFrameBuffer()
    for i in range(20):
        buf.put(np.full((2, 2, 3), i, dtype=np.uint8), i, float(i))
        assert buf.depth() <= 1
    assert buf.dropped == 19


def test_capture_thread_shutdown(tmp_path: Path):
    path = tmp_path / "tiny.mp4"
    w, h, n = 64, 48, 30
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for i in range(n):
        frame = np.full((h, w, 3), i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    from helmet_action.inference.capture import FrameCapture

    # Exercise threaded latest-frame capture + clean shutdown.
    with FrameCapture(str(path), latest_frame=True) as cap:
        item = None
        for _ in range(20):
            item = cap.read()
            if item is not None:
                break
            time.sleep(0.02)
        assert item is not None
        assert cap.buffer_depth() <= 1
    assert cap._thread is None or not cap._thread.is_alive()
    # LatestFrameCapture wrapper also shuts down cleanly.
    wrap = LatestFrameCapture(str(path))
    wrap.start()
    wrap.close()
    assert wrap._inner._thread is None or not wrap._inner._thread.is_alive()


def test_detector_cadence():
    det = FakeDetector([HumanDetection(bbox=(10, 10, 50, 80), confidence=0.8)])
    pose = FakePose()
    tracker = IoUPersonTracker(min_stable_frames=2, max_missing_frames=5)
    pipe = RoiInferencePipeline(
        det,
        tracker,
        pose,
        detector_fps=5,
        pose_fps=20,
        action_fps=20,
        source_fps=20,
        pose_quality_threshold=0.0,
        min_buffer_completeness=0.0,
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    for i in range(12):
        pipe.process_frame(frame, i, i / 20.0)
    assert det.calls == 3
    assert pipe.detector_cadence.interval == 4


def test_pose_only_for_human_track_and_no_person_zero_pose():
    det = FakeDetector([])
    pose = FakePose()
    tracker = IoUPersonTracker(min_stable_frames=1, max_missing_frames=3)
    pipe = RoiInferencePipeline(
        det, tracker, pose, detector_fps=20, pose_fps=20, source_fps=20
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(5):
        _, tracks, _, _ = pipe.process_frame(frame, i, float(i))
        assert tracks == []
    assert pose.calls == 0
    assert pipe.pose_call_count == 0


def test_background_only_pose_not_invoked():
    det = FakeDetector([])
    pose = FakePose()
    tracker = IoUPersonTracker()
    pipe = RoiInferencePipeline(det, tracker, pose, detector_fps=30, pose_fps=30, source_fps=30)
    frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    for i in range(10):
        pipe.process_frame(frame, i, float(i))
    assert pose.calls == 0


def test_roi_coordinates_mapped_to_global():
    local = np.array([[5.0, 7.0], [1.0, 2.0]], dtype=np.float64)
    global_k = local_to_global_keypoints(local, (100, 200, 300, 400))
    assert global_k[0, 0] == pytest.approx(105.0)
    assert global_k[0, 1] == pytest.approx(207.0)
    assert global_k[1, 0] == pytest.approx(101.0)
    assert global_k[1, 1] == pytest.approx(202.0)


def test_roi_margin_clamp():
    xyxy = expand_bbox((2, 2, 20, 40), margin=0.5, frame_w=50, frame_h=60)
    x1, y1, x2, y2 = xyxy
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 50 and y2 <= 60
    assert x2 > x1 and y2 > y1

    frame = np.zeros((60, 50, 3), dtype=np.uint8)
    roi, box = crop_roi(frame, (2, 2, 20, 40), margin=0.5)
    assert roi.shape[0] == box[3] - box[1]
    assert roi.shape[1] == box[2] - box[0]


def test_detector_downscale_bbox_rescale():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    resized, sx, sy = downscale_frame(frame, 640)
    assert max(resized.shape[0], resized.shape[1]) == 640
    bbox = rescale_bbox((10.0, 20.0, 30.0, 40.0), scale_x=sx, scale_y=sy)
    assert bbox[0] == pytest.approx(10.0 * sx)
    assert bbox[1] == pytest.approx(20.0 * sy)
    assert bbox[2] == pytest.approx(30.0 * sx)
    assert bbox[3] == pytest.approx(40.0 * sy)


def test_unstable_track_action_call_count_zero():
    det = FakeDetector([HumanDetection(bbox=(10, 10, 80, 120), confidence=0.9)])
    pose = FakePose(_good_obs())
    action_calls: list[int] = []

    def action_fn(kpts, conf, completeness):
        action_calls.append(1)
        return type(
            "D",
            (),
            {"action": None, "risk_level": "UNKNOWN", "decision_status": "UNKNOWN", "alert": False},
        )()

    tracker = IoUPersonTracker(min_stable_frames=50, max_missing_frames=10)
    pipe = RoiInferencePipeline(
        det,
        tracker,
        pose,
        detector_fps=30,
        pose_fps=30,
        action_fps=30,
        source_fps=30,
        action_callable=action_fn,
        pose_quality_threshold=0.0,
        min_buffer_completeness=0.0,
    )
    pipe.min_frames = 1
    frame = np.zeros((160, 120, 3), dtype=np.uint8)
    for i in range(10):
        pipe.process_frame(frame, i, float(i) / 30.0)
        st = list(pipe.track_states.values())
        if st:
            assert st[0].gate == PipelineDecisionStatus.TRACK_WARMUP
    assert len(action_calls) == 0


def test_low_pose_quality_action_skipped():
    det = FakeDetector([HumanDetection(bbox=(10, 10, 80, 120), confidence=0.9)])
    k = np.zeros((17, 2), dtype=np.float64)
    conf = np.ones(17, dtype=np.float64) * 0.05
    bad = PoseObservation(
        timestamp=0.0,
        frame_index=0,
        track_id=1,
        bbox=(10, 10, 80, 120),
        keypoints=k,
        keypoint_confidence=conf,
        detection_confidence=0.9,
    )
    pose = FakePose(bad)
    action_calls: list[int] = []

    def action_fn(*a, **k):
        action_calls.append(1)
        return type(
            "D",
            (),
            {"action": None, "risk_level": "UNKNOWN", "decision_status": "UNKNOWN", "alert": False},
        )()

    tracker = IoUPersonTracker(min_stable_frames=2, max_missing_frames=10)
    pipe = RoiInferencePipeline(
        det,
        tracker,
        pose,
        detector_fps=30,
        pose_fps=30,
        action_fps=30,
        source_fps=30,
        action_callable=action_fn,
        pose_quality_threshold=0.8,
        min_buffer_completeness=0.0,
    )
    pipe.min_frames = 1
    frame = np.zeros((160, 120, 3), dtype=np.uint8)
    for i in range(8):
        pipe.process_frame(frame, i, float(i) / 30.0)
    assert len(action_calls) == 0
    gates = {st.gate for st in pipe.track_states.values()}
    assert (
        PipelineDecisionStatus.POSE_LOW_CONFIDENCE in gates
        or PipelineDecisionStatus.POSE_INSUFFICIENT in gates
        or PipelineDecisionStatus.TRACK_WARMUP in gates
    )


def test_insufficient_buffer_action_skipped():
    det = FakeDetector([HumanDetection(bbox=(10, 10, 80, 120), confidence=0.9)])
    pose = FakePose(_good_obs())
    action_calls: list[int] = []

    def action_fn(*a, **k):
        action_calls.append(1)
        return type(
            "D",
            (),
            {"action": None, "risk_level": "SAFE", "decision_status": "VALID", "alert": False},
        )()

    tracker = IoUPersonTracker(min_stable_frames=2, max_missing_frames=10)
    pipe = RoiInferencePipeline(
        det,
        tracker,
        pose,
        detector_fps=30,
        pose_fps=30,
        action_fps=30,
        source_fps=30,
        action_callable=action_fn,
        pose_quality_threshold=0.0,
        min_buffer_completeness=0.0,
    )
    pipe.min_frames = 100
    frame = np.zeros((160, 120, 3), dtype=np.uint8)
    for i in range(10):
        pipe.process_frame(frame, i, float(i) / 30.0)
    assert len(action_calls) == 0
    assert any(st.gate == PipelineDecisionStatus.BUFFER_WARMUP for st in pipe.track_states.values())


def test_stable_track_valid_pose_action_runs():
    det = FakeDetector([HumanDetection(bbox=(10, 10, 80, 120), confidence=0.9)])
    pose = FakePose(_good_obs())
    action_calls: list[int] = []

    def action_fn(*a, **k):
        action_calls.append(1)
        return type(
            "D",
            (),
            {"action": None, "risk_level": "SAFE", "decision_status": "VALID", "alert": False},
        )()

    tracker = IoUPersonTracker(min_stable_frames=3, max_missing_frames=10)
    pipe = RoiInferencePipeline(
        det,
        tracker,
        pose,
        detector_fps=30,
        pose_fps=30,
        action_fps=30,
        source_fps=30,
        action_callable=action_fn,
        pose_quality_threshold=0.0,
        min_buffer_completeness=0.0,
    )
    pipe.min_frames = 3
    frame = np.zeros((160, 120, 3), dtype=np.uint8)
    for i in range(12):
        pipe.process_frame(frame, i, float(i) / 30.0)
    assert len(action_calls) > 0
    assert any(st.gate == PipelineDecisionStatus.ACTION_READY for st in pipe.track_states.values())


def test_separate_tracks_separate_buffers():
    dets = [
        HumanDetection(bbox=(10, 10, 50, 80), confidence=0.9),
        HumanDetection(bbox=(100, 10, 150, 80), confidence=0.85),
    ]
    det = FakeDetector(dets)
    pose = FakePose()
    tracker = IoUPersonTracker(min_stable_frames=2, max_missing_frames=10)
    pipe = RoiInferencePipeline(
        det, tracker, pose, detector_fps=30, pose_fps=30, source_fps=30, pose_quality_threshold=0.0
    )
    pipe.min_frames = 1
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    for i in range(5):
        pipe.process_frame(frame, i, float(i) / 30.0)
    ids = pipe.buffers.track_ids()
    assert len(ids) >= 2
    assert ids[0] != ids[1]


def test_track_lost_buffer_cleared():
    det = FakeDetector([HumanDetection(bbox=(10, 10, 50, 80), confidence=0.9)])
    pose = FakePose()
    tracker = IoUPersonTracker(min_stable_frames=1, max_missing_frames=2)
    pipe = RoiInferencePipeline(
        det, tracker, pose, detector_fps=30, pose_fps=30, source_fps=30, pose_quality_threshold=0.0
    )
    pipe.min_frames = 1
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(3):
        pipe.process_frame(frame, i, float(i))
    assert pipe.buffers.track_ids()
    det.detections = []
    for i in range(3, 10):
        pipe.process_frame(frame, i, float(i))
    assert pipe.buffers.track_ids() == []


def test_http_source_parse_and_live_detection():
    src = parse_video_source("http://192.168.0.213:8080/video")
    assert isinstance(src, str)
    assert is_live_source(src)
    assert resolve_latest_frame(src, latest_frame=None, sequential=None) is True


def test_webcam_source_regression():
    src = parse_video_source("0")
    assert src == 0
    assert is_live_source(src)
    assert resolve_latest_frame(0, latest_frame=None, sequential=None) is True


def test_mp4_sequential_regression(tmp_path: Path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"")
    src = parse_video_source(str(path))
    assert resolve_latest_frame(src, latest_frame=None, sequential=None) is False
    assert resolve_latest_frame(src, latest_frame=True, sequential=None) is True
    assert resolve_latest_frame(src, latest_frame=None, sequential=True) is False


def test_invalid_fps_fallback():
    c = CadenceController(-1, 30)
    assert c.interval == 1
    c2 = CadenceController(float("nan"), 30)
    assert c2.interval == 1
    c3 = CadenceController(10, 30)
    assert c3.interval == 3


def test_diagnostic_overlay_missing_fields_no_crash():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    det = FakeDetector([])
    pose = FakePose()
    tracker = IoUPersonTracker()
    pipe = RoiInferencePipeline(det, tracker, pose, source_fps=20)
    vis = draw_pipeline_overlay(frame, [], {}, {}, pipe, debug=True)
    assert vis.shape == frame.shape
    track = HumanTrack(
        track_id=7,
        bbox=(10, 10, 40, 60),
        confidence=0.5,
        age_frames=1,
        missed_frames=0,
        stable=False,
    )
    vis2 = draw_pipeline_overlay(frame, [track], {}, {}, pipe, debug=True)
    assert vis2.shape == frame.shape


def test_pose_not_called_without_track_invariant():
    det = FakeDetector([])
    pose = FakePose()
    tracker = IoUPersonTracker()
    pipe = RoiInferencePipeline(det, tracker, pose, detector_fps=60, pose_fps=60, source_fps=60)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(20):
        pipe.process_frame(frame, i, float(i))
    assert pose.calls == 0
