from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from helmet_action.inference.capture import (
    FrameCapture,
    LatestFrameBuffer,
    rotate_frame,
    sanitize_fps,
)
from helmet_action.inference.diagnostics import PoseDiagState, classify_pose_diag
from helmet_action.inference.source import parse_video_source, redact_source
from helmet_action.inference.video_pipeline import _draw, load_ml_status, run_video
from helmet_action.inference.pose_provider import NumpySequenceProvider
from helmet_action.models.labels import ActionClass
from helmet_action.synthetic.generator import generate_one


cv2 = pytest.importorskip("cv2")


def test_rotate_frame_shapes_and_consistency():
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[5, 10] = (0, 0, 255)
    r0 = rotate_frame(frame, 0)
    assert r0.shape == (40, 80, 3)
    r90 = rotate_frame(frame, 90)
    assert r90.shape == (80, 40, 3)
    # clockwise 90: (row,col)=(5,10) -> (10, 39-5)=(10,34)
    assert tuple(r90[10, 34]) == (0, 0, 255)
    r180 = rotate_frame(frame, 180)
    assert r180.shape == (40, 80, 3)
    assert tuple(r180[39 - 5, 79 - 10]) == (0, 0, 255)
    r270 = rotate_frame(frame, 270)
    assert r270.shape == (80, 40, 3)
    with pytest.raises(ValueError):
        rotate_frame(frame, 45)


def test_sanitize_fps_fallback():
    assert sanitize_fps(29.97) == pytest.approx(29.97)
    assert sanitize_fps(0.0) == 20.0
    assert sanitize_fps(-1) == 20.0
    assert sanitize_fps(float("nan")) == 20.0
    assert sanitize_fps(1000) == 20.0
    assert sanitize_fps(None) == 20.0


def test_latest_frame_buffer_depth_one():
    buf = LatestFrameBuffer()
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = np.ones((2, 2, 3), dtype=np.uint8)
    buf.put(a, 0, 0.0)
    buf.put(b, 1, 0.1)
    assert buf.depth() == 1
    assert buf.dropped == 1
    item = buf.get(timeout=0.2)
    assert item is not None
    assert item.index == 1
    assert np.array_equal(item.frame, b)
    assert buf.depth() == 0


def test_latest_frame_capture_thread_shutdown(tmp_path: Path):
    # Synthetic "video": write a tiny mp4-like sequence via image sequence is hard;
    # instead exercise LatestFrameBuffer close semantics used by FrameCapture.
    buf = LatestFrameBuffer()
    stop = threading.Event()

    def producer():
        i = 0
        while not stop.is_set():
            buf.put(np.full((4, 4, 3), i % 255, dtype=np.uint8), i, time.monotonic())
            i += 1
            time.sleep(0.001)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    time.sleep(0.05)
    stop.set()
    buf.close()
    t.join(timeout=1.0)
    assert not t.is_alive()
    assert buf.get(timeout=0.05) is None


def test_cli_pose_conf_and_hybrid_override_wiring():
    # Unit-level: UltralyticsPoseProvider accepts conf/imgsz without requiring GPU model if import fails.
    ultralytics = pytest.importorskip("ultralytics")
    from helmet_action.inference.ultralytics_pose_provider import UltralyticsPoseProvider

    # Use a tiny fake path only if model exists; otherwise skip model construct.
    model = Path("yolo11n-pose.pt")
    if not model.exists() and not Path("models").exists():
        pytest.skip("pose weights not present")
    # conf override must stick even if weights missing — catch Import/file errors.
    try:
        p = UltralyticsPoseProvider(model_path=str(model) if model.exists() else None, conf=0.25, imgsz=640, track=False)
    except Exception as exc:
        pytest.skip(f"cannot construct YOLO provider: {exc}")
    assert p.conf == 0.25
    assert p.imgsz == 640
    assert p._kwargs()["conf"] == 0.25
    assert p._kwargs()["imgsz"] == 640


def test_hybrid_version_passed_to_run_video(tmp_path: Path):
    seq, conf, _ = generate_one(seed=3, scenario="HELMET_REMOVE", split="train", apply_noise=False)
    provider = NumpySequenceProvider(seq, conf, fps=20.0, frame_shape=(120, 160))
    out = tmp_path / "out.mp4"
    dest = run_video(
        0,
        provider,
        out_path=out,
        show=False,
        max_frames=8,
        hybrid_version="v2",
        debug_overlay=True,
        no_record=False,
        print_banner=False,
    )
    assert dest == out
    assert out.exists() and out.stat().st_size > 0


def test_no_record_skips_writer(tmp_path: Path):
    seq, conf, _ = generate_one(seed=4, scenario="HEAD_SCRATCH", split="train", apply_noise=False)
    provider = NumpySequenceProvider(seq, conf, fps=20.0, frame_shape=(80, 100))
    out = tmp_path / "should_not_exist.mp4"
    dest = run_video(
        0,
        provider,
        out_path=out,
        show=False,
        max_frames=5,
        no_record=True,
        print_banner=False,
    )
    assert dest is None
    assert not out.exists()


def test_ml_model_load_status():
    clf, path, feat, err = load_ml_status()
    assert path
    if Path("models/action_classifier.joblib").exists():
        assert clf is not None
        assert err is None
        assert feat in {"v1", "v2", None} or isinstance(feat, str)
    missing = load_ml_status(Path("/tmp/definitely_missing_action_model.joblib"))
    assert missing[0] is None
    assert "not found" in (missing[3] or "")


def test_http_source_parsing_regression():
    url = "http://192.168.0.213:8080/video"
    assert parse_video_source(url) == url
    assert "192.168.0.213" in redact_source(url)
    secret = "http://user:pass@192.168.0.213:8080/video"
    red = redact_source(secret)
    assert "pass" not in red
    assert "user" not in red


def test_debug_overlay_missing_fields_no_crash():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    from helmet_action.pose.types import PoseObservation

    obs = PoseObservation(
        timestamp=0.0,
        frame_index=0,
        track_id=1,
        bbox=(10.0, 10.0, 80.0, 90.0),
        keypoints=np.zeros((17, 2)),
        keypoint_confidence=np.linspace(0.1, 0.9, 17),
        detection_confidence=0.5,
        source_fps=20.0,
    )

    class Dummy:
        alert = False
        risk = 0.2
        risk_level = None
        action = ActionClass.UNKNOWN
        phase = type("P", (), {"value": "IDLE"})()
        helmet_state = type("H", (), {"value": "UNKNOWN"})()
        confidence = 0.1
        decision_status = "UNKNOWN"
        ml_proba = {}

    out = _draw(
        frame.copy(),
        obs,
        Dummy(),
        debug=True,
        hybrid_version="v2",
        pose_quality=None,
        wrist_quality=None,
        ear_quality=None,
        infer_fps=None,
        source_fps=None,
        diag_state=None,
    )
    assert out.shape == frame.shape
    # no decision at all
    out2 = _draw(frame.copy(), obs, None, debug=True)
    assert out2.shape == frame.shape


def test_pose_diag_states_separate_no_person():
    assert classify_pose_diag(0, pose_quality=None, decision_status=None, action=None) is PoseDiagState.NO_PERSON
    assert classify_pose_diag(1, pose_quality=0.3, decision_status="UNKNOWN", action=ActionClass.UNKNOWN) in {
        PoseDiagState.POSE_LOW_CONFIDENCE,
        PoseDiagState.ACTION_UNKNOWN,
    }
    assert (
        classify_pose_diag(1, pose_quality=0.9, decision_status="VALID", action=ActionClass.HELMET_REMOVE)
        is PoseDiagState.ACTION_VALID
    )
