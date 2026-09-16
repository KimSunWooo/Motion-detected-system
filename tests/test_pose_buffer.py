from __future__ import annotations

import numpy as np

from helmet_action.inference.source import parse_video_source, redact_source
from helmet_action.pose.pose_buffer import TrackPoseBuffer
from helmet_action.pose.types import PoseObservation


def _obs(track_id: int, t: float, marker: float) -> PoseObservation:
    k = np.zeros((17, 2), dtype=np.float64)
    k[5] = [-10.0, 0.0]
    k[6] = [10.0, 0.0]
    k[9] = [marker, float(track_id)]
    return PoseObservation(
        timestamp=t,
        frame_index=int(t * 1000),
        track_id=track_id,
        bbox=(0.0, 0.0, 1.0, 1.0),
        keypoints=k,
        keypoint_confidence=np.ones(17),
        detection_confidence=1.0,
        source_fps=30.0,
    )


def test_tracks_do_not_mix():
    buf = TrackPoseBuffer(window_seconds=2.0, target_fps=20)
    for i in range(10):
        buf.push(_obs(1, i / 30.0, marker=1.0))
        buf.push(_obs(2, i / 30.0, marker=2.0))
    a = buf.get_arrays(1)
    b = buf.get_arrays(2)
    assert a is not None and b is not None
    assert np.allclose(a[0][:, 9, 0], 1.0)
    assert np.allclose(b[0][:, 9, 0], 2.0)
    assert not np.allclose(a[0][:, 9, 1], b[0][:, 9, 1])


def test_window_covers_same_seconds_across_fps():
    spans = {}
    for fps in (15, 24, 30, 60):
        buf = TrackPoseBuffer(window_seconds=2.0, target_fps=20)
        n = int(fps * 3)
        for i in range(n):
            buf.push(_obs(7, i / fps, marker=fps))
        raw = buf.raw(7)
        span = raw[-1].timestamp - raw[0].timestamp
        spans[fps] = span
        assert span <= 2.0 + 1.0 / fps + 1e-6
        assert span >= 2.0 - 2.0 / fps
        packed = buf.get_arrays(7)
        assert packed is not None
        _, _, ts = packed
        assert ts[-1] - ts[0] == span
    assert max(spans.values()) - min(spans.values()) < 0.15


def test_parse_webcam_and_existing_file(tmp_path):
    assert parse_video_source("0") == 0
    assert parse_video_source(1) == 1
    video = tmp_path / "0"
    video.write_bytes(b"not-a-real-video")
    parsed = parse_video_source(str(video))
    assert isinstance(parsed, str)
    assert parsed.endswith("/0") or parsed.endswith("\\0")
    rtsp = "rtsp://user:secret@192.168.0.10:554/stream"
    assert parse_video_source(rtsp) == rtsp
    assert "secret" not in redact_source(rtsp)
    assert "user" not in redact_source(rtsp)
