#!/usr/bin/env python3
"""Optional Ultralytics PoseProvider smoke test. Requires ultralytics extra.

POSE_MODEL_PATH=yolo11n-pose.pt python scripts/smoke_ultralytics.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    try:
        from ultralytics.utils import ASSETS
    except ImportError as exc:
        print("ultralytics_import=FAIL", exc)
        return 2

    from helmet_action.inference.ultralytics_pose_provider import UltralyticsPoseProvider

    model = os.environ.get("POSE_MODEL_PATH")
    print("ultralytics_import=OK")
    detect = UltralyticsPoseProvider(model_path=model, track=False)
    track = UltralyticsPoseProvider(model_path=model, track=True)
    image = str(Path(ASSETS) / "bus.jpg")
    import cv2

    frame = cv2.imread(image)
    if frame is None:
        print("asset_image=FAIL", image)
        return 1
    det_obs = detect.infer_frame(frame, track=False)
    trk_obs = track.infer_frame(frame, track=True)
    print(f"detect_count={len(det_obs)} track_count={len(trk_obs)}")
    if det_obs:
        o = det_obs[0]
        print("keypoint_shape", o.keypoints.shape)
        print("conf_shape", o.keypoint_confidence.shape)
        print("bbox", o.bbox)
        print("detect_track_id", o.track_id)
        assert o.keypoints.shape == (17, 2)
        assert o.keypoint_confidence.shape == (17,)
        assert len(o.bbox) == 4
    if trk_obs:
        print("track_mode_track_id", trk_obs[0].track_id)
    print("smoke=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
