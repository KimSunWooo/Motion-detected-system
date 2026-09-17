"""ROI-based Ultralytics YOLO-Pose estimator."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from helmet_action.config import load_config
from helmet_action.inference.pose_estimator import local_to_global_keypoints
from helmet_action.inference.tracker import HumanTrack
from helmet_action.pose.types import PoseObservation


class UltralyticsRoiPoseEstimator:
    """Run YOLO-Pose on a person ROI and remap keypoints to global coordinates."""

    def __init__(
        self,
        model_path: str | None = None,
        conf: float | None = None,
        imgsz: int | None = None,
        *,
        source_fps: float = 20.0,
        call_counter: list[int] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Install extras with: "
                "pip install ultralytics opencv-python-headless"
            ) from exc
        cfg = load_config()
        path = (
            model_path
            or os.environ.get("POSE_MODEL_PATH")
            or cfg.get("inference.pose.model")
            or cfg.get("pose.model_path", "yolo11n-pose.pt")
        )
        self.model_path = str(path)
        self.conf = float(
            conf if conf is not None else cfg.get("inference.pose.confidence", 0.25)
        )
        self.imgsz = int(imgsz if imgsz is not None else cfg.get("inference.pose.imgsz", 384))
        self.source_fps = float(source_fps)
        self.model = YOLO(path)
        self._call_count = 0
        self._call_counter = call_counter

    @property
    def call_count(self) -> int:
        return int(self._call_count)

    def estimate(
        self,
        roi: np.ndarray,
        track: HumanTrack,
        frame_index: int,
        timestamp: float,
        *,
        roi_xyxy: tuple[int, int, int, int] | None = None,
    ) -> PoseObservation | None:
        """Estimate pose on ROI. ``roi_xyxy`` required for global remapping."""
        self._call_count += 1
        if self._call_counter is not None:
            self._call_counter.append(1)
        if roi is None or roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
            return None
        if roi_xyxy is None:
            # Assume caller already remapped; treat ROI origin as (0,0).
            roi_xyxy = (0, 0, int(roi.shape[1]), int(roi.shape[0]))

        kw: dict[str, Any] = {"verbose": False, "conf": self.conf, "imgsz": int(self.imgsz)}
        results = self.model.predict(roi, **kw)
        if not results:
            return None
        result = results[0]
        kpts = getattr(result, "keypoints", None)
        boxes = getattr(result, "boxes", None)
        if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
            return None
        xy = kpts.xy.cpu().numpy()
        kconf = kpts.conf.cpu().numpy() if kpts.conf is not None else np.ones(xy.shape[:2])
        # Pick the person instance with highest box confidence inside the ROI.
        best_i = 0
        det_conf = float(track.confidence)
        if boxes is not None and boxes.conf is not None and len(boxes.conf) > 0:
            confs = boxes.conf.cpu().numpy()
            best_i = int(np.argmax(confs))
            det_conf = float(confs[best_i])
        k = xy[best_i]
        if k.shape != (17, 2) and k.size >= 34:
            k = k.reshape(-1, 2)[:17]
        local = np.asarray(k[:17], dtype=np.float64)
        global_k = local_to_global_keypoints(local, roi_xyxy)
        conf = np.asarray(kconf[best_i], dtype=np.float64).reshape(-1)[:17]
        if conf.shape != (17,):
            conf = np.pad(conf, (0, max(0, 17 - conf.size)))[:17]
        x1, y1, x2, y2 = track.bbox
        return PoseObservation(
            timestamp=float(timestamp),
            frame_index=int(frame_index),
            track_id=int(track.track_id),
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            keypoints=global_k,
            keypoint_confidence=conf,
            detection_confidence=det_conf,
            source_fps=self.source_fps,
        )
