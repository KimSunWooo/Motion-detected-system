from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np

from helmet_action.config import load_config
from helmet_action.inference.pose_provider import PoseProvider
from helmet_action.pose.types import PoseObservation


class UltralyticsPoseProvider(PoseProvider):
    """Optional Ultralytics YOLO-Pose backend. Model name comes from config / env."""

    def __init__(self, model_path: str | None = None, conf: float | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Synthetic demo still works. "
                "Install extras with: pip install ultralytics opencv-python-headless"
            ) from exc
        cfg = load_config()
        path = model_path or os.environ.get("POSE_MODEL_PATH") or cfg.get("pose.model_path")
        self.conf = float(conf if conf is not None else cfg.get("pose.confidence_threshold", 0.35))
        self.model = YOLO(path)

    def iter_frames(self, source: str | int) -> Iterator[tuple[np.ndarray, list[PoseObservation]]]:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv is required for video IO") from exc

        stream = cv2.VideoCapture(source)
        if not stream.isOpened():
            raise FileNotFoundError(f"cannot open video source: {source}")
        fps = float(stream.get(cv2.CAP_PROP_FPS) or 20.0)
        idx = 0
        try:
            while True:
                ok, frame = stream.read()
                if not ok:
                    break
                results = self.model.track(frame, persist=True, verbose=False, conf=self.conf)
                observations: list[PoseObservation] = []
                if results:
                    r0 = results[0]
                    kpts = getattr(r0, "keypoints", None)
                    boxes = getattr(r0, "boxes", None)
                    if kpts is not None and kpts.xy is not None and len(kpts.xy):
                        xy = kpts.xy.cpu().numpy()
                        kconf = kpts.conf.cpu().numpy() if kpts.conf is not None else np.ones(xy.shape[:2])
                        ids = None
                        if boxes is not None and boxes.id is not None:
                            ids = boxes.id.cpu().numpy().astype(int)
                        bxy = boxes.xyxy.cpu().numpy() if boxes is not None else None
                        bconf = boxes.conf.cpu().numpy() if boxes is not None and boxes.conf is not None else None
                        for i in range(xy.shape[0]):
                            track_id = int(ids[i]) if ids is not None else i + 1
                            bbox = tuple(map(float, bxy[i])) if bxy is not None else (0.0, 0.0, 1.0, 1.0)
                            det = float(bconf[i]) if bconf is not None else 1.0
                            observations.append(
                                PoseObservation(
                                    timestamp=idx / max(fps, 1e-6),
                                    frame_index=idx,
                                    track_id=track_id,
                                    bbox=bbox,
                                    keypoints=xy[i],
                                    keypoint_confidence=kconf[i],
                                    detection_confidence=det,
                                    source_fps=fps,
                                )
                            )
                yield frame, observations
                idx += 1
        finally:
            stream.release()
