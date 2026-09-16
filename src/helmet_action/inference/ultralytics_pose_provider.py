from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np

from helmet_action.config import load_config
from helmet_action.inference.pose_provider import PoseProvider
from helmet_action.pose.types import PoseObservation


class UltralyticsPoseProvider(PoseProvider):
    """Optional Ultralytics YOLO-Pose backend. Model path comes from config / env."""

    def __init__(self, model_path: str | None = None, conf: float | None = None, track: bool = True) -> None:
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
        self.track = bool(track)
        self.model = YOLO(path)

    def _observations(self, result, frame_index: int, fps: float) -> list[PoseObservation]:
        observations: list[PoseObservation] = []
        kpts = getattr(result, "keypoints", None)
        boxes = getattr(result, "boxes", None)
        if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
            return observations
        xy = kpts.xy.cpu().numpy()
        kconf = kpts.conf.cpu().numpy() if kpts.conf is not None else np.ones(xy.shape[:2])
        ids = None
        if boxes is not None and getattr(boxes, "id", None) is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        bxy = boxes.xyxy.cpu().numpy() if boxes is not None else None
        bconf = boxes.conf.cpu().numpy() if boxes is not None and boxes.conf is not None else None
        for i in range(xy.shape[0]):
            track_id = int(ids[i]) if ids is not None else i + 1
            bbox = tuple(map(float, bxy[i])) if bxy is not None else (0.0, 0.0, 1.0, 1.0)
            det = float(bconf[i]) if bconf is not None else 1.0
            k = xy[i]
            if k.shape != (17, 2) and k.size >= 34:
                k = k.reshape(-1, 2)[:17]
            observations.append(
                PoseObservation(
                    timestamp=frame_index / max(fps, 1e-6),
                    frame_index=frame_index,
                    track_id=track_id,
                    bbox=bbox,
                    keypoints=k[:17],
                    keypoint_confidence=np.asarray(kconf[i], dtype=np.float64).reshape(-1)[:17],
                    detection_confidence=det,
                    source_fps=fps,
                )
            )
        return observations

    def infer_frame(self, frame: np.ndarray, frame_index: int = 0, fps: float = 20.0, track: bool | None = None) -> list[PoseObservation]:
        use_track = self.track if track is None else track
        if use_track:
            results = self.model.track(frame, persist=True, verbose=False, conf=self.conf)
        else:
            results = self.model.predict(frame, verbose=False, conf=self.conf)
        if not results:
            return []
        return self._observations(results[0], frame_index, fps)

    def iter_frames(self, source: str | int) -> Iterator[tuple[np.ndarray, list[PoseObservation]]]:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv is required for video IO") from exc

        stream = cv2.VideoCapture(source)
        if not stream.isOpened():
            raise FileNotFoundError("cannot open video source")
        fps = float(stream.get(cv2.CAP_PROP_FPS) or 20.0)
        idx = 0
        try:
            while True:
                ok, frame = stream.read()
                if not ok:
                    break
                yield frame, self.infer_frame(frame, frame_index=idx, fps=fps)
                idx += 1
        finally:
            stream.release()
