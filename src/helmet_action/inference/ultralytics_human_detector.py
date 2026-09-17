"""Ultralytics YOLO person detector (not pose)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from helmet_action.config import load_config
from helmet_action.inference.human_detector import (
    HumanDetection,
    downscale_frame,
    rescale_bbox,
)

# COCO person class id
_PERSON_CLASS_ID = 0


class UltralyticsHumanDetector:
    """Lightweight YOLO detector; runs on a downscaled frame and rescales boxes."""

    def __init__(
        self,
        model_path: str | None = None,
        conf: float | None = None,
        imgsz: int | None = None,
        *,
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
            or os.environ.get("HUMAN_MODEL_PATH")
            or cfg.get("inference.detector.model", "yolo11n.pt")
        )
        self.model_path = str(path)
        self.conf = float(
            conf if conf is not None else cfg.get("inference.detector.confidence", 0.35)
        )
        self.imgsz = int(imgsz if imgsz is not None else cfg.get("inference.detector.imgsz", 640))
        self.model = YOLO(path)
        self._call_count = 0
        self._call_counter = call_counter

    @property
    def call_count(self) -> int:
        return int(self._call_count)

    def detect(self, frame: np.ndarray) -> list[HumanDetection]:
        self._call_count += 1
        if self._call_counter is not None:
            self._call_counter.append(1)
        det_frame, scale_x, scale_y = downscale_frame(frame, self.imgsz)
        kw: dict[str, Any] = {"verbose": False, "conf": self.conf, "classes": [_PERSON_CLASS_ID]}
        # Use the downscaled size as imgsz so Ultralytics does not upscale again.
        dh, dw = int(det_frame.shape[0]), int(det_frame.shape[1])
        kw["imgsz"] = max(dh, dw)
        results = self.model.predict(det_frame, **kw)
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
        out: list[HumanDetection] = []
        for i in range(xyxy.shape[0]):
            if int(clss[i]) != _PERSON_CLASS_ID:
                continue
            bbox = rescale_bbox(tuple(map(float, xyxy[i])), scale_x=scale_x, scale_y=scale_y)
            out.append(
                HumanDetection(
                    bbox=bbox,
                    confidence=float(confs[i]),
                    class_id=int(clss[i]),
                )
            )
        return out
