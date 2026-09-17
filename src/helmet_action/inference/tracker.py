"""Minimal person tracker between detection and ROI pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from helmet_action.config import load_config
from helmet_action.inference.human_detector import HumanDetection


@dataclass
class HumanTrack:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    age_frames: int
    missed_frames: int
    stable: bool


class PersonTracker(Protocol):
    def update(
        self,
        frame: np.ndarray,
        detections: list[HumanDetection] | None,
    ) -> list[HumanTrack]:
        ...


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


@dataclass
class _TrackState:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    age_frames: int = 0
    hits: int = 0
    missed_frames: int = 0


class IoUPersonTracker:
    """Greedy IoU matcher. Detector may be None on non-detection frames (predict-only)."""

    def __init__(
        self,
        *,
        min_stable_frames: int | None = None,
        max_missing_frames: int | None = None,
        iou_match: float | None = None,
    ) -> None:
        cfg = load_config()
        self.min_stable_frames = int(
            min_stable_frames
            if min_stable_frames is not None
            else cfg.get("inference.tracker.min_stable_frames", 5)
        )
        self.max_missing_frames = int(
            max_missing_frames
            if max_missing_frames is not None
            else cfg.get("inference.tracker.max_missing_frames", 10)
        )
        self.iou_match = float(
            iou_match if iou_match is not None else cfg.get("inference.tracker.iou_match", 0.30)
        )
        self._tracks: dict[int, _TrackState] = {}
        self._next_id = 1
        self._lost_ids: list[int] = []

    @property
    def lost_track_ids(self) -> list[int]:
        """Track IDs removed on the most recent update (for buffer cleanup)."""
        return list(self._lost_ids)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._lost_ids = []

    def update(
        self,
        frame: np.ndarray,
        detections: list[HumanDetection] | None,
    ) -> list[HumanTrack]:
        del frame  # reserved for appearance/motion models
        self._lost_ids = []
        if detections is None:
            # Coast: age existing tracks, count miss.
            for tid, st in list(self._tracks.items()):
                st.age_frames += 1
                st.missed_frames += 1
                if st.missed_frames > self.max_missing_frames:
                    self._lost_ids.append(tid)
                    self._tracks.pop(tid, None)
            return self._as_tracks()

        dets = list(detections)
        unmatched_dets = set(range(len(dets)))
        unmatched_tracks = set(self._tracks.keys())
        pairs: list[tuple[float, int, int]] = []
        for tid in self._tracks:
            for di, det in enumerate(dets):
                score = _iou(self._tracks[tid].bbox, det.bbox)
                if score >= self.iou_match:
                    pairs.append((score, tid, di))
        pairs.sort(reverse=True, key=lambda x: x[0])
        used_t: set[int] = set()
        used_d: set[int] = set()
        for score, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(di)
            det = dets[di]
            st = self._tracks[tid]
            st.bbox = det.bbox
            st.confidence = float(det.confidence)
            st.age_frames += 1
            st.hits += 1
            st.missed_frames = 0

        for tid in list(unmatched_tracks):
            st = self._tracks[tid]
            st.age_frames += 1
            st.missed_frames += 1
            if st.missed_frames > self.max_missing_frames:
                self._lost_ids.append(tid)
                self._tracks.pop(tid, None)

        for di in sorted(unmatched_dets):
            det = dets[di]
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _TrackState(
                track_id=tid,
                bbox=det.bbox,
                confidence=float(det.confidence),
                age_frames=1,
                hits=1,
                missed_frames=0,
            )

        return self._as_tracks()

    def _as_tracks(self) -> list[HumanTrack]:
        out: list[HumanTrack] = []
        for st in self._tracks.values():
            out.append(
                HumanTrack(
                    track_id=st.track_id,
                    bbox=st.bbox,
                    confidence=st.confidence,
                    age_frames=st.age_frames,
                    missed_frames=st.missed_frames,
                    stable=st.hits >= self.min_stable_frames and st.missed_frames == 0,
                )
            )
        return out
