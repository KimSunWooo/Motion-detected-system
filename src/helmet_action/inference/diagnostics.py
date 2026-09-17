"""Realtime pose / tracking diagnostics (does not change Hybrid semantics)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from helmet_action.models.labels import ActionClass
from helmet_action.pose.constants import L_EAR, L_ELBOW, L_SHOULDER, L_WRIST, R_EAR, R_ELBOW, R_SHOULDER, R_WRIST
from helmet_action.pose.types import PoseObservation


class PoseDiagState(str, Enum):
    """Diagnostic-only. Does not replace ActionClass / DecisionStatus."""

    NO_PERSON = "NO_PERSON"
    POSE_LOW_CONFIDENCE = "POSE_LOW_CONFIDENCE"
    POSE_VALID = "POSE_VALID"
    ACTION_UNKNOWN = "ACTION_UNKNOWN"
    ACTION_VALID = "ACTION_VALID"


@dataclass
class TrackEvent:
    kind: str  # TRACK_CREATED | TRACK_LOST | TRACK_ID_CHANGED | TRACK_REAPPEARED
    track_id: int
    frame_index: int
    age: int = 0
    last_seen_frame: int = 0
    gap_duration: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "track_id": int(self.track_id),
            "frame_index": int(self.frame_index),
            "age": int(self.age),
            "last_seen_frame": int(self.last_seen_frame),
            "gap_duration": float(self.gap_duration),
            "detail": self.detail,
        }


@dataclass
class PersonPoseDiag:
    track_id: int
    detection_confidence: float
    keypoint_mean: float
    wrist_l: float
    wrist_r: float
    shoulder_l: float
    shoulder_r: float
    ear_l: float
    ear_r: float
    visible_keypoints: int
    bbox_area_ratio: float
    bbox_aspect_ratio: float
    frame_area: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "detection_confidence": float(self.detection_confidence),
            "keypoint_mean": float(self.keypoint_mean),
            "wrist_l": float(self.wrist_l),
            "wrist_r": float(self.wrist_r),
            "shoulder_l": float(self.shoulder_l),
            "shoulder_r": float(self.shoulder_r),
            "ear_l": float(self.ear_l),
            "ear_r": float(self.ear_r),
            "visible_keypoints": int(self.visible_keypoints),
            "bbox_area_ratio": float(self.bbox_area_ratio),
            "bbox_aspect_ratio": float(self.bbox_aspect_ratio),
            "frame_area": float(self.frame_area),
        }


def _conf(obs: PoseObservation, idx: int) -> float:
    c = obs.keypoint_confidence
    if c is None or idx >= len(c):
        return 0.0
    v = float(c[idx])
    return v if np.isfinite(v) else 0.0


def person_pose_diag(obs: PoseObservation, frame_shape: tuple[int, int]) -> PersonPoseDiag:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    area = float(max(h * w, 1))
    x1, y1, x2, y2 = obs.bbox
    bw = max(float(x2 - x1), 1e-6)
    bh = max(float(y2 - y1), 1e-6)
    conf = np.asarray(obs.keypoint_confidence, dtype=np.float64).reshape(-1)
    finite = np.isfinite(conf)
    visible = int(np.sum(finite & (conf >= 0.25)))
    return PersonPoseDiag(
        track_id=int(obs.track_id),
        detection_confidence=float(obs.detection_confidence),
        keypoint_mean=float(np.nanmean(conf)) if conf.size else 0.0,
        wrist_l=_conf(obs, L_WRIST),
        wrist_r=_conf(obs, R_WRIST),
        shoulder_l=_conf(obs, L_SHOULDER),
        shoulder_r=_conf(obs, R_SHOULDER),
        ear_l=_conf(obs, L_EAR),
        ear_r=_conf(obs, R_EAR),
        visible_keypoints=visible,
        bbox_area_ratio=float((bw * bh) / area),
        bbox_aspect_ratio=float(bh / bw),
        frame_area=area,
    )


def classify_pose_diag(
    n_people: int,
    *,
    pose_quality: float | None,
    decision_status: str | None,
    action: ActionClass | str | None,
) -> PoseDiagState:
    if n_people <= 0:
        return PoseDiagState.NO_PERSON
    pq = 1.0 if pose_quality is None else float(pose_quality)
    status = str(decision_status or "")
    if status in {"INSUFFICIENT_POSE", "UNKNOWN", "LOW_CONFIDENCE"} or pq < 0.52:
        if status == "VALID" and pq >= 0.52:
            pass
        else:
            return PoseDiagState.POSE_LOW_CONFIDENCE
    act = action.value if isinstance(action, ActionClass) else (str(action) if action is not None else "")
    if act in {ActionClass.UNKNOWN.value, ActionClass.INSUFFICIENT_POSE.value, ""}:
        if pq >= 0.52 and status == "VALID":
            return PoseDiagState.ACTION_UNKNOWN
        if status in {"UNKNOWN", "INSUFFICIENT_POSE", "LOW_CONFIDENCE"}:
            return PoseDiagState.POSE_LOW_CONFIDENCE if pq < 0.70 else PoseDiagState.ACTION_UNKNOWN
        return PoseDiagState.POSE_VALID
    return PoseDiagState.ACTION_VALID


class TrackEventMonitor:
    """Emit TRACK_* events without mutating TrackPoseBuffer."""

    def __init__(self, lost_after_frames: int = 15) -> None:
        self.lost_after_frames = int(lost_after_frames)
        self._alive: dict[int, dict[str, Any]] = {}
        self._lost: dict[int, dict[str, Any]] = {}
        self.events: list[TrackEvent] = []

    def update(self, track_ids: list[int], frame_index: int, timestamp: float) -> list[TrackEvent]:
        now = set(int(t) for t in track_ids)
        new_events: list[TrackEvent] = []
        for tid in now:
            if tid in self._alive:
                self._alive[tid]["last_seen_frame"] = frame_index
                self._alive[tid]["last_seen_ts"] = timestamp
                self._alive[tid]["age"] = int(self._alive[tid].get("age", 0)) + 1
            elif tid in self._lost:
                gap = frame_index - int(self._lost[tid].get("last_seen_frame", frame_index))
                gap_s = timestamp - float(self._lost[tid].get("last_seen_ts", timestamp))
                ev = TrackEvent(
                    kind="TRACK_REAPPEARED",
                    track_id=tid,
                    frame_index=frame_index,
                    age=0,
                    last_seen_frame=int(self._lost[tid].get("last_seen_frame", frame_index)),
                    gap_duration=float(max(gap_s, 0.0)),
                    detail=f"gap_frames={gap}",
                )
                new_events.append(ev)
                self.events.append(ev)
                self._alive[tid] = {
                    "created_frame": frame_index,
                    "last_seen_frame": frame_index,
                    "last_seen_ts": timestamp,
                    "age": 0,
                }
                self._lost.pop(tid, None)
            else:
                ev = TrackEvent(
                    kind="TRACK_CREATED",
                    track_id=tid,
                    frame_index=frame_index,
                    age=0,
                    last_seen_frame=frame_index,
                )
                new_events.append(ev)
                self.events.append(ev)
                self._alive[tid] = {
                    "created_frame": frame_index,
                    "last_seen_frame": frame_index,
                    "last_seen_ts": timestamp,
                    "age": 0,
                }

        for tid in list(self._alive.keys()):
            if tid in now:
                continue
            meta = self._alive[tid]
            gap = frame_index - int(meta.get("last_seen_frame", frame_index))
            if gap >= self.lost_after_frames:
                ev = TrackEvent(
                    kind="TRACK_LOST",
                    track_id=tid,
                    frame_index=frame_index,
                    age=int(meta.get("age", 0)),
                    last_seen_frame=int(meta.get("last_seen_frame", frame_index)),
                    gap_duration=float(timestamp - float(meta.get("last_seen_ts", timestamp))),
                )
                new_events.append(ev)
                self.events.append(ev)
                self._lost[tid] = meta
                self._alive.pop(tid, None)

        # Soft signal when the set of IDs changes while people remain.
        if new_events and any(e.kind in {"TRACK_CREATED", "TRACK_REAPPEARED"} for e in new_events) and len(now) > 0:
            for e in list(new_events):
                if e.kind == "TRACK_CREATED" and self._lost:
                    # Prefer ID-change wording when a lost ID is replaced.
                    alt = TrackEvent(
                        kind="TRACK_ID_CHANGED",
                        track_id=e.track_id,
                        frame_index=frame_index,
                        age=0,
                        last_seen_frame=e.last_seen_frame,
                        detail="new id while previous tracks were lost",
                    )
                    self.events.append(alt)
                    new_events.append(alt)
                    break
        return new_events


@dataclass
class FpsMeter:
    window: float = 1.0
    _times: list[float] = field(default_factory=list)

    def tick(self) -> float:
        now = time.monotonic()
        self._times.append(now)
        cutoff = now - self.window
        self._times = [t for t in self._times if t >= cutoff]
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        if dt <= 1e-6:
            return 0.0
        return float((len(self._times) - 1) / dt)
