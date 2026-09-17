"""Pipeline decision / gate diagnostics (does not mutate ActionClass semantics)."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PipelineDecisionStatus(str, Enum):
    """Explicit gate status for UI / debug. Separate from ActionClass."""

    NO_PERSON = "NO_PERSON"
    TRACK_WARMUP = "TRACK_WARMUP"
    POSE_MISSING = "POSE_MISSING"
    POSE_LOW_CONFIDENCE = "POSE_LOW_CONFIDENCE"
    POSE_INSUFFICIENT = "POSE_INSUFFICIENT"
    BUFFER_WARMUP = "BUFFER_WARMUP"
    ACTION_READY = "ACTION_READY"


@dataclass
class StageTimer:
    """Rolling average stage latencies in milliseconds."""

    window: int = 60
    _samples: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=60)))

    def __post_init__(self) -> None:
        self._samples = defaultdict(lambda: deque(maxlen=int(self.window)))

    def record(self, name: str, ms: float) -> None:
        self._samples[name].append(float(ms))

    def mean(self, name: str) -> float | None:
        vals = self._samples.get(name)
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def p95(self, name: str) -> float | None:
        vals = list(self._samples.get(name) or ())
        if not vals:
            return None
        vals_sorted = sorted(vals)
        idx = min(len(vals_sorted) - 1, int(round(0.95 * (len(vals_sorted) - 1))))
        return float(vals_sorted[idx])

    def snapshot(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name in sorted(self._samples.keys()):
            m = self.mean(name)
            p = self.p95(name)
            if m is None:
                continue
            out[name] = {"mean_ms": m, "p95_ms": p if p is not None else m, "n": float(len(self._samples[name]))}
        return out

    def format_line(self) -> str:
        parts = []
        for name, stats in self.snapshot().items():
            parts.append(f"{name} {stats['mean_ms']:.1f} ms")
        return " | ".join(parts) if parts else "no samples"


class TimedStage:
    """Context manager that records elapsed ms into a StageTimer."""

    def __init__(self, timer: StageTimer, name: str) -> None:
        self.timer = timer
        self.name = name
        self._t0 = 0.0

    def __enter__(self) -> "TimedStage":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        ms = (time.perf_counter() - self._t0) * 1000.0
        self.timer.record(self.name, ms)


@dataclass
class CallRateMeter:
    """Count calls per second over a sliding window."""

    window: float = 2.0
    _times: list[float] = field(default_factory=list)

    def tick(self, n: int = 1) -> float:
        now = time.monotonic()
        for _ in range(max(0, int(n))):
            self._times.append(now)
        cutoff = now - self.window
        self._times = [t for t in self._times if t >= cutoff]
        if not self._times:
            return 0.0
        dt = self._times[-1] - self._times[0]
        if dt <= 1e-6:
            return float(len(self._times))
        return float(len(self._times) / dt)


@dataclass
class TrackRuntimeState:
    track_id: int
    gate: PipelineDecisionStatus = PipelineDecisionStatus.TRACK_WARMUP
    last_pose_quality: float | None = None
    last_wrist_l: float | None = None
    last_wrist_r: float | None = None
    buffer_frames: int = 0
    buffer_completeness: float = 0.0
    last_action: str | None = None
    last_risk_level: str | None = None
    last_decision_status: str | None = None
    detection_confidence: float | None = None
    age_frames: int = 0
    stable: bool = False
