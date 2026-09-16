from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from helmet_action.config import load_config
from helmet_action.pose.types import PoseObservation


class TrackPoseBuffer:
    """Per-track sliding window of PoseObservation, time-normalized to target FPS."""

    def __init__(
        self,
        window_seconds: float | None = None,
        target_fps: float | None = None,
    ) -> None:
        cfg = load_config()
        self.window_seconds = float(window_seconds if window_seconds is not None else cfg.get("window.seconds", 2.0))
        self.target_fps = float(target_fps if target_fps is not None else cfg.get("window.target_fps", 20))
        self._tracks: dict[int, deque[PoseObservation]] = defaultdict(deque)

    def push(self, obs: PoseObservation) -> None:
        buf = self._tracks[obs.track_id]
        buf.append(obs)
        cutoff = obs.timestamp - self.window_seconds
        while buf and buf[0].timestamp < cutoff:
            buf.popleft()

    def track_ids(self) -> list[int]:
        return [tid for tid, buf in self._tracks.items() if buf]

    def raw(self, track_id: int) -> list[PoseObservation]:
        return list(self._tracks.get(track_id, ()))

    def clear(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(track_id, None)

    def get_arrays(self, track_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return resampled (T,17,2) keypoints, (T,17) conf, (T,) timestamps."""
        buf = self.raw(track_id)
        if len(buf) < 2:
            if not buf:
                return None
            obs = buf[0]
            return (
                obs.keypoints[None, ...],
                obs.keypoint_confidence[None, ...],
                np.array([obs.timestamp], dtype=np.float64),
            )
        times = np.array([o.timestamp for o in buf], dtype=np.float64)
        kpts = np.stack([o.keypoints for o in buf], axis=0)
        conf = np.stack([o.keypoint_confidence for o in buf], axis=0)
        t0, t1 = float(times[0]), float(times[-1])
        duration = max(t1 - t0, 1e-6)
        n = max(2, int(round(duration * self.target_fps)) + 1)
        n = min(n, int(load_config().get("window.max_frames", 80)))
        grid = np.linspace(t0, t1, n)
        k_out = np.empty((n, 17, 2), dtype=np.float64)
        c_out = np.empty((n, 17), dtype=np.float64)
        for j in range(17):
            for axis in (0, 1):
                k_out[:, j, axis] = np.interp(grid, times, kpts[:, j, axis])
            c_out[:, j] = np.interp(grid, times, conf[:, j])
        return k_out, c_out, grid
