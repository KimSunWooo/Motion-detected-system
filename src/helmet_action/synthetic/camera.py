from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class HighAngleCamera:
    """World (Y-up, X-right, Z-depth) → image pixels (v positive downward)."""

    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def factory_ceiling(cls) -> "HighAngleCamera":
        return cls(
            eye=np.array([0.18, 5.70, 1.45], dtype=np.float64),
            target=np.array([0.00, 1.10, 3.50], dtype=np.float64),
            up=np.array([0.00, 1.00, 0.00], dtype=np.float64),
            fx=1180.0,
            fy=1180.0,
            cx=480.0,
            cy=360.0,
        )

    @classmethod
    def from_params(
        cls,
        height: float,
        distance: float,
        pitch_deg: float,
        yaw_deg: float,
        focal: float,
        cx: float = 480.0,
        cy: float = 360.0,
        worker_z: float = 3.50,
    ) -> "HighAngleCamera":
        pitch = np.radians(pitch_deg)
        yaw = np.radians(yaw_deg)
        # Camera sits in front of the worker and looks down at chest height.
        horiz = float(distance) * np.cos(pitch)
        eye = np.array(
            [
                horiz * np.sin(yaw),
                float(height),
                worker_z - horiz * np.cos(yaw),
            ],
            dtype=np.float64,
        )
        target = np.array([0.0, 1.10, worker_z], dtype=np.float64)
        return cls(
            eye=eye,
            target=target,
            up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            fx=float(focal),
            fy=float(focal),
            cx=cx,
            cy=cy,
        )

    def _extrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        eye, target, up = self.eye, self.target, self.up
        z_cam = target - eye
        z_cam = z_cam / np.linalg.norm(z_cam)
        x_cam = np.cross(up, z_cam)
        x_cam = x_cam / (np.linalg.norm(x_cam) + 1e-9)
        y_cam = np.cross(z_cam, x_cam)
        y_cam = y_cam / (np.linalg.norm(y_cam) + 1e-9)
        return np.stack([x_cam, y_cam, z_cam], axis=0), eye

    def project(self, points_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=np.float64)
        r, eye = self._extrinsics()
        cam = np.einsum("ij,...j->...i", r, pts - eye)
        z = np.clip(cam[..., 2], 1e-6, None)
        u = self.fx * (cam[..., 0] / z) + self.cx
        v = -self.fy * (cam[..., 1] / z) + self.cy
        return np.stack([u, v], axis=-1)

    def to_dict(self) -> dict:
        return {
            "eye": self.eye.tolist(),
            "target": self.target.tolist(),
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }
