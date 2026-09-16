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
    height: float = 5.70
    distance: float = 3.50
    pitch_deg: float = 42.0
    yaw_deg: float = 0.0
    worker_z: float = 3.50

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
            height=float(height),
            distance=float(distance),
            pitch_deg=float(pitch_deg),
            yaw_deg=float(yaw_deg),
            worker_z=float(worker_z),
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
            "height": float(self.height),
            "distance": float(self.distance),
            "pitch_deg": float(self.pitch_deg),
            "yaw_deg": float(self.yaw_deg),
            "focal": float(self.fx),
            "worker_z": float(self.worker_z),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HighAngleCamera":
        if "height" in d and "pitch_deg" in d:
            return cls.from_params(
                height=float(d["height"]),
                distance=float(d.get("distance", 3.5)),
                pitch_deg=float(d["pitch_deg"]),
                yaw_deg=float(d.get("yaw_deg", 0.0)),
                focal=float(d.get("focal", d.get("fx", 1180.0))),
                cx=float(d.get("cx", 480.0)),
                cy=float(d.get("cy", 360.0)),
                worker_z=float(d.get("worker_z", 3.5)),
            )
        return cls(
            eye=np.asarray(d["eye"], dtype=np.float64),
            target=np.asarray(d["target"], dtype=np.float64),
            up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            fx=float(d["fx"]),
            fy=float(d.get("fy", d["fx"])),
            cx=float(d.get("cx", 480.0)),
            cy=float(d.get("cy", 360.0)),
        )
