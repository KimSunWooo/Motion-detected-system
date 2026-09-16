from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class NoiseParams:
    gaussian_sigma: float = 0.0016
    dropout_prob: float = 0.02
    low_conf_prob: float = 0.04
    occlusion_prob: float = 0.03
    jitter: float = 0.002
    scale_jitter: float = 0.03
    translation_jitter: float = 4.0
    frame_drop_prob: float = 0.02
    temporal_stretch: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def apply_pose_noise(
    seq_px: np.ndarray,
    rng: np.random.Generator,
    noise: NoiseParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy keypoints (T,17,2), confidence (T,17))."""
    seq = np.asarray(seq_px, dtype=np.float64).copy()
    t, j, _ = seq.shape
    conf = np.ones((t, j), dtype=np.float64)

    seq += rng.normal(0.0, noise.gaussian_sigma * 40.0, size=seq.shape)
    seq += rng.normal(0.0, noise.jitter * 40.0, size=seq.shape)

    scale = 1.0 + rng.uniform(-noise.scale_jitter, noise.scale_jitter)
    shift = rng.uniform(-noise.translation_jitter, noise.translation_jitter, size=2)
    seq = seq * scale + shift

    drop = rng.random((t, j)) < noise.dropout_prob
    low = rng.random((t, j)) < noise.low_conf_prob
    conf[low] = rng.uniform(0.12, 0.28, size=int(low.sum())) if low.any() else conf[low]
    conf[drop] = 0.0
    seq[drop] = np.nan

    if rng.random() < noise.occlusion_prob:
        joint = int(rng.integers(0, j))
        start = int(rng.integers(0, max(t - 4, 1)))
        length = int(rng.integers(2, 6))
        seq[start : start + length, joint] = np.nan
        conf[start : start + length, joint] = 0.0

    keep = rng.random(t) >= noise.frame_drop_prob
    if keep.sum() < 8:
        keep[:] = True
    seq = seq[keep]
    conf = conf[keep]

    stretch = float(np.clip(noise.temporal_stretch, 0.6, 1.6))
    if abs(stretch - 1.0) > 0.05 and seq.shape[0] >= 8:
        n_new = max(8, int(round(seq.shape[0] * stretch)))
        src = np.linspace(0, seq.shape[0] - 1, seq.shape[0])
        dst = np.linspace(0, seq.shape[0] - 1, n_new)
        out = np.empty((n_new, j, 2), dtype=np.float64)
        c_out = np.empty((n_new, j), dtype=np.float64)
        for ji in range(j):
            for axis in (0, 1):
                col = seq[:, ji, axis]
                finite = np.isfinite(col)
                if finite.sum() < 2:
                    out[:, ji, axis] = col[0] if col.size else np.nan
                else:
                    out[:, ji, axis] = np.interp(dst, src[finite], col[finite])
            c_out[:, ji] = np.interp(dst, src, np.nan_to_num(conf[:, ji], nan=0.0))
        seq, conf = out, c_out
    return seq, conf
