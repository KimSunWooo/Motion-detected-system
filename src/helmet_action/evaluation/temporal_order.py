"""Temporal-order variants of a pose sequence (no new generator templates)."""

from __future__ import annotations

import numpy as np


def reverse_sequence(keypoints: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(keypoints)[::-1].copy(), np.asarray(confidence)[::-1].copy()


def shuffle_phase_blocks(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    n_blocks: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(keypoints)
    c = np.asarray(confidence)
    t = k.shape[0]
    n_blocks = int(np.clip(n_blocks, 2, max(t // 4, 2)))
    edges = np.linspace(0, t, n_blocks + 1).astype(int)
    blocks = [(k[edges[i] : edges[i + 1]], c[edges[i] : edges[i + 1]]) for i in range(n_blocks)]
    rng = np.random.default_rng(seed)
    order = np.arange(n_blocks)
    rng.shuffle(order)
    k_out = np.concatenate([blocks[i][0] for i in order], axis=0)
    c_out = np.concatenate([blocks[i][1] for i in order], axis=0)
    return k_out, c_out


def partial_start(keypoints: np.ndarray, confidence: np.ndarray, frac: float = 0.40) -> tuple[np.ndarray, np.ndarray]:
    t = max(8, int(round(len(keypoints) * float(frac))))
    t = min(t, len(keypoints))
    return np.asarray(keypoints)[:t].copy(), np.asarray(confidence)[:t].copy()


def partial_end(keypoints: np.ndarray, confidence: np.ndarray, frac: float = 0.40) -> tuple[np.ndarray, np.ndarray]:
    t = max(8, int(round(len(keypoints) * float(frac))))
    t = min(t, len(keypoints))
    return np.asarray(keypoints)[-t:].copy(), np.asarray(confidence)[-t:].copy()


def temporal_variants(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    seed: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    k = np.asarray(keypoints)
    c = np.asarray(confidence)
    return {
        "NORMAL": (k.copy(), c.copy()),
        "REVERSED": reverse_sequence(k, c),
        "SHUFFLED": shuffle_phase_blocks(k, c, seed=seed),
        "PARTIAL_START": partial_start(k, c, 0.40),
        "PARTIAL_END": partial_end(k, c, 0.40),
    }
