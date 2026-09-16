from __future__ import annotations

import numpy as np

from helmet_action.features.temporal_features import extract_feature_vector
from helmet_action.pose.constants import L_SHOULDER, R_SHOULDER
from helmet_action.pose.geometry import compute_shoulder_width
from helmet_action.pose.normalizer import normalize_keypoints
from helmet_action.synthetic.scenarios import generate_idle_sequence, generate_scratch_sequence


def test_normalized_shoulder_width_is_one():
    seq = generate_scratch_sequence()
    norm, infos = normalize_keypoints(seq)
    widths = compute_shoulder_width(norm)
    assert np.allclose(widths, 1.0, atol=1e-5)
    assert all(info.scale > 1.0 for info in infos)


def test_translation_invariance():
    seq = generate_scratch_sequence(seed=7)
    shifted = seq + np.array([120.0, -80.0])
    a, _ = normalize_keypoints(seq)
    b, _ = normalize_keypoints(shifted)
    assert np.allclose(a, b, atol=1e-6)


def test_uniform_scale_invariance():
    seq = generate_idle_sequence(seed=3)
    scaled = seq * 1.7
    fa = extract_feature_vector(normalize_keypoints(seq)[0])
    fb = extract_feature_vector(normalize_keypoints(scaled)[0])
    # ignore n_frames which is identical anyway
    rel = np.max(np.abs(fa - fb) / (np.abs(fa) + 1e-6))
    assert rel < 0.08, rel
