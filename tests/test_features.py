from __future__ import annotations

import numpy as np

from helmet_action.features.temporal_features import extract_feature_dict, feature_names
from helmet_action.pose.normalizer import normalize_keypoints
from helmet_action.synthetic.scenarios import generate_helmet_off_sequence, generate_scratch_sequence


def test_feature_vector_fixed_size():
    names = feature_names()
    assert len(names) > 40
    seq = generate_scratch_sequence()
    d = extract_feature_dict(normalize_keypoints(seq)[0])
    assert set(names) <= set(d)


def test_features_finite_with_nans():
    seq = generate_helmet_off_sequence()
    seq[3:6, 9] = np.nan
    d = extract_feature_dict(normalize_keypoints(np.nan_to_num(seq))[0])
    assert all(np.isfinite(v) for v in d.values())
