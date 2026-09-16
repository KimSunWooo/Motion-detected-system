from __future__ import annotations

import numpy as np

from helmet_action.synthetic.generator import generate_one
from helmet_action.synthetic.scenarios import generate_helmet_off_sequence, generate_scratch_sequence


def test_same_seed_reproducible():
    a, ca, ma = generate_one(seed=42, scenario="HELMET_REMOVE", split="train", apply_noise=True)
    b, cb, mb = generate_one(seed=42, scenario="HELMET_REMOVE", split="train", apply_noise=True)
    assert np.allclose(a, b)
    assert np.allclose(ca, cb)
    assert ma.body == mb.body
    assert ma.action == mb.action


def test_canonical_demo_generators_stable():
    s1 = generate_scratch_sequence(seed=7)
    s2 = generate_scratch_sequence(seed=7)
    h1 = generate_helmet_off_sequence(seed=11)
    h2 = generate_helmet_off_sequence(seed=11)
    assert np.allclose(s1, s2)
    assert np.allclose(h1, h2)
    assert s1.shape[-2] == 17
    assert h1.shape[0] == 64
