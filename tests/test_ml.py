from __future__ import annotations

from pathlib import Path

import numpy as np

from helmet_action.models.temporal_classifier import SklearnActionClassifier, train_sklearn_classifier
from helmet_action.models.training import vectorize_dataset
from helmet_action.synthetic.generator import generate_one


def _tiny_xy(n: int = 24):
    scenarios = ["IDLE", "HEAD_SCRATCH", "HELMET_REMOVE", "HELMET_ADJUST"] * (n // 4)
    kpts, confs, labels = [], [], []
    for i, sc in enumerate(scenarios):
        seq, conf, meta = generate_one(seed=1000 + i, scenario=sc, split="train", apply_noise=False)
        kpts.append(seq)
        confs.append(conf)
        labels.append(meta.label)
    X = vectorize_dataset(kpts, confs)
    return X, np.array(labels), kpts, confs


def test_model_save_load_predictions_match(tmp_path: Path):
    X, y, kpts, confs = _tiny_xy(24)
    clf = train_sklearn_classifier(X, y)
    p1 = [clf.predict(kpts[i], confs[i]) for i in range(len(kpts))]
    pr1 = [clf.predict_proba(kpts[i], confs[i]) for i in range(len(kpts))]
    path = clf.save(tmp_path / "action_classifier.joblib")
    loaded = SklearnActionClassifier.load(path)
    p2 = [loaded.predict(kpts[i], confs[i]) for i in range(len(kpts))]
    pr2 = [loaded.predict_proba(kpts[i], confs[i]) for i in range(len(kpts))]
    assert p1 == p2
    for a, b in zip(pr1, pr2):
        for k in a:
            assert abs(a[k] - b[k]) < 1e-9


def test_trained_model_scratch_vs_remove_if_present():
    from helmet_action.config import repo_root
    from helmet_action.synthetic.scenarios import (
        generate_helmet_off_sequence,
        generate_idle_sequence,
        generate_scratch_sequence,
    )

    path = repo_root() / "models" / "action_classifier.joblib"
    if not path.exists():
        return
    clf = SklearnActionClassifier.load(path)
    p_s = clf.predict_proba(generate_scratch_sequence())
    p_h = clf.predict_proba(generate_helmet_off_sequence())
    p_i = clf.predict_proba(generate_idle_sequence())
    assert p_s.get("HELMET_REMOVE", 0) < 0.5
    assert p_h.get("HELMET_REMOVE", 0) > 0.4
    assert p_i.get("HELMET_REMOVE", 0) < 0.4
