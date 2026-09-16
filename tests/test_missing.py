from __future__ import annotations

import numpy as np

from helmet_action.models.hybrid import HybridActionClassifier
from helmet_action.models.labels import ActionClass
from helmet_action.pose.confidence import assess_pose_quality, prepare_sequence
from helmet_action.pose.types import PoseQuality
from helmet_action.synthetic.scenarios import generate_scratch_sequence


def test_missing_keypoints_do_not_crash():
    seq = generate_scratch_sequence()
    seq[:, 9] = np.nan
    seq[:, 10] = np.nan
    conf = np.ones((seq.shape[0], 17))
    conf[:, 9] = 0.0
    conf[:, 10] = 0.0
    repaired, quality = prepare_sequence(seq, conf)
    assert repaired.shape[1] == 17
    dec = HybridActionClassifier(ml=None).predict(seq, conf)
    assert dec.action in (ActionClass.INSUFFICIENT_POSE, ActionClass.UNKNOWN, ActionClass.IDLE)


def test_low_confidence_yields_unknown_or_insufficient():
    seq = generate_scratch_sequence()
    conf = np.full((seq.shape[0], 17), 0.05)
    quality = assess_pose_quality(seq, conf)
    assert quality.quality is PoseQuality.INSUFFICIENT_POSE
    dec = HybridActionClassifier(ml=None).predict(seq, conf)
    assert dec.action is ActionClass.INSUFFICIENT_POSE
    assert dec.quality == "INSUFFICIENT_POSE"
    assert dec.helmet_state.value == "UNKNOWN"
