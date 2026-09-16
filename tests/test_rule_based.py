from __future__ import annotations

import numpy as np

from helmet_action.models.hybrid import HybridActionClassifier
from helmet_action.models.labels import ActionClass, ActionLabel
from helmet_action.models.rule_based import RuleBasedActionClassifier, classify_pose_sequence
from helmet_action.synthetic.scenarios import (
    generate_helmet_off_sequence,
    generate_idle_sequence,
    generate_scratch_sequence,
)


def test_rule_scratch_not_helmet_off():
    seq = generate_scratch_sequence()
    result = RuleBasedActionClassifier().predict(seq)
    assert result.label is ActionLabel.SCRATCH
    assert result.label is not ActionLabel.HELMET_OFF


def test_rule_helmet_off():
    seq = generate_helmet_off_sequence()
    result = classify_pose_sequence(seq)
    assert result.label is ActionLabel.HELMET_OFF
    assert result.confidence >= 0.55


def test_rule_idle_not_removal():
    seq = generate_idle_sequence()
    result = classify_pose_sequence(seq)
    assert result.label is ActionLabel.NO_CONTACT


def test_hybrid_scratch_low_remove_prob():
    seq = generate_scratch_sequence()
    dec = HybridActionClassifier(ml=None).predict(seq)
    assert dec.action is not ActionClass.HELMET_REMOVE
    assert dec.risk < 0.5


def test_hybrid_helmet_remove_high_risk():
    seq = generate_helmet_off_sequence()
    dec = HybridActionClassifier(ml=None).predict(seq)
    assert dec.action is ActionClass.HELMET_REMOVE
    assert dec.risk >= 0.5


def test_idle_not_false_positive():
    seq = generate_idle_sequence()
    dec = HybridActionClassifier(ml=None).predict(seq)
    assert dec.action is not ActionClass.HELMET_REMOVE
    assert dec.risk < 0.3
