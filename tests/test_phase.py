from __future__ import annotations

from helmet_action.models.labels import RemovalPhase
from helmet_action.state.action_state_machine import infer_phases
from helmet_action.synthetic.generator import generate_one
from helmet_action.synthetic.scenarios import generate_helmet_off_sequence, generate_scratch_sequence


def _order_ok(history: list[str]) -> bool:
    wanted = [
        RemovalPhase.HAND_APPROACH.value,
        RemovalPhase.HELMET_GRASP.value,
        RemovalPhase.LIFT_OR_SEPARATE.value,
    ]
    idxs = [history.index(p) if p in history else 10**9 for p in wanted]
    return idxs[0] < idxs[2] or idxs[1] < idxs[2]


def test_helmet_remove_reaches_confirmed():
    seq = generate_helmet_off_sequence()
    trace = infer_phases(seq)
    assert trace.phase is RemovalPhase.REMOVAL_CONFIRMED
    assert trace.saw_grasp and trace.saw_lift
    assert _order_ok(trace.history)


def test_scratch_and_adjust_do_not_confirm_removal():
    scratch = infer_phases(generate_scratch_sequence())
    assert scratch.phase is not RemovalPhase.REMOVAL_CONFIRMED
    adj, _, _ = generate_one(seed=21, scenario="HELMET_ADJUST", split="train", apply_noise=False)
    adjust = infer_phases(adj)
    assert adjust.phase is not RemovalPhase.REMOVAL_CONFIRMED
