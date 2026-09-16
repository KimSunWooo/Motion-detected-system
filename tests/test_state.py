from __future__ import annotations

from helmet_action.state.helmet_state import (
    ActionEvent,
    DummyHelmetPresenceDetector,
    HelmetState,
    HelmetStateMachine,
)


def test_dummy_detector_always_unknown():
    det = DummyHelmetPresenceDetector()
    obs = det.detect(None, (0, 0, 10, 10))
    assert obs.state is HelmetState.UNKNOWN


def test_unknown_not_flipped_by_pose_action():
    sm = HelmetStateMachine()
    assert sm.state is HelmetState.UNKNOWN
    sm.update(ActionEvent.REMOVE_CONFIRMED)
    assert sm.state is HelmetState.UNKNOWN


def test_worn_to_not_worn_after_confirmed_remove():
    sm = HelmetStateMachine(HelmetState.WORN)
    sm.update(ActionEvent.REMOVE_CONFIRMED)
    assert sm.state is HelmetState.NOT_WORN
    sm.update(ActionEvent.PUT_ON_CONFIRMED)
    assert sm.state is HelmetState.WORN
