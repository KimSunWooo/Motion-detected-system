"""RiskLevel is not a class label. UNKNOWN is not SAFE."""

from __future__ import annotations

from collections import defaultdict, deque

from helmet_action.models.labels import ActionClass, RiskLevel, SAFE_CONFIRMED_ACTIONS


SAFE_ACTIONS = SAFE_CONFIRMED_ACTIONS


class RiskEscalator:
    """Per-track hysteresis: a single window cannot jump to ALERT."""

    def __init__(self, min_windows_for_alert: int = 2, history: int = 6) -> None:
        self.min_windows_for_alert = max(int(min_windows_for_alert), 2)
        self.history = int(history)
        self._buf: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=self.history))
        self._level: dict[int, RiskLevel] = {}

    def peek(self, track_id: int = 0) -> RiskLevel:
        return self._level.get(int(track_id), RiskLevel.UNKNOWN)

    def history_for(self, track_id: int = 0) -> list[str]:
        return list(self._buf[int(track_id)])

    def reset(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._buf.clear()
            self._level.clear()
            return
        tid = int(track_id)
        self._buf.pop(tid, None)
        self._level.pop(tid, None)

    def update(self, proposed: RiskLevel | str, track_id: int = 0) -> RiskLevel:
        tid = int(track_id)
        proposed_level = proposed if isinstance(proposed, RiskLevel) else RiskLevel(str(proposed))
        buf = self._buf[tid]
        buf.append(proposed_level.value)
        current = self._level.get(tid, RiskLevel.UNKNOWN)

        if proposed_level is RiskLevel.UNKNOWN:
            # Stay UNKNOWN; never collapse into SAFE.
            self._level[tid] = RiskLevel.UNKNOWN
            return RiskLevel.UNKNOWN

        if proposed_level is RiskLevel.SAFE:
            self._level[tid] = RiskLevel.SAFE
            return RiskLevel.SAFE

        if proposed_level is RiskLevel.WATCH:
            if current is RiskLevel.ALERT:
                self._level[tid] = RiskLevel.ALERT
                return RiskLevel.ALERT
            self._level[tid] = RiskLevel.WATCH
            return RiskLevel.WATCH

        # proposed ALERT: require consecutive supporting windows.
        n_support = 0
        for item in reversed(buf):
            if item in (RiskLevel.ALERT.value, RiskLevel.WATCH.value):
                n_support += 1
            else:
                break
        if n_support >= self.min_windows_for_alert:
            self._level[tid] = RiskLevel.ALERT
            return RiskLevel.ALERT
        self._level[tid] = RiskLevel.WATCH
        return RiskLevel.WATCH


def propose_risk_level(
    *,
    action: ActionClass,
    remove_score: float,
    pose_quality: float,
    phase_score: float,
    ml_remove: float,
    watch_threshold: float,
    alert_threshold: float,
    pose_poor: float = 0.52,
    safety_unknown: bool = False,
) -> RiskLevel:
    if safety_unknown or action is ActionClass.INSUFFICIENT_POSE:
        return RiskLevel.UNKNOWN
    if pose_quality < pose_poor and action is not ActionClass.HELMET_REMOVE:
        if action.value in SAFE_ACTIONS and pose_quality < 0.40:
            return RiskLevel.UNKNOWN
        if action is ActionClass.UNKNOWN:
            return RiskLevel.UNKNOWN
    if action is ActionClass.HELMET_REMOVE and remove_score >= alert_threshold and pose_quality >= pose_poor:
        return RiskLevel.ALERT
    if (
        remove_score >= watch_threshold
        or (ml_remove >= 0.75 and phase_score >= 0.35)
        or action is ActionClass.HELMET_REMOVE
    ):
        return RiskLevel.WATCH
    if action is ActionClass.UNKNOWN:
        return RiskLevel.UNKNOWN
    if action.value in SAFE_ACTIONS:
        return RiskLevel.SAFE
    return RiskLevel.UNKNOWN


def apply_escalation_to_action(
    action: ActionClass,
    risk_level: RiskLevel,
    proposed: RiskLevel,
) -> tuple[ActionClass, RiskLevel]:
    """A demoted ALERT becomes WATCH/UNKNOWN, never SAFE."""
    if risk_level is RiskLevel.WATCH and action is ActionClass.HELMET_REMOVE:
        return ActionClass.UNKNOWN, RiskLevel.WATCH
    if risk_level is RiskLevel.UNKNOWN and action.value in SAFE_ACTIONS and proposed is RiskLevel.UNKNOWN:
        return ActionClass.UNKNOWN, RiskLevel.UNKNOWN
    return action, risk_level
