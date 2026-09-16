from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np


class HelmetState(str, Enum):
    UNKNOWN = "UNKNOWN"
    WORN = "WORN"
    NOT_WORN = "NOT_WORN"


class ActionEvent(str, Enum):
    NONE = "NONE"
    REMOVE_INTENT = "REMOVE_INTENT"
    REMOVE_CONFIRMED = "REMOVE_CONFIRMED"
    PUT_ON_INTENT = "PUT_ON_INTENT"
    PUT_ON_CONFIRMED = "PUT_ON_CONFIRMED"


@dataclass
class HelmetPresenceObservation:
    state: HelmetState
    confidence: float
    note: str = ""


class HelmetPresenceDetector(Protocol):
    def detect(self, frame: np.ndarray, person_bbox: tuple[float, float, float, float]) -> HelmetPresenceObservation:
        ...


class DummyHelmetPresenceDetector:
    """No detector weights shipped. Always UNKNOWN — never invent WORN/NOT_WORN."""

    def detect(self, frame: np.ndarray, person_bbox: tuple[float, float, float, float]) -> HelmetPresenceObservation:
        return HelmetPresenceObservation(
            state=HelmetState.UNKNOWN,
            confidence=0.0,
            note="no helmet object-detector weights; pose cannot observe helmet presence at rest",
        )


class HelmetStateMachine:
    def __init__(self, initial: HelmetState = HelmetState.UNKNOWN) -> None:
        self.state = initial

    def update(
        self,
        event: ActionEvent = ActionEvent.NONE,
        detector: HelmetPresenceObservation | None = None,
    ) -> HelmetState:
        if detector is not None and detector.state is not HelmetState.UNKNOWN:
            self.state = detector.state
        if self.state is HelmetState.WORN and event is ActionEvent.REMOVE_CONFIRMED:
            self.state = HelmetState.NOT_WORN
        elif self.state is HelmetState.NOT_WORN and event is ActionEvent.PUT_ON_CONFIRMED:
            self.state = HelmetState.WORN
        # UNKNOWN: pose action alone never asserts worn vs not-worn.
        return self.state


class AlertGate:
    def __init__(self, enter: float = 0.75, exit: float = 0.45, window: int = 5, hits: int = 3) -> None:
        self.enter = enter
        self.exit = exit
        self.window = window
        self.hits = hits
        self.alert = False
        self._buf: list[float] = []

    def update(self, risk: float) -> bool:
        self._buf.append(float(risk))
        self._buf = self._buf[-self.window :]
        n_high = sum(v >= self.enter for v in self._buf)
        if self.alert:
            if risk < self.exit and n_high == 0:
                self.alert = False
        else:
            if n_high >= self.hits or (len(self._buf) == 1 and risk >= self.enter):
                self.alert = True
        return self.alert
