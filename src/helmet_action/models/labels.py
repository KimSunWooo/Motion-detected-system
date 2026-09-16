from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class BaselineActionLabel(str, Enum):
    """Original rule-based labels — kept for dashboard / self-test compatibility."""

    SCRATCH = "scratch"
    HELMET_OFF = "helmet_off"
    NO_CONTACT = "no_contact"
    UNKNOWN_CONTACT = "unknown_contact"


class ActionClass(str, Enum):
    IDLE = "IDLE"
    HEAD_SCRATCH = "HEAD_SCRATCH"
    HEAD_TOUCH = "HEAD_TOUCH"
    HELMET_ADJUST = "HELMET_ADJUST"
    HELMET_REMOVE = "HELMET_REMOVE"
    HELMET_PUT_ON = "HELMET_PUT_ON"
    UNKNOWN = "UNKNOWN"
    INSUFFICIENT_POSE = "INSUFFICIENT_POSE"


class RemovalPhase(str, Enum):
    IDLE = "IDLE"
    HAND_APPROACH = "HAND_APPROACH"
    HELMET_GRASP = "HELMET_GRASP"
    LIFT_OR_SEPARATE = "LIFT_OR_SEPARATE"
    REMOVAL_CONFIRMED = "REMOVAL_CONFIRMED"


LABEL_KO = {
    BaselineActionLabel.SCRATCH: "단순 머리 긁기 (정상)",
    BaselineActionLabel.HELMET_OFF: "안전모 벗기 시도 (예방 알림)",
    BaselineActionLabel.NO_CONTACT: "머리 비접촉 (동작 없음)",
    BaselineActionLabel.UNKNOWN_CONTACT: "머리 접촉 · 판정 보류",
}

ACTION_KO = {
    ActionClass.IDLE: "비접촉 / 대기",
    ActionClass.HEAD_SCRATCH: "머리 긁기",
    ActionClass.HEAD_TOUCH: "머리·얼굴 접촉",
    ActionClass.HELMET_ADJUST: "안전모 고쳐 쓰기",
    ActionClass.HELMET_REMOVE: "안전모 벗기",
    ActionClass.HELMET_PUT_ON: "안전모 착용",
    ActionClass.UNKNOWN: "판정 보류",
    ActionClass.INSUFFICIENT_POSE: "관절 정보 부족",
}

# Confirmed non-remove actions. UNKNOWN / INSUFFICIENT_POSE are abstentions, not "safe".
SAFE_CONFIRMED_ACTIONS = frozenset(
    {
        ActionClass.IDLE.value,
        ActionClass.HEAD_SCRATCH.value,
        ActionClass.HEAD_TOUCH.value,
        ActionClass.HELMET_ADJUST.value,
        ActionClass.HELMET_PUT_ON.value,
    }
)


ML_CLASSES = [
    ActionClass.IDLE,
    ActionClass.HEAD_SCRATCH,
    ActionClass.HEAD_TOUCH,
    ActionClass.HELMET_ADJUST,
    ActionClass.HELMET_REMOVE,
    ActionClass.HELMET_PUT_ON,
    ActionClass.UNKNOWN,
]

SCENARIO_TO_CLASS = {
    "IDLE": ActionClass.IDLE,
    "HEAD_SCRATCH": ActionClass.HEAD_SCRATCH,
    "ONE_HAND_HEAD_TOUCH": ActionClass.HEAD_TOUCH,
    "TWO_HAND_HEAD_TOUCH": ActionClass.HEAD_TOUCH,
    "FACE_TOUCH": ActionClass.HEAD_TOUCH,
    "HELMET_ADJUST": ActionClass.HELMET_ADJUST,
    "WIPE_SWEAT": ActionClass.HEAD_TOUCH,
    "LOOK_DOWN": ActionClass.IDLE,
    "RAISE_ARMS": ActionClass.UNKNOWN,
    "STRETCH": ActionClass.UNKNOWN,
    "PHONE_NEAR_HEAD": ActionClass.HEAD_TOUCH,
    "HELMET_REMOVE": ActionClass.HELMET_REMOVE,
    "HELMET_PUT_ON": ActionClass.HELMET_PUT_ON,
    "UNKNOWN_RANDOM_MOTION": ActionClass.UNKNOWN,
}


def baseline_to_action(label: BaselineActionLabel) -> ActionClass:
    return {
        BaselineActionLabel.SCRATCH: ActionClass.HEAD_SCRATCH,
        BaselineActionLabel.HELMET_OFF: ActionClass.HELMET_REMOVE,
        BaselineActionLabel.NO_CONTACT: ActionClass.IDLE,
        BaselineActionLabel.UNKNOWN_CONTACT: ActionClass.UNKNOWN,
    }[label]


def action_to_baseline(label: ActionClass) -> BaselineActionLabel:
    return {
        ActionClass.HEAD_SCRATCH: BaselineActionLabel.SCRATCH,
        ActionClass.HELMET_REMOVE: BaselineActionLabel.HELMET_OFF,
        ActionClass.IDLE: BaselineActionLabel.NO_CONTACT,
        ActionClass.HEAD_TOUCH: BaselineActionLabel.UNKNOWN_CONTACT,
        ActionClass.HELMET_ADJUST: BaselineActionLabel.UNKNOWN_CONTACT,
        ActionClass.HELMET_PUT_ON: BaselineActionLabel.UNKNOWN_CONTACT,
        ActionClass.UNKNOWN: BaselineActionLabel.UNKNOWN_CONTACT,
        ActionClass.INSUFFICIENT_POSE: BaselineActionLabel.UNKNOWN_CONTACT,
    }[label]


# Back-compat alias used by the original script / dashboard.
ActionLabel = BaselineActionLabel


@dataclass
class FeatureReport:
    bbox_frames: int = 0
    center_frames: int = 0
    both_ear_frames: int = 0
    active_wrist: str = "none"
    scratch_radius: float = 0.0
    scratch_std: float = 0.0
    n_oscillations: int = 0
    pause_detected: bool = False
    pause_std: float = 0.0
    dx_spread: float = 0.0
    co_rise_y: float = 0.0
    radial_expand: float = 0.0
    head_scale_up: float = 0.0
    wrist_spread: float = 0.0
    d_wrist_pause: float = 0.0
    d_wrist_late: float = 0.0


@dataclass
class ClassificationResult:
    label: BaselineActionLabel
    confidence: float
    features: FeatureReport
    explanation: list[str] = field(default_factory=list)
    frame_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label.value,
            "label_ko": LABEL_KO[self.label],
            "confidence": self.confidence,
            "features": asdict(self.features),
            "explanation": self.explanation,
            "frame_labels": self.frame_labels,
        }
