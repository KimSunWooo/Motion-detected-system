"""Trajectory-family catalogue and train/test holdout helpers."""

from __future__ import annotations

from typing import Any

SCENARIO_FAMILIES: dict[str, list[str]] = {
    "HELMET_REMOVE": ["REMOVE_A", "REMOVE_B", "REMOVE_C"],
    "HELMET_PUT_ON": ["PUTON_A", "PUTON_B"],
    "HELMET_ADJUST": ["ADJUST_A", "ADJUST_B"],
    "HEAD_SCRATCH": ["SCRATCH_A", "SCRATCH_B"],
    "ONE_HAND_HEAD_TOUCH": ["TOUCH_A"],
    "TWO_HAND_HEAD_TOUCH": ["TOUCH_A", "TOUCH_B"],
    "FACE_TOUCH": ["TOUCH_A"],
    "WIPE_SWEAT": ["WIPE_A"],
    "PHONE_NEAR_HEAD": ["PHONE_A"],
    "STRETCH": ["STRETCH_A"],
    "RAISE_ARMS": ["RAISE_A"],
    "LOOK_DOWN": ["IDLE_A"],
    "IDLE": ["IDLE_A"],
    "UNKNOWN_RANDOM_MOTION": ["RAND_A"],
}

DEFAULT_TRAIN_FAMILIES = [
    "REMOVE_A",
    "REMOVE_B",
    "TOUCH_A",
    "ADJUST_A",
    "SCRATCH_A",
    "PUTON_A",
    "WIPE_A",
    "PHONE_A",
    "STRETCH_A",
    "RAISE_A",
    "IDLE_A",
    "RAND_A",
]

DEFAULT_HOLDOUT_FAMILIES = [
    "REMOVE_C",
    "TOUCH_B",
    "ADJUST_B",
    "SCRATCH_B",
    "PUTON_B",
]


def families_for_scenario(scenario: str) -> list[str]:
    return list(SCENARIO_FAMILIES.get(scenario, [f"{scenario}_A"]))


def choose_family(
    scenario: str,
    split: str,
    rng,
    train_families: list[str] | None = None,
    holdout_families: list[str] | None = None,
) -> str:
    available = families_for_scenario(scenario)
    train_f = set(train_families or DEFAULT_TRAIN_FAMILIES)
    hold_f = set(holdout_families or DEFAULT_HOLDOUT_FAMILIES)
    if split == "test":
        held = [f for f in available if f in hold_f]
        pool = held if held else [f for f in available if f not in train_f] or available
    else:
        allowed = [f for f in available if f in train_f and f not in hold_f]
        pool = allowed if allowed else [f for f in available if f not in hold_f] or available
    idx = int(rng.integers(0, len(pool)))
    return pool[idx]


def family_split_report(train: list[str], validation: list[str], test: list[str]) -> dict[str, Any]:
    tr, va, te = set(train), set(validation), set(test)
    holdout_in_train = sorted((tr & te) & set(DEFAULT_HOLDOUT_FAMILIES))
    return {
        "train_families": sorted(tr),
        "validation_families": sorted(va),
        "test_families": sorted(te),
        "train_test_overlap": sorted(tr & te),
        "holdout_families_in_train": holdout_in_train,
        "holdout_leakage": len(holdout_in_train),
    }
