from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from helmet_action.config import load_config
from helmet_action.pose.constants import FACE_IDX, L_EAR, L_SHOULDER, L_WRIST, NOSE, R_EAR, R_SHOULDER, R_WRIST
from helmet_action.synthetic.augmentation import NoiseParams, apply_pose_noise
from helmet_action.synthetic.camera import HighAngleCamera
from helmet_action.synthetic.scenarios import SCENARIOS
from helmet_action.synthetic.skeleton import (
    BodyParams,
    apply_body_params,
    canonical_pose_3d,
    place_arm_3d,
    scale_head_toward_camera,
)

GENERATOR_VERSION = "1.0.0"


def _lerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    return (1.0 - u) * a + u * b


def _ease(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _uniform(rng: np.random.Generator, lo_hi: list[float] | tuple[float, float]) -> float:
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    return float(rng.uniform(lo, hi))


@dataclass
class ActionParams:
    speed: float = 1.0
    pause: float = 0.15
    lift_height: float = 0.22
    wrist_separation: float = 0.08
    amplitude: float = 1.0
    start_phase: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SampleMeta:
    seed: int
    split: str
    scenario: str
    label: str
    generator_version: str
    camera: dict
    body: dict
    action: dict
    noise: dict
    n_frames: int


def _sample_range(rng: np.random.Generator, section: dict[str, Any], key: str, fallback: list[float]) -> float:
    return _uniform(rng, section.get(key, fallback))


def sample_body(rng: np.random.Generator, split: str) -> BodyParams:
    cfg = load_config()
    sec = cfg.section("synthetic.ood.body") if split == "test" else cfg.section("synthetic.body")
    handed = "left" if rng.random() < 0.35 else "right"
    return BodyParams(
        height_scale=_sample_range(rng, sec, "height_scale", [0.92, 1.10]),
        shoulder_width_scale=_sample_range(rng, sec, "shoulder_width_scale", [0.88, 1.14]),
        arm_length_scale=_sample_range(rng, sec, "arm_length_scale", [0.90, 1.12]),
        forearm_scale=_sample_range(rng, sec, "forearm_scale", [0.90, 1.12]),
        head_size_scale=_sample_range(rng, sec, "head_size_scale", [0.90, 1.12]),
        handedness=handed,
    )


def sample_camera(rng: np.random.Generator, split: str) -> HighAngleCamera:
    cfg = load_config()
    sec = cfg.section("synthetic.ood.camera") if split == "test" else cfg.section("synthetic.camera")
    return HighAngleCamera.from_params(
        height=_sample_range(rng, sec, "height", [3.8, 6.4]),
        distance=_sample_range(rng, sec, "distance", [2.6, 4.4]),
        pitch_deg=_sample_range(rng, sec, "pitch_deg", [32.0, 52.0]),
        yaw_deg=_sample_range(rng, sec, "yaw_deg", [-12.0, 12.0]),
        focal=_sample_range(rng, sec, "focal", [900.0, 1400.0]),
        cx=480.0 + float(rng.uniform(-30, 30)),
        cy=360.0 + float(rng.uniform(-20, 20)),
        worker_z=float(rng.uniform(3.2, 4.2)),
    )


def sample_action(rng: np.random.Generator, split: str) -> ActionParams:
    cfg = load_config()
    sec = cfg.section("synthetic.ood.action") if split == "test" else cfg.section("synthetic.action")
    return ActionParams(
        speed=_sample_range(rng, sec, "speed", [0.8, 1.25]),
        pause=_sample_range(rng, sec, "pause", [0.08, 0.22]),
        lift_height=_sample_range(rng, sec, "lift_height", [0.14, 0.30]),
        wrist_separation=_sample_range(rng, sec, "wrist_separation", [0.05, 0.14]),
        amplitude=_sample_range(rng, sec, "amplitude", [0.70, 1.20]),
        start_phase=float(rng.uniform(0.0, 0.25)),
    )


def sample_noise(rng: np.random.Generator, split: str) -> NoiseParams:
    cfg = load_config()
    sec = cfg.section("synthetic.ood.noise") if split == "test" else cfg.section("synthetic.noise")
    return NoiseParams(
        gaussian_sigma=_sample_range(rng, sec, "gaussian_sigma", [0.0012, 0.0022])
        if isinstance(sec.get("gaussian_sigma"), list)
        else float(sec.get("gaussian_sigma", 0.0016)),
        dropout_prob=float(sec.get("dropout_prob", 0.02)),
        low_conf_prob=float(sec.get("low_conf_prob", 0.04)),
        occlusion_prob=float(sec.get("occlusion_prob", 0.03)),
        jitter=float(sec.get("jitter", 0.002)),
        scale_jitter=float(sec.get("scale_jitter", 0.03)),
        translation_jitter=float(sec.get("translation_jitter", 4.0)),
        frame_drop_prob=float(sec.get("frame_drop_prob", 0.02)),
        temporal_stretch=float(rng.uniform(0.75, 1.30 if split != "test" else 1.55)),
    )


def _rest(body: BodyParams) -> np.ndarray:
    return apply_body_params(canonical_pose_3d(), body)


def _project_loop(
    n: int,
    body: BodyParams,
    cam: HighAngleCamera,
    rng: np.random.Generator,
    pose_fn,
    world_sigma: float = 0.0014,
) -> np.ndarray:
    seq = np.zeros((n, 17, 2), dtype=np.float64)
    for i in range(n):
        pose = _rest(body)
        pose_fn(i, pose)
        pose += rng.normal(0.0, world_sigma, size=pose.shape)
        seq[i] = cam.project(pose)
    return seq


def _phase_u(i: int, n: int, action: ActionParams, start: float, end: float) -> float:
    """Map frame i into [0,1] over a fraction of the clip, with speed/start_phase."""
    a = (start + action.start_phase) * n
    b = min(n - 1, (end + action.start_phase * 0.3) * n / max(action.speed, 0.4))
    if b <= a + 1:
        b = a + max(4, 0.15 * n)
    return _ease((i - a) / max(b - a, 1.0))


def generate_scenario_sequence(
    scenario: str,
    rng: np.random.Generator,
    body: BodyParams,
    cam: HighAngleCamera,
    action: ActionParams,
    n_frames: int,
) -> np.ndarray:
    rest = _rest(body)
    amp = action.amplitude
    hand = body.handedness
    scalp = 0.5 * (rest[L_EAR] + rest[R_EAR]) + np.array([0.0, 0.012, 0.015])
    grab_l = rest[L_EAR] + np.array([-0.03, 0.01, 0.02])
    grab_r = rest[R_EAR] + np.array([0.03, 0.01, 0.02])
    nose = rest[NOSE] + np.array([0.0, -0.02, -0.02])
    raise_end = int(n_frames * (0.14 + 0.08 / action.speed))
    pause_end = raise_end + max(4, int(n_frames * action.pause))
    lift_end = min(n_frames - 2, pause_end + int(n_frames * (0.35 + 0.1 * amp)))

    def idle_fn(i, pose):
        ph = 2 * math.pi * i / max(12.0, 18.0 / action.speed)
        wr = rest[R_WRIST] + np.array([0.03 * math.sin(ph), 0.04 * math.sin(ph), 0.0]) * amp
        place_arm_3d(pose, "right", wr, body)

    def scratch_fn(i, pose):
        side = hand
        home = rest[R_WRIST] if side == "right" else rest[L_WRIST]
        if i < raise_end:
            u = _ease(i / max(raise_end - 1, 1))
            w = _lerp(home, scalp, u)
        elif i < n_frames - 8:
            t = (i - raise_end) / 20.0 * action.speed
            rad = 0.016 * amp
            w = scalp + np.array(
                [
                    rad * math.sin(2 * math.pi * 3.4 * t),
                    0.35 * rad * math.cos(2 * math.pi * 3.4 * t),
                    0.4 * rad * math.sin(2 * math.pi * 4.0 * t),
                ]
            )
        else:
            u = _ease((i - (n_frames - 8)) / 8.0)
            w = _lerp(scalp, home, u)
        place_arm_3d(pose, side, w, body)

    def one_touch_fn(i, pose):
        side = hand
        home = rest[R_WRIST] if side == "right" else rest[L_WRIST]
        target = scalp + np.array([0.02 if side == "right" else -0.02, 0.0, 0.0])
        if i < raise_end:
            w = _lerp(home, target, _ease(i / max(raise_end - 1, 1)))
        elif i < pause_end + 10:
            w = target + rng.normal(0, 0.002, size=3)
        else:
            w = _lerp(target, home, _ease((i - pause_end - 10) / max(n_frames - pause_end - 11, 1)))
        place_arm_3d(pose, side, w, body)

    def two_touch_fn(i, pose):
        tl = scalp + np.array([-0.04, 0.01, 0.01])
        tr = scalp + np.array([0.04, 0.01, 0.01])
        if i < raise_end:
            u = _ease(i / max(raise_end - 1, 1))
            wl, wr = _lerp(rest[L_WRIST], tl, u), _lerp(rest[R_WRIST], tr, u)
        elif i < pause_end + 8:
            wl, wr = tl + rng.normal(0, 0.002, 3), tr + rng.normal(0, 0.002, 3)
        else:
            u = _ease((i - pause_end - 8) / max(n_frames - pause_end - 9, 1))
            wl, wr = _lerp(tl, rest[L_WRIST], u), _lerp(tr, rest[R_WRIST], u)
        place_arm_3d(pose, "left", wl, body)
        place_arm_3d(pose, "right", wr, body)

    def face_fn(i, pose):
        side = hand
        home = rest[R_WRIST] if side == "right" else rest[L_WRIST]
        if i < raise_end:
            w = _lerp(home, nose, _ease(i / max(raise_end - 1, 1)))
        elif i < pause_end + 6:
            w = nose + rng.normal(0, 0.003, 3)
        else:
            w = _lerp(nose, home, _ease((i - pause_end - 6) / max(n_frames - pause_end - 7, 1)))
        place_arm_3d(pose, side, w, body)

    def adjust_fn(i, pose):
        # Grasp brim then small settle — no lift, no spread.
        if i < raise_end:
            u = _ease(i / max(raise_end - 1, 1))
            wl, wr = _lerp(rest[L_WRIST], grab_l, u), _lerp(rest[R_WRIST], grab_r, u)
        elif i < pause_end + 6:
            wig = 0.008 * amp
            wl = grab_l + np.array([wig * math.sin(i / 2.0), 0.004 * math.sin(i / 3.0), 0.0])
            wr = grab_r + np.array([-wig * math.sin(i / 2.0), 0.004 * math.cos(i / 3.0), 0.0])
        else:
            u = _ease((i - pause_end - 6) / max(n_frames - pause_end - 7, 1))
            wl, wr = _lerp(grab_l, rest[L_WRIST], u), _lerp(grab_r, rest[R_WRIST], u)
        place_arm_3d(pose, "left", wl, body)
        place_arm_3d(pose, "right", wr, body)

    def wipe_fn(i, pose):
        side = hand
        home = rest[R_WRIST] if side == "right" else rest[L_WRIST]
        brow = rest[NOSE] + np.array([0.0, 0.04, 0.0])
        across = brow + np.array([0.08 if side == "left" else -0.08, 0.0, 0.0])
        if i < raise_end:
            w = _lerp(home, brow, _ease(i / max(raise_end - 1, 1)))
        elif i < pause_end + 8:
            u = (i - raise_end) / max(pause_end + 8 - raise_end, 1)
            w = _lerp(brow, across, _ease(u))
        else:
            w = _lerp(across, home, _ease((i - pause_end - 8) / max(n_frames - pause_end - 9, 1)))
        place_arm_3d(pose, side, w, body)

    def look_fn(i, pose):
        u = 0.5 * (1 - math.cos(2 * math.pi * i / max(n_frames, 2)))
        for idx in FACE_IDX:
            pose[idx] = pose[idx] + np.array([0.0, -0.06 * u * amp, 0.04 * u])
        idle_fn(i, pose)

    def raise_fn(i, pose):
        up_l = rest[L_WRIST] + np.array([-0.05, 0.45 * amp, -0.05])
        up_r = rest[R_WRIST] + np.array([0.05, 0.45 * amp, -0.05])
        u = 0.5 * (1 - math.cos(2 * math.pi * i / max(n_frames, 2)))
        place_arm_3d(pose, "left", _lerp(rest[L_WRIST], up_l, u), body)
        place_arm_3d(pose, "right", _lerp(rest[R_WRIST], up_r, u), body)

    def stretch_fn(i, pose):
        out_l = rest[L_SHOULDER] + np.array([-0.45 * amp, -0.05, 0.0])
        out_r = rest[R_SHOULDER] + np.array([0.45 * amp, -0.05, 0.0])
        u = 0.5 * (1 - math.cos(2 * math.pi * i / max(n_frames, 2)))
        place_arm_3d(pose, "left", _lerp(rest[L_WRIST], out_l, u), body)
        place_arm_3d(pose, "right", _lerp(rest[R_WRIST], out_r, u), body)

    def phone_fn(i, pose):
        side = hand
        home = rest[R_WRIST] if side == "right" else rest[L_WRIST]
        ear = rest[R_EAR] if side == "right" else rest[L_EAR]
        target = ear + np.array([0.04 if side == "right" else -0.04, -0.02, -0.03])
        if i < raise_end:
            w = _lerp(home, target, _ease(i / max(raise_end - 1, 1)))
        elif i < n_frames - 8:
            w = target + rng.normal(0, 0.002, 3)
        else:
            w = _lerp(target, home, _ease((i - (n_frames - 8)) / 8.0))
        place_arm_3d(pose, side, w, body)

    def remove_fn(i, pose):
        amount = 0.0
        if i < raise_end:
            u = _ease(i / max(raise_end - 1, 1))
            wl, wr = _lerp(rest[L_WRIST], grab_l, u), _lerp(rest[R_WRIST], grab_r, u)
        elif i < pause_end:
            wl = grab_l + rng.normal(0.0, 0.0012, size=3)
            wr = grab_r + rng.normal(0.0, 0.0012, size=3)
        else:
            u = _ease(min(1.0, (i - pause_end) / max(lift_end - pause_end - 1, 1)))
            sep = action.wrist_separation
            h = action.lift_height
            wl = grab_l + np.array([-sep * u, h * u, -0.10 * u])
            wr = grab_r + np.array([sep * u, h * u, -0.10 * u])
            amount = 0.55 * u * amp
        place_arm_3d(pose, "left", wl, body)
        place_arm_3d(pose, "right", wr, body)
        scale_head_toward_camera(pose, amount)

    def put_on_fn(i, pose):
        # Reverse of removal: start high/spread, settle onto ears, then down.
        sep = action.wrist_separation
        h = action.lift_height
        high_l = grab_l + np.array([-sep, h, -0.10])
        high_r = grab_r + np.array([sep, h, -0.10])
        settle = raise_end
        hold = pause_end
        if i < settle:
            u = _ease(i / max(settle - 1, 1))
            wl, wr = _lerp(high_l, grab_l, u), _lerp(high_r, grab_r, u)
            scale_head_toward_camera(pose, 0.45 * (1 - u))
        elif i < hold:
            wl = grab_l + rng.normal(0, 0.0012, 3)
            wr = grab_r + rng.normal(0, 0.0012, 3)
        else:
            u = _ease((i - hold) / max(n_frames - hold - 1, 1))
            wl, wr = _lerp(grab_l, rest[L_WRIST], u), _lerp(grab_r, rest[R_WRIST], u)
        place_arm_3d(pose, "left", wl, body)
        place_arm_3d(pose, "right", wr, body)

    def random_fn(i, pose):
        ph = i / max(n_frames, 1)
        wl = rest[L_WRIST] + amp * np.array(
            [0.12 * math.sin(4 * ph + 0.2), 0.18 * math.sin(3 * ph), 0.05 * math.cos(5 * ph)]
        )
        wr = rest[R_WRIST] + amp * np.array(
            [0.10 * math.cos(3.2 * ph), 0.16 * math.sin(4.1 * ph + 1.0), 0.04 * math.sin(2 * ph)]
        )
        place_arm_3d(pose, "left", wl, body)
        place_arm_3d(pose, "right", wr, body)

    fns = {
        "IDLE": idle_fn,
        "HEAD_SCRATCH": scratch_fn,
        "ONE_HAND_HEAD_TOUCH": one_touch_fn,
        "TWO_HAND_HEAD_TOUCH": two_touch_fn,
        "FACE_TOUCH": face_fn,
        "HELMET_ADJUST": adjust_fn,
        "WIPE_SWEAT": wipe_fn,
        "LOOK_DOWN": look_fn,
        "RAISE_ARMS": raise_fn,
        "STRETCH": stretch_fn,
        "PHONE_NEAR_HEAD": phone_fn,
        "HELMET_REMOVE": remove_fn,
        "HELMET_PUT_ON": put_on_fn,
        "UNKNOWN_RANDOM_MOTION": random_fn,
    }
    if scenario not in fns:
        raise ValueError(f"Unknown scenario {scenario}")
    fn = fns[scenario]
    return _project_loop(n_frames, body, cam, rng, fn)


def generate_one(
    seed: int,
    scenario: str,
    split: str = "train",
    n_frames: int | None = None,
    apply_noise: bool = True,
) -> tuple[np.ndarray, np.ndarray, SampleMeta]:
    from helmet_action.models.labels import SCENARIO_TO_CLASS

    rng = np.random.default_rng(int(seed))
    cfg = load_config()
    n = int(n_frames or cfg.get("synthetic.n_frames", 60))
    n = int(np.clip(n / max(sample_action(np.random.default_rng(seed + 99), split).speed, 0.5), 36, 80))
    body = sample_body(rng, split)
    cam = sample_camera(rng, split)
    action = sample_action(rng, split)
    noise = sample_noise(rng, split)
    seq = generate_scenario_sequence(scenario, rng, body, cam, action, n)
    conf = np.ones((seq.shape[0], 17), dtype=np.float64)
    if apply_noise:
        seq, conf = apply_pose_noise(seq, rng, noise)
    # fill leftover nans for storage (confidence already marks them)
    nan_mask = ~np.isfinite(seq)
    if nan_mask.any():
        seq = np.where(nan_mask, 0.0, seq)
    label = SCENARIO_TO_CLASS[scenario].value
    meta = SampleMeta(
        seed=int(seed),
        split=split,
        scenario=scenario,
        label=label,
        generator_version=str(cfg.get("synthetic.generator_version", GENERATOR_VERSION)),
        camera=cam.to_dict(),
        body=body.to_dict(),
        action=action.to_dict(),
        noise=noise.to_dict(),
        n_frames=int(seq.shape[0]),
    )
    return seq, conf, meta


def scenario_mix(quick: bool = False) -> list[str]:
    # Over-represent the hard negatives vs HELMET_REMOVE.
    weights = {
        "IDLE": 10,
        "HEAD_SCRATCH": 12,
        "ONE_HAND_HEAD_TOUCH": 6,
        "TWO_HAND_HEAD_TOUCH": 10,
        "FACE_TOUCH": 5,
        "HELMET_ADJUST": 12,
        "WIPE_SWEAT": 5,
        "LOOK_DOWN": 4,
        "RAISE_ARMS": 4,
        "STRETCH": 4,
        "PHONE_NEAR_HEAD": 5,
        "HELMET_REMOVE": 14,
        "HELMET_PUT_ON": 6,
        "UNKNOWN_RANDOM_MOTION": 3,
    }
    if quick:
        return [
            "IDLE",
            "HEAD_SCRATCH",
            "HELMET_ADJUST",
            "TWO_HAND_HEAD_TOUCH",
            "HELMET_REMOVE",
            "HELMET_PUT_ON",
            "FACE_TOUCH",
        ]
    bag: list[str] = []
    for k, w in weights.items():
        bag.extend([k] * w)
    return bag
