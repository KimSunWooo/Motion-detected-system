from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from helmet_action.config import load_config
from helmet_action.pose.constants import FACE_IDX, L_EAR, L_SHOULDER, L_WRIST, NOSE, R_EAR, R_SHOULDER, R_WRIST
from helmet_action.synthetic.augmentation import NoiseParams, apply_pose_noise
from helmet_action.synthetic.camera import HighAngleCamera
from helmet_action.synthetic.families import choose_family
from helmet_action.synthetic.kinematics import constrain_path, interp_ease, lerp, rotate_head
from helmet_action.synthetic.skeleton import (
    BodyParams,
    apply_body_params,
    canonical_pose_3d,
    place_arm_3d,
    scale_head_toward_camera,
)

GENERATOR_VERSION = "2.0.0"


def _uniform(rng: np.random.Generator, lo_hi: list[float] | tuple[float, float]) -> float:
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    return float(rng.uniform(lo, hi))


def _sample_range(rng: np.random.Generator, section: dict[str, Any], key: str, fallback: list[float]) -> float:
    val = section.get(key, fallback)
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return _uniform(rng, val)
    return float(val)


@dataclass
class ActionParams:
    speed: float = 1.0
    pause: float = 0.15
    lift_height: float = 0.22
    wrist_separation: float = 0.08
    amplitude: float = 1.0
    start_phase: float = 0.0
    family: str = ""
    variant: str = ""
    dominant: str = "right"
    grasp_mode: str = "sides"
    lift_dir: str = "up"
    pause_mode: str = "short"
    speed_mode: str = "normal"
    head_turn_deg: float = 0.0
    one_then_two: bool = False
    stutter: bool = False
    ease: str = "cubic"

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
    family: str = ""
    fps: float = 20.0
    occlusion: dict = field(default_factory=dict)
    variant: str = ""
    pair_id: str = ""
    pair_role: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


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
        neck_length_scale=_sample_range(rng, sec, "neck_length_scale", [0.90, 1.12]),
        body_yaw_deg=_sample_range(rng, sec, "body_yaw_deg", [-18.0, 18.0]),
        body_pitch_deg=_sample_range(rng, sec, "body_pitch_deg", [-8.0, 8.0]),
        torso_lean=_sample_range(rng, sec, "torso_lean", [-0.08, 0.12]),
        handedness=handed,
    )


def sample_camera(
    rng: np.random.Generator,
    split: str,
    overrides: dict[str, float] | None = None,
) -> HighAngleCamera:
    cfg = load_config()
    sec = cfg.section("synthetic.ood.camera") if split == "test" else cfg.section("synthetic.camera")
    ov = overrides or {}
    return HighAngleCamera.from_params(
        height=ov.get("height", _sample_range(rng, sec, "height", [3.8, 6.4])),
        distance=ov.get("distance", _sample_range(rng, sec, "distance", [2.6, 4.4])),
        pitch_deg=ov.get("pitch_deg", _sample_range(rng, sec, "pitch_deg", [32.0, 52.0])),
        yaw_deg=ov.get("yaw_deg", _sample_range(rng, sec, "yaw_deg", [-12.0, 12.0])),
        focal=ov.get("focal", _sample_range(rng, sec, "focal", [900.0, 1400.0])),
        cx=480.0 + float(rng.uniform(-30, 30)),
        cy=360.0 + float(rng.uniform(-20, 20)),
        worker_z=float(rng.uniform(3.2, 4.2)),
    )


def _style_for_family(family: str, rng: np.random.Generator, handed: str) -> dict[str, Any]:
    if family == "REMOVE_A":
        return {
            "variant": "two_hand_sides_up",
            "grasp_mode": "sides",
            "lift_dir": "up",
            "pause_mode": "short" if rng.random() < 0.7 else "none",
            "speed_mode": rng.choice(["normal", "slow"]),
            "one_then_two": False,
            "dominant": handed,
            "head_turn_deg": float(rng.uniform(-6, 6)),
            "ease": "cubic",
        }
    if family == "REMOVE_B":
        return {
            "variant": "one_then_two_brim_left",
            "grasp_mode": "brim",
            "lift_dir": rng.choice(["up", "left"]),
            "pause_mode": rng.choice(["short", "none"]),
            "speed_mode": rng.choice(["normal", "fast"]),
            "one_then_two": True,
            "dominant": handed,
            "head_turn_deg": float(rng.uniform(-8, 8)),
            "ease": "bezier",
        }
    if family == "REMOVE_C":
        return {
            "variant": "holdout_lateral_stutter",
            "grasp_mode": rng.choice(["sides", "brim"]),
            "lift_dir": rng.choice(["right", "left"]),
            "pause_mode": rng.choice(["stutter", "none", "short"]),
            "speed_mode": rng.choice(["slow", "fast", "normal"]),
            "one_then_two": bool(rng.random() < 0.6),
            "dominant": "left" if handed == "right" else "right",
            "head_turn_deg": float(rng.uniform(-22, 22)),
            "ease": rng.choice(["cubic", "smoothstep", "ease_out"]),
        }
    if family == "ADJUST_B":
        return {"variant": "near_remove_adjust", "grasp_mode": "sides", "pause_mode": "short", "dominant": handed}
    if family == "TOUCH_B":
        return {"variant": "two_hand_hold_one_leave", "dominant": handed}
    if family == "SCRATCH_B":
        return {"variant": "high_freq_oscillation", "dominant": handed}
    return {"variant": family.lower(), "dominant": handed, "grasp_mode": "sides"}


def sample_action(
    rng: np.random.Generator,
    split: str,
    family: str = "",
    handed: str = "right",
) -> ActionParams:
    cfg = load_config()
    sec = cfg.section("synthetic.ood.action") if split == "test" else cfg.section("synthetic.action")
    style = _style_for_family(family, rng, handed) if family else {}
    speed_mode = style.get("speed_mode", "normal")
    speed = _sample_range(rng, sec, "speed", [0.8, 1.25])
    if speed_mode == "slow":
        speed *= 0.62
    elif speed_mode == "fast":
        speed *= 1.45
    pause = _sample_range(rng, sec, "pause", [0.08, 0.22])
    pause_mode = style.get("pause_mode", "short")
    if pause_mode == "none":
        pause *= 0.15
    elif pause_mode == "stutter":
        pause *= 1.35
    return ActionParams(
        speed=float(np.clip(speed, 0.35, 2.2)),
        pause=float(pause),
        lift_height=_sample_range(rng, sec, "lift_height", [0.14, 0.30]),
        wrist_separation=_sample_range(rng, sec, "wrist_separation", [0.05, 0.14]),
        amplitude=_sample_range(rng, sec, "amplitude", [0.70, 1.20]),
        start_phase=float(rng.uniform(0.0, 0.18)),
        family=family,
        variant=str(style.get("variant", "")),
        dominant=str(style.get("dominant", handed)),
        grasp_mode=str(style.get("grasp_mode", "sides")),
        lift_dir=str(style.get("lift_dir", "up")),
        pause_mode=pause_mode,
        speed_mode=speed_mode,
        head_turn_deg=float(style.get("head_turn_deg", 0.0)),
        one_then_two=bool(style.get("one_then_two", False)),
        stutter=pause_mode == "stutter",
        ease=str(style.get("ease", "cubic")),
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
        consecutive_drop_prob=0.05 if split == "test" else 0.03,
        conf_flicker_prob=0.07 if split == "test" else 0.04,
        keypoint_swap_prob=0.02 if split == "test" else 0.01,
        track_interrupt_prob=0.03 if split == "test" else 0.015,
        bbox_jitter=4.0 if split == "test" else 2.5,
        ear_conf_degrade=0.22 if split == "test" else 0.14,
        person_scale_jitter=0.04 if split == "test" else 0.02,
    )


def _rest(body: BodyParams) -> np.ndarray:
    return apply_body_params(canonical_pose_3d(), body)


def _anchors(rest: np.ndarray) -> dict[str, np.ndarray]:
    scalp = 0.5 * (rest[L_EAR] + rest[R_EAR]) + np.array([0.0, 0.012, 0.015])
    return {
        "scalp": scalp,
        "grab_l": rest[L_EAR] + np.array([-0.03, 0.01, 0.02]),
        "grab_r": rest[R_EAR] + np.array([0.03, 0.01, 0.02]),
        "brim_l": rest[NOSE] + np.array([-0.06, 0.03, -0.04]),
        "brim_r": rest[NOSE] + np.array([0.06, 0.03, -0.04]),
        "nose": rest[NOSE] + np.array([0.0, -0.02, -0.02]),
        "brow": rest[NOSE] + np.array([0.0, 0.04, 0.0]),
        "home_l": rest[L_WRIST].copy(),
        "home_r": rest[R_WRIST].copy(),
    }


def _segment_u(i: int, a: int, b: int, kind: str) -> float:
    if b <= a:
        return 1.0
    return interp_ease((i - a) / max(b - a, 1), kind)


def _lift_offset(action: ActionParams, u: float) -> np.ndarray:
    h = action.lift_height * u
    sep = action.wrist_separation * u
    lat = 0.0
    if action.lift_dir == "left":
        lat = -0.16 * u
    elif action.lift_dir == "right":
        lat = 0.16 * u
    return np.array([lat, h, -0.10 * u]), sep


def _wrist_pair_at(
    i: int,
    n: int,
    rest: np.ndarray,
    anc: dict[str, np.ndarray],
    action: ActionParams,
    scenario: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return left/right wrist 3D and helmet-scale amount for frame i."""
    ease = action.ease
    amp = action.amplitude
    hand = action.dominant or "right"
    raise_end = int(n * (0.16 + 0.07 / max(action.speed, 0.4)))
    pause_end = raise_end + max(3, int(n * action.pause))
    if action.stutter:
        pause_end = raise_end + max(6, int(n * action.pause * 1.4))
    lift_end = min(n - 2, pause_end + int(n * (0.34 + 0.08 * amp)))
    home_l, home_r = anc["home_l"], anc["home_r"]
    g_l = anc["brim_l"] if action.grasp_mode == "brim" else anc["grab_l"]
    g_r = anc["brim_r"] if action.grasp_mode == "brim" else anc["grab_r"]
    amount = 0.0

    if scenario == "IDLE":
        ph = 2 * math.pi * i / max(12.0, 18.0 / action.speed)
        wr = home_r + np.array([0.03 * math.sin(ph), 0.04 * math.sin(ph), 0.0]) * amp
        return home_l, wr, 0.0

    if scenario == "HEAD_SCRATCH":
        side_home = home_r if hand == "right" else home_l
        scalp = anc["scalp"]
        freq = 5.1 if action.variant == "high_freq_oscillation" else 3.4
        if i < raise_end:
            w = lerp(side_home, scalp, _segment_u(i, 0, raise_end, ease))
        elif i < n - 8:
            t = (i - raise_end) / 20.0 * action.speed
            rad = 0.016 * amp
            w = scalp + np.array(
                [rad * math.sin(2 * math.pi * freq * t), 0.35 * rad * math.cos(2 * math.pi * freq * t), 0.4 * rad * math.sin(2 * math.pi * 4.0 * t)]
            )
        else:
            w = lerp(scalp, side_home, _segment_u(i, n - 8, n - 1, "ease_out"))
        return (home_l, w, 0.0) if hand == "right" else (w, home_r, 0.0)

    if scenario in ("ONE_HAND_HEAD_TOUCH", "FACE_TOUCH"):
        side_home = home_r if hand == "right" else home_l
        target = anc["scalp"] + np.array([0.02 if hand == "right" else -0.02, 0.0, 0.0])
        if scenario == "FACE_TOUCH":
            target = anc["nose"]
        if i < raise_end:
            w = lerp(side_home, target, _segment_u(i, 0, raise_end, ease))
        elif i < pause_end + 10:
            w = target + rng.normal(0, 0.002, size=3)
        else:
            w = lerp(target, side_home, _segment_u(i, pause_end + 10, n - 1, "ease_out"))
        return (home_l, w, 0.0) if hand == "right" else (w, home_r, 0.0)

    if scenario == "TWO_HAND_HEAD_TOUCH":
        tl = anc["scalp"] + np.array([-0.04, 0.01, 0.01])
        tr = anc["scalp"] + np.array([0.04, 0.01, 0.01])
        if i < raise_end:
            u = _segment_u(i, 0, raise_end, ease)
            return lerp(home_l, tl, u), lerp(home_r, tr, u), 0.0
        if i < pause_end + 8:
            # Harder negative: short hold, then one hand leaves (TOUCH_B).
            if action.variant == "two_hand_hold_one_leave" and i > pause_end + 2:
                leave = lerp(tl, home_l, _segment_u(i, pause_end + 2, n - 1, "ease_out"))
                return leave, tr + rng.normal(0, 0.002, 3), 0.0
            return tl + rng.normal(0, 0.002, 3), tr + rng.normal(0, 0.002, 3), 0.0
        u = _segment_u(i, pause_end + 8, n - 1, "ease_out")
        return lerp(tl, home_l, u), lerp(tr, home_r, u), 0.0

    if scenario == "HELMET_ADJUST":
        # Grasp brim then settle. Harder ADJUST_B has a tiny upward nudge still
        # well below lift thresholds, then returns to rest (image-y increases).
        tiny_h = 0.018 * amp if action.variant == "near_remove_adjust" else 0.0
        tiny_sep = 0.022 * amp if action.variant == "near_remove_adjust" else 0.0
        if i < raise_end:
            u = _segment_u(i, 0, raise_end, ease)
            return lerp(home_l, g_l, u), lerp(home_r, g_r, u), 0.0
        if i < pause_end + 6:
            wig = 0.006 * amp
            extra = np.array([0.0, tiny_h, 0.0])
            wl = g_l + extra + np.array([-tiny_sep * 0.35 + wig * math.sin(i / 2.0), 0.003 * math.sin(i / 3.0), 0.0])
            wr = g_r + extra + np.array([tiny_sep * 0.35 - wig * math.sin(i / 2.0), 0.003 * math.cos(i / 3.0), 0.0])
            return wl, wr, 0.0
        u = _segment_u(i, pause_end + 6, n - 1, "ease_out")
        return lerp(g_l + np.array([0.0, tiny_h, 0.0]), home_l, u), lerp(g_r + np.array([0.0, tiny_h, 0.0]), home_r, u), 0.0

    if scenario == "WIPE_SWEAT":
        side_home = home_r if hand == "right" else home_l
        brow = anc["brow"]
        across = brow + np.array([0.09 if hand == "left" else -0.09, -0.02, 0.0])
        if i < raise_end:
            w = lerp(side_home, brow, _segment_u(i, 0, raise_end, ease))
        elif i < pause_end + 8:
            w = lerp(brow, across, _segment_u(i, raise_end, pause_end + 8, ease))
        else:
            w = lerp(across, side_home, _segment_u(i, pause_end + 8, n - 1, "ease_out"))
        return (home_l, w, 0.0) if hand == "right" else (w, home_r, 0.0)

    if scenario == "LOOK_DOWN":
        ph = 2 * math.pi * i / max(12.0, 18.0 / action.speed)
        wr = home_r + np.array([0.03 * math.sin(ph), 0.04 * math.sin(ph), 0.0]) * amp
        return home_l, wr, 0.0

    if scenario == "RAISE_ARMS":
        u = 0.5 * (1 - math.cos(2 * math.pi * i / max(n, 2)))
        up_l = home_l + np.array([-0.05, 0.45 * amp, -0.05])
        up_r = home_r + np.array([0.05, 0.45 * amp, -0.05])
        return lerp(home_l, up_l, u), lerp(home_r, up_r, u), 0.0

    if scenario == "STRETCH":
        # Hands above head but no ear grasp.
        u = 0.5 * (1 - math.cos(2 * math.pi * i / max(n, 2)))
        out_l = rest[L_SHOULDER] + np.array([-0.18 * amp, 0.38 * amp, -0.04])
        out_r = rest[R_SHOULDER] + np.array([0.18 * amp, 0.38 * amp, -0.04])
        return lerp(home_l, out_l, u), lerp(home_r, out_r, u), 0.0

    if scenario == "PHONE_NEAR_HEAD":
        side_home = home_r if hand == "right" else home_l
        ear = rest[R_EAR] if hand == "right" else rest[L_EAR]
        target = ear + np.array([0.04 if hand == "right" else -0.04, -0.02, -0.03])
        if i < raise_end:
            w = lerp(side_home, target, _segment_u(i, 0, raise_end, ease))
        elif i < n - 8:
            w = target + rng.normal(0, 0.0015, 3)
        else:
            w = lerp(target, side_home, _segment_u(i, n - 8, n - 1, "ease_out"))
        return (home_l, w, 0.0) if hand == "right" else (w, home_r, 0.0)

    if scenario == "HELMET_REMOVE":
        join = raise_end if not action.one_then_two else int(raise_end * 0.55)
        if i < join:
            u = _segment_u(i, 0, join, ease)
            if action.one_then_two:
                if action.dominant == "left":
                    return lerp(home_l, g_l, u), home_r, 0.0
                return home_l, lerp(home_r, g_r, u), 0.0
            return lerp(home_l, g_l, u), lerp(home_r, g_r, u), 0.0
        if i < raise_end:
            u = _segment_u(i, join, raise_end, ease)
            if action.one_then_two:
                if action.dominant == "left":
                    return g_l, lerp(home_r, g_r, u), 0.0
                return lerp(home_l, g_l, u), g_r, 0.0
            return lerp(home_l, g_l, u), lerp(home_r, g_r, u), 0.0
        if i < pause_end:
            if action.stutter and (i // 3) % 2 == 0:
                return g_l + rng.normal(0, 0.001, 3), g_r + rng.normal(0, 0.001, 3), 0.0
            return g_l + rng.normal(0, 0.0012, 3), g_r + rng.normal(0, 0.0012, 3), 0.0
        u = _segment_u(i, pause_end, lift_end, ease)
        off, sep = _lift_offset(action, u)
        wl = g_l + np.array([-sep, 0.0, 0.0]) + off
        wr = g_r + np.array([sep, 0.0, 0.0]) + off
        amount = 0.55 * u * amp
        return wl, wr, amount

    if scenario == "HELMET_PUT_ON":
        sep = action.wrist_separation
        h = action.lift_height
        high_l = g_l + np.array([-sep, h, -0.10])
        high_r = g_r + np.array([sep, h, -0.10])
        if i < raise_end:
            u = _segment_u(i, 0, raise_end, ease)
            return lerp(high_l, g_l, u), lerp(high_r, g_r, u), 0.45 * (1 - u)
        if i < pause_end:
            return g_l + rng.normal(0, 0.0012, 3), g_r + rng.normal(0, 0.0012, 3), 0.0
        u = _segment_u(i, pause_end, n - 1, "ease_out")
        return lerp(g_l, home_l, u), lerp(g_r, home_r, u), 0.0

    # UNKNOWN_RANDOM_MOTION
    ph = i / max(n, 1)
    wl = home_l + amp * np.array([0.12 * math.sin(4 * ph + 0.2), 0.18 * math.sin(3 * ph), 0.05 * math.cos(5 * ph)])
    wr = home_r + amp * np.array([0.10 * math.cos(3.2 * ph), 0.16 * math.sin(4.1 * ph + 1.0), 0.04 * math.sin(2 * ph)])
    return wl, wr, 0.0


def generate_poses_3d(
    scenario: str,
    rng: np.random.Generator,
    body: BodyParams,
    action: ActionParams,
    n_frames: int,
    fps: float = 20.0,
) -> np.ndarray:
    rest = _rest(body)
    anc = _anchors(rest)
    dt = 1.0 / max(fps, 1.0)
    left = np.zeros((n_frames, 3), dtype=np.float64)
    right = np.zeros((n_frames, 3), dtype=np.float64)
    amount = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        wl, wr, am = _wrist_pair_at(i, n_frames, rest, anc, action, scenario, rng)
        left[i], right[i], amount[i] = wl, wr, am
    left = constrain_path(left, dt)
    right = constrain_path(right, dt)
    poses = np.zeros((n_frames, 17, 3), dtype=np.float64)
    for i in range(n_frames):
        pose = rest.copy()
        pose += rng.normal(0.0, 0.00035, size=pose.shape)
        place_arm_3d(pose, "left", left[i], body)
        place_arm_3d(pose, "right", right[i], body)
        if amount[i] > 0:
            scale_head_toward_camera(pose, float(amount[i]))
        if scenario in ("HELMET_REMOVE", "LOOK_DOWN") and abs(action.head_turn_deg) > 1.0:
            u = i / max(n_frames - 1, 1)
            rotate_head(pose, action.head_turn_deg * u, pitch_deg=6.0 * u if scenario == "LOOK_DOWN" else 0.0)
        if scenario == "LOOK_DOWN":
            u = 0.5 * (1 - math.cos(2 * math.pi * i / max(n_frames, 2)))
            for idx in FACE_IDX:
                pose[idx] = pose[idx] + np.array([0.0, -0.06 * u * action.amplitude, 0.04 * u])
        poses[i] = pose
    return poses


def generate_scenario_sequence(
    scenario: str,
    rng: np.random.Generator,
    body: BodyParams,
    cam: HighAngleCamera,
    action: ActionParams,
    n_frames: int,
    fps: float = 20.0,
    world_sigma: float = 0.0014,
) -> np.ndarray:
    poses = generate_poses_3d(scenario, rng, body, action, n_frames, fps=fps)
    seq = np.zeros((n_frames, 17, 2), dtype=np.float64)
    for i in range(n_frames):
        pose = poses[i] + rng.normal(0.0, world_sigma, size=poses[i].shape)
        seq[i] = cam.project(pose)
    return seq


def _choose_family(rng: np.random.Generator, scenario: str, split: str) -> str:
    cfg = load_config()
    return choose_family(
        scenario,
        split,
        rng,
        train_families=list(cfg.get("synthetic.train_families") or []),
        holdout_families=list(cfg.get("synthetic.holdout_families") or []),
    )


def generate_one(
    seed: int,
    scenario: str,
    split: str = "train",
    n_frames: int | None = None,
    apply_noise: bool = True,
    family: str | None = None,
    fps: float | None = None,
    camera_overrides: dict[str, float] | None = None,
    body: BodyParams | None = None,
    cam: HighAngleCamera | None = None,
    action: ActionParams | None = None,
    noise: NoiseParams | None = None,
    pair_id: str = "",
    pair_role: str = "",
) -> tuple[np.ndarray, np.ndarray, SampleMeta]:
    from helmet_action.models.labels import SCENARIO_TO_CLASS

    rng = np.random.default_rng(int(seed))
    cfg = load_config()
    fps_v = float(fps if fps is not None else cfg.get("synthetic.fps", 20.0))
    fam = family or _choose_family(rng, scenario, split)
    body = body or sample_body(rng, split)
    cam = cam or sample_camera(rng, split, overrides=camera_overrides)
    action = action or sample_action(rng, split, family=fam, handed=body.handedness)
    if not action.family:
        action.family = fam
    noise = noise or sample_noise(rng, split)
    duration = 3.0 / max(action.speed, 0.45)
    n = int(n_frames or round(duration * fps_v))
    n = int(np.clip(n, 28, 120 if fps_v >= 45 else 80))
    seq = generate_scenario_sequence(scenario, rng, body, cam, action, n, fps=fps_v)
    conf = np.ones((seq.shape[0], 17), dtype=np.float64)
    if apply_noise:
        seq, conf = apply_pose_noise(seq, rng, noise)
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
        family=fam,
        fps=fps_v,
        occlusion={"dropout_prob": noise.dropout_prob, "occlusion_prob": noise.occlusion_prob},
        variant=action.variant,
        pair_id=pair_id,
        pair_role=pair_role,
    )
    return seq, conf, meta


def generate_counterfactual_pair(
    seed: int,
    kind: str = "adjust_vs_remove",
    split: str = "train",
    n_frames: int | None = None,
    shared_frac: float = 0.62,
) -> tuple[tuple[np.ndarray, np.ndarray, SampleMeta], tuple[np.ndarray, np.ndarray, SampleMeta]]:
    """Shared person/camera/noise/prefix; last phase differs.

    kind: adjust_vs_remove | touch_vs_remove
    """
    rng = np.random.default_rng(int(seed))
    body = sample_body(rng, split)
    cam = sample_camera(rng, split)
    noise = sample_noise(rng, split)
    action_rm = sample_action(rng, split, family="REMOVE_A", handed=body.handedness)
    action_rm.pause_mode = "short"
    action_rm.one_then_two = False
    action_rm.lift_dir = "up"
    action_neg = sample_action(rng, split, family="ADJUST_A" if kind == "adjust_vs_remove" else "TOUCH_A", handed=body.handedness)
    fps = 20.0
    n = int(n_frames or 60)
    scen_neg = "HELMET_ADJUST" if kind == "adjust_vs_remove" else "TWO_HAND_HEAD_TOUCH"
    # Independent 3D then copy prefix wrists conceptually by regenerating with same rng
    # for the shared portion: build both fully, then splice projected+noise prefix.
    rng_a = np.random.default_rng(int(seed) + 17)
    rng_b = np.random.default_rng(int(seed) + 17)
    seq_neg = generate_scenario_sequence(scen_neg, rng_a, body, cam, action_neg, n, fps=fps)
    seq_rm = generate_scenario_sequence("HELMET_REMOVE", rng_b, body, cam, action_rm, n, fps=fps)
    split_i = int(np.clip(shared_frac * n, 8, n - 6))
    seq_rm = seq_rm.copy()
    seq_rm[:split_i] = seq_neg[:split_i]
    rng_n = np.random.default_rng(int(seed) + 91)
    noisy_neg, conf_neg = apply_pose_noise(seq_neg, rng_n, noise)
    rng_n2 = np.random.default_rng(int(seed) + 91)
    noisy_rm, conf_rm = apply_pose_noise(seq_rm, rng_n2, noise)
    # Force identical noise on the shared prefix after independent stretching.
    t_share = min(split_i, noisy_neg.shape[0], noisy_rm.shape[0])
    noisy_rm = noisy_rm.copy()
    conf_rm = conf_rm.copy()
    noisy_rm[:t_share] = noisy_neg[:t_share]
    conf_rm[:t_share] = conf_neg[:t_share]
    pair_id = f"cf_{seed}_{kind}"
    from helmet_action.models.labels import SCENARIO_TO_CLASS

    def _meta(scen, seq, action, role):
        return SampleMeta(
            seed=int(seed),
            split=split,
            scenario=scen,
            label=SCENARIO_TO_CLASS[scen].value,
            generator_version=GENERATOR_VERSION,
            camera=cam.to_dict(),
            body=body.to_dict(),
            action=action.to_dict(),
            noise=noise.to_dict(),
            n_frames=int(seq.shape[0]),
            family=action.family,
            fps=fps,
            variant=action.variant,
            pair_id=pair_id,
            pair_role=role,
        )

    a = (noisy_neg, conf_neg, _meta(scen_neg, noisy_neg, action_neg, "negative"))
    b = (noisy_rm, conf_rm, _meta("HELMET_REMOVE", noisy_rm, action_rm, "remove"))
    return a, b


def scenario_mix(quick: bool = False) -> list[str]:
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
