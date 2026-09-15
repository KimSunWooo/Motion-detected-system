#!/usr/bin/env python3
"""하이 앵글 CCTV용 2D 스켈레톤 동작 분류 프로토타입.

공장 천장/벽면 상단에서 작업자를 대각선으로 내려다보는 구도(YOLOv8-Pose 17점)를
전제로, '단순 머리 긁기(정상)'와 '안전모 벗기 시도(예방 알림)'을 구분한다.

왜 눈높이 Y축 상승을 쓰지 않는가
--------------------------------
하이 앵글에서는 손이 머리로 가는 동작이 화면의 '위'가 아니라
카메라(렌즈) 쪽으로 다가가는 Z 이동이다. 투영 평면에서는
키포인트 덩어리가 커지는 팽창(scale-up)이나, 양손목이 귀 쪽에서
좌우로 벌어지는 변화로 관측된다.

정규화
------
목~골반 길이는 투시 왜곡으로 심하게 단축되므로 사용하지 않는다.
양쪽 어깨 픽셀 거리만을 scale=1.0 으로 쓴다.

    neck  = (L_shoulder + R_shoulder) / 2
    s     = ||L_shoulder − R_shoulder||
    p̂    = (p − neck) / s
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

import matplotlib

if os.environ.get("DISPLAY") in (None, "") or os.environ.get("MPLBACKEND") == "Agg":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

_KOREAN_FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"


def _configure_korean_font() -> None:
    if Path(_KOREAN_FONT).exists():
        font_manager.fontManager.addfont(_KOREAN_FONT)
        name = font_manager.FontProperties(fname=_KOREAN_FONT).get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


_configure_korean_font()

# ---------------------------------------------------------------------------
# COCO / YOLOv8-Pose 17 키포인트
# ---------------------------------------------------------------------------
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

SKELETON_BONES = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE),
    (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE),
    (R_KNEE, R_ANKLE),
    (NOSE, L_EYE),
    (NOSE, R_EYE),
    (L_EYE, L_EAR),
    (R_EYE, R_EAR),
]

FACE_IDX = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)

# ---------------------------------------------------------------------------
# 하이 앵글 머리 영역 (정규화 공간, 어깨너비 = 1.0)
# 위에서 내려다보면 두상이 넓게 보이므로 bbox 를 눈높이보다 넓고 둥글게 잡는다.
# ---------------------------------------------------------------------------
HEAD_HALF_WIDTH = 0.52
HEAD_HALF_HEIGHT = 0.46
CENTER_RADIUS = 0.24  # 긁기: 두상 중심부
EAR_RADIUS = 0.24  # 안전모: 귀/챙 모서리

# ---------------------------------------------------------------------------
# 분류 임계값 — 전부 어깨너비 단위 또는 무차원
# ---------------------------------------------------------------------------
MIN_CENTER_FRAMES = 8
MIN_EAR_FRAMES = 8
PAUSE_FRAMES = 8

TAU_SCRATCH_RADIUS = 0.11  # 긁기 궤적의 국소 반경 상한
TAU_SCRATCH_STD = 0.028  # 그 반경 안에서의 고주파 진동 (std)
MIN_OSCILLATIONS = 4

TAU_PAUSE_STD = 0.016  # 양손목 파지 후 일시 정지
TAU_SPREAD = 0.10  # 양손목 사이 거리 증가량
TAU_SCALE_UP = 0.07  # 머리 겉보기 크기(귀 간격) 상대 증가율


class ActionLabel(str, Enum):
    SCRATCH = "scratch"
    HELMET_OFF = "helmet_off"
    NO_CONTACT = "no_contact"
    UNKNOWN_CONTACT = "unknown_contact"


LABEL_KO = {
    ActionLabel.SCRATCH: "단순 머리 긁기 (정상)",
    ActionLabel.HELMET_OFF: "안전모 벗기 시도 (예방 알림)",
    ActionLabel.NO_CONTACT: "머리 비접촉 (동작 없음)",
    ActionLabel.UNKNOWN_CONTACT: "머리 접촉 · 판정 보류",
}


# ===========================================================================
# 하이 앵글 핀홀 카메라 (천장/벽면 상단 → 대각선 하향)
# ===========================================================================
@dataclass
class HighAngleCamera:
    """세계좌표(Y-up, X-right, Z-실내깊이)를 픽셀로 투영한다.

    카메라가 작업자보다 높고 앞쪽(벽)에 있으므로, 머리·어깨는 렌즈에
    가깝고 골반·다리는 멀다. 원근 나눗셈으로 다리가 짧게 겹쳐 보인다.
    """

    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def factory_ceiling(cls) -> "HighAngleCamera":
        # 높이 4.5 m, 작업자까지 수평 약 3.3 m → 하향각 ≈ 45°
        return cls(
            eye=np.array([0.18, 5.70, 1.45], dtype=np.float64),
            target=np.array([0.00, 1.10, 3.50], dtype=np.float64),
            up=np.array([0.00, 1.00, 0.00], dtype=np.float64),
            fx=1180.0,
            fy=1180.0,
            cx=480.0,
            cy=360.0,
        )

    def _extrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        eye, target, up = self.eye, self.target, self.up
        z_cam = target - eye
        z_cam = z_cam / np.linalg.norm(z_cam)
        x_cam = np.cross(up, z_cam)
        x_cam = x_cam / np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        y_cam = y_cam / np.linalg.norm(y_cam)
        # world → camera: Xc = R (Xw - eye), rows are camera axes
        r = np.stack([x_cam, y_cam, z_cam], axis=0)
        return r, eye

    def project(self, points_xyz: np.ndarray) -> np.ndarray:
        """(..., 3) 세계좌표 → (..., 2) 픽셀. v 는 아래가 양수."""
        pts = np.asarray(points_xyz, dtype=np.float64)
        r, eye = self._extrinsics()
        cam = np.einsum("ij,...j->...i", r, pts - eye)
        z = np.clip(cam[..., 2], 1e-6, None)
        u = self.fx * (cam[..., 0] / z) + self.cx
        # 카메라 Y 위쪽(+)을 이미지 아래가 양수가 되도록 뒤집는다.
        v = -self.fy * (cam[..., 1] / z) + self.cy
        return np.stack([u, v], axis=-1)


def canonical_pose_3d() -> np.ndarray:
    """서 있는 작업자의 세계좌표 (m). 원점은 발 사이 바닥, Y가 키."""
    p = np.zeros((17, 3), dtype=np.float64)
    # 얼굴이 카메라(작은 Z) 쪽을 향하게 두어 정수리와 얼굴이 함께 보이게 한다.
    p[NOSE] = (0.00, 1.66, 3.44)
    p[L_EYE] = (-0.035, 1.68, 3.45)
    p[R_EYE] = (0.035, 1.68, 3.45)
    p[L_EAR] = (-0.090, 1.64, 3.54)
    p[R_EAR] = (0.090, 1.64, 3.54)
    p[L_SHOULDER] = (-0.205, 1.42, 3.52)
    p[R_SHOULDER] = (0.205, 1.42, 3.52)
    p[L_ELBOW] = (-0.27, 1.12, 3.54)
    p[R_ELBOW] = (0.27, 1.12, 3.54)
    p[L_WRIST] = (-0.25, 0.86, 3.55)
    p[R_WRIST] = (0.25, 0.86, 3.55)
    p[L_HIP] = (-0.13, 0.94, 3.50)
    p[R_HIP] = (0.13, 0.94, 3.50)
    p[L_KNEE] = (-0.13, 0.50, 3.52)
    p[R_KNEE] = (0.13, 0.50, 3.52)
    p[L_ANKLE] = (-0.12, 0.07, 3.50)
    p[R_ANKLE] = (0.12, 0.07, 3.50)
    return p


# ===========================================================================
# 조건 3: 어깨너비만 사용하는 정규화
# ===========================================================================
@dataclass
class NormalizeInfo:
    origin: np.ndarray  # 목 = 어깨 중점 (픽셀)
    scale: float  # 어깨너비 (픽셀). 정규화 단위 1.0


def _as_sequence(keypoints: np.ndarray) -> np.ndarray:
    arr = np.asarray(keypoints, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    arr = arr[..., :2]
    if arr.shape[-2] != 17:
        raise ValueError(f"COCO 17 키포인트가 필요합니다. shape={arr.shape}")
    return arr


def compute_neck(kpts: np.ndarray) -> np.ndarray:
    return 0.5 * (kpts[..., L_SHOULDER, :] + kpts[..., R_SHOULDER, :])


def compute_shoulder_width(kpts: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        kpts[..., L_SHOULDER, :] - kpts[..., R_SHOULDER, :], axis=-1
    )


def normalize_keypoints(keypoints: np.ndarray) -> tuple[np.ndarray, list[NormalizeInfo]]:
    """픽셀 좌표 → 목 원점 / 어깨너비 단위.

    하이 앵글에서는 몸통 길이가 프레임마다 심하게 변하므로
    torso length 를 scale 로 쓰지 않는다. 어깨가 거의 겹치면
    (측면 가림) 수치 폭주만 막기 위해 epsilon 을 둔다.

        p̂ = (p − neck) / max(||Ls − Rs||, ε)
    """
    seq = _as_sequence(keypoints)
    neck = compute_neck(seq)
    width = compute_shoulder_width(seq)
    scale = np.maximum(width, 1e-6)
    normalized = (seq - neck[:, None, :]) / scale[:, None, None]
    infos = [
        NormalizeInfo(origin=neck[i], scale=float(scale[i]))
        for i in range(seq.shape[0])
    ]
    return normalized, infos


# ===========================================================================
# 동적 머리 영역: 중심부 vs 귀 모서리
# ===========================================================================
@dataclass
class HeadRegions:
    """정규화 공간의 머리 bbox · 중심원 · 좌/우 귀 원."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    center: np.ndarray
    center_r: float
    left_ear: np.ndarray
    right_ear: np.ndarray
    ear_r: float

    def as_xywh(self) -> tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max - self.x_min, self.y_max - self.y_min

    def in_bbox(self, p: np.ndarray) -> bool:
        return self.x_min <= float(p[0]) <= self.x_max and self.y_min <= float(p[1]) <= self.y_max

    def in_center(self, p: np.ndarray) -> bool:
        return float(np.linalg.norm(p - self.center)) <= self.center_r

    def in_left_ear(self, p: np.ndarray) -> bool:
        return float(np.linalg.norm(p - self.left_ear)) <= self.ear_r

    def in_right_ear(self, p: np.ndarray) -> bool:
        return float(np.linalg.norm(p - self.right_ear)) <= self.ear_r


def head_center_norm(pose: np.ndarray) -> np.ndarray:
    return 0.5 * (pose[L_EAR] + pose[R_EAR])


def head_scale_norm(pose: np.ndarray) -> float:
    """겉보기 머리 크기 ≈ 정규화된 귀-귀 거리. 카메라로 다가오면 커진다."""
    return float(np.linalg.norm(pose[L_EAR] - pose[R_EAR]))


def compute_head_regions(pose_norm: np.ndarray) -> HeadRegions:
    """어깨너비에 비례한 동적 머리 영역.

    bbox 중심은 귀 중점(두상 중심). 하이 앵글에서 안전모가 가장 크게
    보이는 지점이다. 절대 픽셀 상수는 사용하지 않는다.
    """
    c = head_center_norm(pose_norm)
    return HeadRegions(
        x_min=float(c[0] - HEAD_HALF_WIDTH),
        y_min=float(c[1] - HEAD_HALF_HEIGHT),
        x_max=float(c[0] + HEAD_HALF_WIDTH),
        y_max=float(c[1] + HEAD_HALF_HEIGHT),
        center=c,
        center_r=CENTER_RADIUS,
        left_ear=pose_norm[L_EAR].copy(),
        right_ear=pose_norm[R_EAR].copy(),
        ear_r=EAR_RADIUS,
    )


# ===========================================================================
# 조건 2: 하이 앵글 특화 분류
# ===========================================================================
@dataclass
class FeatureReport:
    center_frames: int = 0
    both_ear_frames: int = 0
    active_wrist: str = "none"
    scratch_radius: float = 0.0
    scratch_std: float = 0.0
    n_oscillations: int = 0
    pause_detected: bool = False
    pause_std: float = 0.0
    wrist_spread: float = 0.0
    head_scale_up: float = 0.0
    d_wrist_pause: float = 0.0
    d_wrist_late: float = 0.0


@dataclass
class ClassificationResult:
    label: ActionLabel
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


def _std(x: np.ndarray) -> float:
    if np.asarray(x).size < 2:
        return 0.0
    return float(np.std(np.asarray(x, dtype=np.float64), ddof=0))


def _smooth(y: np.ndarray, k: int = 5) -> np.ndarray:
    if y.size < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="same")


def _zero_crossings(y: np.ndarray, deadband: float = 0.010) -> int:
    sm = _smooth(y)
    if sm.size < 3:
        return 0
    d = np.diff(sm)
    d = d[np.abs(d) > deadband]
    if d.size < 2:
        return 0
    s = np.sign(d)
    return int(np.sum(s[1:] * s[:-1] < 0))


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    best = (0, 0)
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if start is not None and len(mask) - start > best[1] - best[0]:
        best = (start, len(mask))
    return best


def _first_pause(xy: np.ndarray, window: int = PAUSE_FRAMES, tau: float = TAU_PAUSE_STD) -> tuple[int | None, float]:
    """(T, 2) 궤적에서 창 표준편차(두 축 RMS)가 tau 미만인 첫 구간."""
    t = xy.shape[0]
    if t < window:
        s = float(np.sqrt(_std(xy[:, 0]) ** 2 + _std(xy[:, 1]) ** 2))
        return (0 if s < tau else None), s
    best = float("inf")
    first = None
    for i in range(0, t - window + 1):
        w = xy[i : i + window]
        s = float(np.sqrt(_std(w[:, 0]) ** 2 + _std(w[:, 1]) ** 2))
        best = min(best, s)
        if first is None and s < tau:
            first = i
    return first, best


def extract_features(seq_norm: np.ndarray) -> FeatureReport:
    """하이 앵글 판정 통계량.

    긁기
        한 손목이 두상 *중심원* 안에 머물며, 국소 반경은 작고
        (x,y) 표준편차·영점교차는 크다 → 짧은 고주파 진동.

    안전모 벗기
        왼손목∈왼귀원 AND 오른손목∈오른귀원 으로 파지(일시 정지)한 뒤
        양손목 거리 증가(벌어짐) 또는 귀 간격 / 어깨너비 증가(팽창).
        팽창은 헬멧이 천장 카메라 쪽으로 들어 올려질 때 생긴다.
    """
    t = seq_norm.shape[0]
    regions = [compute_head_regions(seq_norm[i]) for i in range(t)]
    lw = seq_norm[:, L_WRIST]
    rw = seq_norm[:, R_WRIST]

    in_c_l = np.array([regions[i].in_center(lw[i]) for i in range(t)])
    in_c_r = np.array([regions[i].in_center(rw[i]) for i in range(t)])
    in_el = np.array([regions[i].in_left_ear(lw[i]) for i in range(t)])
    in_er = np.array([regions[i].in_right_ear(rw[i]) for i in range(t)])
    both_ears = in_el & in_er

    feat = FeatureReport()
    c_l0, c_l1 = _longest_run(in_c_l)
    c_r0, c_r1 = _longest_run(in_c_r)
    if (c_r1 - c_r0) >= (c_l1 - c_l0):
        feat.active_wrist = "right"
        c0, c1 = c_r0, c_r1
        active = rw
    else:
        feat.active_wrist = "left"
        c0, c1 = c_l0, c_l1
        active = lw
    feat.center_frames = c1 - c0

    e0, e1 = _longest_run(both_ears)
    feat.both_ear_frames = e1 - e0

    if feat.center_frames >= 3:
        dwell = active[c0:c1]
        centroid = dwell.mean(axis=0)
        rad = np.linalg.norm(dwell - centroid, axis=1)
        feat.scratch_radius = float(np.quantile(rad, 0.85))
        feat.scratch_std = float(np.sqrt(_std(dwell[:, 0]) ** 2 + _std(dwell[:, 1]) ** 2))
        feat.n_oscillations = _zero_crossings(dwell[:, 0]) + _zero_crossings(dwell[:, 1])

    if feat.both_ear_frames >= 3:
        # 파지 구간 + 그 이후(헬멧이 bbox 밖으로 팽창해도 추적)
        trail_end = min(t, e1 + 16)
        pair = np.stack([lw[e0:trail_end], rw[e0:trail_end]], axis=1)  # (T, 2, 2)
        mean_xy = pair.mean(axis=1)
        pause_i, pause_std = _first_pause(mean_xy)
        feat.pause_std = pause_std
        feat.pause_detected = pause_i is not None
        if pause_i is None:
            pause_i, pause_n = 0, min(PAUSE_FRAMES, pair.shape[0])
        else:
            pause_n = min(PAUSE_FRAMES, pair.shape[0] - pause_i)

        d = np.linalg.norm(lw[e0:trail_end] - rw[e0:trail_end], axis=1)
        hs = np.array([head_scale_norm(seq_norm[i]) for i in range(e0, trail_end)])
        d_pause = float(np.median(d[pause_i : pause_i + pause_n]))
        h_pause = float(np.median(hs[pause_i : pause_i + pause_n]))
        late = slice(pause_i + pause_n, None)
        if d[late].size == 0:
            late = slice(pause_i, None)
        feat.d_wrist_pause = d_pause
        feat.d_wrist_late = float(np.max(d[late])) if d[late].size else d_pause
        feat.wrist_spread = feat.d_wrist_late - d_pause
        h_late = float(np.max(hs[late])) if hs[late].size else h_pause
        feat.head_scale_up = (h_late / max(h_pause, 1e-6)) - 1.0
    return feat


def classify_pose_sequence(keypoints: np.ndarray) -> ClassificationResult:
    """시퀀스 분류. 안전 우선으로 헬멧 탈착을 먼저 검사한다.

    수식 요약
    --------
    정규화        p̂ = (p − neck) / ||Ls − Rs||

    긁기 (정상)
        한 손목이 중심원(r=0.24)에 T≥Tmin 체류
        radius_85% = Q_0.85( ||ŵ − mean(ŵ)|| )  ≤  τ_radius
        σ_xy = sqrt(σx² + σy²)                 ≥  τ_std
        osc  = ZC(ŵx) + ZC(ŵy)                 ≥  N_osc
        (반경은 작고 진동은 큼 = 제자리 고주파)

    안전모 벗기 (경보)
        왼손목∈왼귀원 ∧ 오른손목∈오른귀원, T≥Tmin
        파지 창에서 σ_xy < τ_pause
        이후  spread = max||ŵL−ŵR|| − med_pause(||ŵL−ŵR||)  > τ_spread
          또는  scale = max(d_ear)/med_pause(d_ear) − 1      > τ_scale
        d_ear 는 이미 어깨너비로 나눠진 귀 간격이므로,
        어깨는 그대로인데 헬멧만 렌즈로 다가오면 scale-up 이 관측된다.
    """
    seq_norm, _ = normalize_keypoints(keypoints)
    feat = extract_features(seq_norm)
    notes: list[str] = [
        "하이 앵글 정규화: p̂ = (p − neck) / shoulder_width  (torso 길이 미사용).",
        f"중심부 체류={feat.center_frames}fr ({feat.active_wrist}), "
        f"양귀 동시 파지={feat.both_ear_frames}fr.",
    ]

    helmet_rule = (
        feat.both_ear_frames >= MIN_EAR_FRAMES
        and feat.pause_detected
        and (feat.wrist_spread > TAU_SPREAD or feat.head_scale_up > TAU_SCALE_UP)
    )
    scratch_rule = (
        feat.center_frames >= MIN_CENTER_FRAMES
        and feat.scratch_radius <= TAU_SCRATCH_RADIUS
        and feat.scratch_std >= TAU_SCRATCH_STD
        and feat.n_oscillations >= MIN_OSCILLATIONS
        and feat.both_ear_frames < MIN_EAR_FRAMES
    )

    if feat.center_frames < MIN_CENTER_FRAMES and feat.both_ear_frames < MIN_EAR_FRAMES:
        notes.append("머리 중심부·귀 모서리 모두 최소 체류 미달 → 비접촉.")
        label, conf = ActionLabel.NO_CONTACT, 0.95
    elif helmet_rule:
        conf = float(
            np.clip(
                0.60
                + 0.20 * min(max(feat.wrist_spread, 0) / (2 * TAU_SPREAD), 1)
                + 0.20 * min(max(feat.head_scale_up, 0) / (2 * TAU_SCALE_UP), 1),
                0.60,
                0.99,
            )
        )
        notes.append(
            "통계량: "
            f"pause_std={feat.pause_std:.4f} (< {TAU_PAUSE_STD}), "
            f"spread={feat.wrist_spread:.3f} (τ={TAU_SPREAD}), "
            f"head_scale_up={feat.head_scale_up:.3f} (τ={TAU_SCALE_UP})."
        )
        notes.append(
            "규칙 HELMET_OFF: 양손목이 귀 모서리에서 멈춘 뒤 "
            "손목 간격이 벌어지거나 귀 간격이 어깨너비 대비 팽창 "
            "(헬멧이 천장 카메라 쪽으로 들어 올려짐)."
        )
        notes.append("예방 알림: 안전모 벗기 시도로 판정합니다.")
        label = ActionLabel.HELMET_OFF
    elif scratch_rule:
        conf = float(
            np.clip(
                0.58
                + 0.22 * min(feat.scratch_std / (2 * TAU_SCRATCH_STD), 1)
                + 0.20 * min(feat.n_oscillations / 10.0, 1),
                0.58,
                0.99,
            )
        )
        notes.append(
            "통계량: "
            f"radius_85%={feat.scratch_radius:.3f} (≤ {TAU_SCRATCH_RADIUS}), "
            f"σ_xy={feat.scratch_std:.4f} (≥ {TAU_SCRATCH_STD}), "
            f"osc={feat.n_oscillations} (≥ {MIN_OSCILLATIONS})."
        )
        notes.append(
            "규칙 SCRATCH: 한 손목만 두상 중심에서 짧은 반경의 고주파 진동. "
            "양손 귀 파지/팽창이 없으므로 정상 긁기로 봅니다."
        )
        notes.append("정상: 알림을 울리지 않습니다.")
        label = ActionLabel.SCRATCH
    else:
        conf = 0.40
        notes.append(
            "통계량: "
            f"radius={feat.scratch_radius:.3f}, σ_xy={feat.scratch_std:.4f}, "
            f"osc={feat.n_oscillations}, spread={feat.wrist_spread:.3f}, "
            f"scale_up={feat.head_scale_up:.3f}."
        )
        notes.append("중심 진동과 양귀 팽창 모두 임계값을 못 넘겨 판정을 보류합니다.")
        label = ActionLabel.UNKNOWN_CONTACT

    timeline = _frame_timeline(seq_norm, label)
    return ClassificationResult(label, conf, feat, notes, timeline)


def _frame_timeline(seq_norm: np.ndarray, final: ActionLabel) -> list[str]:
    t = seq_norm.shape[0]
    regions = [compute_head_regions(seq_norm[i]) for i in range(t)]
    labels = ["idle"] * t
    for i, rg in enumerate(regions):
        lw, rw = seq_norm[i, L_WRIST], seq_norm[i, R_WRIST]
        if rg.in_left_ear(lw) and rg.in_right_ear(rw):
            labels[i] = "ear_grasp"
        elif rg.in_center(lw) or rg.in_center(rw):
            labels[i] = "center"
    if final in (ActionLabel.SCRATCH, ActionLabel.HELMET_OFF):
        key = "center" if final is ActionLabel.SCRATCH else "ear_grasp"
        for i, v in enumerate(labels):
            if v == key:
                labels[i] = final.value
    return labels


def print_classification(name: str, result: ClassificationResult) -> None:
    print("=" * 72)
    print(f"[시퀀스] {name}")
    print(f"  판정     : {LABEL_KO[result.label]}  ({result.label.value})")
    print(f"  신뢰도   : {result.confidence:.2f}")
    print("  판단 근거:")
    for line in result.explanation:
        print(f"    - {line}")
    print("=" * 72)


# ===========================================================================
# 조건 1: 하이 앵글 정적 스켈레톤 시각화
# ===========================================================================
def project_canonical(camera: HighAngleCamera | None = None) -> np.ndarray:
    cam = camera or HighAngleCamera.factory_ceiling()
    return cam.project(canonical_pose_3d())


def _bent_elbow(shoulder: np.ndarray, wrist: np.ndarray, outward: float) -> np.ndarray:
    mid = 0.5 * (shoulder + wrist)
    vec = wrist - shoulder
    length = np.linalg.norm(vec)
    if length < 1e-8:
        return mid
    perp = np.array([vec[1], -vec[0]]) / length
    return mid + perp * outward


def plot_static_skeleton(
    keypoints: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str = "하이 앵글 CCTV 상체 스켈레톤 (어깨너비 = 1.0)",
) -> plt.Axes:
    """투시 왜곡이 반영된 정적 2D 스켈레톤.

    머리·어깨는 넓고, 골반·다리는 머리 쪽으로 단축·중첩되어 보인다.
    """
    _configure_korean_font()
    if keypoints is None:
        pix = project_canonical()
        pose, _ = normalize_keypoints(pix)
        pose = pose[0]
    else:
        pose = _as_sequence(keypoints)[0]
        if np.linalg.norm(pose[L_SHOULDER] - pose[R_SHOULDER]) > 2.5:
            pose, _ = normalize_keypoints(pose)
            pose = pose[0]

    created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.4, 8.4), facecolor="#0b1220")
        fig.patch.set_facecolor("#0b1220")

    ax.set_facecolor("#10182a")
    rg = compute_head_regions(pose)
    x, y, w, h = rg.as_xywh()
    ax.add_patch(
        Rectangle(
            (x, y), w, h, fill=True, facecolor="#f5c518", alpha=0.08,
            edgecolor="#f5c518", linewidth=1.6, linestyle="--",
            label="머리 Bounding Box", zorder=1,
        )
    )
    ax.add_patch(
        Circle(
            rg.center, rg.center_r, fill=True, facecolor="#3dd68c", alpha=0.16,
            edgecolor="#3dd68c", linewidth=1.4, linestyle=":",
            label="긁기 중심부", zorder=1,
        )
    )
    ax.add_patch(
        Circle(
            rg.left_ear, rg.ear_r, fill=False, edgecolor="#ff8fab",
            linewidth=1.4, linestyle=":", label="귀 모서리 (파지)", zorder=1,
        )
    )
    ax.add_patch(
        Circle(rg.right_ear, rg.ear_r, fill=False, edgecolor="#ff8fab", linewidth=1.4, linestyle=":")
    )

    neck = np.array([0.0, 0.0])
    for a, b in SKELETON_BONES:
        ax.plot(
            [pose[a, 0], pose[b, 0]], [pose[a, 1], pose[b, 1]],
            color="#5ee0ff", lw=2.5, solid_capstyle="round", zorder=2,
        )
    ax.plot(
        [pose[L_SHOULDER, 0], neck[0], pose[R_SHOULDER, 0]],
        [pose[L_SHOULDER, 1], neck[1], pose[R_SHOULDER, 1]],
        color="#5ee0ff", lw=2.5, zorder=2,
    )
    ax.plot([neck[0], pose[NOSE, 0]], [neck[1], pose[NOSE, 1]], color="#5ee0ff", lw=2.5, zorder=2)

    colors = np.full(17, "#d7ecff")
    colors[[L_SHOULDER, R_SHOULDER]] = "#7cffb2"
    colors[[L_WRIST, R_WRIST]] = "#ffd166"
    colors[list(FACE_IDX)] = "#ff8fab"
    ax.scatter(pose[:, 0], pose[:, 1], c=colors, s=52, zorder=4, edgecolors="#0b1220", linewidths=0.6)
    ax.scatter([0], [0], c="#7cffb2", s=70, zorder=5, edgecolors="#0b1220", label="목 (원점)")

    ax.annotate(
        "",
        xy=(pose[R_SHOULDER, 0], pose[R_SHOULDER, 1] - 0.06),
        xytext=(pose[L_SHOULDER, 0], pose[L_SHOULDER, 1] - 0.06),
        arrowprops=dict(arrowstyle="<->", color="#7cffb2", lw=1.6),
    )
    ax.text(
        0.0, float(pose[L_SHOULDER, 1] - 0.14),
        "Shoulder Width = 1.0  (유일한 정규화 단위)",
        ha="center", color="#7cffb2", fontsize=9,
    )

    hip_y = float(0.5 * (pose[L_HIP, 1] + pose[R_HIP, 1]))
    ank_y = float(0.5 * (pose[L_ANKLE, 1] + pose[R_ANKLE, 1]))
    ax.annotate(
        "투시 왜곡: 골반·다리가 머리 쪽으로 단축",
        xy=(0.02, hip_y),
        xytext=(0.85, (hip_y + ank_y) * 0.5),
        color="#c9d6f0",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#8aa0c4"),
    )
    ax.text(
        -1.55, float(pose[NOSE, 1]) - 0.35,
        "CCTV 하이 앵글\n(대각선 하향)",
        color="#f5c518", fontsize=9, ha="left", va="top",
    )
    ax.add_patch(
        FancyArrowPatch(
            (-1.35, float(pose[NOSE, 1]) - 0.55),
            (-0.15, float(pose[NOSE, 1]) - 0.05),
            arrowstyle="-|>", mutation_scale=12, color="#f5c518", lw=1.3,
        )
    )

    labels = {
        NOSE: "코",
        L_SHOULDER: "왼어깨",
        R_SHOULDER: "오른어깨",
        L_WRIST: "왼손목",
        R_WRIST: "오른손목",
        L_HIP: "골반",
        L_ANKLE: "발",
    }
    for idx, name in labels.items():
        ax.text(pose[idx, 0] + 0.05, pose[idx, 1], name, color="#c9d6f0", fontsize=8, va="center")

    ax.set_title(title, color="#f2f6ff", fontsize=13, pad=12)
    ax.set_xlabel("X (어깨너비 단위, 목=0)", color="#9db0d0")
    ax.set_ylabel("Y (이미지 아래가 +)", color="#9db0d0")
    ax.tick_params(colors="#8aa0c4")
    for spine in ax.spines.values():
        spine.set_color("#2a3a58")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlim(-1.7, 1.7)
    y_bottom = max(float(pose[L_ANKLE, 1]), float(pose[R_ANKLE, 1])) + 0.35
    ax.set_ylim(y_bottom, float(rg.y_min) - 0.25)
    ax.grid(True, color="#1c2a44", linestyle=":", linewidth=0.8)
    ax.legend(loc="lower right", facecolor="#152038", edgecolor="#2a3a58", labelcolor="#e8eefc", fontsize=8)
    if created:
        ax.figure.tight_layout()
    return ax


def plot_sequence_strip(
    seq_pixels: np.ndarray,
    frame_indices: Iterable[int],
    title: str,
    result: ClassificationResult,
    out_path: Path | None = None,
) -> plt.Figure:
    _configure_korean_font()
    idx = list(frame_indices)
    fig, axes = plt.subplots(1, len(idx), figsize=(3.15 * len(idx), 5.8), facecolor="#0b1220")
    if len(idx) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0b1220")
    seq_norm, _ = normalize_keypoints(seq_pixels)
    accent = {
        ActionLabel.SCRATCH: "#3dd68c",
        ActionLabel.HELMET_OFF: "#ff5c5c",
        ActionLabel.NO_CONTACT: "#8aa0c4",
        ActionLabel.UNKNOWN_CONTACT: "#f5c518",
    }[result.label]

    for ax, fi in zip(axes, idx):
        ax.set_facecolor("#10182a")
        pose = seq_norm[fi]
        rg = compute_head_regions(pose)
        x, y, w, h = rg.as_xywh()
        ax.add_patch(Rectangle((x, y), w, h, fill=True, facecolor="#f5c518", alpha=0.10, edgecolor="#f5c518", lw=1.0, ls="--"))
        ax.add_patch(Circle(rg.center, rg.center_r, fill=False, edgecolor="#3dd68c", lw=1.0, ls=":"))
        ax.add_patch(Circle(rg.left_ear, rg.ear_r, fill=False, edgecolor="#ff8fab", lw=1.0, ls=":"))
        ax.add_patch(Circle(rg.right_ear, rg.ear_r, fill=False, edgecolor="#ff8fab", lw=1.0, ls=":"))
        for a, b in SKELETON_BONES:
            ax.plot([pose[a, 0], pose[b, 0]], [pose[a, 1], pose[b, 1]], color="#5ee0ff", lw=1.8)
        ax.scatter(pose[:, 0], pose[:, 1], c="#d7ecff", s=18, zorder=3)
        ax.scatter(pose[[L_WRIST, R_WRIST], 0], pose[[L_WRIST, R_WRIST], 1], c=accent, s=42, zorder=4)
        ax.set_title(f"t = {fi}", color="#e8eefc", fontsize=10)
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(1.35, -0.95)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(f"{title}\n판정: {LABEL_KO[result.label]}", color=accent, fontsize=13, y=0.98)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    return fig


def plot_feature_timeline(
    seq_pixels: np.ndarray,
    title: str,
    out_path: Path | None = None,
) -> plt.Figure:
    _configure_korean_font()
    seq_norm, _ = normalize_keypoints(seq_pixels)
    t = np.arange(seq_norm.shape[0])
    d_wrist = np.linalg.norm(seq_norm[:, L_WRIST] - seq_norm[:, R_WRIST], axis=1)
    h_scale = np.array([head_scale_norm(seq_norm[i]) for i in range(len(t))])
    regions = [compute_head_regions(seq_norm[i]) for i in range(len(t))]
    rw = seq_norm[:, R_WRIST]
    dist_center = np.array([np.linalg.norm(rw[i] - regions[i].center) for i in range(len(t))])

    fig, axes = plt.subplots(2, 1, figsize=(10.6, 6.3), sharex=True, facecolor="#0b1220")
    fig.patch.set_facecolor("#0b1220")
    for ax in axes:
        ax.set_facecolor("#10182a")
        ax.tick_params(colors="#8aa0c4")
        for spine in ax.spines.values():
            spine.set_color("#2a3a58")
        ax.grid(True, color="#1c2a44", linestyle=":", lw=0.8)

    axes[0].plot(t, dist_center, color="#ffd166", lw=2.0, label="오른손목 → 두상 중심")
    axes[0].axhline(CENTER_RADIUS, color="#3dd68c", ls="--", lw=1.2, label=f"중심원 r={CENTER_RADIUS}")
    axes[0].set_ylabel("거리 (어깨너비 단위)", color="#9db0d0")
    axes[0].legend(facecolor="#152038", edgecolor="#2a3a58", labelcolor="#e8eefc")
    axes[0].set_title(title, color="#f2f6ff")

    axes[1].plot(t, d_wrist, color="#ff8fab", lw=2.0, label="양손목 사이 거리")
    axes[1].plot(t, h_scale, color="#f5c518", lw=2.0, label="귀 간격 (머리 겉보기 크기)")
    axes[1].set_xlabel("프레임", color="#9db0d0")
    axes[1].set_ylabel("정규화 길이", color="#9db0d0")
    axes[1].legend(facecolor="#152038", edgecolor="#2a3a58", labelcolor="#e8eefc")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    return fig


# ===========================================================================
# 하이 앵글 가짜 시계열 (3D 동작 → 핀홀 투영)
# ===========================================================================
def _lerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    return (1.0 - u) * a + u * b


def _ease(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _place_arm_3d(pose: np.ndarray, side: str, wrist: np.ndarray) -> None:
    sh_i, el_i, wr_i = (L_SHOULDER, L_ELBOW, L_WRIST) if side == "left" else (R_SHOULDER, R_ELBOW, R_WRIST)
    pose[wr_i] = wrist
    sh = pose[sh_i]
    mid = 0.5 * (sh + wrist)
    # 팔꿈치를 몸 바깥·약간 뒤로
    sign = -1.0 if side == "left" else 1.0
    pose[el_i] = mid + np.array([0.06 * sign, 0.0, -0.04])


def _scale_head_toward_camera(pose: np.ndarray, amount: float) -> None:
    """헬멧이 렌즈 쪽으로 들어 올려질 때: 얼굴점을 카메라(작은 Z, 큰 Y) 쪽으로 팽창."""
    c = 0.5 * (pose[L_EAR] + pose[R_EAR])
    cam_dir = np.array([0.04, 0.89, -0.45], dtype=np.float64)  # 천장 카메라 대략 방향
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    for idx in FACE_IDX:
        radial = pose[idx] - c
        pose[idx] = c + radial * (1.0 + 0.85 * amount) + cam_dir * (0.18 * amount)


def generate_scratch_sequence(
    n_frames: int = 60,
    fps: int = 30,
    seed: int = 7,
    pixel_scale: float = 1.0,
    pixel_shift: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """한 손목이 두상 중심으로 이동한 뒤 짧은 반경에서 고주파 진동."""
    rng = np.random.default_rng(seed)
    cam = HighAngleCamera.factory_ceiling()
    rest = canonical_pose_3d()
    # 두상 중심 = 귀 중점. 하이 앵글에서 bbox 중심부와 일치시킨다.
    scalp = 0.5 * (rest[L_EAR] + rest[R_EAR]) + np.array([0.00, 0.012, 0.015])
    seq = np.zeros((n_frames, 17, 2), dtype=np.float64)
    raise_end, scratch_end = 12, 50
    for i in range(n_frames):
        pose = rest.copy()
        if i < raise_end:
            u = _ease(i / max(raise_end - 1, 1))
            wrist = _lerp(rest[R_WRIST], scalp, u)
        elif i < scratch_end:
            t = (i - raise_end) / fps
            # 반경 ~2 cm, 3.6 Hz — 투영 후에도 중심원 안의 고주파
            wrist = scalp + np.array(
                [
                    0.018 * math.sin(2 * math.pi * 3.6 * t),
                    0.006 * math.cos(2 * math.pi * 3.6 * t),
                    0.008 * math.sin(2 * math.pi * 4.1 * t + 0.4),
                ]
            )
        else:
            u = _ease((i - scratch_end) / max(n_frames - scratch_end - 1, 1))
            wrist = _lerp(scalp, rest[R_WRIST], u)
        _place_arm_3d(pose, "right", wrist)
        pose += rng.normal(0.0, 0.0015, size=pose.shape)
        pix = cam.project(pose) * pixel_scale + np.asarray(pixel_shift)
        seq[i] = pix
    return seq


def generate_helmet_off_sequence(
    n_frames: int = 64,
    seed: int = 11,
    pixel_scale: float = 1.0,
    pixel_shift: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """양손목이 귀 모서리로 이동 → 일시 정지 → 벌어짐 + 머리 팽창(카메라 접근)."""
    rng = np.random.default_rng(seed)
    cam = HighAngleCamera.factory_ceiling()
    rest = canonical_pose_3d()
    grab_l = rest[L_EAR] + np.array([-0.03, 0.01, 0.02])
    grab_r = rest[R_EAR] + np.array([0.03, 0.01, 0.02])
    seq = np.zeros((n_frames, 17, 2), dtype=np.float64)
    raise_end, pause_end, lift_end = 11, 22, 52
    for i in range(n_frames):
        pose = rest.copy()
        amount = 0.0
        if i < raise_end:
            u = _ease(i / max(raise_end - 1, 1))
            wl = _lerp(rest[L_WRIST], grab_l, u)
            wr = _lerp(rest[R_WRIST], grab_r, u)
        elif i < pause_end:
            wl = grab_l + rng.normal(0.0, 0.0012, size=3)
            wr = grab_r + rng.normal(0.0, 0.0012, size=3)
        else:
            u = _ease(min(1.0, (i - pause_end) / max(lift_end - pause_end - 1, 1)))
            # 양손이 귀를 잡고 바깥·위·카메라 쪽으로 벌리며 들어 올림
            wl = grab_l + np.array([-0.07 * u, 0.10 * u, -0.12 * u])
            wr = grab_r + np.array([0.07 * u, 0.10 * u, -0.12 * u])
            amount = 0.55 * u
        _place_arm_3d(pose, "left", wl)
        _place_arm_3d(pose, "right", wr)
        _scale_head_toward_camera(pose, amount)
        pose += rng.normal(0.0, 0.0014, size=pose.shape)
        seq[i] = cam.project(pose) * pixel_scale + np.asarray(pixel_shift)
    return seq


def generate_idle_sequence(
    n_frames: int = 40,
    seed: int = 3,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cam = HighAngleCamera.factory_ceiling()
    rest = canonical_pose_3d()
    seq = np.zeros((n_frames, 17, 2), dtype=np.float64)
    for i in range(n_frames):
        pose = rest.copy()
        phase = 2 * math.pi * i / 16.0
        wr = rest[R_WRIST] + np.array([0.03 * math.sin(phase), 0.04 * math.sin(phase), 0.0])
        _place_arm_3d(pose, "right", wr)
        pose += rng.normal(0.0, 0.0015, size=pose.shape)
        seq[i] = cam.project(pose)
    return seq


# ===========================================================================
# 대시보드 / 자가 검증
# ===========================================================================
def sequence_payload(name: str, seq_pixels: np.ndarray, expected: ActionLabel) -> dict:
    result = classify_pose_sequence(seq_pixels)
    seq_norm, infos = normalize_keypoints(seq_pixels)
    frames = []
    for i, pose in enumerate(seq_norm):
        rg = compute_head_regions(pose)
        x, y, w, h = rg.as_xywh()
        frames.append(
            {
                "keypoints": pose.tolist(),
                "bbox": [rg.x_min, rg.y_min, rg.x_max, rg.y_max],
                "center": rg.center.tolist(),
                "center_r": rg.center_r,
                "left_ear": rg.left_ear.tolist(),
                "right_ear": rg.right_ear.tolist(),
                "ear_r": rg.ear_r,
                "scale_px": infos[i].scale,
                "origin_px": infos[i].origin.tolist(),
                "state": result.frame_labels[i] if i < len(result.frame_labels) else "idle",
            }
        )
    return {
        "name": name,
        "expected": expected.value,
        "expected_ko": LABEL_KO[expected],
        "result": result.to_dict(),
        "match": result.label == expected,
        "n_frames": int(seq_pixels.shape[0]),
        "bones": SKELETON_BONES,
        "frames": frames,
    }


def build_demo_scenarios() -> list[dict]:
    """필수 2개(긁기/벗기) + 스케일 불변 + 비접촉."""
    scratch = generate_scratch_sequence()
    helmet = generate_helmet_off_sequence()
    scratch_far = generate_scratch_sequence(pixel_scale=0.55, pixel_shift=(80.0, 40.0), seed=7)
    helmet_near = generate_helmet_off_sequence(pixel_scale=1.45, pixel_shift=(-60.0, 30.0), seed=11)
    return [
        sequence_payload("하이 앵글 · 단순 머리 긁기", scratch, ActionLabel.SCRATCH),
        sequence_payload("하이 앵글 · 안전모 벗기", helmet, ActionLabel.HELMET_OFF),
        sequence_payload("원거리 축소 카메라 · 긁기", scratch_far, ActionLabel.SCRATCH),
        sequence_payload("근거리 확대 카메라 · 벗기", helmet_near, ActionLabel.HELMET_OFF),
        sequence_payload("팔만 흔들기 · 머리 비접촉", generate_idle_sequence(), ActionLabel.NO_CONTACT),
    ]


def render_all_figures(out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7.4, 8.4), facecolor="#0b1220")
    fig.patch.set_facecolor("#0b1220")
    plot_static_skeleton(ax=ax)
    fig.tight_layout()
    p = out_dir / "static_skeleton.png"
    fig.savefig(p, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths["static_skeleton"] = str(p)

    scratch = generate_scratch_sequence()
    helmet = generate_helmet_off_sequence()
    r_s = classify_pose_sequence(scratch)
    r_h = classify_pose_sequence(helmet)

    p = out_dir / "scratch_strip.png"
    fig = plot_sequence_strip(scratch, [0, 12, 24, 36, 50], "하이 앵글 · 머리 긁기", r_s, p)
    plt.close(fig)
    paths["scratch_strip"] = str(p)

    p = out_dir / "helmet_strip.png"
    fig = plot_sequence_strip(helmet, [0, 11, 20, 36, 52], "하이 앵글 · 안전모 벗기", r_h, p)
    plt.close(fig)
    paths["helmet_strip"] = str(p)

    p = out_dir / "scratch_timeline.png"
    fig = plot_feature_timeline(scratch, "긁기: 한 손목이 중심원에 들어가 고주파 진동", p)
    plt.close(fig)
    paths["scratch_timeline"] = str(p)

    p = out_dir / "helmet_timeline.png"
    fig = plot_feature_timeline(helmet, "벗기: 양손목 거리 증가 + 귀 간격 팽창(카메라 접근)", p)
    plt.close(fig)
    paths["helmet_timeline"] = str(p)
    return paths


def run_self_test(verbose: bool = True) -> bool:
    print(
        """
[하이 앵글 알고리즘 개요]
  카메라   : 천장/벽면 상단, 대각선 하향 (하향각 ≈ 45°)
  정규화   : p̂ = (p − neck) / ||L_shoulder − R_shoulder||
             ※ 목~골반 길이는 투시 단축 때문에 사용하지 않음

  긁기     : 한 손목 ∈ 두상 중심원(r=0.24)
             국소 반경 ≤ {:.2f}  이면서  σ_xy ≥ {:.3f},  영점교차 ≥ {}

  안전모   : 왼손목∈왼귀원 ∧ 오른손목∈오른귀원 → 일시정지
             이후 양손목 거리 증가 > {:.2f}
             또는 귀간격/어깨너비 상대증가 > {:.2f}  (렌즈 방향 팽창)
""".format(
            TAU_SCRATCH_RADIUS, TAU_SCRATCH_STD, MIN_OSCILLATIONS, TAU_SPREAD, TAU_SCALE_UP
        )
    )
    ok = True
    for sc in build_demo_scenarios():
        result = ClassificationResult(
            label=ActionLabel(sc["result"]["label"]),
            confidence=sc["result"]["confidence"],
            features=FeatureReport(**sc["result"]["features"]),
            explanation=sc["result"]["explanation"],
            frame_labels=sc["result"]["frame_labels"],
        )
        if verbose:
            print_classification(sc["name"], result)
            print(f"  기대 라벨 : {sc['expected_ko']}  →  일치={sc['match']}\n")
        if not sc["match"]:
            ok = False
            print(f"[FAIL] {sc['name']}: expected {sc['expected']} got {result.label.value}")
    print("[SELF-TEST]", "PASS" if ok else "FAIL")
    return ok


def maybe_show(fig: plt.Figure | None = None) -> None:
    if os.environ.get("DISPLAY") and matplotlib.get_backend().lower() != "agg":
        plt.show()
    elif fig is not None:
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="하이 앵글 CCTV 안전모/긁기 분류 프로토타입")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)

    passed = run_self_test(verbose=True)
    out_dir = Path(args.out)
    if not args.skip_plots:
        paths = render_all_figures(out_dir)
        print("\n[시각화 저장]")
        for k, v in paths.items():
            print(f"  {k}: {v}")
        fig, ax = plt.subplots(figsize=(7.4, 8.4), facecolor="#0b1220")
        fig.patch.set_facecolor("#0b1220")
        plot_static_skeleton(ax=ax)
        fig.tight_layout()
        maybe_show(fig)

    if args.serve:
        from dashboard_server import serve

        serve(host=args.host, port=args.port, out_dir=out_dir)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
