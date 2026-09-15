#!/usr/bin/env python3
"""공장 작업자 2D 스켈레톤 기반 동작 분류 프로토타입.

YOLOv8-Pose / MediaPipe 형식의 17개 COCO 키포인트를 입력으로,
'단순 머리 긁기(정상)'와 '안전모 벗기 시도(예방 알림)'을 구분한다.

핵심 제약
---------
* 절대 픽셀 좌표(y=100 등)로 판단하지 않는다.
* 어깨너비(Shoulder Width)를 기준 단위(scale = 1.0)로 정규화한다.
* 머리 Bounding Box 는 목(Neck)과 어깨너비에 비례해 매 프레임 동적으로 계산한다.

이미지 좌표 규약
----------------
OpenCV / COCO 와 동일하게 원점은 좌상단, Y축은 아래 방향이 양수이다.
따라서 '위쪽으로 이동'은 Y 좌표가 *감소*하는 것과 같다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

import matplotlib

# 서버/헤드리스 환경에서도 이미지를 저장할 수 있도록 비대화형 백엔드를 우선 사용한다.
if os.environ.get("DISPLAY") in (None, "") or os.environ.get("MPLBACKEND") == "Agg":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle

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
# COCO / YOLOv8-Pose 17 키포인트 인덱스
# ---------------------------------------------------------------------------
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

KEYPOINT_NAMES = [
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
]

# 상체가 한눈에 보이도록 그리는 스켈레톤 연결 (목은 어깨 중점으로 파생)
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

# ---------------------------------------------------------------------------
# 정규화 공간(어깨너비 = 1.0)에서 정의한 머리 영역 비율
# 성인 상체 비율: 머리 높이는 대략 어깨너비의 0.9~1.1 배, 너비는 0.7~0.9 배.
# ---------------------------------------------------------------------------
HEAD_HALF_WIDTH = 0.42  # bbox 가로 반폭  (단위: 어깨너비)
HEAD_ABOVE_NECK = 1.05  # 목에서 정수리까지 (Y 음수 방향)
HEAD_BELOW_NECK = 0.18  # 목 아래 챙/윗목까지 포함 (손을 챙에 올리는 동작 허용)
HEAD_CROWN_OFFSET = 0.28  # 눈/귀보다 정수리가 위에 있는 정도

# ---------------------------------------------------------------------------
# 분류 임계값 — 전부 정규화 좌표(어깨너비 단위) 또는 무차원 통계량
# ---------------------------------------------------------------------------
MIN_CONTACT_FRAMES = 8  # 머리 영역 최소 체류 프레임 (~0.27s @ 30fps)
PAUSE_FRAMES = 8  # 안전모 파지 후 '일시 정지'로 볼 최소 프레임
RISE_FRAMES_MIN = 6

TAU_PAUSE_STD = 0.018  # 일시 정지: 손목 Y 표준편차 상한
TAU_SCRATCH_STD = 0.045  # 긁기 진동: 손목 Y 표준편차 하한
TAU_HEAD_STILL_STD = 0.022  # 긁기 시 머리 고정으로 볼 표준편차 상한
TAU_RISE = 0.12  # 동시 상승량 하한 (어깨너비의 12%)
TAU_CORR = 0.50  # 손목-머리 상승 구간 피어슨 상관계수
MIN_OSCILLATIONS = 3  # 긁기 방향 전환 최소 횟수


class ActionLabel(str, Enum):
    SCRATCH = "scratch"  # 단순 머리 긁기 (정상)
    HELMET_OFF = "helmet_off"  # 안전모 벗기 시도 (예방 알림)
    NO_CONTACT = "no_contact"  # 손목이 머리 영역에 들어오지 않음
    UNKNOWN_CONTACT = "unknown_contact"  # 접촉은 있으나 두 동작 모두 미충족


LABEL_KO = {
    ActionLabel.SCRATCH: "단순 머리 긁기 (정상)",
    ActionLabel.HELMET_OFF: "안전모 벗기 시도 (예방 알림)",
    ActionLabel.NO_CONTACT: "머리 비접촉 (동작 없음)",
    ActionLabel.UNKNOWN_CONTACT: "머리 접촉 · 판정 보류",
}


# ===========================================================================
# 조건 3: 체형 / 카메라 거리 정규화
# ===========================================================================
@dataclass
class NormalizeInfo:
    """한 프레임의 정규화 파라미터.

    origin : 목(양쪽 어깨 중점). 모든 좌표의 원점.
    scale  : 어깨너비. 정규화 공간에서 1.0 에 해당.
    torso_length : 목~골반 중점 거리. 어깨가 가려진 경우 scale 대체값.
    """

    origin: np.ndarray
    scale: float
    torso_length: float
    shoulder_width: float


def _as_sequence(keypoints: np.ndarray) -> np.ndarray:
    """(17, 2|3) → (1, 17, 2), (T, 17, 2|3) → (T, 17, 2)."""
    arr = np.asarray(keypoints, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.shape[-1] >= 2:
        arr = arr[..., :2]
    if arr.shape[-2] != 17:
        raise ValueError(f"COCO 17 키포인트가 필요합니다. 받은 shape={arr.shape}")
    return arr


def compute_neck(keypoints: np.ndarray) -> np.ndarray:
    """양쪽 어깨 중점 = 목. shape (..., 2)."""
    kpts = np.asarray(keypoints, dtype=np.float64)[..., :2]
    return 0.5 * (kpts[..., L_SHOULDER, :] + kpts[..., R_SHOULDER, :])


def compute_mid_hip(keypoints: np.ndarray) -> np.ndarray:
    kpts = np.asarray(keypoints, dtype=np.float64)[..., :2]
    return 0.5 * (kpts[..., L_HIP, :] + kpts[..., R_HIP, :])


def compute_shoulder_width(keypoints: np.ndarray) -> np.ndarray:
    kpts = np.asarray(keypoints, dtype=np.float64)[..., :2]
    delta = kpts[..., L_SHOULDER, :] - kpts[..., R_SHOULDER, :]
    return np.linalg.norm(delta, axis=-1)


def compute_torso_length(keypoints: np.ndarray) -> np.ndarray:
    neck = compute_neck(keypoints)
    hip = compute_mid_hip(keypoints)
    return np.linalg.norm(neck - hip, axis=-1)


def normalize_keypoints(keypoints: np.ndarray) -> tuple[np.ndarray, list[NormalizeInfo]]:
    """픽셀 좌표 → 목 원점 / 어깨너비 단위의 상대 좌표.

    변환식
        p_hat = (p - neck) / max(shoulder_width, epsilon)
    어깨가 거의 겹쳐 보이면(측면·가림) torso_length 로 대체한다.

    Returns
    -------
    normalized : (T, 17, 2)  정규화 좌표. 목이 (0, 0), 어깨너비가 1.0.
    infos      : 프레임별 원점·스케일 (역변환 및 bbox 픽셀 투영에 사용)
    """
    seq = _as_sequence(keypoints)  # (T, 17, 2)
    t = seq.shape[0]
    ls = seq[:, L_SHOULDER, :]
    rs = seq[:, R_SHOULDER, :]
    neck = 0.5 * (ls + rs)
    shoulder_width = np.linalg.norm(ls - rs, axis=1)
    mid_hip = 0.5 * (seq[:, L_HIP, :] + seq[:, R_HIP, :])
    torso = np.linalg.norm(neck - mid_hip, axis=1)

    scale = np.where(shoulder_width > 1e-6, shoulder_width, torso)
    scale = np.maximum(scale, 1e-6)

    normalized = (seq - neck[:, None, :]) / scale[:, None, None]

    infos = [
        NormalizeInfo(
            origin=neck[i],
            scale=float(scale[i]),
            torso_length=float(torso[i]),
            shoulder_width=float(shoulder_width[i]),
        )
        for i in range(t)
    ]
    return normalized, infos


def denormalize_points(
    points_norm: np.ndarray, info: NormalizeInfo
) -> np.ndarray:
    """정규화 좌표를 해당 프레임의 픽셀 좌표로 되돌린다."""
    return np.asarray(points_norm, dtype=np.float64) * info.scale + info.origin


# ===========================================================================
# 동적 머리 Bounding Box (목 기준, 어깨너비 비례)
# ===========================================================================
@dataclass
class HeadBBox:
    """정규화 공간의 축정렬 머리 영역. y_min 이 화면 상단(정수리)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains(self, point: np.ndarray) -> bool:
        x, y = float(point[0]), float(point[1])
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def as_xywh(self) -> tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max - self.x_min, self.y_max - self.y_min

    def to_pixel(self, info: NormalizeInfo) -> "HeadBBox":
        corners = np.array(
            [[self.x_min, self.y_min], [self.x_max, self.y_max]], dtype=np.float64
        )
        pix = denormalize_points(corners, info)
        return HeadBBox(
            x_min=float(min(pix[0, 0], pix[1, 0])),
            y_min=float(min(pix[0, 1], pix[1, 1])),
            x_max=float(max(pix[0, 0], pix[1, 0])),
            y_max=float(max(pix[0, 1], pix[1, 1])),
        )


def estimate_head_top_norm(pose_norm: np.ndarray) -> np.ndarray:
    """정규화 포즈에서 '머리 상단(정수리)' 좌표를 추정한다.

    얼굴 키포인트 중 가장 위(최소 Y)에서 정수리 오프셋을 더한다.
    안전모 벗기 시뮬레이션에서는 얼굴 점이 함께 올라가므로 이 점도 같이 상승한다.
    """
    face = pose_norm[[NOSE, L_EYE, R_EYE, L_EAR, R_EAR], :]
    xy = face[np.argmin(face[:, 1])]
    return np.array([float(pose_norm[NOSE, 0]), float(xy[1] - HEAD_CROWN_OFFSET)])


def compute_head_bbox_norm(pose_norm: np.ndarray) -> HeadBBox:
    """목(원점)과 코의 X 를 기준으로 어깨너비 비례 머리 bbox 를 만든다.

    정규화 공간에서 목 = (0, 0) 이므로 절대 픽셀 상수는 일절 사용하지 않는다.
        x ∈ [nose_x − 0.42, nose_x + 0.42]
        y ∈ [−1.05, +0.18]
    """
    cx = float(pose_norm[NOSE, 0])
    if not np.isfinite(cx):
        cx = 0.0
    return HeadBBox(
        x_min=cx - HEAD_HALF_WIDTH,
        y_min=-HEAD_ABOVE_NECK,
        x_max=cx + HEAD_HALF_WIDTH,
        y_max=HEAD_BELOW_NECK,
    )


# ===========================================================================
# 조건 2: 판단 기준 수식화
# ===========================================================================
@dataclass
class FeatureReport:
    """한 시퀀스에서 추출한 판정용 통계량 (모두 정규화 단위 / 무차원)."""

    contact_frames: int = 0
    active_wrist: str = "none"
    wrist_y_std: float = 0.0
    wrist_y_std_early: float = 0.0
    head_y_std: float = 0.0
    n_oscillations: int = 0
    rise_wrist: float = 0.0  # 양수 = 화면 위쪽으로 이동량
    rise_head: float = 0.0
    corr_wrist_head: float = 0.0
    pause_detected: bool = False
    co_rising: bool = False


@dataclass
class ClassificationResult:
    label: ActionLabel
    confidence: float
    features: FeatureReport
    explanation: list[str] = field(default_factory=list)
    frame_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = {
            "label": self.label.value,
            "label_ko": LABEL_KO[self.label],
            "confidence": self.confidence,
            "features": asdict(self.features),
            "explanation": self.explanation,
            "frame_labels": self.frame_labels,
        }
        return payload


def _std(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=0))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return 0.0
    if _std(a) < 1e-12 or _std(b) < 1e-12:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def _smooth(y: np.ndarray, k: int = 5) -> np.ndarray:
    if y.size < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")


def _zero_crossings(y: np.ndarray, deadband: float = 0.012) -> int:
    """평활화된 Y의 1차 차분 부호 반전 횟수 = 상하 진동(긁기) 대리 지표.

    deadband 미만의 차분은 센서 노이즈로 보고 무시한다.
    """
    sm = _smooth(y)
    if sm.size < 3:
        return 0
    d = np.diff(sm)
    d = d[np.abs(d) > deadband]
    if d.size < 2:
        return 0
    signs = np.sign(d)
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def _first_pause(y: np.ndarray, window: int = PAUSE_FRAMES, tau: float = TAU_PAUSE_STD) -> tuple[int | None, float]:
    """체류 구간에서 손목 Y 표준편차가 tau 미만인 첫 창 = 파지 후 일시 정지."""
    if y.size < window:
        s = _std(y)
        return (0 if s < tau else None), s
    best_std = float("inf")
    first = None
    for i in range(0, y.size - window + 1):
        s = _std(y[i : i + window])
        if s < best_std:
            best_std = s
        if first is None and s < tau:
            first = i
    return first, float(best_std)


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """True 구간의 가장 긴 (start, end) exclusive-end 인덱스를 반환."""
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


def _pick_active_wrist(
    seq_norm: np.ndarray, bboxes: list[HeadBBox]
) -> tuple[np.ndarray, str, np.ndarray]:
    """머리 bbox 안에 더 오래 머문 손목을 활성 손목으로 선택."""
    t = seq_norm.shape[0]
    left = seq_norm[:, L_WRIST, :]
    right = seq_norm[:, R_WRIST, :]
    in_l = np.array([bboxes[i].contains(left[i]) for i in range(t)])
    in_r = np.array([bboxes[i].contains(right[i]) for i in range(t)])
    if int(in_r.sum()) >= int(in_l.sum()):
        return right, "right", in_r
    return left, "left", in_l


def extract_features(seq_norm: np.ndarray) -> FeatureReport:
    """정규화된 (T, 17, 2) 시퀀스에서 판정 통계량을 계산한다.

    접촉 시작부터 시퀀스 끝(+짧은 꼬리)까지를 본다. 안전모를 들어 올리면
    손목이 원래 머리 bbox 위로 빠져나가기 때문에, 최장 True 구간에만
    묶으면 상승 벡터가 잘린다.
    """
    t = seq_norm.shape[0]
    bboxes = [compute_head_bbox_norm(seq_norm[i]) for i in range(t)]
    head_top = np.stack([estimate_head_top_norm(seq_norm[i]) for i in range(t)])
    wrist, which, in_head = _pick_active_wrist(seq_norm, bboxes)

    feat = FeatureReport(active_wrist=which)
    start, end = _longest_true_run(in_head)
    feat.contact_frames = end - start
    if feat.contact_frames < 1:
        return feat

    trail_end = min(t, end + 12)
    wy_contact = wrist[start:end, 1]
    hy_contact = head_top[start:end, 1]
    wy_all = wrist[start:trail_end, 1]
    hy_all = head_top[start:trail_end, 1]

    entry_trim = min(PAUSE_FRAMES, max(3, feat.contact_frames // 4))
    wy_dwell = wy_contact[entry_trim:] if wy_contact.size > entry_trim else wy_contact
    hy_dwell = hy_contact[entry_trim:] if hy_contact.size > entry_trim else hy_contact
    feat.wrist_y_std = _std(wy_dwell)
    feat.head_y_std = _std(hy_dwell)
    feat.n_oscillations = _zero_crossings(wy_dwell)

    pause_i, min_pause_std = _first_pause(wy_all)
    feat.wrist_y_std_early = min_pause_std
    feat.pause_detected = pause_i is not None

    if pause_i is None:
        pause_i = 0
        pause_n = min(PAUSE_FRAMES, wy_all.size)
    else:
        pause_n = min(PAUSE_FRAMES, wy_all.size - pause_i)

    late_w = wy_all[pause_i + pause_n :]
    late_h = hy_all[pause_i + pause_n :]
    if late_w.size == 0:
        late_w = wy_all[pause_i:]
        late_h = hy_all[pause_i:]
    baseline_w = float(np.median(wy_all[pause_i : pause_i + pause_n]))
    baseline_h = float(np.median(hy_all[pause_i : pause_i + pause_n]))
    feat.rise_wrist = float(baseline_w - np.min(late_w)) if late_w.size else 0.0
    feat.rise_head = float(baseline_h - np.min(late_h)) if late_h.size else 0.0
    feat.corr_wrist_head = _pearson(late_w, late_h)
    feat.co_rising = (
        feat.rise_wrist > TAU_RISE
        and feat.rise_head > TAU_RISE
        and feat.corr_wrist_head > TAU_CORR
    )
    return feat


def classify_pose_sequence(keypoints: np.ndarray) -> ClassificationResult:
    """시퀀스 전체를 보고 두 동작을 구분하는 핵심 알고리즘.

    판단 로직 (안전 우선: 헬멧 탈착을 먼저 검사)
    ------------------------------------------
    [공통] 정규화 후 손목이 동적 머리 bbox 에 MIN_CONTACT_FRAMES 이상 체류.

    [안전모 벗기]
        1) 체류 초반 손목 Y 표준편차가 작다 (일시 정지 / 파지).
        2) 이후 손목 Y 와 머리 상단 Y 가 동시에 감소 (화면 위쪽 상승 벡터).
        3) 상승량 > τ_rise 이고 두 궤적의 피어슨 상관이 > τ_corr.

    [머리 긁기]
        1) 체류 구간 손목 Y 표준편차가 크다 (상하 진동).
        2) 머리 상단 Y 표준편차는 작다 (머리는 고정).
        3) 차분 영점교차(진동 횟수) >= MIN_OSCILLATIONS.
        4) 머리 상승량이 τ_rise 미만 (헬멧을 들어 올리지 않음).
    """
    seq_norm, _infos = normalize_keypoints(keypoints)
    feat = extract_features(seq_norm)
    notes: list[str] = []

    notes.append(
        "정규화: p̂ = (p − neck) / shoulder_width  "
        f"(shoulder=1.0, torso는 대체 스케일)."
    )
    notes.append(
        f"활성 손목={feat.active_wrist}, 머리영역 최장 체류={feat.contact_frames} 프레임."
    )

    if feat.contact_frames < MIN_CONTACT_FRAMES:
        notes.append(
            f"체류 프레임 {feat.contact_frames} < {MIN_CONTACT_FRAMES} → 비접촉."
        )
        timeline = _frame_timeline(seq_norm, ActionLabel.NO_CONTACT)
        return ClassificationResult(
            ActionLabel.NO_CONTACT, 0.95, feat, notes, timeline
        )

    notes.append(
        "통계량 (모두 어깨너비 단위): "
        f"std(wrist_y)={feat.wrist_y_std:.4f}, "
        f"std(wrist_y)_early={feat.wrist_y_std_early:.4f}, "
        f"std(head_y)={feat.head_y_std:.4f}, "
        f"oscillations={feat.n_oscillations}, "
        f"rise_wrist={feat.rise_wrist:.4f}, "
        f"rise_head={feat.rise_head:.4f}, "
        f"corr={feat.corr_wrist_head:.3f}."
    )

    helmet_rule = (
        feat.pause_detected
        and feat.rise_wrist > TAU_RISE
        and feat.rise_head > TAU_RISE
        and feat.corr_wrist_head > TAU_CORR
    )
    scratch_rule = (
        feat.wrist_y_std >= TAU_SCRATCH_STD
        and feat.head_y_std <= TAU_HEAD_STILL_STD
        and feat.n_oscillations >= MIN_OSCILLATIONS
        and feat.rise_head < TAU_RISE
    )

    if helmet_rule:
        # 신뢰도: 상승량·상관이 임계값을 얼마나 여유 있게 넘었는지
        conf = float(
            np.clip(
                0.55
                + 0.20 * min(feat.rise_head / (2 * TAU_RISE), 1.0)
                + 0.15 * min(max(feat.corr_wrist_head, 0.0), 1.0)
                + 0.10 * float(feat.pause_detected),
                0.55,
                0.99,
            )
        )
        notes.append(
            "규칙 HELMET_OFF 충족: "
            f"일시정지(std_early={feat.wrist_y_std_early:.4f} < {TAU_PAUSE_STD}) "
            f"+ 동시 상승(Δŷ_wrist={feat.rise_wrist:.3f}, Δŷ_head={feat.rise_head:.3f} "
            f"> {TAU_RISE}) + corr={feat.corr_wrist_head:.2f} > {TAU_CORR}."
        )
        notes.append("예방 알림: 작업자가 안전모를 벗으려는 동작으로 판정합니다.")
        label = ActionLabel.HELMET_OFF
    elif scratch_rule:
        conf = float(
            np.clip(
                0.55
                + 0.20 * min(feat.wrist_y_std / (2 * TAU_SCRATCH_STD), 1.0)
                + 0.15 * min(feat.n_oscillations / 8.0, 1.0)
                + 0.10 * (1.0 - min(feat.head_y_std / TAU_HEAD_STILL_STD, 1.0)),
                0.55,
                0.99,
            )
        )
        notes.append(
            "규칙 SCRATCH 충족: "
            f"손목 진동 std={feat.wrist_y_std:.4f} ≥ {TAU_SCRATCH_STD}, "
            f"머리 고정 std={feat.head_y_std:.4f} ≤ {TAU_HEAD_STILL_STD}, "
            f"영점교차={feat.n_oscillations} ≥ {MIN_OSCILLATIONS}, "
            f"머리 상승={feat.rise_head:.3f} < {TAU_RISE}."
        )
        notes.append("정상: 단순 머리 긁기로 판정합니다. 알림을 울리지 않습니다.")
        label = ActionLabel.SCRATCH
    else:
        conf = 0.40
        notes.append(
            "두 규칙 모두 미충족 → 머리 접촉은 있으나 동작 유형을 단정하지 않습니다."
        )
        label = ActionLabel.UNKNOWN_CONTACT

    timeline = _frame_timeline(seq_norm, label)
    return ClassificationResult(label, conf, feat, notes, timeline)


def _frame_timeline(seq_norm: np.ndarray, final: ActionLabel) -> list[str]:
    """시각화용 프레임 라벨. 접촉 전에는 idle, 접촉 후 최종 라벨을 입힌다."""
    t = seq_norm.shape[0]
    bboxes = [compute_head_bbox_norm(seq_norm[i]) for i in range(t)]
    _, _, in_head = _pick_active_wrist(seq_norm, bboxes)
    start, end = _longest_true_run(in_head)
    labels = ["idle"] * t
    if end <= start:
        return labels
    contact_label = (
        final.value
        if final in (ActionLabel.SCRATCH, ActionLabel.HELMET_OFF)
        else "contact"
    )
    for i in range(start, end):
        labels[i] = contact_label
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
# 조건 1: 정적 스켈레톤 시각화
# ===========================================================================
def canonical_pose_norm() -> np.ndarray:
    """정규화 공간의 정면 상체 스탠딩 포즈. 목=(0,0), 어깨너비=1.0."""
    pose = np.zeros((17, 2), dtype=np.float64)
    pose[NOSE] = (0.00, -0.55)
    pose[L_EYE] = (-0.10, -0.60)
    pose[R_EYE] = (0.10, -0.60)
    pose[L_EAR] = (-0.18, -0.52)
    pose[R_EAR] = (0.18, -0.52)
    pose[L_SHOULDER] = (-0.50, 0.00)
    pose[R_SHOULDER] = (0.50, 0.00)
    pose[L_ELBOW] = (-0.68, 0.58)
    pose[R_ELBOW] = (0.68, 0.58)
    pose[L_WRIST] = (-0.72, 1.08)
    pose[R_WRIST] = (0.72, 1.08)
    pose[L_HIP] = (-0.34, 1.38)
    pose[R_HIP] = (0.34, 1.38)
    pose[L_KNEE] = (-0.36, 2.28)
    pose[R_KNEE] = (0.36, 2.28)
    pose[L_ANKLE] = (-0.34, 3.18)
    pose[R_ANKLE] = (0.34, 3.18)
    return pose


def pose_norm_to_pixels(
    pose_norm: np.ndarray,
    scale_px: float,
    origin_px: tuple[float, float],
) -> np.ndarray:
    """정규화 포즈를 픽셀로 투영. scale_px = 어깨너비(px) = 카메라 거리 대리변수."""
    origin = np.asarray(origin_px, dtype=np.float64)
    return pose_norm * float(scale_px) + origin


def _bent_elbow(shoulder: np.ndarray, wrist: np.ndarray, outward: float) -> np.ndarray:
    """어깨-손목 중점에서 바깥쪽으로 꺾인 팔꿈치. 시각화용 단순 IK."""
    mid = 0.5 * (shoulder + wrist)
    vec = wrist - shoulder
    length = np.linalg.norm(vec)
    if length < 1e-8:
        return mid
    # 이미지 좌표에서 왼쪽으로 90도 회전 (x, y) → (y, -x)
    perp = np.array([vec[1], -vec[0]]) / length
    return mid + perp * outward


def plot_static_skeleton(
    keypoints: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str = "정규화 상체 스켈레톤 (어깨너비 = 1.0)",
    show_bbox: bool = True,
    show_labels: bool = True,
    highlight_wrist: str | None = None,
) -> plt.Axes:
    """가상의 2D 상체 스켈레톤을 한 장의 정적 이미지로 플롯한다.

    관절을 선으로 연결해 머리·어깨·팔·골반 형태가 보이도록 한다.
    좌표는 정규화 공간이 기본이며, 머리 bbox 도 같은 단위로 겹쳐 그린다.
    """
    _configure_korean_font()
    pose = canonical_pose_norm() if keypoints is None else _as_sequence(keypoints)[0]
    # 이미 정규화된 포즈라고 가정. 픽셀 입력이면 정규화한다.
    if np.linalg.norm(pose[L_SHOULDER] - pose[R_SHOULDER]) > 2.5:
        pose, _ = normalize_keypoints(pose)
        pose = pose[0]

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 9.2), facecolor="#0b1220")
        fig.patch.set_facecolor("#0b1220")
    else:
        fig = ax.figure

    ax.set_facecolor("#10182a")
    bbox = compute_head_bbox_norm(pose)
    neck = np.array([0.0, 0.0])
    head_top = estimate_head_top_norm(pose)

    if show_bbox:
        x, y, w, h = bbox.as_xywh()
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                fill=True,
                facecolor="#f5c518",
                alpha=0.12,
                edgecolor="#f5c518",
                linewidth=1.8,
                linestyle="--",
                label="동적 머리 Bounding Box",
                zorder=1,
            )
        )

    for a, b in SKELETON_BONES:
        ax.plot(
            [pose[a, 0], pose[b, 0]],
            [pose[a, 1], pose[b, 1]],
            color="#5ee0ff",
            linewidth=2.4,
            solid_capstyle="round",
            zorder=2,
        )
    # 목 파생점 연결
    ax.plot(
        [pose[L_SHOULDER, 0], neck[0], pose[R_SHOULDER, 0]],
        [pose[L_SHOULDER, 1], neck[1], pose[R_SHOULDER, 1]],
        color="#5ee0ff",
        linewidth=2.4,
        zorder=2,
    )
    ax.plot(
        [neck[0], pose[NOSE, 0]],
        [neck[1], pose[NOSE, 1]],
        color="#5ee0ff",
        linewidth=2.4,
        zorder=2,
    )

    joint_colors = np.full(17, "#d7ecff")
    joint_colors[[L_SHOULDER, R_SHOULDER]] = "#7cffb2"
    joint_colors[[L_WRIST, R_WRIST]] = "#ffd166"
    joint_colors[NOSE] = "#ff8fab"
    if highlight_wrist == "right":
        joint_colors[R_WRIST] = "#ff4d4d"
    elif highlight_wrist == "left":
        joint_colors[L_WRIST] = "#ff4d4d"

    ax.scatter(
        pose[:, 0],
        pose[:, 1],
        c=joint_colors,
        s=48,
        zorder=4,
        edgecolors="#0b1220",
        linewidths=0.6,
    )
    ax.scatter(
        [neck[0]], [neck[1]], c="#7cffb2", s=70, zorder=5, edgecolors="#0b1220", label="목 (원점)"
    )
    ax.scatter(
        [head_top[0]],
        [head_top[1]],
        c="#f5c518",
        s=70,
        marker="^",
        zorder=5,
        label="머리 상단 (추정)",
    )

    # 어깨너비 = 1.0 스케일 막대
    ax.annotate(
        "",
        xy=(pose[R_SHOULDER, 0], -0.08),
        xytext=(pose[L_SHOULDER, 0], -0.08),
        arrowprops=dict(arrowstyle="<->", color="#7cffb2", lw=1.6),
    )
    ax.text(
        0.0,
        -0.18,
        "Shoulder Width = 1.0  (정규화 단위)",
        ha="center",
        va="top",
        color="#7cffb2",
        fontsize=9,
    )

    if show_labels:
        label_map = {
            NOSE: "코",
            L_SHOULDER: "왼어깨",
            R_SHOULDER: "오른어깨",
            L_ELBOW: "왼팔꿈치",
            R_ELBOW: "오른팔꿈치",
            L_WRIST: "왼손목",
            R_WRIST: "오른손목",
            L_HIP: "왼골반",
            R_HIP: "오른골반",
        }
        for idx, name in label_map.items():
            ax.text(
                pose[idx, 0] + 0.06,
                pose[idx, 1],
                name,
                color="#c9d6f0",
                fontsize=8,
                va="center",
            )

    ax.set_title(title, color="#f2f6ff", fontsize=13, pad=12)
    ax.set_xlabel("X (어깨너비 단위, 목=0)", color="#9db0d0")
    ax.set_ylabel("Y (아래가 + / 위가 −)", color="#9db0d0")
    ax.tick_params(colors="#8aa0c4")
    for spine in ax.spines.values():
        spine.set_color("#2a3a58")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(3.5, -1.5)
    ax.grid(True, color="#1c2a44", linestyle=":", linewidth=0.8)
    ax.legend(loc="lower right", facecolor="#152038", edgecolor="#2a3a58", labelcolor="#e8eefc")

    if created_fig:
        fig.tight_layout()
    return ax


def plot_sequence_strip(
    seq_pixels: np.ndarray,
    infos: list[NormalizeInfo],
    frame_indices: Iterable[int],
    title: str,
    result: ClassificationResult,
    out_path: Path | None = None,
) -> plt.Figure:
    """시퀀스에서 몇 프레임을 골라 스틱피겨 스트립으로 그린다."""
    _configure_korean_font()
    idx = list(frame_indices)
    n = len(idx)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 6.4), facecolor="#0b1220")
    if n == 1:
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
        bbox = compute_head_bbox_norm(pose)
        x, y, w, h = bbox.as_xywh()
        ax.add_patch(
            Rectangle(
                (x, y), w, h, fill=True, facecolor="#f5c518", alpha=0.12,
                edgecolor="#f5c518", linewidth=1.2, linestyle="--",
            )
        )
        for a, b in SKELETON_BONES:
            ax.plot(
                [pose[a, 0], pose[b, 0]],
                [pose[a, 1], pose[b, 1]],
                color="#5ee0ff",
                lw=2.0,
            )
        ax.scatter(pose[:, 0], pose[:, 1], c="#d7ecff", s=22, zorder=3)
        ax.scatter(pose[R_WRIST, 0], pose[R_WRIST, 1], c=accent, s=46, zorder=4)
        ax.scatter(*estimate_head_top_norm(pose), c="#f5c518", s=36, marker="^", zorder=4)
        ax.set_title(f"t = {fi}", color="#e8eefc", fontsize=10)
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(3.5, -1.6)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(
        f"{title}\n판정: {LABEL_KO[result.label]}",
        color=accent,
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    return fig


def plot_feature_timeline(
    seq_pixels: np.ndarray,
    result: ClassificationResult,
    title: str,
    out_path: Path | None = None,
) -> plt.Figure:
    """손목·머리 Y 궤적과 머리영역 체류를 한 장에 그려 판정 근거를 보여 준다."""
    _configure_korean_font()
    seq_norm, _ = normalize_keypoints(seq_pixels)
    t = np.arange(seq_norm.shape[0])
    bboxes = [compute_head_bbox_norm(seq_norm[i]) for i in range(len(t))]
    wrist, which, in_head = _pick_active_wrist(seq_norm, bboxes)
    head_y = np.array([estimate_head_top_norm(seq_norm[i])[1] for i in range(len(t))])

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True, facecolor="#0b1220")
    fig.patch.set_facecolor("#0b1220")
    for ax in axes:
        ax.set_facecolor("#10182a")
        ax.tick_params(colors="#8aa0c4")
        for spine in ax.spines.values():
            spine.set_color("#2a3a58")
        ax.grid(True, color="#1c2a44", linestyle=":", linewidth=0.8)

    axes[0].plot(t, wrist[:, 1], color="#ffd166", lw=2.0, label=f"{which} wrist ŷ")
    axes[0].plot(t, head_y, color="#f5c518", lw=2.0, label="머리 상단 ŷ")
    start, end = _longest_true_run(in_head)
    if end > start:
        axes[0].axvspan(start, end - 1, color="#f5c518", alpha=0.12, zorder=0)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("정규화 Y (위가 작음)", color="#9db0d0")
    axes[0].legend(facecolor="#152038", edgecolor="#2a3a58", labelcolor="#e8eefc")
    axes[0].set_title(title, color="#f2f6ff")

    # rolling std of wrist y
    win = 8
    rstd = np.array(
        [_std(wrist[max(0, i - win + 1) : i + 1, 1]) for i in range(len(t))]
    )
    axes[1].plot(t, rstd, color="#5ee0ff", lw=2.0, label=f"rolling std(wrist_y, w={win})")
    axes[1].axhline(TAU_SCRATCH_STD, color="#3dd68c", ls="--", lw=1.2, label=f"τ_scratch={TAU_SCRATCH_STD}")
    axes[1].axhline(TAU_PAUSE_STD, color="#ff5c5c", ls="--", lw=1.2, label=f"τ_pause={TAU_PAUSE_STD}")
    axes[1].set_xlabel("프레임", color="#9db0d0")
    axes[1].set_ylabel("표준편차 (어깨너비 단위)", color="#9db0d0")
    axes[1].legend(facecolor="#152038", edgecolor="#2a3a58", labelcolor="#e8eefc")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    return fig


# ===========================================================================
# 시뮬레이션 데이터 (가짜 키포인트 시퀀스)
# ===========================================================================
def _lerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    return (1.0 - u) * a + u * b


def _ease(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _apply_right_arm(pose: np.ndarray, wrist: np.ndarray) -> None:
    pose[R_WRIST] = wrist
    pose[R_ELBOW] = _bent_elbow(pose[R_SHOULDER], wrist, outward=0.22)


def _shift_head(pose: np.ndarray, dy: float) -> None:
    """얼굴 키포인트를 수직 이동. 안전모(머리 상단) 상승을 모사."""
    for idx in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR):
        pose[idx, 1] += dy


def generate_scratch_sequence(
    n_frames: int = 60,
    fps: int = 30,
    scale_px: float = 180.0,
    origin_px: tuple[float, float] = (360.0, 240.0),
    seed: int = 7,
) -> np.ndarray:
    """오른손목이 머리 bbox 에 들어간 뒤 Y축으로 진동하고, 머리는 거의 고정."""
    rng = np.random.default_rng(seed)
    rest = canonical_pose_norm()
    target_wrist = np.array([0.28, -0.42])  # 머리 우측
    seq = np.zeros((n_frames, 17, 2), dtype=np.float64)

    raise_end, scratch_end = 12, 50
    for i in range(n_frames):
        pose = rest.copy()
        if i < raise_end:
            u = _ease(i / max(raise_end - 1, 1))
            wrist = _lerp(rest[R_WRIST], target_wrist, u)
        elif i < scratch_end:
            # 3.4 Hz 상하 진동, 진폭 ≈ 0.09 어깨너비. X 는 소폭.
            t = (i - raise_end) / fps
            wrist = target_wrist + np.array(
                [0.025 * math.sin(2 * math.pi * 1.3 * t), 0.09 * math.sin(2 * math.pi * 3.4 * t)]
            )
        else:
            u = _ease((i - scratch_end) / max(n_frames - scratch_end - 1, 1))
            last = target_wrist + np.array([0.0, 0.09 * math.sin(2 * math.pi * 3.4 * (scratch_end - raise_end) / fps)])
            wrist = _lerp(last, rest[R_WRIST], u)
        _apply_right_arm(pose, wrist)
        pose += rng.normal(0.0, 0.006, size=pose.shape)
        # 어깨너비가 1.0 근처를 유지하도록 어깨는 고정에 가깝게
        pose[L_SHOULDER] = rest[L_SHOULDER] + rng.normal(0.0, 0.003, size=2)
        pose[R_SHOULDER] = rest[R_SHOULDER] + rng.normal(0.0, 0.003, size=2)
        seq[i] = pose_norm_to_pixels(pose, scale_px, origin_px)
    return seq


def generate_helmet_off_sequence(
    n_frames: int = 60,
    fps: int = 30,
    scale_px: float = 180.0,
    origin_px: tuple[float, float] = (360.0, 240.0),
    seed: int = 11,
) -> np.ndarray:
    """손목이 정수리로 들어가 일시 정지한 뒤, 손목과 머리 상단이 함께 상승."""
    rng = np.random.default_rng(seed)
    rest = canonical_pose_norm()
    grab = np.array([0.12, -0.72])  # 정수리/챙 부근
    seq = np.zeros((n_frames, 17, 2), dtype=np.float64)

    raise_end, pause_end, lift_end = 10, 20, 48
    lift_amount = -0.42  # 정규화 Y 감소 = 위쪽

    for i in range(n_frames):
        pose = rest.copy()
        head_dy = 0.0
        if i < raise_end:
            u = _ease(i / max(raise_end - 1, 1))
            wrist = _lerp(rest[R_WRIST], grab, u)
        elif i < pause_end:
            # 일시 정지: 파지. 매우 작은 떨림만.
            wrist = grab + rng.normal(0.0, 0.004, size=2)
        elif i < lift_end:
            u = _ease((i - pause_end) / max(lift_end - pause_end - 1, 1))
            wrist = grab + np.array([0.02 * u, lift_amount * u])
            head_dy = lift_amount * u
        else:
            u = 1.0
            wrist = grab + np.array([0.02, lift_amount])
            head_dy = lift_amount
        _apply_right_arm(pose, wrist)
        _shift_head(pose, head_dy)
        pose += rng.normal(0.0, 0.005, size=pose.shape)
        pose[L_SHOULDER] = rest[L_SHOULDER] + rng.normal(0.0, 0.003, size=2)
        pose[R_SHOULDER] = rest[R_SHOULDER] + rng.normal(0.0, 0.003, size=2)
        seq[i] = pose_norm_to_pixels(pose, scale_px, origin_px)
    return seq


def generate_idle_sequence(
    n_frames: int = 45,
    scale_px: float = 180.0,
    origin_px: tuple[float, float] = (360.0, 240.0),
    seed: int = 3,
) -> np.ndarray:
    """팔을 몸통 옆에서 흔들기만 하고 머리에는 닿지 않는 정상 보행 유사 동작."""
    rng = np.random.default_rng(seed)
    rest = canonical_pose_norm()
    seq = np.zeros((n_frames, 17, 2), dtype=np.float64)
    for i in range(n_frames):
        pose = rest.copy()
        phase = 2 * math.pi * i / 18.0
        wrist = rest[R_WRIST] + np.array([0.06 * math.sin(phase), 0.10 * math.sin(phase)])
        _apply_right_arm(pose, wrist)
        pose += rng.normal(0.0, 0.005, size=pose.shape)
        seq[i] = pose_norm_to_pixels(pose, scale_px, origin_px)
    return seq


def generate_rest_on_helmet_sequence(
    n_frames: int = 50,
    scale_px: float = 180.0,
    origin_px: tuple[float, float] = (360.0, 240.0),
    seed: int = 19,
) -> np.ndarray:
    """안전모에 손을 올리기만 하고 들어 올리지는 않음 → unknown_contact 기대."""
    rng = np.random.default_rng(seed)
    rest = canonical_pose_norm()
    grab = np.array([0.14, -0.70])
    seq = np.zeros((n_frames, 17, 2), dtype=np.float64)
    raise_end = 12
    for i in range(n_frames):
        pose = rest.copy()
        if i < raise_end:
            u = _ease(i / max(raise_end - 1, 1))
            wrist = _lerp(rest[R_WRIST], grab, u)
        else:
            wrist = grab + rng.normal(0.0, 0.005, size=2)
        _apply_right_arm(pose, wrist)
        pose += rng.normal(0.0, 0.004, size=pose.shape)
        seq[i] = pose_norm_to_pixels(pose, scale_px, origin_px)
    return seq


# ===========================================================================
# 시퀀스 → 대시보드 JSON
# ===========================================================================
def sequence_payload(name: str, seq_pixels: np.ndarray, expected: ActionLabel) -> dict:
    result = classify_pose_sequence(seq_pixels)
    seq_norm, infos = normalize_keypoints(seq_pixels)
    frames = []
    for i, pose in enumerate(seq_norm):
        bbox = compute_head_bbox_norm(pose)
        head_top = estimate_head_top_norm(pose)
        frames.append(
            {
                "keypoints": pose.tolist(),
                "bbox": [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max],
                "head_top": head_top.tolist(),
                "neck": [0.0, 0.0],
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
    """체형·카메라 거리가 다른 여러 스케일로 동일 동작이 같은 라벨이 나오는지 검증."""
    cameras = [
        ("근거리·대형", 240.0, (420.0, 260.0)),
        ("원거리·소형", 95.0, (180.0, 150.0)),
        ("측면 이동 카메라", 160.0, (520.0, 310.0)),
    ]
    scenarios: list[dict] = []
    for tag, scale, origin in cameras:
        scenarios.append(
            sequence_payload(
                f"머리 긁기 · {tag} (scale={scale:.0f}px)",
                generate_scratch_sequence(scale_px=scale, origin_px=origin, seed=7),
                ActionLabel.SCRATCH,
            )
        )
        scenarios.append(
            sequence_payload(
                f"안전모 벗기 · {tag} (scale={scale:.0f}px)",
                generate_helmet_off_sequence(scale_px=scale, origin_px=origin, seed=11),
                ActionLabel.HELMET_OFF,
            )
        )
    scenarios.append(
        sequence_payload(
            "팔 흔들기 · 머리 비접촉",
            generate_idle_sequence(),
            ActionLabel.NO_CONTACT,
        )
    )
    scenarios.append(
        sequence_payload(
            "안전모에 손만 올림 · 미상승",
            generate_rest_on_helmet_sequence(),
            ActionLabel.UNKNOWN_CONTACT,
        )
    )
    return scenarios


def render_all_figures(out_dir: Path) -> dict[str, str]:
    """대시보드/README 용 matplotlib 이미지를 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7.2, 9.0), facecolor="#0b1220")
    fig.patch.set_facecolor("#0b1220")
    plot_static_skeleton(ax=ax)
    fig.tight_layout()
    p = out_dir / "static_skeleton.png"
    fig.savefig(p, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths["static_skeleton"] = str(p)

    scratch = generate_scratch_sequence(scale_px=180, origin_px=(360, 240))
    helmet = generate_helmet_off_sequence(scale_px=180, origin_px=(360, 240))
    r_scratch = classify_pose_sequence(scratch)
    r_helmet = classify_pose_sequence(helmet)

    p = out_dir / "scratch_strip.png"
    fig = plot_sequence_strip(
        scratch, [], [0, 12, 22, 34, 48], "단순 머리 긁기 시뮬레이션", r_scratch, p
    )
    plt.close(fig)
    paths["scratch_strip"] = str(p)

    p = out_dir / "helmet_strip.png"
    fig = plot_sequence_strip(
        helmet, [], [0, 10, 18, 32, 48], "안전모 벗기 시뮬레이션", r_helmet, p
    )
    plt.close(fig)
    paths["helmet_strip"] = str(p)

    p = out_dir / "scratch_timeline.png"
    fig = plot_feature_timeline(scratch, r_scratch, "긁기: 손목 Y 분산 ↑ / 머리 고정", p)
    plt.close(fig)
    paths["scratch_timeline"] = str(p)

    p = out_dir / "helmet_timeline.png"
    fig = plot_feature_timeline(helmet, r_helmet, "안전모: 일시정지 후 손목·머리 동시 상승", p)
    plt.close(fig)
    paths["helmet_timeline"] = str(p)

    return paths


def run_self_test(verbose: bool = True) -> bool:
    """가짜 데이터로 분류기가 스케일 불변인지 검증한다."""
    print("\n[알고리즘 개요]")
    print(
        """
  좌표 정규화
      neck = (L_shoulder + R_shoulder) / 2
      scale = ||L_shoulder − R_shoulder||     # 어깨너비 = 1.0
      p̂ = (p − neck) / scale

  동적 머리 bbox (정규화 공간, 목 = 원점)
      x ∈ [nose_x − 0.42, nose_x + 0.42]
      y ∈ [−1.05, +0.18]

  긁기 (정상)
      손목 ∈ bbox 유지,  std(wrist_y) ≥ {:.3f},
      std(head_y) ≤ {:.3f},  진동횟수 ≥ {},  rise_head < {:.2f}

  안전모 벗기 (예방 알림)
      초반 일시정지 std(wrist_y)_early < {:.3f}
      이후 rise_wrist, rise_head > {:.2f}  그리고  corr(wrist_y, head_y) > {:.2f}
      (rise = Δy의 부호 반전: 이미지에서 y 감소 = 위쪽)
""".format(
            TAU_SCRATCH_STD,
            TAU_HEAD_STILL_STD,
            MIN_OSCILLATIONS,
            TAU_RISE,
            TAU_PAUSE_STD,
            TAU_RISE,
            TAU_CORR,
        )
    )

    scenarios = build_demo_scenarios()
    ok = True
    for sc in scenarios:
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
    """디스플레이가 있으면 창을 띄우고, 헤드리스면 건너뛴다."""
    if os.environ.get("DISPLAY") and matplotlib.get_backend().lower() != "agg":
        plt.show()
    elif fig is not None:
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="2D 스켈레톤 안전모/긁기 동작 분류 프로토타입")
    parser.add_argument("--out", default="outputs", help="시각화 저장 디렉터리")
    parser.add_argument("--serve", action="store_true", help="대시보드 HTTP 서버 시작")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    passed = run_self_test(verbose=True)
    if not args.skip_plots:
        paths = render_all_figures(out_dir)
        print("\n[시각화 저장]")
        for k, v in paths.items():
            print(f"  {k}: {v}")
        # 조건 1: 정적 스켈레톤을 화면에 출력 시도
        fig, ax = plt.subplots(figsize=(7.2, 9.0), facecolor="#0b1220")
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
