"""Baseline rule features — preserved from the original classifier."""

from __future__ import annotations

import numpy as np

from helmet_action.config import load_config
from helmet_action.models.labels import ActionLabel, ClassificationResult, FeatureReport
from helmet_action.pose.constants import L_WRIST, R_WRIST
from helmet_action.pose.geometry import compute_head_regions, head_scale_norm
from helmet_action.pose.normalizer import normalize_keypoints


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


def _rules() -> dict:
    cfg = load_config()
    return {
        "pause_frames": int(cfg.get("rules.pause_frames", 8)),
        "pause_std": float(cfg.get("rules.helmet_remove.pause_std", 0.016)),
        "min_center": int(cfg.get("rules.min_center_frames", 8)),
        "min_ear": int(cfg.get("rules.min_ear_frames", 8)),
        "scratch_radius": float(cfg.get("rules.scratch.radius", 0.11)),
        "scratch_std": float(cfg.get("rules.scratch.std", 0.028)),
        "min_osc": int(cfg.get("rules.scratch.min_oscillations", 4)),
        "dx": float(cfg.get("rules.helmet_remove.dx_spread", 0.08)),
        "rise": float(cfg.get("rules.helmet_remove.co_rise_y", 0.05)),
        "radial": float(cfg.get("rules.helmet_remove.radial", 0.08)),
        "scale": float(cfg.get("rules.helmet_remove.scale_up", 0.07)),
    }


def _first_pause(xy: np.ndarray, window: int | None = None, tau: float | None = None) -> tuple[int | None, float]:
    r = _rules()
    window = window if window is not None else r["pause_frames"]
    tau = tau if tau is not None else r["pause_std"]
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
    r = _rules()
    t = seq_norm.shape[0]
    regions = [compute_head_regions(seq_norm[i]) for i in range(t)]
    lw = seq_norm[:, L_WRIST]
    rw = seq_norm[:, R_WRIST]
    centers = np.stack([rg.center for rg in regions])

    in_box_l = np.array([regions[i].in_bbox(lw[i]) for i in range(t)])
    in_box_r = np.array([regions[i].in_bbox(rw[i]) for i in range(t)])
    in_c_l = np.array([regions[i].in_center(lw[i]) for i in range(t)])
    in_c_r = np.array([regions[i].in_center(rw[i]) for i in range(t)])
    in_el = np.array([regions[i].in_left_ear(lw[i]) for i in range(t)])
    in_er = np.array([regions[i].in_right_ear(rw[i]) for i in range(t)])
    both_ears = in_el & in_er

    feat = FeatureReport()
    b_l0, b_l1 = _longest_run(in_box_l)
    b_r0, b_r1 = _longest_run(in_box_r)
    if (b_r1 - b_r0) >= (b_l1 - b_l0):
        feat.active_wrist = "right"
        b0, b1 = b_r0, b_r1
        active = rw
    else:
        feat.active_wrist = "left"
        b0, b1 = b_l0, b_l1
        active = lw
    feat.bbox_frames = b1 - b0

    c_l0, c_l1 = _longest_run(in_c_l)
    c_r0, c_r1 = _longest_run(in_c_r)
    feat.center_frames = max(c_l1 - c_l0, c_r1 - c_r0)

    e0, e1 = _longest_run(both_ears)
    feat.both_ear_frames = e1 - e0

    if feat.bbox_frames >= 3 and np.isfinite(active[b0:b1]).all():
        dwell = active[b0:b1]
        centroid = dwell.mean(axis=0)
        rad = np.linalg.norm(dwell - centroid, axis=1)
        feat.scratch_radius = float(np.quantile(rad, 0.85))
        feat.scratch_std = float(np.sqrt(_std(dwell[:, 0]) ** 2 + _std(dwell[:, 1]) ** 2))
        feat.n_oscillations = _zero_crossings(dwell[:, 0]) + _zero_crossings(dwell[:, 1])

    if feat.both_ear_frames >= 3:
        trail_end = min(t, e1 + 16)
        sl = slice(e0, trail_end)
        mean_xy = 0.5 * (lw[sl] + rw[sl])
        pause_i, pause_std = _first_pause(mean_xy)
        feat.pause_std = pause_std
        feat.pause_detected = pause_i is not None
        if pause_i is None:
            pause_i, pause_n = 0, min(r["pause_frames"], mean_xy.shape[0])
        else:
            pause_n = min(r["pause_frames"], mean_xy.shape[0] - pause_i)

        late = slice(pause_i + pause_n, None)
        if mean_xy[late].size == 0:
            late = slice(pause_i, None)

        dx = np.abs(rw[sl, 0] - lw[sl, 0])
        dy_mean = mean_xy[:, 1]
        r_l = np.linalg.norm(lw[sl] - centers[sl], axis=1)
        r_r = np.linalg.norm(rw[sl] - centers[sl], axis=1)
        r_mean = 0.5 * (r_l + r_r)
        d = np.linalg.norm(lw[sl] - rw[sl], axis=1)
        hs = np.array([head_scale_norm(seq_norm[i]) for i in range(e0, trail_end)])

        dx_p = float(np.nanmedian(dx[pause_i : pause_i + pause_n]))
        y_p = float(np.nanmedian(dy_mean[pause_i : pause_i + pause_n]))
        r_p = float(np.nanmedian(r_mean[pause_i : pause_i + pause_n]))
        d_p = float(np.nanmedian(d[pause_i : pause_i + pause_n]))
        h_p = float(np.nanmedian(hs[pause_i : pause_i + pause_n]))

        feat.dx_spread = float(np.nanmax(dx[late]) - dx_p) if dx[late].size else 0.0
        feat.co_rise_y = float(y_p - np.nanmin(dy_mean[late])) if dy_mean[late].size else 0.0
        feat.radial_expand = float(np.nanmax(r_mean[late]) - r_p) if r_mean[late].size else 0.0
        feat.d_wrist_pause = d_p
        feat.d_wrist_late = float(np.nanmax(d[late])) if d[late].size else d_p
        feat.wrist_spread = feat.d_wrist_late - d_p
        h_late = float(np.nanmax(hs[late])) if hs[late].size else h_p
        feat.head_scale_up = (h_late / max(h_p, 1e-6)) - 1.0
    return feat


def classify_pose_sequence(keypoints: np.ndarray) -> ClassificationResult:
    r = _rules()
    seq_norm, _ = normalize_keypoints(keypoints)
    feat = extract_features(seq_norm)
    notes: list[str] = [
        "정규화: p̂ = (p − neck) / shoulder_width. 하체·골반 길이는 쓰지 않는다.",
        f"bbox 체류={feat.bbox_frames}fr ({feat.active_wrist}), 양귀 파지={feat.both_ear_frames}fr.",
    ]

    expansion = (
        feat.dx_spread > r["dx"]
        or feat.co_rise_y > r["rise"]
        or feat.radial_expand > r["radial"]
        or feat.head_scale_up > r["scale"]
    )
    helmet_rule = feat.both_ear_frames >= r["min_ear"] and feat.pause_detected and expansion
    scratch_rule = (
        feat.bbox_frames >= r["min_center"]
        and feat.scratch_radius <= r["scratch_radius"]
        and feat.scratch_std >= r["scratch_std"]
        and feat.n_oscillations >= r["min_osc"]
        and feat.both_ear_frames < r["min_ear"]
    )

    if feat.bbox_frames < r["min_center"] and feat.both_ear_frames < r["min_ear"]:
        notes.append("머리 bbox 최소 체류 미달 → 비접촉.")
        label, conf = ActionLabel.NO_CONTACT, 0.95
    elif helmet_rule:
        conf = float(
            np.clip(
                0.55
                + 0.12 * min(max(feat.dx_spread, 0) / (2 * r["dx"]), 1)
                + 0.12 * min(max(feat.co_rise_y, 0) / (2 * r["rise"]), 1)
                + 0.12 * min(max(feat.radial_expand, 0) / (2 * r["radial"]), 1)
                + 0.12 * min(max(feat.head_scale_up, 0) / (2 * r["scale"]), 1),
                0.55,
                0.99,
            )
        )
        notes.append(
            "하이브리드 통계량: "
            f"pause_σ={feat.pause_std:.4f}, "
            f"Δ|x_R−x_L|={feat.dx_spread:.3f} (τ={r['dx']}), "
            f"−Δȳ={feat.co_rise_y:.3f} (τ={r['rise']}), "
            f"Δr={feat.radial_expand:.3f} (τ={r['radial']}), "
            f"Δd_ear={feat.head_scale_up:.3f} (τ={r['scale']})."
        )
        notes.append(
            "규칙 HELMET_OFF: 양손이 귀 부근에서 멈춘 뒤 "
            "X축 벌어짐 · 동반 Y상승(y 감소) · 방사형/겉보기 팽창 중 하나가 발생."
        )
        notes.append("예방 알림: 안전모 벗기 시도로 판정합니다.")
        label = ActionLabel.HELMET_OFF
    elif scratch_rule:
        conf = float(
            np.clip(
                0.58
                + 0.22 * min(feat.scratch_std / (2 * r["scratch_std"]), 1)
                + 0.20 * min(feat.n_oscillations / 10.0, 1),
                0.58,
                0.99,
            )
        )
        notes.append(
            "통계량: "
            f"r_85%={feat.scratch_radius:.3f} (≤ {r['scratch_radius']}), "
            f"σ_xy={feat.scratch_std:.4f} (≥ {r['scratch_std']}), "
            f"osc={feat.n_oscillations} (≥ {r['min_osc']})."
        )
        notes.append(
            "규칙 SCRATCH: 한쪽 손목만 bbox 안에서 좁은 반경의 고주파수 진동. "
            "양손 협응·팽창이 없어 정상 긁기로 봅니다."
        )
        notes.append("정상: 알림을 울리지 않습니다.")
        label = ActionLabel.SCRATCH
    else:
        conf = 0.40
        notes.append(
            "통계량: "
            f"r={feat.scratch_radius:.3f}, σ_xy={feat.scratch_std:.4f}, "
            f"osc={feat.n_oscillations}, Δx={feat.dx_spread:.3f}, "
            f"−Δȳ={feat.co_rise_y:.3f}, Δr={feat.radial_expand:.3f}."
        )
        notes.append("긁기 진동과 양손 팽창 모두 임계값을 못 넘겨 판정을 보류합니다.")
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
        elif rg.in_bbox(lw) or rg.in_bbox(rw):
            labels[i] = "bbox"
    if final in (ActionLabel.SCRATCH, ActionLabel.HELMET_OFF):
        key = "bbox" if final is ActionLabel.SCRATCH else "ear_grasp"
        for i, v in enumerate(labels):
            if v == key:
                labels[i] = final.value
    return labels


class RuleBasedActionClassifier:
    """Baseline geometric classifier. Do not delete — used as a safety veto."""

    name = "rule_based"

    def predict(self, keypoints: np.ndarray) -> ClassificationResult:
        return classify_pose_sequence(keypoints)

    def predict_label(self, keypoints: np.ndarray) -> str:
        return self.predict(keypoints).label.value
