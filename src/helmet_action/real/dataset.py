"""Real pose sequence IO. RGB frames are never used as training features."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from helmet_action.config import repo_root

LABEL_ALIASES = {
    "helmet_remove": "HELMET_REMOVE",
    "helmet_put_on": "HELMET_PUT_ON",
    "helmet_puton": "HELMET_PUT_ON",
    "helmet_adjust": "HELMET_ADJUST",
    "head_scratch": "HEAD_SCRATCH",
    "head_touch": "HEAD_TOUCH",
    "wipe_sweat": "WIPE_SWEAT",
    "raise_arms": "RAISE_ARMS",
    "phone_near_head": "PHONE_NEAR_HEAD",
    "idle": "IDLE",
    "HELMET_REMOVE": "HELMET_REMOVE",
    "HELMET_PUT_ON": "HELMET_PUT_ON",
    "HELMET_ADJUST": "HELMET_ADJUST",
    "HEAD_SCRATCH": "HEAD_SCRATCH",
    "HEAD_TOUCH": "HEAD_TOUCH",
    "WIPE_SWEAT": "HEAD_TOUCH",
    "RAISE_ARMS": "UNKNOWN",
    "PHONE_NEAR_HEAD": "HEAD_TOUCH",
    "IDLE": "IDLE",
}

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def anonymize_subject(raw: str) -> str:
    text = str(raw).strip()
    m = re.fullmatch(r"(?i)p(\d+)", text)
    if m:
        return f"P{int(m.group(1)):03d}"
    m = re.fullmatch(r"(?i)person[_-]?(\d+)", text)
    if m:
        return f"P{int(m.group(1)):03d}"
    digits = re.findall(r"\d+", text)
    if digits:
        return f"P{int(digits[0]):03d}"
    slug = re.sub(r"[^A-Za-z0-9]+", "", text)[:8].upper() or "UNK"
    return f"P_{slug}"


def canonical_label(name: str) -> str:
    key = str(name).strip()
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    lowered = key.lower().replace("-", "_").replace(" ", "_")
    return LABEL_ALIASES.get(lowered, key.upper())


def folder_action_name(name: str) -> str:
    """Keep the collection folder name (WIPE_SWEAT) distinct from ML class mapping."""
    key = str(name).strip()
    lowered = key.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "helmet_remove": "HELMET_REMOVE",
        "helmet_put_on": "HELMET_PUT_ON",
        "helmet_puton": "HELMET_PUT_ON",
        "helmet_adjust": "HELMET_ADJUST",
        "head_scratch": "HEAD_SCRATCH",
        "head_touch": "HEAD_TOUCH",
        "wipe_sweat": "WIPE_SWEAT",
        "raise_arms": "RAISE_ARMS",
        "phone_near_head": "PHONE_NEAR_HEAD",
        "idle": "IDLE",
    }
    if key in mapping.values() or key in LABEL_ALIASES and key.isupper():
        return key if key.isupper() else LABEL_ALIASES[key]
    return mapping.get(lowered, key.upper())


@dataclass
class RealSequence:
    keypoints: np.ndarray
    confidences: np.ndarray
    timestamps: np.ndarray
    bbox: np.ndarray
    track_ids: np.ndarray
    label: str
    subject_id: str
    source_video: str
    fps: float
    camera_id: str = "unknown"
    detection_confidence: np.ndarray | None = None
    frame_count: int = 0
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float64)
        self.confidences = np.asarray(self.confidences, dtype=np.float64)
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.bbox = np.asarray(self.bbox, dtype=np.float64)
        self.track_ids = np.asarray(self.track_ids, dtype=np.int32)
        if self.detection_confidence is None:
            self.detection_confidence = np.ones(len(self.keypoints), dtype=np.float64)
        else:
            self.detection_confidence = np.asarray(self.detection_confidence, dtype=np.float64)
        self.frame_count = int(self.keypoints.shape[0])
        self.subject_id = anonymize_subject(self.subject_id)
        self.label = folder_action_name(self.label)

    @property
    def ml_label(self) -> str:
        return canonical_label(self.label)

    def to_meta(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "ml_label": self.ml_label,
            "subject_id": self.subject_id,
            "source_video": self.source_video,
            "fps": float(self.fps),
            "frame_count": int(self.frame_count),
            "camera_id": self.camera_id,
            "n_tracks": int(len(set(map(int, self.track_ids.tolist())))) if self.track_ids.size else 0,
            "notes": list(self.notes),
        }


def parse_raw_video_path(path: Path, raw_root: Path) -> dict[str, str]:
    rel = path.resolve().relative_to(raw_root.resolve())
    parts = list(rel.parts)
    if len(parts) < 3:
        # subject/label/file
        if len(parts) == 2:
            subject, fname = parts[0], parts[1]
            return {
                "subject_id": anonymize_subject(subject),
                "label": folder_action_name(Path(fname).stem.split("_")[0]),
                "camera_id": "unknown",
                "source_video": str(rel),
            }
        raise ValueError(f"expected subject/label/video under {raw_root}, got {rel}")
    subject = parts[0]
    # subject / [camera] / label / file  OR  subject / label / file
    if folder_action_name(parts[1]) in LABEL_ALIASES.values() or parts[1].lower().replace("-", "_") in {
        "helmet_remove",
        "helmet_put_on",
        "helmet_adjust",
        "head_scratch",
        "head_touch",
        "wipe_sweat",
        "raise_arms",
        "phone_near_head",
        "idle",
    }:
        camera = "unknown"
        label = folder_action_name(parts[1])
    else:
        camera = str(parts[1])
        label = folder_action_name(parts[2]) if len(parts) >= 3 else "UNKNOWN"
    return {
        "subject_id": anonymize_subject(subject),
        "label": label,
        "camera_id": camera,
        "source_video": str(rel),
    }


def iter_raw_videos(raw_root: Path) -> Iterator[Path]:
    raw_root = Path(raw_root)
    if not raw_root.exists():
        return
        yield  # pragma: no cover — keeps this a generator
    for path in sorted(raw_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            yield path


def default_raw_root() -> Path:
    return repo_root() / "data" / "real" / "raw"


def default_processed_root() -> Path:
    return repo_root() / "data" / "real" / "processed"


def sequence_id(seq: RealSequence, index: int) -> str:
    stem = Path(seq.source_video).stem
    return f"{seq.subject_id}_{seq.label}_{stem}_t{int(seq.track_ids[0]) if seq.track_ids.size else 0}_{index:04d}"


def save_sequence(processed_root: Path, seq: RealSequence, index: int) -> Path:
    processed_root = Path(processed_root)
    seq_dir = processed_root / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    sid = sequence_id(seq, index)
    npz_path = seq_dir / f"{sid}.npz"
    meta_path = seq_dir / f"{sid}.json"
    np.savez_compressed(
        npz_path,
        keypoints=seq.keypoints.astype(np.float32),
        confidences=seq.confidences.astype(np.float32),
        timestamps=seq.timestamps.astype(np.float64),
        bbox=seq.bbox.astype(np.float32),
        track_ids=seq.track_ids.astype(np.int32),
        detection_confidence=seq.detection_confidence.astype(np.float32),
        label=np.array(seq.label),
        ml_label=np.array(seq.ml_label),
        subject_id=np.array(seq.subject_id),
        source_video=np.array(seq.source_video),
        fps=np.array(float(seq.fps)),
        camera_id=np.array(seq.camera_id),
    )
    meta_path.write_text(json.dumps(seq.to_meta(), indent=2, ensure_ascii=False), encoding="utf-8")
    return npz_path


def load_sequence(path: Path) -> RealSequence:
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    def _s(key: str, default: str = "") -> str:
        if key not in data.files:
            return default
        val = data[key]
        if val.shape == ():
            return str(val.item())
        return str(val)

    return RealSequence(
        keypoints=data["keypoints"],
        confidences=data["confidences"],
        timestamps=data["timestamps"] if "timestamps" in data.files else np.arange(len(data["keypoints"])),
        bbox=data["bbox"] if "bbox" in data.files else np.zeros((len(data["keypoints"]), 4)),
        track_ids=data["track_ids"] if "track_ids" in data.files else np.ones(len(data["keypoints"]), dtype=np.int32),
        label=_s("label"),
        subject_id=_s("subject_id", "P000"),
        source_video=_s("source_video"),
        fps=float(data["fps"].item()) if "fps" in data.files else 20.0,
        camera_id=_s("camera_id", "unknown"),
        detection_confidence=data["detection_confidence"] if "detection_confidence" in data.files else None,
    )


def write_dataset_metadata(processed_root: Path, records: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> Path:
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_sequences": len(records),
        "subjects": sorted({r.get("subject_id") for r in records if r.get("subject_id")}),
        "labels": sorted({r.get("label") for r in records if r.get("label")}),
        "cameras": sorted({r.get("camera_id") for r in records if r.get("camera_id")}),
        "leave_one_subject_ready": True,
        "zero_shot_only": True,
        "records": records,
    }
    if extra:
        payload.update(extra)
    dest = processed_root / "metadata.json"
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest


def list_processed_sequences(processed_root: Path) -> list[Path]:
    seq_dir = Path(processed_root) / "sequences"
    if not seq_dir.exists():
        return []
    return sorted(seq_dir.glob("*.npz"))


def load_processed_dataset(processed_root: Path) -> list[RealSequence]:
    out = []
    errors = []
    for path in list_processed_sequences(processed_root):
        try:
            out.append(load_sequence(path))
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return out
