"""Hand-written real-capture manifest (CSV). Videos themselves are not committed."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

MANIFEST_COLUMNS = [
    "subject_id",
    "video_file",
    "label",
    "camera_pitch",
    "camera_distance",
    "dominant_hand",
    "speed",
    "occlusion",
    "notes",
]

MANIFEST_LABELS = (
    "HELMET_REMOVE",
    "HELMET_PUT_ON",
    "HELMET_ADJUST",
    "HEAD_SCRATCH",
    "HEAD_TOUCH",
    "WIPE_SWEAT",
    "RAISE_ARMS",
    "PHONE_NEAR_HEAD",
)

CAMERA_DISTANCE = ("near", "medium", "far")
SPEED = ("slow", "normal", "fast")
DOMINANT_HAND = ("left", "right", "both")
OCCLUSION = ("none", "partial", "ear", "wrist", "body")

EXAMPLE_ROWS = [
    {
        "subject_id": "P001",
        "video_file": "P001/helmet_remove/take_01.mp4",
        "label": "HELMET_REMOVE",
        "camera_pitch": "45",
        "camera_distance": "medium",
        "dominant_hand": "right",
        "speed": "normal",
        "occlusion": "none",
        "notes": "lateral lift",
    },
    {
        "subject_id": "P001",
        "video_file": "P001/helmet_adjust/take_01.mp4",
        "label": "HELMET_ADJUST",
        "camera_pitch": "40",
        "camera_distance": "near",
        "dominant_hand": "left",
        "speed": "slow",
        "occlusion": "partial",
        "notes": "",
    },
]


class ManifestError(ValueError):
    pass


def default_template_path() -> Path:
    from helmet_action.config import repo_root

    return repo_root() / "docs" / "real_dataset_manifest_template.csv"


def write_manifest_template(path: Path | None = None) -> Path:
    dest = Path(path) if path is not None else default_template_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in EXAMPLE_ROWS:
            writer.writerow(row)
    return dest


def parse_manifest(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ManifestError("manifest has no header")
        missing = [c for c in MANIFEST_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ManifestError(f"manifest missing columns: {missing}")
        for i, raw in enumerate(reader, start=2):
            row = {k: str(raw.get(k, "") or "").strip() for k in MANIFEST_COLUMNS}
            if not any(row.values()):
                continue
            _validate_row(row, line=i)
            rows.append(row)
    return rows


def _validate_row(row: dict[str, str], line: int) -> None:
    if not row["subject_id"]:
        raise ManifestError(f"line {line}: subject_id is required")
    if not row["video_file"]:
        raise ManifestError(f"line {line}: video_file is required")
    label = row["label"].upper().replace("-", "_").replace(" ", "_")
    if label not in MANIFEST_LABELS:
        raise ManifestError(f"line {line}: label {row['label']!r} not in {MANIFEST_LABELS}")
    row["label"] = label
    if row["camera_distance"] and row["camera_distance"].lower() not in CAMERA_DISTANCE:
        raise ManifestError(f"line {line}: camera_distance must be near|medium|far")
    row["camera_distance"] = row["camera_distance"].lower()
    if row["speed"] and row["speed"].lower() not in SPEED:
        raise ManifestError(f"line {line}: speed must be slow|normal|fast")
    row["speed"] = row["speed"].lower()
    if row["dominant_hand"] and row["dominant_hand"].lower() not in DOMINANT_HAND:
        raise ManifestError(f"line {line}: dominant_hand must be left|right|both")
    row["dominant_hand"] = row["dominant_hand"].lower()
    if row["occlusion"]:
        row["occlusion"] = row["occlusion"].lower()
    pitch = row["camera_pitch"]
    if pitch:
        try:
            float(pitch)
        except ValueError as exc:
            raise ManifestError(f"line {line}: camera_pitch must be numeric degrees") from exc
