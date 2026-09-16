#!/usr/bin/env python3
"""Extract YOLO pose sequences from real videos. Does not train.

PYTHONPATH=src python scripts/extract_real_pose.py \\
  --input data/real/raw --output data/real/processed --pose-model yolo11n-pose.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.real.dataset import (  # noqa: E402
    RealSequence,
    default_processed_root,
    default_raw_root,
    iter_raw_videos,
    parse_raw_video_path,
    save_sequence,
    write_dataset_metadata,
)


def _extract_video(path: Path, raw_root: Path, provider, min_frames: int = 8) -> tuple[list[RealSequence], dict]:
    info = parse_raw_video_path(path, raw_root)
    tracks: dict[int, dict[str, list]] = defaultdict(lambda: {
        "k": [],
        "c": [],
        "t": [],
        "bbox": [],
        "det": [],
        "idx": [],
    })
    n_frames = 0
    n_empty = 0
    fps = 20.0
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless is required to extract real pose") from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], {"source_video": str(path), "error": "cannot open video", "n_frames": 0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            obs = provider.infer_frame(frame, frame_index=n_frames, fps=fps)
            n_frames += 1
            if not obs:
                n_empty += 1
                continue
            for o in obs:
                bucket = tracks[int(o.track_id)]
                bucket["k"].append(o.keypoints)
                bucket["c"].append(o.keypoint_confidence)
                bucket["t"].append(o.timestamp)
                bucket["bbox"].append(o.bbox)
                bucket["det"].append(o.detection_confidence)
                bucket["idx"].append(o.frame_index)
    finally:
        cap.release()

    sequences: list[RealSequence] = []
    for tid, bucket in tracks.items():
        if len(bucket["k"]) < min_frames:
            continue
        sequences.append(
            RealSequence(
                keypoints=np.stack(bucket["k"], axis=0),
                confidences=np.stack(bucket["c"], axis=0),
                timestamps=np.asarray(bucket["t"], dtype=np.float64),
                bbox=np.asarray(bucket["bbox"], dtype=np.float64),
                track_ids=np.full(len(bucket["k"]), int(tid), dtype=np.int32),
                label=info["label"],
                subject_id=info["subject_id"],
                source_video=info["source_video"],
                fps=fps,
                camera_id=info["camera_id"],
                detection_confidence=np.asarray(bucket["det"], dtype=np.float64),
                notes=[],
            )
        )
    stats = {
        "source_video": info["source_video"],
        "subject_id": info["subject_id"],
        "label": info["label"],
        "camera_id": info["camera_id"],
        "fps": fps,
        "frame_count": n_frames,
        "empty_frames": n_empty,
        "n_sequences": len(sequences),
        "pose_fail_rate": float(n_empty / max(n_frames, 1)),
    }
    return sequences, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=default_raw_root())
    parser.add_argument("--output", type=Path, default=default_processed_root())
    parser.add_argument("--pose-model", type=str, default=None)
    args = parser.parse_args(argv)

    videos = list(iter_raw_videos(args.input))
    if not videos:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "sequences").mkdir(parents=True, exist_ok=True)
        write_dataset_metadata(args.output, [], extra={"status": "REAL DATASET: NOT AVAILABLE", "n_videos": 0})
        print("REAL DATASET: NOT AVAILABLE (no videos under", args.input, ")")
        return 0

    from helmet_action.inference.ultralytics_pose_provider import UltralyticsPoseProvider

    provider = UltralyticsPoseProvider(model_path=args.pose_model, track=True)
    records = []
    video_stats = []
    index = 0
    for video in videos:
        print(f"extract {video}", flush=True)
        try:
            seqs, stats = _extract_video(video, args.input, provider)
        except Exception as exc:
            video_stats.append({"source_video": str(video), "error": str(exc)})
            continue
        video_stats.append(stats)
        for seq in seqs:
            save_sequence(args.output, seq, index)
            records.append(seq.to_meta())
            index += 1
    write_dataset_metadata(args.output, records, extra={"videos": video_stats, "pose_model": args.pose_model})
    print(f"wrote {len(records)} sequences to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
