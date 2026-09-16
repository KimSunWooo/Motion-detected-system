#!/usr/bin/env python3
"""Run the hybrid action detector on a webcam, video file, or RTSP stream.

python scripts/run_video.py --source sample.mp4 --pose-model yolo11n-pose.pt
python scripts/run_video.py --source 0
python scripts/run_video.py --source rtsp://192.168.0.10:554/stream
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.inference.source import parse_video_source, redact_source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="video path, webcam index, or RTSP URL")
    parser.add_argument("--pose-model", default=os.environ.get("POSE_MODEL_PATH") or None)
    parser.add_argument("--out", default="outputs/annotated.mp4")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--detect-only", action="store_true", help="disable tracking; use detect mode")
    args = parser.parse_args(argv)

    source = parse_video_source(args.source)
    try:
        from helmet_action.inference.ultralytics_pose_provider import UltralyticsPoseProvider
        from helmet_action.inference.video_pipeline import run_video
    except ImportError as exc:
        print(exc)
        print("Ultralytics is optional. Synthetic dashboard still runs without it.")
        return 1

    provider = UltralyticsPoseProvider(model_path=args.pose_model, track=not args.detect_only)
    print(f"source={redact_source(source)}")
    dest = run_video(source, provider, out_path=args.out, show=args.show, max_frames=args.max_frames)
    print(f"wrote {dest}")
    print("Helmet State is UNKNOWN unless a real HelmetPresenceDetector is wired in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
