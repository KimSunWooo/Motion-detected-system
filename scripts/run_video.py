#!/usr/bin/env python3
"""Run the hybrid action detector on a webcam, video file, HTTP, or RTSP stream.

Examples:
  python scripts/run_video.py --source 0 --pose-model yolo11n-pose.pt --latest-frame --debug-overlay --show
  python scripts/run_video.py --source "http://192.168.0.213:8080/video" --rotate 90 --latest-frame --show
  python scripts/run_video.py --source sample.mp4 --pose-model yolo11n-pose.pt --out outputs/sample_annotated.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.config import load_config  # noqa: E402
from helmet_action.inference.capture import ROTATE_CHOICES  # noqa: E402
from helmet_action.inference.source import parse_video_source, redact_source  # noqa: E402
from helmet_action.inference.video_pipeline import load_ml_status, run_video  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    default_conf = float(cfg.get("pose.confidence_threshold", 0.35))
    default_hybrid = str(cfg.get("decision.hybrid_version", "v1"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="video path, webcam index, HTTP, or RTSP URL")
    parser.add_argument("--pose-model", default=os.environ.get("POSE_MODEL_PATH") or None)
    parser.add_argument("--out", default="outputs/annotated.mp4")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--detect-only", action="store_true", help="disable tracking; use detect mode")
    parser.add_argument("--rotate", type=int, choices=list(ROTATE_CHOICES), default=0)
    parser.add_argument("--pose-conf", type=float, default=None, help=f"override pose conf (default config {default_conf})")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference image size")
    parser.add_argument("--hybrid-version", choices=["v1", "v2"], default=None, help=f"default from config: {default_hybrid}")
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--no-record", action="store_true", help="do not create VideoWriter even if --out is set")
    parser.add_argument("--latest-frame", action="store_true", help="drop backlog; always infer the newest frame")
    args = parser.parse_args(argv)

    source = parse_video_source(args.source)
    try:
        from helmet_action.inference.ultralytics_pose_provider import UltralyticsPoseProvider
    except ImportError as exc:
        print(exc)
        print("Ultralytics is optional. Synthetic dashboard still runs without it.")
        return 1

    conf = float(args.pose_conf) if args.pose_conf is not None else None
    provider = UltralyticsPoseProvider(
        model_path=args.pose_model,
        conf=conf,
        track=not args.detect_only,
        imgsz=args.imgsz,
    )
    ml, ml_path, feat_ver, ml_err = load_ml_status()
    hybrid_version = args.hybrid_version or default_hybrid

    print(f"source={redact_source(source)}")
    dest = run_video(
        source,
        provider,
        out_path=None if args.no_record else args.out,
        show=args.show,
        max_frames=args.max_frames,
        rotate=int(args.rotate),
        latest_frame=bool(args.latest_frame),
        hybrid_version=hybrid_version,
        debug_overlay=bool(args.debug_overlay),
        no_record=bool(args.no_record),
        ml=ml,
        print_banner=True,
    )
    if dest is not None:
        print(f"wrote {dest}")
    elif args.no_record:
        print("recording disabled (--no-record)")
    print("Helmet State is UNKNOWN unless a real HelmetPresenceDetector is wired in.")
    if ml is None:
        print(f"ML Loaded: NO ({ml_err})")
    else:
        print(f"ML Loaded: YES ({ml_path}, feature={feat_ver})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
