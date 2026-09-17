#!/usr/bin/env python3
"""Run the hybrid action detector on a webcam, video file, HTTP, or RTSP stream.

Examples:
  python scripts/run_video.py --source 0 --pipeline roi-pose --latest-frame --debug-overlay --show
  python scripts/run_video.py --source sample.mp4 --pipeline full-pose --sequential --out outputs/sample.mp4
  python scripts/run_video.py --source "http://192.168.0.213:8080/video" \\
      --human-model yolo11n.pt --pose-model yolo11n-pose.pt \\
      --detector-fps 5 --pose-fps 10 --roi-margin 0.15 --latest-frame --show
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
    default_pipeline = str(cfg.get("inference.pipeline", "roi-pose"))
    default_det_conf = float(cfg.get("inference.detector.confidence", 0.35))
    default_pose_conf_roi = float(cfg.get("inference.pose.confidence", 0.25))
    default_det_imgsz = int(cfg.get("inference.detector.imgsz", 640))
    default_pose_imgsz = int(cfg.get("inference.pose.imgsz", 384))
    default_det_fps = float(cfg.get("inference.detector.fps", 5))
    default_pose_fps = float(cfg.get("inference.pose.fps", 10))
    default_roi_margin = float(cfg.get("inference.pose.roi_margin", 0.15))

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="video path, webcam index, HTTP, or RTSP URL")
    parser.add_argument(
        "--pipeline",
        choices=["roi-pose", "full-pose"],
        default=default_pipeline,
        help="roi-pose (v0.5) or full-pose (baseline)",
    )
    parser.add_argument("--human-model", default=os.environ.get("HUMAN_MODEL_PATH") or None)
    parser.add_argument("--pose-model", default=os.environ.get("POSE_MODEL_PATH") or None)
    parser.add_argument("--out", default="outputs/annotated.mp4")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--detect-only", action="store_true", help="full-pose: disable tracking")
    parser.add_argument("--rotate", type=int, choices=list(ROTATE_CHOICES), default=0)
    parser.add_argument("--pose-conf", type=float, default=None)
    parser.add_argument("--detector-conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None, help="full-pose YOLO imgsz")
    parser.add_argument("--detector-imgsz", type=int, default=None)
    parser.add_argument("--pose-imgsz", type=int, default=None)
    parser.add_argument("--detector-fps", type=float, default=None)
    parser.add_argument("--pose-fps", type=float, default=None)
    parser.add_argument("--roi-margin", type=float, default=None)
    parser.add_argument("--hybrid-version", choices=["v1", "v2"], default=None)
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--latest-frame", action="store_true", help="force latest-frame mode")
    parser.add_argument("--sequential", action="store_true", help="force sequential (offline) mode")
    parser.add_argument(
        "--realtime",
        action="store_true",
        dest="latest_frame",
        help="alias for --latest-frame",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        dest="sequential",
        help="alias for --sequential",
    )
    args = parser.parse_args(argv)

    source = parse_video_source(args.source)
    hybrid_version = args.hybrid_version or default_hybrid
    ml, ml_path, feat_ver, ml_err = load_ml_status()

    print(f"source={redact_source(source)}")
    print(f"pipeline={args.pipeline}")

    if args.pipeline == "roi-pose":
        from helmet_action.inference.pipeline import run_roi_video

        dest, stats = run_roi_video(
            source,
            human_model=args.human_model,
            pose_model=args.pose_model,
            detector_conf=args.detector_conf if args.detector_conf is not None else default_det_conf,
            pose_conf=args.pose_conf if args.pose_conf is not None else default_pose_conf_roi,
            detector_imgsz=args.detector_imgsz if args.detector_imgsz is not None else default_det_imgsz,
            pose_imgsz=args.pose_imgsz if args.pose_imgsz is not None else default_pose_imgsz,
            detector_fps=args.detector_fps if args.detector_fps is not None else default_det_fps,
            pose_fps=args.pose_fps if args.pose_fps is not None else default_pose_fps,
            roi_margin=args.roi_margin if args.roi_margin is not None else default_roi_margin,
            hybrid_version=hybrid_version,
            rotate=int(args.rotate),
            latest_frame=True if args.latest_frame else None,
            sequential=True if args.sequential else None,
            out_path=None if args.no_record else args.out,
            show=args.show,
            max_frames=args.max_frames,
            no_record=bool(args.no_record),
            debug_overlay=bool(args.debug_overlay),
            ml=ml,
            print_banner=True,
        )
        if dest is not None:
            print(f"wrote {dest}")
        elif args.no_record:
            print("recording disabled (--no-record)")
        print(
            f"stats: cam={stats.camera_fps:.1f} render={stats.render_fps:.1f} "
            f"det={stats.detector_fps:.1f} pose={stats.pose_fps:.1f} act={stats.action_fps:.1f} "
            f"dropped={stats.dropped_frames} e2e_ms={stats.mean_e2e_ms}"
        )
    else:
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
            imgsz=args.imgsz if args.imgsz is not None else args.pose_imgsz,
        )
        dest = run_video(
            source,
            provider,
            out_path=None if args.no_record else args.out,
            show=args.show,
            max_frames=args.max_frames,
            rotate=int(args.rotate),
            latest_frame=bool(args.latest_frame) and not bool(args.sequential),
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
