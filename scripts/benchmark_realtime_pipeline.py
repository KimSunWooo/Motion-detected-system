#!/usr/bin/env python3
"""Benchmark full-pose vs roi-pose realtime pipelines.

Example:
  python scripts/benchmark_realtime_pipeline.py --source test.mp4 --max-frames 60
  python scripts/benchmark_realtime_pipeline.py --source test.mp4 --compare
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.config import load_config  # noqa: E402
from helmet_action.inference.source import parse_video_source, redact_source  # noqa: E402


def _run_roi(source, args, cfg) -> dict:
    from helmet_action.inference.pipeline import run_roi_video

    t0 = time.perf_counter()
    _, stats = run_roi_video(
        source,
        human_model=args.human_model,
        pose_model=args.pose_model,
        detector_conf=args.detector_conf,
        pose_conf=args.pose_conf,
        detector_imgsz=args.detector_imgsz,
        pose_imgsz=args.pose_imgsz,
        detector_fps=args.detector_fps,
        pose_fps=args.pose_fps,
        roi_margin=args.roi_margin,
        rotate=args.rotate,
        latest_frame=args.latest_frame,
        sequential=args.sequential,
        out_path=None,
        show=False,
        max_frames=args.max_frames,
        no_record=True,
        debug_overlay=False,
        print_banner=False,
    )
    wall = time.perf_counter() - t0
    return {
        "pipeline": "roi-pose",
        "wall_s": wall,
        "resolution": stats.resolution,
        "source_fps": stats.source_fps,
        "capture_fps": stats.camera_fps,
        "render_fps": stats.render_fps,
        "detector_calls_per_sec": stats.detector_fps,
        "pose_calls_per_sec": stats.pose_fps,
        "action_calls_per_sec": stats.action_fps,
        "detector_calls": stats.detector_calls,
        "pose_calls": stats.pose_calls,
        "action_calls": stats.action_calls,
        "mean_detector_ms": stats.mean_detector_ms,
        "p95_detector_ms": stats.p95_detector_ms,
        "mean_pose_ms": stats.mean_pose_ms,
        "p95_pose_ms": stats.p95_pose_ms,
        "mean_action_ms": stats.mean_action_ms,
        "mean_e2e_ms": stats.mean_e2e_ms,
        "dropped_frames": stats.dropped_frames,
        "average_active_tracks": stats.active_tracks,
        "stages": stats.stage_snapshot,
    }


def _run_full(source, args) -> dict:
    from helmet_action.inference.capture import FrameCapture
    from helmet_action.inference.diagnostics import FpsMeter
    from helmet_action.inference.ultralytics_pose_provider import UltralyticsPoseProvider

    provider = UltralyticsPoseProvider(
        model_path=args.pose_model,
        conf=args.pose_conf,
        track=True,
        imgsz=args.pose_imgsz or args.imgsz,
    )
    use_latest = bool(args.latest_frame) and not bool(args.sequential)
    meter = FpsMeter()
    pose_latencies: list[float] = []
    n = 0
    dropped = 0
    resolution = None
    source_fps = 20.0
    t0 = time.perf_counter()
    with FrameCapture(source, rotate=args.rotate, latest_frame=use_latest) as capture:
        capture.start()
        source_fps = capture.source_fps
        while True:
            item = capture.read()
            if item is None:
                break
            frame = item.frame
            resolution = (int(frame.shape[1]), int(frame.shape[0]))
            t1 = time.perf_counter()
            provider.infer_frame(frame, frame_index=item.index, fps=source_fps)
            pose_latencies.append((time.perf_counter() - t1) * 1000.0)
            meter.tick()
            dropped = capture.dropped_frames()
            n += 1
            if args.max_frames is not None and n >= args.max_frames:
                break
    wall = time.perf_counter() - t0
    pose_latencies_sorted = sorted(pose_latencies)
    p95 = pose_latencies_sorted[int(0.95 * (len(pose_latencies_sorted) - 1))] if pose_latencies_sorted else None
    mean = sum(pose_latencies) / len(pose_latencies) if pose_latencies else None
    render_fps = 0.0
    if len(meter._times) >= 2:
        dt = meter._times[-1] - meter._times[0]
        if dt > 1e-6:
            render_fps = (len(meter._times) - 1) / dt
    return {
        "pipeline": "full-pose",
        "wall_s": wall,
        "resolution": resolution,
        "source_fps": source_fps,
        "capture_fps": render_fps,
        "render_fps": render_fps,
        "detector_calls_per_sec": 0.0,
        "pose_calls_per_sec": render_fps,
        "action_calls_per_sec": 0.0,
        "detector_calls": 0,
        "pose_calls": n,
        "action_calls": 0,
        "mean_detector_ms": None,
        "p95_detector_ms": None,
        "mean_pose_ms": mean,
        "p95_pose_ms": p95,
        "mean_action_ms": None,
        "mean_e2e_ms": mean,
        "dropped_frames": dropped,
        "average_active_tracks": None,
        "stages": {},
    }


def _print_report(data: dict) -> None:
    print("=== Benchmark ===")
    print(f"Pipeline:           {data['pipeline']}")
    print(f"Resolution:         {data.get('resolution')}")
    print(f"Source FPS:         {data.get('source_fps')}")
    print(f"Capture FPS:        {data.get('capture_fps')}")
    print(f"Render FPS:         {data.get('render_fps')}")
    print(f"Detector calls/sec: {data.get('detector_calls_per_sec')}")
    print(f"Pose calls/sec:     {data.get('pose_calls_per_sec')}")
    print(f"Action calls/sec:   {data.get('action_calls_per_sec')}")
    print(f"Mean detector ms:   {data.get('mean_detector_ms')}")
    print(f"P95 detector ms:    {data.get('p95_detector_ms')}")
    print(f"Mean pose ms:       {data.get('mean_pose_ms')}")
    print(f"P95 pose ms:        {data.get('p95_pose_ms')}")
    print(f"Mean action ms:     {data.get('mean_action_ms')}")
    print(f"Dropped frames:     {data.get('dropped_frames')}")
    print(f"Avg active tracks:  {data.get('average_active_tracks')}")
    print(f"End-to-end ms:      {data.get('mean_e2e_ms')}")
    print(f"Wall seconds:       {data.get('wall_s')}")
    print()


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--pipeline", choices=["roi-pose", "full-pose"], default="roi-pose")
    parser.add_argument("--compare", action="store_true", help="run both pipelines")
    parser.add_argument("--human-model", default=None)
    parser.add_argument("--pose-model", default=None)
    parser.add_argument("--detector-conf", type=float, default=None)
    parser.add_argument("--pose-conf", type=float, default=None)
    parser.add_argument("--detector-imgsz", type=int, default=None)
    parser.add_argument("--pose-imgsz", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--detector-fps", type=float, default=None)
    parser.add_argument("--pose-fps", type=float, default=None)
    parser.add_argument("--roi-margin", type=float, default=None)
    parser.add_argument("--rotate", type=int, default=0)
    parser.add_argument("--latest-frame", action="store_true")
    parser.add_argument("--sequential", action="store_true", default=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    source = parse_video_source(args.source)
    print(f"source={redact_source(source)}")

    results = []
    pipelines = ["full-pose", "roi-pose"] if args.compare else [args.pipeline]
    for name in pipelines:
        if name == "roi-pose":
            data = _run_roi(source, args, cfg)
        else:
            data = _run_full(source, args)
        results.append(data)
        _print_report(data)

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
