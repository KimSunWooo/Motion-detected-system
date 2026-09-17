# v0.5 Inference Architecture Report

## 1. 기존 architecture

```
Full Frame → YOLO Pose (+ track) → TrackPoseBuffer → Hybrid Action → Rendering
```

고해상도 IP Webcam에서 full-frame pose를 매 프레임 돌리면 backlog가 누적되고,
배경에 false skeleton이 붙을 수 있으며, pose jitter로 action 평가가 무의미해짐.

## 2. 변경 architecture

```
Camera → LatestFrameCapture
      → Downscaled HumanDetector (cadenced)
      → PersonTracker (stable gate)
      → ROI crop (+ margin)
      → PoseEstimator (ROI only)
      → PoseQualityGate
      → TrackPoseBuffer
      → Existing Hybrid / Sklearn / Rule
```

CLI: `--pipeline roi-pose` (default) vs `--pipeline full-pose` (baseline).

## 3. 변경 파일 목록

| Path | Role |
|---|---|
| `config/default.yaml` | `inference:` block (detector/tracker/pose/action cadence) |
| `src/helmet_action/inference/human_detector.py` | HumanDetection + downscale/rescale |
| `src/helmet_action/inference/ultralytics_human_detector.py` | yolo11n person detector |
| `src/helmet_action/inference/tracker.py` | HumanTrack + IoUPersonTracker |
| `src/helmet_action/inference/pose_estimator.py` | ROI helpers + PoseEstimator protocol |
| `src/helmet_action/inference/ultralytics_roi_pose.py` | ROI YOLO-Pose → global coords |
| `src/helmet_action/inference/pipeline.py` | RoiInferencePipeline + run_roi_video |
| `src/helmet_action/inference/diagnostics_v5.py` | PipelineDecisionStatus + StageTimer |
| `src/helmet_action/inference/capture.py` | LatestFrameCapture, EOF mark_ended |
| `scripts/run_video.py` | pipeline / detector / pose CLI |
| `scripts/benchmark_realtime_pipeline.py` | full vs ROI benchmark |
| `tests/test_roi_pipeline.py` | §31 gate / cadence / invariant tests |
| `README.md` | v0.5 architecture docs |

## 4. 유지한 기존 component

Synthetic generator, PoseObservation, TrackPoseBuffer, normalization / confidence repair,
PoseQualityScore, Feature V1/V2, SklearnActionClassifier, RuleBasedActionClassifier,
HybridActionClassifier (V1/V2), ActionPhaseMachine, HelmetStateMachine, safety gates,
real pose dataset pipeline, evaluation metrics, HGB baseline weights,
`UltralyticsPoseProvider` full-pose baseline, TemporalActionModel Protocol.

## 5. 제거하거나 deprecated한 component

없음. Full-frame pose는 `--pipeline full-pose` baseline으로 보존.

## 6–13. Benchmark (CPU, 1920×1080 controlled sample)

환경: CPU ultralytics, sample image with people pasted onto 1080p canvas.

| Metric | Full Pose | ROI Pose (det@5, pose@10) |
|---|---|---|
| Mean pose stage | 22.5 ms (full frame) | 10.0 ms (ROI) |
| P95 pose | 23.9 ms | 10.4 ms |
| Mean detector | n/a | 22.0 ms |
| Est. compute ms / camera-sec @30 FPS | ~674 | ~210 (3.2× less) |
| Pipeline throughput (40f) | — | ~70 FPS wall |
| Detector / Pose / Action calls (40f @30src) | pose every frame | det=7, pose=42, action=0* |

\* action gated until buffer warmup (`min_frames`).

Synthetic blob video (no real person): ROI correctly invoked **0 pose calls**
(No HumanTrack → No Pose), capture throughput ~106 FPS vs full-pose ~36 FPS.

## 14. Dropped frame count

Sequential offline bench: 0 drops. Latest-frame LIVE mode overwrites unread frames
(`LatestFrameBuffer` depth ≤ 1); backlog cannot grow unbounded.

## 15. Tracking stability

IoU tracker with `min_stable_frames=5`, `max_missing_frames=10`.
Unstable tracks emit `TRACK_WARMUP` and never call action.
Unit tests cover track ID continuity and buffer clear on loss.

## 16. Person false negative

Synthetic non-person blobs are not detected (expected). Real person FN depends on
`detector.confidence` / `imgsz` / cadence — measure on IP Webcam protocol (§35).

## 17. Background false pose

Invariant enforced + tested: **no HumanTrack → pose call count = 0**.
Pose runs only on tracker-approved ROI.

## 18. Test 결과

```
PYTHONPATH=src python -m pytest tests -q
→ 100 passed (incl. 20 new ROI tests), prior suite intact
```

## 19. 현재 bottleneck

On CPU 1080p with people present: **detector (~22 ms)** and **ROI pose (~10 ms)**
dominate when both fire. Cadence (det 5 FPS, pose 10 FPS) keeps camera-second
budget ~3× below full-frame-every-frame. Next levers: lower detector imgsz,
skip pose on unstable tracks if needed, GPU if available. Do not retune Hybrid.

## 20. Temporal Model 적용 준비 여부

**Not yet.** Hooks (`TemporalActionModel`, TrackPoseBuffer → `[T,17,3]`) remain.
Apply LSTM/TCN only after: stable detection/tracking, acceptable wrist/shoulder
confidence, no camera backlog, validated real-video pipeline.
