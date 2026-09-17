# Motion-detected-system

산업 현장 CCTV 영상에서 작업자의 **안전모 탈착 행동을 Pose 기반으로 분석**하기 위한 실시간 행동 인식 프로젝트입니다.

이 프로젝트의 핵심 목표는 단순히 한 프레임에서 안전모의 존재 여부를 판별하는 것이 아니라,

> 작업자가 안전모를 벗으려고 하는 과정과 동작의 흐름을 시간축으로 분석하는 것

입니다.

이를 위해 사람을 먼저 탐지하고, 탐지된 사람의 Skeleton을 추출한 뒤, 일정 시간 동안의 관절 움직임을 누적하여 행동을 판단하는 구조로 설계하고 있습니다.

---

## 프로젝트 기획 배경

산업 현장에서 안전모 착용 여부는 중요한 안전 관리 요소입니다.

일반적인 Object Detection 방식은 특정 시점에서:

```text
Helmet
No Helmet
```

을 판단하는 데 적합하지만,

```text
손을 머리로 이동
→ 안전모를 잡음
→ 들어 올림
→ 안전모 제거
```

처럼 **행동이 진행되는 과정 자체**를 판단하기에는 한계가 있습니다.

본 프로젝트는 이러한 문제를 다음 두 영역으로 분리했습니다.

### 1. Action Recognition

작업자의 Skeleton 움직임을 시간축으로 분석하여 다음 행동을 구분합니다.

```text
IDLE
HEAD_TOUCH
HEAD_SCRATCH
HELMET_ADJUST
HELMET_REMOVE
HELMET_PUT_ON
```

### 2. Helmet State Detection

향후 별도의 Helmet Detection 모델을 통해:

```text
WORN
NOT_WORN
UNKNOWN
```

상태를 판단합니다.

즉 최종 구조는:

```text
행동 분석
+
안전모 객체 상태 분석
```

을 결합하는 것을 목표로 합니다.

---

# 전체 시스템 기획

프로젝트는 다음 흐름을 기준으로 설계했습니다.

```text
Camera / CCTV
        ↓
Human Detection
        ↓
Person Tracking
        ↓
Human ROI
        ↓
Pose Estimation
        ↓
Skeleton Sequence
        ↓
Temporal Action Analysis
        ↓
Safety Decision
```

중요한 설계 원칙은 세 가지입니다.

### 1. Human을 먼저 찾는다

전체 화면에 Pose 모델을 실행하는 대신,

```text
Frame
↓
Human Detection
↓
Human ROI
```

순서로 사람의 위치를 먼저 찾습니다.

Pose Estimator는 Human으로 확인된 영역에 대해서만 동작하도록 설계합니다.

이를 통해:

* 불필요한 Pose 연산 감소
* 배경 영역에 잘못된 Skeleton 생성 방지
* 고해상도 CCTV 환경 대응

을 목표로 합니다.

---

### 2. Skeleton을 사람 단위로 관리한다

탐지된 사람마다 고유한 Track ID를 유지합니다.

```text
Human #1
→ Pose Sequence #1

Human #2
→ Pose Sequence #2
```

각 사람의 Skeleton은 일정 시간 동안 별도의 Buffer에 누적됩니다.

현재 `TrackPoseBuffer`는 약 1~2초 구간의 Pose Sequence를 관리하며,

```text
Frame 1 Skeleton
Frame 2 Skeleton
Frame 3 Skeleton
...
Frame N Skeleton
```

형태의 시간축 데이터를 Action Model에 전달합니다.

---

### 3. 행동은 한 프레임이 아니라 시간의 흐름으로 판단한다

예를 들어 안전모 제거 행동은 단순히:

```text
손이 머리 근처에 있음
```

으로 판단하지 않습니다.

대신 다음과 같은 흐름을 봅니다.

```text
IDLE
  ↓
HAND_APPROACH
  ↓
HELMET_GRASP
  ↓
LIFT / SEPARATION
  ↓
REMOVAL_CONFIRMED
```

이를 통해:

```text
머리 긁기
머리 만지기
안전모 조정
안전모 제거
```

처럼 서로 비슷한 동작을 구분하는 것을 목표로 합니다.

---

# Synthetic Skeleton 기반 학습

실제 산업 CCTV 데이터를 대량으로 확보하기 전 단계에서 먼저 수학적으로 Skeleton Sequence를 생성했습니다.

Synthetic Generator는 다음 조건을 변화시킵니다.

```text
신체 비율
팔 길이
어깨 너비
카메라 높이
카메라 각도
거리
동작 속도
손의 이동 방향
노이즈
관절 누락
가림
프레임 드롭
```

그리고 다음과 같은 행동을 생성합니다.

```text
IDLE
HEAD_SCRATCH
HEAD_TOUCH
HELMET_ADJUST
HELMET_REMOVE
HELMET_PUT_ON
RAISE_ARMS
FACE_TOUCH
WIPE_SWEAT
UNKNOWN_RANDOM_MOTION
...
```

이 데이터는 실제 RGB 영상이 아니라:

```text
[T, 17, 2]
```

형태의 시간축 Skeleton 데이터입니다.

---

# 현재 Action Model

현재 모델은 Skeleton Sequence에서 움직임의 특징을 수학적으로 추출합니다.

예:

```text
손목 ↔ 머리 거리
손목 ↔ 귀 거리
양 손목 간 거리
팔꿈치 각도
손의 이동 속도
손의 가속도
이동 방향
정지 시간
머리 주변 체류 시간
양손 동시 움직임
```

이를 기반으로:

```text
Rule-based Model
+
HistGradientBoosting
+
Action Phase Machine
```

결과를 결합합니다.

현재 구조는 향후 Temporal Model과 비교하기 위한 baseline 역할도 합니다.

향후에는 동일한 Skeleton Sequence를 그대로:

```text
[T, 17, 3]
```

형태로 LSTM / TCN 등에 입력하여 시간 흐름 자체를 학습시키는 방향을 계획하고 있습니다.

---

# v0.5 Realtime Inference Pipeline

실제 카메라 테스트를 진행하면서 Full-frame Pose 방식보다 효율적인 실시간 구조가 필요하다고 판단했습니다.

따라서 v0.5부터 다음 구조를 사용합니다.

```text
High Resolution Camera
        ↓
LatestFrameCapture
        ↓
Downscaled Frame
        ↓
Human Detector
        ↓
Person Tracker
        ↓
Human ROI
        ↓
ROI Pose Estimator
        ↓
Pose Quality
        ↓
TrackPoseBuffer
        ↓
Action Model
```

## 왜 ROI Pose 방식을 선택했는가?

초기 구조에서는:

```text
Full Frame
→ YOLO Pose
```

를 사용했습니다.

하지만 실제 CCTV 시스템에서는 영상 전체보다 **사람이 존재하는 영역만 분석하는 것이 더 효율적**이라고 판단했습니다.

따라서 현재는:

```text
전체 화면
↓
저해상도 Human Detection
↓
Human Bounding Box
↓
원본 영상에서 Human ROI Crop
↓
Pose Estimation
```

구조를 사용합니다.

이를 통해 고해상도 입력에서도 분석 비용을 줄이는 것을 목표로 합니다.

---

# 실시간 처리 전략

이 프로젝트는 모든 Camera Frame을 반드시 분석하는 방식으로 설계하지 않습니다.

예를 들어 Camera가 30 FPS이고 AI 분석 속도가 10 FPS라면:

```text
Camera
1 2 3 4 5 6 7 8 9 ...

AI
1       4       7 ...
```

형태로 필요한 최신 Frame을 선택하여 처리합니다.

오래된 Frame을 Queue에 계속 쌓지 않고:

```text
Latest Frame
```

을 우선 처리합니다.

따라서 목표는:

```text
모든 Frame 분석
```

이 아니라:

```text
현재 장면을 낮은 Latency로 지속 분석
```

입니다.

Detector, Pose, Action Model 역시 서로 다른 속도로 실행할 수 있습니다.

예:

```text
Camera       30 FPS
Tracker      30 FPS
Detector      5 FPS
Pose         10 FPS
Action        5~10 FPS
```

---

# 현재 개발 단계

| Stage | 내용                              | 상태          |
| ----- | ------------------------------- | ----------- |
| 1     | Rule-based Pose Prototype       | DONE        |
| 2     | Synthetic Skeleton Generator    | DONE        |
| 3     | Feature 기반 ML Model             | DONE        |
| 4     | Hybrid Action Decision          | DONE        |
| 5     | Synthetic OOD / Stress Test     | DONE        |
| 6     | Real Pose Dataset Pipeline      | DONE        |
| 7     | Realtime ROI Inference Pipeline | DONE        |
| 8     | Live Camera Validation          | IN PROGRESS |
| 9     | Real-world Pose Dataset 확대      | PLANNED     |
| 10    | LSTM / TCN Temporal Model       | PLANNED     |
| 11    | Helmet Presence Detector        | PLANNED     |

현재는 **실제 카메라에서 Human → Skeleton Sequence를 안정적으로 확보하는 단계**입니다.

---

# Live Camera 테스트

실시간 카메라 테스트는 `live_cam.py`를 사용합니다.

먼저 환경을 준비합니다.

```bash
source opencv/bin/activate
export PYTHONPATH=src
```

필요한 라이브러리 확인:

```bash
python3 -c "import cv2; import ultralytics; print('environment OK')"
```

---

## 노트북 Webcam

```bash
PYTHONPATH=src python3 live_cam.py \
  --source 0
```

ROI Pipeline:

```bash
PYTHONPATH=src python3 live_cam.py \
  --source 0 \
  --pipeline roi-pose \
  --human-model yolo11n.pt \
  --pose-model yolo11n-pose.pt \
  --detector-fps 5 \
  --pose-fps 10 \
  --roi-margin 0.15 \
  --latest-frame \
  --debug-overlay
```

---

## Android IP Webcam

스마트폰과 PC를 동일한 네트워크에 연결합니다.

예:

```text
http://192.168.0.213:8080/
```

영상 Stream:

```text
http://192.168.0.213:8080/video
```

실행:

```bash
PYTHONPATH=src python3 live_cam.py \
  --source "http://192.168.0.213:8080/video" \
  --pipeline roi-pose \
  --human-model yolo11n.pt \
  --pose-model yolo11n-pose.pt \
  --detector-fps 5 \
  --pose-fps 10 \
  --roi-margin 0.15 \
  --latest-frame \
  --debug-overlay
```

영상 방향이 돌아가 있다면:

```bash
--rotate 90
```

또는:

```bash
--rotate 270
```

을 사용할 수 있습니다.

---

# 테스트 영상 저장

실시간 분석 결과를 저장할 수 있습니다.

```bash
PYTHONPATH=src python3 live_cam.py \
  --source "http://192.168.0.213:8080/video" \
  --pipeline roi-pose \
  --human-model yolo11n.pt \
  --pose-model yolo11n-pose.pt \
  --detector-fps 5 \
  --pose-fps 10 \
  --roi-margin 0.15 \
  --latest-frame \
  --debug-overlay \
  --out outputs/live_cam_test.mp4
```

저장 결과:

```text
outputs/live_cam_test.mp4
```

---

# Live Camera 검증 시나리오

초기 실시간 테스트에서는 Action 정확도보다 먼저 다음 Pipeline을 확인합니다.

```text
Camera
↓
Human Detection
↓
Tracking
↓
Skeleton
```

테스트 동작:

```text
정지
→ 좌우 이동
→ 카메라 접근
→ 카메라에서 멀어짐
→ 팔 들어올리기
→ 머리에 손 대기
→ 방향 전환
```

확인 항목:

```text
영상이 실시간으로 자연스럽게 유지되는가?

Human Bounding Box가 사람을 따라가는가?

동일한 사람의 Track ID가 유지되는가?

Human 영역에 Skeleton이 정상적으로 생성되는가?

사람이 움직여도 Skeleton이 따라가는가?
```

이 과정이 안정화된 이후 다음 행동을 검증합니다.

```text
HEAD_TOUCH
HEAD_SCRATCH
HELMET_ADJUST
HELMET_REMOVE
```

---

# 프로젝트가 지향하는 최종 구조

```text
Industrial CCTV
        ↓
Human Detection
        ↓
Person Tracking
        ↓
Human ROI Pose
        ↓
Skeleton Temporal Sequence
        ↓
Temporal Action Model
        ↓
HELMET_REMOVE / ADJUST / SCRATCH / TOUCH
        │
        └──────────────┐
                       │
Helmet Detector        │
WORN / NOT_WORN        │
        │              │
        └───────┬──────┘
                ↓
          Safety Decision
                ↓
        Alert / Monitoring
```

최종적으로는 **객체 인식과 Skeleton 기반 행동 분석을 결합한 산업 안전 모니터링 시스템**을 목표로 합니다.

현재 단계에서는 안전모 자체의 Object Detection보다 먼저,

> 사람을 안정적으로 찾고 → Skeleton을 확보하고 → Skeleton의 시간적 움직임을 분석할 수 있는 기반

을 만드는 데 집중하고 있습니다.
