# Motion-detected-system

> **경고 / Warning**
>
> - **Synthetic evaluation results are NOT real-world CCTV performance.**
> - **Pose alone cannot determine static helmet presence.**
> - **Helmet State remains UNKNOWN without a helmet-presence detector.**

공사장·제조·산업 현장 CCTV에서 **작업자가 안전모를 벗으려 하는지** 를 자세(pose) 시계열로 판단하는 시스템입니다.

이 저장소는 실제 현장 영상을 대량으로 확보하기 전 단계의 **Pose Action Model** 입니다.
인체 관절의 기하학, 궤적, 속도/가속도, 양손 협응, 동작 phase, Synthetic Pose Sequence를 사용합니다.

## 프로젝트 목표

최종적으로 알고 싶은 것은 두 가지입니다.

1. 작업자가 안전모를 벗으려고 하는가? (**ACTION MODEL**)
2. 작업자가 지금 안전모를 착용하지 않은 상태인가? (**HELMET STATE MODEL**)

두 문제는 분리합니다. Pose 좌표만으로는

* 안전모를 쓰고 가만히 있는 사람
* 안전모를 쓰지 않고 가만히 있는 사람

을 구분할 수 없습니다. 현재 단계에 안전모 Object Detection weight가 없으므로
Helmet State는 기본적으로 **UNKNOWN** 입니다. 가짜 착용 검출을 만들지 않습니다.

탈착 행동이 높은 신뢰도로 완료되고, 이전 상태가 WORN 이었다면 State Machine이 NOT_WORN으로 갱신할 수 있습니다.
기본 초기 상태는 UNKNOWN이라 Pose만으로 착용 여부를 확정하지 않습니다.

## 현재 지원 범위

* COCO 17 keypoint, 목 = 어깨 중점, 어깨너비 정규화
* 동적 머리 bbox / 귀 영역 / wrist-head 거리
* Rule-based baseline (`scratch` vs `helmet_off` vs `no_contact`)
* 하이 앵글 핀홀 카메라 + canonical 3D pose + synthetic scratch / helmet-off / idle
* Domain-randomized synthetic dataset (14 시나리오)
* Engineered temporal feature + HistGradientBoosting 분류기
* Rule + ML + Temporal State Machine 하이브리드 판정
* Track ID별 pose buffer, keypoint 신뢰도 / 결측 처리
* Flask 대시보드 (기존 호환)
* Ultralytics Pose 영상 입력 (optional)
* DummyHelmetPresenceDetector → 항상 UNKNOWN

지원하지 않는 것:

* 실제 현장 정확도 주장
* Pose만으로 WORN / NOT_WORN 확정
* 한 프레임만 보고 탈착 확정

## 전체 Architecture

```
Video / Webcam / RTSP / Synthetic
        │
        ▼
 PoseProvider  (Ultralytics optional, NumpySequence for demo)
        │  PoseObservation(timestamp, track_id, bbox, keypoints, conf)
        ▼
 TrackPoseBuffer  (track_id별 1.5~2.5초 sliding window, FPS 정규화)
        │
        ▼
 Keypoint repair  VALID / LOW_CONFIDENCE / MISSING
        │  부족하면 INSUFFICIENT_POSE / UNKNOWN  (정상도 위험도 확정하지 않음)
        ▼
 Feature extractor  (neck origin + shoulder-width 정규화 공간)
        │
        ├─► RuleBasedActionClassifier     baseline
        ├─► SklearnActionClassifier       lightweight ML
        └─► ActionPhaseMachine            IDLE→APPROACH→GRASP→LIFT→CONFIRMED
        │
        ▼
 Hybrid decision + hysteresis
        │
        ├─► Action event (REMOVE_INTENT / REMOVE_CONFIRMED / …)
        └─► HelmetStateMachine  +  HelmetPresenceDetector (지금은 Dummy=UNKNOWN)
```

향후 LSTM / TCN / ST-GCN / Transformer 는 `TemporalActionModel` 구현체만 교체하면 됩니다.
Rule baseline과 feature, state machine은 유지합니다.

## Pose 기반 탈착 행동 판단 원리

정규화 (하체 미사용 — 하이 앵글에서 몸통이 심하게 단축됨):

```
neck = (L_shoulder + R_shoulder) / 2
p̂    = (p − neck) / ||L_shoulder − R_shoulder||
```

머리 bbox는 목 원점과 어깨너비에 비례합니다. `center = (nose_x, −0.30)`, 반폭 `(0.52, 0.42)`.

안전모 벗기는 한 threshold가 아니라 phase로 봅니다.

```
IDLE → HAND_APPROACH → HELMET_GRASP → LIFT_OR_SEPARATE → REMOVAL_CONFIRMED
```

* APPROACH: 손목이 머리/귀로 접근, 거리 derivative < 0, 팔꿈치 각도 변화
* GRASP: 양손이 좌/우 귀 부근에서 잠시 정지
* LIFT: 양손 separation 증가 **또는** 동반 상승 **또는** 방사형 팽창 / 겉보기 머리 크기 증가
* CONFIRMED: 위 phase가 시간 순서로 연속 발생

손이 머리 근처에 있다는 이유만으로 탈착으로 보지 않습니다.
긁기, 양손 머리 접촉, 안전모 고쳐 쓰기는 별도 negative class 입니다.

## 왜 Pose만으로 안전모 미착용을 직접 판단할 수 없는가

정지한 사람의 관절 좌표에는 “안전모 객체”가 없습니다.
쓴 사람과 안 쓴 사람의 idle pose는 동일할 수 있습니다.
그래서 Helmet State는 별도 `HelmetPresenceDetector.detect(frame, person_bbox)` 인터페이스를 기다립니다.
weight가 없는 현재 Dummy 구현은 항상 UNKNOWN을 반환합니다.

## 실행

```bash
python3 -m pip install -r requirements.txt
export PYTHONPATH=src

# 기존 엔트리 (Rule baseline self-test + 그림)
python3 pose_action_classifier.py --out outputs

# 대시보드
python3 pose_action_classifier.py --serve --port 8765
# 또는
python3 dashboard_server.py
```

## Synthetic Data 생성

```bash
python scripts/generate_dataset.py --samples 5000 --output data/synthetic
python scripts/generate_dataset.py --samples 500 --quick
```

저장 형식:

```
data/synthetic/train.npz
data/synthetic/validation.npz
data/synthetic/test.npz
data/synthetic/metadata.json
```

npz 키: `keypoints (N,T,17,2)`, `confidences (N,T,17)`, `labels`, `scenarios`, `lengths`, `seeds`.

metadata에는 random seed, generator version, scenario, camera / body / action / noise parameter가 들어갑니다.

Train/Val은 in-distribution 카메라·신체·노이즈 범위입니다.
**Test는 일부러 OOD** 입니다: 다른 pitch/distance, 더 높은 keypoint noise, 다른 action speed, 다른 body proportion.
같은 generator parameter를 shuffle만 해서 나누지 않습니다.
Synthetic 99% 정확도는 현장 성능이 아닙니다.

시나리오:

`IDLE`, `HEAD_SCRATCH`, `ONE_HAND_HEAD_TOUCH`, `TWO_HAND_HEAD_TOUCH`, `FACE_TOUCH`,
`HELMET_ADJUST`, `WIPE_SWEAT`, `LOOK_DOWN`, `RAISE_ARMS`, `STRETCH`,
`PHONE_NEAR_HEAD`, `HELMET_REMOVE`, `HELMET_PUT_ON`, `UNKNOWN_RANDOM_MOTION`

사람(키, 어깨너비, 팔 길이, 손잡이), 카메라(높이, 거리, pitch, yaw, focal),
행동(속도, pause, lift, 진폭), Pose noise(가우시안, dropout, occlusion, jitter, frame drop, FPS/시간축 왜곡)
를 sequence마다 샘플링합니다.

## 수학적 Feature

위치 (정규화 공간): wrist→head, wrist→ear, wrist separation, wrist height relative to shoulder,
elbow, head scale, shoulder orientation.

각도: left/right elbow, upper-arm / forearm, wrist-head direction.

속도 `v(t)=p(t)-p(t-1)`, 가속도 `a(t)=v(t)-v(t-1)`.

추가: radial velocity toward/away from head, vertical wrist velocity, wrist separation velocity,
bilateral symmetry, left/right wrist velocity cosine similarity, pause duration,
head/ear dwell, approach/lift duration, motion energy, trajectory variance, directional reversal.

Rule baseline 통계량도 그대로 사용합니다: scratch radius/std/oscillation, pause, wrist spread,
co-rise, radial expansion, apparent head scale.

## Rule-Based Baseline

`RuleBasedActionClassifier` = 기존 `classify_pose_sequence()`.

* 긁기: 한쪽 손목만 bbox, 좁은 반경, 고주파 진동, 양귀 파지 아님
* 안전모 벗기: 양손목 귀 부근 정지 후 (1) X 간격 (2) 동반 y 감소 (3) 방사형/scale-up 중 하나
* 비접촉 / 판정 보류

이 규칙은 삭제하지 않습니다. ML이 높더라도 관절이 없거나 phase 순서가 불가능하면 UNKNOWN으로 내립니다.

## ML Model

Sequence → feature vector → `HistGradientBoostingClassifier` (기본).

```
P(IDLE), P(HEAD_SCRATCH), P(HEAD_TOUCH), P(HELMET_ADJUST),
P(HELMET_REMOVE), P(HELMET_PUT_ON), P(UNKNOWN)
```

`predict()` 와 `predict_proba()` 를 모두 제공합니다.
모델 파일: `models/action_classifier.joblib`

## State Machine

HelmetState: `UNKNOWN | WORN | NOT_WORN`

ActionEvent: `NONE | REMOVE_INTENT | REMOVE_CONFIRMED | PUT_ON_INTENT | PUT_ON_CONFIRMED`

* WORN + REMOVE_CONFIRMED → NOT_WORN
* NOT_WORN + PUT_ON_CONFIRMED → WORN
* UNKNOWN에서는 Pose action만으로 착용 여부를 바꾸지 않음

Hysteresis: `risk_enter=0.75`, `risk_exit=0.45`, 최근 N window 투표.

## 학습 / 평가

```bash
export PYTHONPATH=src
python scripts/generate_dataset.py --samples 500 --output data/synthetic --seed 42
python scripts/train_action_model.py --data data/synthetic --output models/action_classifier.joblib
python scripts/evaluate_action_model.py --data data/synthetic --model models/action_classifier.joblib
python scripts/evaluate_action_model.py --data data/synthetic/test.npz --model models/action_classifier.joblib
```

`--calibrate` 는 validation split으로 sigmoid calibration을 붙이는 옵션입니다.
`predict_proba` 는 현장 confidence가 아닙니다. Synthetic 확률 분포일 뿐입니다.

평가 출력: Accuracy, Macro Precision/Recall/F1, per-class Precision/Recall/F1, Confusion Matrix,
HELMET_REMOVE binary TP/FP/TN/FN, FNR, FPR, PR-AUC, hard-negative → REMOVE 수.

숫자는 synthetic OOD test에 대한 것입니다. **현장 정확도로 인용하지 마세요.**

## 실제 영상 실행

```bash
pip install ultralytics opencv-python-headless
export POSE_MODEL_PATH=yolo11n-pose.pt
python scripts/smoke_ultralytics.py
python scripts/run_video.py --source sample.mp4 --pose-model "$POSE_MODEL_PATH"
python scripts/run_video.py --source 0 --pose-model "$POSE_MODEL_PATH"
python scripts/run_video.py --source rtsp://192.168.0.10:554/stream --pose-model "$POSE_MODEL_PATH"
```

모델 이름은 코드에 고정하지 않습니다. `--pose-model` 또는 `POSE_MODEL_PATH` 또는 `config/default.yaml` 의 `pose.model_path` 를 사용합니다.
정수 source는 웹캠, 존재하는 파일 경로는 파일이 우선입니다. RTSP userinfo는 로그에 출력하지 않습니다.
Ultralytics가 없으면 synthetic demo / dashboard는 그대로 동작합니다.

오버레이: Track ID, Action, Action Probability, Helmet State, Removal Risk, skeleton.
Helmet detector가 없으면 Helmet State는 **UNKNOWN** 이라고 표시됩니다.

## 테스트

```bash
PYTHONPATH=src python -m pytest tests -q
```

정규화 후 어깨너비 ≈ 1, translation/scale 불변, scratch가 HELMET_REMOVE로 가지 않음,
helmet-off의 removal probability, idle 오탐 없음, keypoint missing 시 crash 없음,
confidence 부족 시 UNKNOWN, 동일 seed 재현, 모델 save/load 후 예측 동일,
track buffer 격리, Helmet UNKNOWN 유지.

## Artifact policy

- `data/synthetic/*.npz` 와 `metadata.json` 은 Git에 올리지 않습니다. seed로 재생성하세요.
- `models/action_classifier.joblib` 는 작은 sklearn 모델만 demo로 포함합니다. YOLO weight(`.pt`)나 대용량 네트워크는 Git에 올리지 마세요.
- `outputs/`, `.venv*`, `__pycache__`, `.pytest_cache` 는 ignore 됩니다.

## 설정

임계값은 `config/default.yaml` 에 있습니다. pose confidence, window seconds, head geometry,
scratch/helmet rules, alert/clear, synthetic camera/noise.

## 패키지 구조

```
src/helmet_action/
  pose/          constants, geometry, normalizer, pose_buffer, confidence
  features/      baseline extractor, temporal features
  synthetic/     camera, skeleton, generator, scenarios, augmentation
  models/        rule_based, temporal_classifier, hybrid, training
  state/         action_state_machine, helmet_state
  inference/     pose_provider, ultralytics, video_pipeline
  visualization/ plots, dashboard payloads
scripts/         generate_dataset, train, evaluate, run_video, smoke_ultralytics
tests/
config/default.yaml
pose_action_classifier.py   # 기존 CLI 호환 엔트리
dashboard_server.py
```

## 현재 한계

* 실제 공사장 CCTV 데이터가 없음. 학습/평가는 synthetic.
* 안전모 객체 검출기가 없음. 착용 상태는 UNKNOWN.
* 2D pose는 가림, 측면, 저해상도에서 손목/귀가 자주 사라짐.
* High-angle 투영을 수학적으로 흉내 내지만 실제 렌즈/왜곡/압축 노이즈와는 다름.
* ML은 가벼운 tabular classifier. Sequence 딥러닝은 인터페이스만 준비됨.

## 향후 Real Dataset Fine-tuning 계획

1. 현장 클립에 track_id + 시간 구간 단위로 ACTION 라벨 (scratch/adjust/remove/put-on/other)
2. 별도 helmet detector 학습 또는 기존 모델 연결 → `HelmetPresenceDetector`
3. Synthetic pretrain → 실제 pose sequence로 `TemporalActionModel` fine-tune
   (LSTM / TCN / ST-GCN / Transformer 중 데이터량에 맞는 것)
4. 오탐 클립(고쳐 쓰기, 땀 닦기, 통화)을 hard negative로 재학습
5. 알람 hysteresis를 현장 FPS/작업 밀도에 맞게 재조정
6. False Negative Rate(벗기를 놓침)를 현장 기준으로 별도 모니터링

## 의존성

필수: `numpy`, `matplotlib`, `flask`, `pyyaml`, `scikit-learn`, `joblib`

선택: `ultralytics`, `opencv-python-headless` (실제 영상)
