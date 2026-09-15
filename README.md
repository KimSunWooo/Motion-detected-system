# 공장 작업자 안전모 탈착 예방 — 2D 스켈레톤 동작 분류

YOLOv8-Pose / MediaPipe 형식의 **17개 COCO 키포인트** 시계열로, 공장 작업자의 **단순 머리 긁기(정상)** 와 **안전모 벗기 시도(예방 알림)** 를 구분하는 Python 프로토타입입니다.

절대 픽셀 좌표(`y=100` 같은 값)는 사용하지 않습니다. 모든 판단은 **어깨너비(Shoulder Width)를 1.0으로 둔 상대 좌표**에서 이루어집니다.

## 실행

```bash
python3 -m pip install -r requirements.txt
python3 pose_action_classifier.py --out outputs
python3 pose_action_classifier.py --serve --port 8765
```

브라우저에서 `http://127.0.0.1:8765` 로 대시보드를 엽니다. 헤드리스 환경에서는 matplotlib 창 대신 `outputs/` PNG 와 웹 캔버스가 시각화 역할을 합니다.

## 의존성

| 패키지 | 용도 |
| --- | --- |
| `numpy` | 키포인트 정규화, 분산·상관·영점교차 |
| `matplotlib` | 정적 스켈레톤·시퀀스 스트립·궤적 플롯 |
| `flask` | 시뮬레이션 대시보드 (선택) |

## 알고리즘 요약

1. **정규화**  
   `neck = (L_shoulder + R_shoulder) / 2`  
   `scale = ||L_shoulder − R_shoulder||` (어깨가 가려지면 목–골반 길이로 대체)  
   `p̂ = (p − neck) / scale`

2. **동적 머리 Bounding Box** (목 원점, 어깨너비 단위)  
   `x ∈ [nose_x − 0.42, nose_x + 0.42]`, `y ∈ [−1.05, +0.18]`  
   이미지 좌표에서 Y는 아래가 양수이므로, 정수리는 음수 쪽에 있습니다.

3. **긁기 (정상)**  
   손목이 머리 영역에 들어간 뒤 `std(wrist_y)` 가 크고, `std(head_y)` 는 작으며, Y 차분의 영점교차(상하 진동)가 충분하고, 머리 상승량은 작다.

4. **안전모 벗기 (예방 알림)**  
   손목이 머리 영역에서 **일시 정지**(초반 분산 낮음)한 뒤, 손목과 머리 상단이 **동시에 Y 감소(화면 위쪽)** 하고 두 궤적의 피어슨 상관이 높다.

## 파일

- `pose_action_classifier.py` — 정규화, bbox, 분류기, 시뮬레이션, matplotlib 시각화
- `dashboard_server.py` — 시나리오 재생 대시보드
- `templates/index.html` — 캔버스 스켈레톤 플레이어
