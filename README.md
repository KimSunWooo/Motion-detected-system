# 하이 앵글 CCTV · 안전모 탈착 / 머리 긁기 분류

공장 **천장·벽면 상단 CCTV**처럼 작업자를 대각선으로 내려다보는 구도에서, YOLOv8-Pose 17 키포인트로 **단순 머리 긁기(정상)** 와 **안전모 벗기(예방 알림)** 를 구분하는 프로토타입입니다.

눈높이 카메라처럼 “손목 Y가 올라간다”로 판단하지 않습니다. 하이 앵글에서 손이 머리로 가는 동작은 렌즈 쪽으로 다가가는 **팽창(scale-up)** 과 **양손목이 귀 쪽에서 벌어지는 변화**로 관측됩니다.

## 실행

```bash
python3 -m pip install -r requirements.txt
python3 pose_action_classifier.py --out outputs
python3 pose_action_classifier.py --serve --port 8765
```

`http://127.0.0.1:8765` 에서 하이 앵글 스켈레톤 재생과 판정 근거를 볼 수 있습니다.

## 의존성

- `numpy` — 투영, 정규화, 분산·영점교차
- `matplotlib` — 투시 왜곡 스켈레톤·궤적 플롯
- `flask` — 대시보드 (선택)

## 알고리즘

1. **정규화 (어깨너비만)**  
   `neck = (L_shoulder + R_shoulder) / 2`  
   `p̂ = (p − neck) / ||L_shoulder − R_shoulder||`  
   목~골반 길이는 투시로 심하게 단축되므로 스케일로 쓰지 않습니다.

2. **긁기**  
   한쪽 손목이 두상 **중심원**에 들어간 뒤, 짧은 반경에서 고주파 진동 (`σ_xy` 큼, 영점교차 다수).

3. **안전모 벗기**  
   양손목이 **귀 모서리**로 이동해 일시 정지한 뒤, 손목 간격이 벌어지거나 귀 간격/어깨너비가 커집니다 (헬멧이 천장 카메라 쪽으로 들어 올려짐).

## 파일

- `pose_action_classifier.py` — 핀홀 하이 앵글 투영, 정규화, 분류, mock 시계열
- `dashboard_server.py` / `templates/index.html` — 재생 대시보드
