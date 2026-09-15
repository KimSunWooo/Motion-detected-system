# 하이 앵글 CCTV · 안전모 탈착 / 머리 긁기 분류

공장 상단 CCTV처럼 작업자를 **대각선으로 내려다보는** 구도에서 YOLOv8-Pose 17 키포인트로 **단순 머리 긁기(정상)** 와 **안전모 벗기(예방 알림)** 를 구분합니다.

핵심 스크립트 `pose_action_classifier.py` 는 **numpy + matplotlib** 만으로 동작합니다. Flask 대시보드는 선택입니다.

## 실행

```bash
python3 -m pip install -r requirements.txt
python3 pose_action_classifier.py --out outputs
python3 pose_action_classifier.py --serve --port 8765   # 선택
```

## 하이브리드 판정

정규화 (하체 미사용):

```
neck = (L_shoulder + R_shoulder) / 2
p̂    = (p − neck) / ||L_shoulder − R_shoulder||
```

머리 bbox 는 목 원점과 어깨너비에 비례합니다. `center = (nose_x, −0.30)`, 반폭 `(0.52, 0.42)`.

| 동작 | 조건 |
| --- | --- |
| 긁기 | 한쪽 손목만 bbox 진입 → 좁은 반경에서 고주파 진동 |
| 안전모 벗기 | 양손목이 귀 부근에서 일시 정지한 뒤, 아래 **OR** |
| | (1) 양손목 X 간격 증가 |
| | (2) 두 손목이 함께 y 감소 (화면 위쪽) |
| | (3) 머리 기준 방사형 팽창 / 귀 간격 scale-up |

## 의존성

- `numpy`, `matplotlib` — 필수
- `flask` — `--serve` 대시보드만
