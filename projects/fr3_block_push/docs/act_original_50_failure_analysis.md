# 기존 50개 데이터 기반 ACT 실패 분석

## 대상

- 데이터셋: `datasets/data_success_50_v3`
- 총 데이터: 50 episodes, 12,316 frames, 20Hz
- 기존 체크포인트: `outputs/train/act_fr3_success_v3/checkpoints/005000/pretrained_model`
- 평가 환경: FR3 MuJoCo block-push, 동일한 초기 block 위치에서 시작

## 데이터셋 자체의 상태

기존 데이터의 demonstration 종료 위치는 대체로 정상 범위였다.

| 항목 | 값 |
|---|---:|
| 시작 goal distance | 약 0.180m |
| 종료 goal distance 최소 | 0.0022m |
| 종료 goal distance 중앙값 | 0.0094m |
| 종료 goal distance 최대 | 0.0176m |
| episode 길이 | 7.35~18.9초 |

따라서 저장된 demonstration이 전부 실패한 것은 아니었다. 실제 조작자는 대부분 block을 goal 안쪽까지 밀고 성공으로 저장했다.

## 반복적으로 나타난 추론 오류

기존 ACT 평가에서는 여러 episode가 거의 동일하게 다음 패턴을 보였다.

```text
시작 block x ≈ 0.620
중간까지 이동
block x ≈ 0.736~0.739에서 정지
goal x = 0.800
최종 goal distance ≈ 0.062m
```

관절 action은 계속 출력되고 있었기 때문에 actuator나 MuJoCo stepping이 멈춘 것은 아니었다. 정책이 목표 앞에서 후반부 push action을 충분히 출력하지 못한 것이 핵심이었다.

## 지속 오류의 원인

### 1. 접근 방향의 다양성 부족

기존 50개는 대부분 정면에서 비슷한 방식으로 접근했다. block이 오른쪽 또는 왼쪽으로 회전하거나 drift했을 때, 어느 면으로 이동해 다시 goal에 넣어야 하는지 보여주는 demonstration이 부족했다.

### 2. 후반부 보정 action 부족

goal에 진입하기 직전에는 단순한 전진이 아니라 다음과 같은 조작이 필요하다.

- block의 회전 방향 확인
- 오른쪽 또는 왼쪽 면으로 재접근
- y 방향 drift 보정
- goal 내부에서 최종 정렬

기존 데이터는 성공 종료 위치는 좋았지만, 이 후반부 보정 trajectory의 종류가 충분하지 않았다.

### 3. goal distance만으로는 접근 의미를 표현하기 어려움

기존 데이터에는 `observation.goal_distance`가 있었지만, 이것은 거리 크기만 나타낸다. 오른쪽 면으로 접근했는지, 왼쪽 면으로 접근했는지, 정면에서 계속 밀었는지는 표현하지 못한다.

참고로 이후 추가한 `goal_delta`는 목표와 block의 signed 위치 차이이며, 접근 방식 자체의 라벨은 아니다. 접근 방식은 이미지와 action sequence에 나타나므로 demonstration 다양성이 중요하다.

### 4. 동일한 초기 상태에서의 제한된 일반화

기존 평가에서 매 episode가 동일한 초기 상태에서 시작되었고, 정책도 결정론적으로 같은 action 패턴을 반복했다. 작은 시각·동작 오차가 생겨도 다른 접근 전략으로 전환할 데이터가 부족했다.

### 5. action chunk 후반부 학습 부족

ACT는 action chunk를 예측한다. 데이터가 충분히 다양하지 않으면 chunk 초반의 전진은 학습하지만, chunk 후반의 방향 전환·보정 action은 평균화되어 목표 앞에서 멈추는 경향이 나타날 수 있다.

## 해결을 위한 데이터 수집

기존 50개를 폐기하지 않고, 다음 demonstration을 추가했다.

```text
right 접근 30개
center 접근 30개
left 접근 30개
```

여기서 right/left/center는 최종 block 좌표가 아니라 접근 방식이다.

- right: block이 오른쪽으로 회전할 때 오른쪽 면으로 재접근
- left: block이 왼쪽으로 회전할 때 왼쪽 면으로 재접근
- center: 정면 면을 유지하며 goal로 진입

기존 50개와 추가 90개를 합쳐 총 140 episode, 37,478 frame의 ACT 데이터셋을 만들었다. episode별 접근 방향은 `meta/approach_labels.jsonl`에 자동 기록했다.

## 개선 결과

140개 데이터로 재학습한 ACT는 평가 5개 episode에서 다음 결과를 보였다.

- 성공: 5/5
- 성공률: 100%
- 성공 시점: 약 255 steps, 약 12.75초
- 최종 block 위치: 약 `(0.769, -0.002)`
- 최종 goal distance: 약 `0.031m`

추론 중 y 방향 drift가 약 `+0.026m`까지 발생했지만, 이후 다시 중앙으로 보정하고 goal에 진입했다. 이는 기존 50개에서 부족했던 접근 방향과 보정 trajectory가 추가 데이터로 보완되었음을 보여준다.

## 결론

기존 50개 데이터의 실패는 데이터가 모두 잘못 저장되었기 때문이 아니었다. demonstration은 성공 위치까지 도달했지만, 접근 방향과 후반부 보정 action의 다양성이 부족했다. 그 결과 ACT는 초반 전진은 수행했지만 목표 앞에서 반복적으로 정지했다.

따라서 문제 해결을 위해 데이터를 무작정 늘린 것이 아니라, 동일한 task 안에서 right/center/left 접근 방식을 일관된 기준으로 나누어 추가 수집했다. 140개로 확장한 뒤 ACT가 5/5 성공한 결과는 데이터의 양뿐 아니라 접근 trajectory의 일관성과 다양성이 핵심이었다는 결론을 뒷받침한다.
