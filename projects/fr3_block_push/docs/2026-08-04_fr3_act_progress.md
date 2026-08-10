# FR3 ACT 실험 진행 정리 — 2026-08-04

## 데이터 수집

- 실행 스크립트: `./data`
- 현재 기본 수집 목표: 50 episodes
- 기존 저장 위치: `datasets/data_50`
- 엄격한 성공 기준으로 추가 수집할 위치: `datasets/data_success_50_v3`
- 실패 데이터 저장 위치: `datasets/data_success_50_v3_fail`
- 데이터 형식: LeRobot
- 샘플링: 20 Hz
- 이미지: `observation.images.top`, 기본 96×96 RGB
- 상태: FR3 7관절 `float32` (라디안)
- action: FR3 7관절 position target `float32` (라디안)
- 추가 저장값: `observation.ee_pose`/`action.ee_pose` (`[qw,qx,qy,qz,x,y,z]`)
- 추가 저장값: `observation.goal_distance` (block-goal XY Euclidean distance)
- 기존 `data_50`: 50 episodes, 8648 frames

### 키 조작

- `Z`: 에피소드 시작
- `S`: 현재 에피소드를 성공 데이터셋에 저장
- `X`: 현재 에피소드를 실패 데이터셋에 저장
- `R`: 폐기 후 리셋
- `V`: operator camera 전환
- `SPACE`: 상태 출력
- `Q/ESC`: 종료

## 성공 판정

기존 평가 기준은 목표 중심 기준 약 ±3.5 cm, 속도 ≤0.01 m/s, 0.5초 유지였다.
성공 판정은 색상이 아니라 MuJoCo body 좌표와 블록 속도로 계산한다.

시연 종료 위치의 분산을 줄이기 위해 앞으로 **수집용 성공 기준만 강화**했다.

```text
collection success_margin = 0.025 m
허용 중심 오차 ≈ ±0.015 m
```

평가용 기본 기준은 기존 값(`0.005 m`, 약 ±3.5 cm)을 유지한다. 따라서 기존 `data_50`은 보존하고, 새 데이터는 목표 중심에 가깝고 정지한 성공 시연만 저장한다.

## ACT 학습

- 50개 데이터 학습 checkpoint:
  `outputs/train/act_fr3_test_50/checkpoints/005000/pretrained_model`
- 학습 설정: ACT, `chunk_size=64`, `n_action_steps=32`, temporal ensemble OFF
- 입력: 이미지 1장 + 7관절 상태
- 정규화: LeRobot dataset statistics 기반 MEAN/STD
- 50개 데이터 학습은 5000 step까지 정상 완료

## ACT 추론

- 추론 코드: `projects/fr3_block_push/scripts/eval_act.py`
- MuJoCo 직접 제어: `TeleopRunner.apply_policy_action()`
- 정책 action 주기: 20 Hz
- MuJoCo timestep: 0.001초, action 하나를 50 substep 동안 유지
- `--n-action-steps`로 queue 길이를 추론 시 변경 가능

실험 결과:

- `n_action_steps=32`: 목표 앞 `x≈0.749`
- `n_action_steps=8`: `x≈0.760~0.764`
- 성공 기준 최소 x는 약 `0.765`
- 8-step이 더 가까이 가지만 목표 직전에서 정지/진동

현재 가장 가능성 높은 원인은 목표를 못 찾는 것이 아니라, 성공 시연의 종료 위치가 분산되어 ACT가 목표 직전의 보수적인 action을 학습하는 것과, 단일 관측(`n_obs_steps=1`)으로 접촉/속도를 추정하기 어려운 점이다.

## 디버그 로그와 경로 plot

추론 시 다음 파일이 생성된다.

```text
outputs/eval/<run>/debug_steps.jsonl
outputs/eval/<run>/debug_steps.csv
outputs/eval/<run>/trajectories_xy.png
```

로그에는 block/goal XY, 목표 거리, action 범위, simulation time, 종료 이유가 포함된다.

기존 데이터셋에는 블록 XY/속도가 직접 저장되지 않아, 50개 demonstration의 최종 block 위치 분포를 정확히 계산할 수 없다. 새 수집에는 block-goal XY distance가 추가되며, 정확한 방향/속도 분석에는 향후 block XY와 block speed를 별도 feature로 추가할 수 있다.

새 수집부터는 EE pose와 block-goal XY distance도 함께 저장한다. 기존 `observation.state`와 `action` joint 값은 그대로 유지하므로, 나중에 joint 기반 또는 EE 기반 학습 구성을 선택할 수 있다. EE action은 joint target을 MuJoCo forward kinematics로 변환한 값이다.

## 다음 실행

1. 새 엄격한 성공 기준으로 `./data`를 실행해 성공 시연을 추가 수집
2. 새 데이터로 ACT 재학습
3. `n_action_steps=8` 및 `2` 비교 평가
4. debug CSV와 XY plot으로 목표 근처 정지/진동 분석
