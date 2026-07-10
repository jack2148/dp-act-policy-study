# Diffusion Policy — Code Analysis

Repo: huggingface/lerobot (submodule) — path: `lerobot/policies/diffusion/`
목적: 논문 수식/개념 ↔ 코드 위치 매핑. 나중에 ACT 코드 분석이랑 나란히 비교.

---

## 1. Config (`configuration_diffusion.py`)

| 파라미터 | 논문 표기 | 의미 | 오늘 학습한 트레이드오프 |
|---|---|---|---|
| `n_obs_steps` | $T_o$ | 현재 시점 기준 과거 관측 몇 스텝 볼지 (미래 방향 없음) | — |
| `horizon` | $T_p$ | 한 번에 예측하는 action sequence 길이 | 길수록 temporal consistency ↑ |
| `n_action_steps` | $T_a$ | 실제 environment에서 실행하는 action 개수 | 작을수록 반응성 ↑, rollout 연산비용 ↑ / 크면 open-loop 위험 |
| `num_train_timesteps` | K | DDPM 전체 diffusion step 수 | — |

## 2. Core class (`modeling_diffusion.py`) — `DiffusionPolicy`

| 함수 | 논문 대응 | 하는 일 | 확인한 것 |
|---|---|---|---|
| `forward()` | Eq.3 (training loss) | random k 샘플링 + noise 예측, MSE loss | |
| `conditional_sample()` | Eq.1/4 (inference) | K-step denoising loop | |
| `select_action()` | receding horizon control (3.3절) | horizon만큼 예측 후 n_action_steps만 실행, 나머지 버림 | |

## 3. Sub-modules
- Vision encoder: (파일 경로) — ResNet + spatial softmax pooling (GAP 대체 이유: ?)
- Noise predictor: (파일 경로) — Unet1D or Transformer, config 플래그로 분기

## 4. 오늘 정리한 개념 (날짜: 2026-07-XX)
- n_action_steps ↔ horizon 트레이드오프: 일관성(긴 horizon) vs 반응성(짧은 n_action_steps)
- n_action_steps는 학습 연산량이 아니라 **inference/rollout 연산량**에 영향
- (다음에 채울 것) ACT의 temporal ensembling과 비교 — 왜 DP는 "버리기", ACT는 "겹쳐서 평균내기"를 택했는가

## 5. ACT와 비교 예정 항목 (나중에 채움)
- [ ] chunk 실행 방식: DP(일부 실행+버림) vs ACT(temporal ensemble)
- [ ] 생성 모델: DDPM vs CVAE — 학습/추론 비용 구조 차이
- [ ] 아키텍처: CNN+FiLM vs Transformer encoder-decoder