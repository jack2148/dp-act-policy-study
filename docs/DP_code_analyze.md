# Diffusion Policy — Code Analysis

Repo: huggingface/lerobot (submodule) — path: `lerobot/policies/diffusion/`
목적: 논문 수식/개념 ↔ 코드 위치 매핑. 나중에 ACT 코드 분석이랑 나란히 비교.

---

## 1. Config (`configuration_diffusion.py`)

| 파라미터 | 논문 표기 | 의미 | 오늘 학습한 트레이드오프 |
|---|---|---|---|
| `n_obs_steps` | $T_o$ | 현재 시점 기준 과거 관측 몇 스텝 볼지 (미래 방향 없음) | Push-T 기본 설정은 2이며, 두 관측 간 차분으로 속도 변화를 간접적으로 활용할 수 있다. |
| `horizon` | $T_p$ | 한 번에 예측하는 action sequence 길이 | 길수록 temporal consistency ↑ |
| `n_action_steps` | $T_a$ | 실제 environment에서 실행하는 action 개수 | 작을수록 반응성 ↑, rollout 연산비용 ↑ / 크면 open-loop 위험 |
| `num_train_timesteps` | K | DDPM 전체 diffusion step 수 | 클수록 denoising 단계가 세분화되지만 rollout 추론 비용도 증가한다. |

## 2. Core class (`modeling_diffusion.py`) — `DiffusionPolicy`

| 함수 | 논문 대응 | 하는 일 | 확인한 것 |
|---|---|---|---|
| `forward()` | Eq.3 (training loss) | random k 샘플링 + noise 예측, MSE loss | `modeling_diffusion.py`의 `DiffusionPolicy.forward()`가 배치 action에 noise를 더하고 predictor의 noise estimate와 MSE를 계산한다. |
| `conditional_sample()` | Eq.1/4 (inference) | K-step denoising loop | `DiffusionPolicy.conditional_sample()`이 scheduler timesteps를 순회하며 noise predictor와 scheduler step으로 action trajectory를 복원한다. |
| `select_action()` | receding horizon control (3.3절) | horizon만큼 예측 후 n_action_steps만 실행, 나머지 버림 | `DiffusionPolicy.select_action()`이 새 trajectory에서 실행 구간만 반환하며, 다음 관측에서 다시 계획한다. |

## 3. Sub-modules
- Vision encoder: `lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py`의 `DiffusionRgbEncoder` — ResNet feature map 뒤에 `SpatialSoftmax`를 두어 채널별 salient keypoint의 2D 위치를 보존한다. 전역 평균 풀링보다 물체·그리퍼의 위치 정보를 action conditioning에 직접 전달하기 위한 선택이다.
- Noise predictor: 같은 파일의 `DiffusionUNet`/`ConditionalUnet1D` 계열 — 이 로컬 LeRobot 버전에서는 FiLM conditioning을 사용하는 1D convolutional U-Net이 action-sequence noise를 예측한다. Transformer predictor 여부는 이 실험에서 사용한 config로는 미검증이다.

## 4. 확인한 개념
- n_action_steps ↔ horizon 트레이드오프: 일관성(긴 horizon) vs 반응성(짧은 n_action_steps)
- n_action_steps는 학습 연산량이 아니라 **inference/rollout 연산량**에 영향
- DP는 최근 관측으로 조건화한 trajectory의 앞부분만 실행하고 재계획해, 관측 변화에 빠르게 반응한다. 반대로 ACT의 temporal ensembling은 여러 예측을 부드럽게 만들려는 설계지만, 본 Push-T 실험에서는 서로 다른 action mode의 평균화가 성능 병목으로 관찰됐다.

## 5. ACT와의 비교

| 항목 | DP | ACT | 이 실험에서의 관찰 |
|---|---|---|---|
| Chunk 실행 | 일부 실행 후 새 관측으로 재계획 | 기본 설정은 겹친 chunk를 temporal ensemble | ACT ensemble OFF·`n_action_steps=32`가 12%로 기본 ACT 2%보다 높았다. |
| 생성 모델 | iterative DDPM denoising | CVAE latent에서 한 번에 chunk 생성 | DP는 rollout당 반복 추론 비용이 크지만 Push-T에서 24%를 기록했다. |
| 아키텍처 | ResNet + spatial softmax + FiLM-conditioned 1D U-Net | Transformer encoder-decoder | 정확한 ACT 구현 세부 매핑은 이 문서에서는 미검증이며, 결과 해석은 실행 전략 ablation에 한정한다. |
