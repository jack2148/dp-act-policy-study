# ACT vs Diffusion Policy for Manipulation

Controlled ACT and Diffusion Policy experiments across Push-T policy analysis and teleoperation-derived Franka FR3 MuJoCo manipulation.

> Research narrative: teleoperation demonstrations → LeRobot datasets → ACT / Diffusion Policy → policy-execution analysis → trajectory coverage → data-efficient demonstration collection.

This repository is a reproducible research record, not a claim that one policy is universally superior. It studies how execution strategy affects closed-loop behavior and investigates how demonstration distribution may influence learned manipulation policies.

## Overview
- Push-T: controlled 50-episode ACT execution-strategy ablations.
- FR3 MuJoCo: teleoperation-to-dataset pipeline and fixed-initial-condition ACT/DP pilot.
- Current focus: identifying underrepresented task conditions to guide more data-efficient demonstrations.

## Research questions
1. How does policy execution strategy affect closed-loop imitation-learning performance?
2. How does demonstration trajectory diversity affect manipulation performance?
3. How do ACT and Diffusion Policy behave under controlled manipulation settings?
4. Which trajectory or task conditions are underrepresented in collected demonstrations?

An adaptive collection algorithm is not yet implemented; it is the next research direction.

## Key findings

### Execution strategy affects ACT
In the Push-T 50-episode evaluation, disabling temporal ensembling and executing 32 actions per inference (n_action_steps=32) produced **12%** ACT success, versus **2%** for default ACT. Diffusion Policy reached **24%** under its own observation and execution configuration. Because representations and inference settings differ, this is behavior analysis rather than a definitive head-to-head benchmark.

### FR3 pilot reveals policy-dependent behavior under fixed conditions
A fixed-condition FR3 pilot showed substantially different learned behavior between ACT and Diffusion Policy on the same balanced-140 teleoperation dataset. ACT reached 5/5 and Diffusion Policy 1/5 after 5k training steps. Because the evaluation uses a single initial state, n=5 rollouts, limited training seeds, and no matched dataset-composition ablation, this result supports only a fixed-condition behavioral comparison between the two policies. It does not establish a causal effect of demonstration coverage or generalization performance. These limitations motivate the current investigation into how trajectory coverage and demonstration distribution relate to policy behavior.

### Data efficiency is the next question
Current work analyzes trajectory/task-region coverage and underrepresented conditions in the collected demonstrations. The next step is to relate these coverage patterns to condition-wise policy performance before designing targeted collection.

## Experiment A — Push-T policy-execution analysis

| Item | Value |
|---|---|
| Task | lerobot/pusht / lerobot/pusht_keypoints |
| Evaluation | 50 episodes, pc_success |
| Diffusion Policy | image + agent position, partial execution |
| ACT | keypoints + agent position, chunk size 64 |
| Hardware | Ubuntu 22.04, RTX 4060 8 GB |
| Framework | LeRobot 0.6.1 |

| Configuration | Ensemble | n_action_steps | Final success |
|---|---:|---:|---:|
| Diffusion Policy | — | partial execution | **24%** |
| ACT baseline | on (0.01) | 1 | **2%** |
| ACT, no ensemble | off | 32 | **12%** |

In this setup, changing the ACT execution strategy—disabling temporal ensembling and executing longer action segments—improved closed-loop success. Whether the gain comes primarily from temporal ensembling, action-horizon choice, or their interaction remains a follow-up question. Single-frame ACT observation (`n_obs_steps=1`) may also omit velocity/momentum; this remains a hypothesis rather than an isolated causal conclusion.

<details><summary>Detailed ACT ablations</summary>

| Run | Batch | LR | KL | Ensemble | Steps/action | Final SR | Avg reward |
|---|---:|---:|---:|---|---:|---:|---:|
| Pixel baseline | 8 | 1e-5 | 10 | on | 1 | 2% | 25.19 |
| Keypoints batch-8 | 8 | 1e-5 | 10 | on | 1 | 0% | 16.31 |
| Keypoints v2 | 64 | 3e-5 | 10 | on | 1 | 2% | 22.98 |
| Max batch | 128 | 6e-5 | 10 | on | 1 | 0% | 24.77 |
| No ensemble | 64 | 3e-5 | 10 | off | 32 | **12%** | **69.96** |
| KL weight 1 | 64 | 3e-5 | 1 | on | 1 | 0% | 24.85 |

The max-batch run reached training loss 0.068 without improving closed-loop success. The batch-8 comparison is inconclusive because batch size and learning rate were confounded with input representation.
</details>

## Experiment B — FR3 MuJoCo teleoperation and evaluation

OMY-L100 leader → ROS 2 / ZeroMQ bridge → FR3 MuJoCo controller → 20 Hz LeRobot demonstrations → ACT / Diffusion Policy training and evaluation.

See [OMY_FRANKA_TELEOP](https://github.com/jack2148/OMY_FRANKA_TELEOP). This repository covers simulation-stage FR3 tooling; physical FR3 deployment is not claimed.

### Dataset and task
- 20 Hz recordings: observation.images.top (96×96 RGB), 7-DoF FR3 joint state, 7-DoF absolute joint-target action.
- Balanced-140: 140 episodes / 37,478 frames spanning left, center, and right approach regions.
- Success: block-goal XY distance ≤ 0.035 m; five pilot episodes share one fixed initial state.

### Controlled 5k-step pilot
| Metric | ACT | Diffusion Policy |
|---|---:|---:|
| Training steps / batch | 5,000 / 8 | 5,000 / 8 |
| Evaluation episodes | 5 | 5 |
| Success episodes | **5/5** | 1/5 |
| Success rate | **100%** | 20% |
| Mean final goal distance | **0.0312 m** | 0.0778 m |
| Mean maximum lateral deviation | 0.0270 m | 0.0795 m |
| Mean block path length | 0.1791 m | 0.2603 m |

Diffusion Policy's dominant failure mode was lateral drift and unstable approach. These are fixed-condition behavioral observations. Equal training steps do not imply equal optimization or convergence, and the pilot does not support claims about randomized-state generalization.

These pilot observations do not establish a coverage effect, but they motivate a more controlled study of how demonstration distribution and task-condition coverage relate to policy behavior.

**Limitations:** deterministic fixed initial condition, n=5, limited pilot scale, and no randomized generalization benchmark. See the [verified analysis](projects/fr3_block_push/docs/act_vs_dp_5k_comparison.md).

<p align="center"><img src="results/ACT_sucess_data_140.gif" width="480" alt="ACT rollout on balanced-140 FR3 data"></p>

## Current research — data-efficient demonstrations
The active investigation analyzes trajectory and task-region coverage in the collected demonstrations and explores how underrepresented conditions can be identified for targeted additional collection. The next evaluation stage will extend policy assessment beyond the current fixed-state baseline toward condition-wise performance analysis. Planned progression: **observation → coverage metric → targeted collection → policy retraining and evaluation**, with fixed-baseline evaluation at each stage. The focus is improving the demonstration-collection loop, not online policy adaptation. Coverage-aware collection is not yet an implemented algorithm or achieved benchmark.

## Repository map
projects/fr3_block_push/ — FR3 scene, teleoperation, recording, evaluation, tests
  analysis/ — ACT/DP comparison and summary scripts
  dataset/ — LeRobot recording and validation
  scripts/ — collection, preparation, and evaluation entry points
docs/ — implementation and experiment notes
references/ — archived Push_MuJoCo material
lerobot/ — LeRobot source (git submodule)
results/ — rollout GIFs and visualizations

## Reproduction
Clone with submodules, create a Python 3.10 environment, install LeRobot with dataset/training extras and gym-pusht, then run:

    lerobot-train --policy.type=diffusion --env.type=pusht --dataset.repo_id=lerobot/pusht --policy.push_to_hub=false --wandb.enable=false --steps=100000
    lerobot-eval --policy.path=outputs/train/<run>/checkpoints/last/pretrained_model --env.type=pusht --eval.n_episodes=50 --device=cuda
    ./train_act_balanced   # FR3 balanced-140 ACT (default: 5,000 steps, batch 8)
    ./train_dp_balanced    # matching FR3 Diffusion Policy baseline
    python3 projects/fr3_block_push/analysis/compare_act_dp_5k.py
    python3 -m unittest discover -s projects/fr3_block_push/tests -v

The FR3 ACT training configuration is preserved in [train_act_balanced](train_act_balanced). For teleoperation and recording, see [projects/fr3_block_push/README.md](projects/fr3_block_push/README.md).

<details>
<summary>Policy Background &amp; Paper Notes</summary>

- [Diffusion Policy](https://arxiv.org/abs/2303.04137) — iterative denoising of action sequences.
- [ACT](https://arxiv.org/abs/2304.13705) — CVAE/Transformer action chunks with optional temporal ensembling.
- [LeRobot](https://github.com/huggingface/lerobot).
- Code walkthrough: [docs/DP_code_analyze.md](docs/DP_code_analyze.md).

</details>

## References
- [Diffusion Policy paper](https://arxiv.org/abs/2303.04137)
- [ACT paper](https://arxiv.org/abs/2304.13705)
- [LeRobot](https://github.com/huggingface/lerobot)
- [Official Diffusion Policy repository](https://github.com/real-stanford/diffusion_policy)
- [LeRobot PR #314](https://github.com/huggingface/lerobot/pull/314)
