# ACT vs Diffusion Policy for Manipulation

A comparative study of Action Chunking with Transformers (ACT) and Diffusion Policy (DP) in Push-T and MuJoCo FR3 manipulation. It connects policy-behavior analysis with teleoperation-derived demonstrations and ongoing research on demonstration coverage and data-efficient collection.

## Overview

This repository compares two visuomotor imitation-learning approaches:

- **ACT**, which predicts action chunks with a transformer and can optionally smooth overlapping predictions through temporal ensembling.
- **Diffusion Policy**, which generates action sequences through conditional denoising.

The work combines a Push-T benchmark study with a fixed-condition MuJoCo FR3 block-pushing pilot. Push-T is used to examine policy and execution behavior; the FR3 study tests policies trained from teleoperation-derived demonstrations and motivates the current question of which trajectories and task conditions the dataset covers. Results are reported with their protocol constraints and evidence status: retained raw measurements where available, and explicitly labeled historical records where raw artifacts are no longer present.

## Research Questions

1. How do ACT and Diffusion Policy behave under limited demonstration data and training budgets?
2. Which implementation and inference choices materially change ACT performance?
3. How does Diffusion Policy performance vary across the evaluated FR3 training checkpoints?
4. How can trajectory and task coverage be characterized to guide more data-efficient demonstration collection?

## Experimental Setup

### Push-T

Push-T provides a controlled planar manipulation benchmark for comparing ACT variants and Diffusion Policy.

- **ACT baseline:** original ACT settings used in this study.
- **Modified ACT:** temporal ensembling was changed and <code>n_action_steps</code> was increased from 1 to 32.
- **Diffusion Policy:** a 100k-step result is retained as a historical record from the initial experiment log and tracked GIF.

Because the modified ACT run changed two factors at once, it is an exploratory comparison rather than a controlled ablation. The contribution of either change cannot be isolated from this run.

### FR3 MuJoCo manipulation

The MuJoCo FR3 pilot uses a block-pushing task and the <code>data_act_balanced_140</code> dataset of 140 teleoperation-derived demonstrations. Evaluation was conducted from one fixed initial condition with five trials per checkpoint.

Two thresholds serve different purposes:

- **0.025 m:** collection-only criterion.
- **0.035 m:** evaluation success threshold.

These thresholds are not interchangeable. Additional setup, dataset, and evaluation details are documented in [the FR3 project README](projects/fr3_block_push/README.md).

## Key Results

### Push-T policy analysis

| Policy / configuration | Successes | Success rate | Evidence status |
|---|---:|---:|---|
| ACT baseline | 1 / 50 | 2% | Raw result retained |
| Modified ACT: temporal ensembling changed and <code>n_action_steps</code> 1 → 32 | 6 / 50 | 12% | Raw result retained; exploratory, two factors changed |
| Diffusion Policy, 100k steps | 12 / 50 | 24% | Historical record; raw metric/checkpoint not retained |

The 24% Diffusion Policy value is historically documented from the initial record and its tracked GIF, but it is **not raw-verified** in the current repository. The GIF is qualitative evidence only and must not be treated as a substitute for a retained metric or checkpoint.

#### Qualitative rollout examples

<table>
  <tr>
    <th>Diffusion Policy, 100k (historical)</th>
    <th>ACT baseline</th>
    <th>Modified ACT</th>
  </tr>
  <tr>
    <td><img src="results/100k_result.gif" alt="Historical Push-T Diffusion Policy 100k rollout" width="280"></td>
    <td><img src="results/act_v2_100k_result.gif" alt="Push-T ACT baseline rollout" width="280"></td>
    <td><img src="results/act_noensemble_100k_result.gif" alt="Push-T modified ACT rollout" width="280"></td>
  </tr>
</table>

These GIFs illustrate individual rollouts; they do not establish the quantitative success rates.

### FR3 fixed-condition pilot

| Policy / checkpoint | Trials succeeded | Mean final object-to-goal distance |
|---|---:|---:|
| ACT 5k | 5 / 5 | 0.0312 m |
| DP 5k | 1 / 5 | 0.0778 m |
| DP 10k | 2 / 5 | 0.0578 m |
| DP 20k, Run 1 | 4 / 5 | 0.0406 m (raw: 0.04058144 m) |
| DP 20k, Run 2 | 4 / 5 | 0.0342 m (raw: 0.03418998 m) |

All rows are fixed-condition pilot observations with n = 5 repeated rollouts per reported evaluation. DP 5k and 10k used batch size 8, while DP 20k used batch size 4; therefore, the checkpoint comparison is observational rather than a controlled training-step scaling experiment.

The two DP 20k evaluations used the same checkpoint and fixed protocol. Legacy Diffusion Policy inference was unseeded, so the runs are reported separately; they are not pooled as 8 / 10 and their distances are not averaged.

Run 1 remains reproducible from retained machine-generated per-episode artifacts although its original step-level CSV was overwritten. Run 2's exact value comes from the current raw CSV.

ACT achieved **5 / 5 under the tested fixed initial condition**. This is a pilot result, not evidence of general 100% task performance.

#### Qualitative rollout example

![FR3 ACT fixed-condition rollout from data_act_balanced_140](results/ACT_sucess_data_140.gif)

This rollout does not demonstrate performance under randomized conditions or generalization.

## Key Observations

- In Push-T, the retained ACT measurements improved from 1 / 50 to 6 / 50 after temporal ensembling and action-horizon behavior were both changed. The result motivates controlled follow-up experiments, but it does not identify which change produced the difference.
- The historical Push-T Diffusion Policy record is numerically higher than the retained ACT results, but its raw metric and checkpoint are unavailable. It should not be interpreted as a fully reproducible head-to-head result.
- In the FR3 fixed-condition pilot, ACT 5k met the evaluation criterion in all five tested trials.
- The evaluated DP checkpoints improved from 1 / 5 at 5k to 4 / 5 in each 20k run, while mean final distance decreased. This pattern is observational because training settings were not fully controlled across budgets.
- Variation between the two unseeded DP 20k runs shows why repeated, seeded evaluation is needed even under a fixed starting condition.

## Current Research Direction

Current work extends ACT / Diffusion Policy manipulation-behavior analysis and teleoperation-derived demonstration collection toward **trajectory coverage, task-condition coverage, demonstration-distribution analysis, and data-efficient collection**. It is investigating whether underrepresented trajectory or task conditions can be identified and used to guide future targeted collection, followed by retraining and evaluation.

This repository does **not** yet implement an adaptive collection algorithm, establish a causal effect of coverage, provide a completed condition-performance model, or demonstrate generalized performance improvement.

## Limitations & Next Steps

The present evidence has several important limits:

- FR3 evaluation uses one fixed initial condition and only five trials per evaluation.
- Training and evaluation use limited seeds; legacy DP inference was unseeded.
- No randomized initial-state, disturbance, or out-of-distribution generalization evaluation has been completed.
- The modified Push-T ACT experiment changed two factors simultaneously.
- The Push-T DP 100k number is historical and cannot currently be raw-verified.
- The FR3 DP checkpoint comparison changes both training steps and batch size at 20k.

Priority follow-up work:

1. Seed inference and repeat each evaluation across multiple independent runs.
2. Randomize initial object and goal poses and report condition-stratified results.
3. Run one-factor ACT ablations for temporal ensembling and <code>n_action_steps</code>.
4. Repeat Diffusion Policy training budgets with matched batch size, data, seeds, and evaluation.
5. Define prospective trajectory-coverage and task-coverage measures before testing collection strategies.
6. Retain machine-readable metrics, checkpoints, configuration snapshots, and rollout metadata for every reported result.

## Technical Details

### Experimental context

| Item | Push-T study |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 4060 8 GB |
| Framework | LeRobot 0.6.1 |
| Dataset | <code>lerobot/pusht</code> or <code>lerobot/pusht_keypoints</code>, depending on the ACT run |
| Evaluation | 50 episodes, <code>pc_success</code> |

### Policy notes

| Topic | ACT | Diffusion Policy |
|---|---|---|
| Output | Direct action chunks | Denoised action sequence |
| Temporal behavior | Chunk execution; optional temporal ensembling of overlapping predictions | Receding-horizon execution from a generated sequence |
| Main sensitivity examined here | Temporal ensembling and <code>n_action_steps</code> | Training budget and stochastic inference |
| Evaluation concern | Coupled configuration changes can confound attribution | Unseeded inference can change repeated outcomes |

A more detailed Diffusion Policy implementation walkthrough is available in [DP code analysis](docs/DP_code_analyze.md).

### Push-T ACT experiments

| Run | Input | Batch | LR | KL weight | Temporal ensemble | <code>n_action_steps</code> | Final SR | Peak SR | Final avg. sum reward | Final avg. max reward |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Pixel | Image + agent position | 8 | 1e-5 | 10 | On (0.01) | 1 | 2% | 2% | 25.19 | 0.355 |
| Batch-8 keypoints | Keypoints + agent position | 8 | 1e-5 | 10 | On (0.01) | 1 | 0% | 2% | 16.31 | 0.174 |
| Baseline (v2) | Keypoints + agent position | 64 | 3e-5 | 10 | On (0.01) | 1 | 2% | 2% | 22.98 | 0.328 |
| Max-batch | Keypoints + agent position | 128 | 6e-5 | 10 | On (0.01) | 1 | 0% | 2% | 24.77 | 0.373 |
| Modified | Keypoints + agent position | 64 | 3e-5 | 10 | Off | 32 | 12% | 12% | 69.96 | 0.595 |
| KL-weight-1 | Keypoints + agent position | 64 | 3e-5 | 1 | On (0.01) | 1 | 0% | Not recovered | 24.85 | 0.280 |

The modified run changed temporal ensembling and <code>n_action_steps</code> together, so it is not a one-variable causal ablation. The Batch-8 keypoints run is inconclusive because batch size and learning rate differ from the reference configuration. For KL-weight-1, the checkpoint-by-checkpoint curve could not be reconstructed from the training log; the final values come from the retained evaluation result.

### Implementation mapping

| Concept | LeRobot location | Local note |
|---|---|---|
| ACT action chunking | <code>lerobot/src/lerobot/policies/act/configuration_act.py</code> | <code>chunk_size</code> controls the predicted action-sequence length |
| ACT temporal ensembling | <code>lerobot/src/lerobot/policies/act/modeling_act.py</code> | <code>ACTTemporalEnsembler</code> combines overlapping chunks |
| ACT CVAE and KL loss | <code>lerobot/src/lerobot/policies/act/modeling_act.py</code> | Reconstruction loss is combined with weighted KL loss |
| Diffusion inference and loss | <code>lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py</code> | Conditional denoising and training objective |

### FR3 dataset and training configuration

The FR3 pilot used <code>data_act_balanced_140</code> and training seed 1000. The Diffusion Policy training configurations were:

| Checkpoint budget | Batch size | Dataset | Training seed |
|---:|---:|---|---:|
| DP 5k | 8 | data_act_balanced_140 | 1000 |
| DP 10k | 8 | data_act_balanced_140 | 1000 |
| DP 20k | 4 | data_act_balanced_140 | 1000 |

### Repository map

~~~
.
├── train_act_balanced/          # ACT training entry points and configuration
├── train_dp_balanced/           # Diffusion Policy training entry points and configuration
├── projects/
│   └── fr3_block_push/          # FR3 task documentation and robot-side workflow
├── data_efficiency/             # Coverage and distribution-analysis research
├── docs/
│   └── DP_code_analyze.md       # Diffusion Policy implementation notes
└── results/                     # Tracked qualitative rollout GIFs
~~~

## Reproduction

Commands and environments vary between the Push-T and FR3 studies. Review script arguments before launching training or robot control, and follow the task-specific documentation for hardware safety.

### Historical Push-T environment

The following setup records the LeRobot-based Push-T environment used for these experiments. Dependency versions and CLI options may require adaptation on a current installation.

~~~bash
git clone --recurse-submodules https://github.com/jack2148/dp-act-policy-study.git
cd dp-act-policy-study/lerobot
conda create -n lerobot python=3.10
conda activate lerobot
pip install -e .
pip install 'lerobot[dataset]'
pip install 'lerobot[training]'
pip install gym-pusht
~~~

### Diffusion Policy training

Historical Push-T command:

~~~bash
lerobot-train \
  --policy.type=diffusion \
  --env.type=pusht \
  --dataset.repo_id=lerobot/pusht \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --steps=100000
~~~

### ACT training

Historical keypoint baseline command:

~~~bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/pusht_keypoints \
  --output_dir=outputs/train/act_pusht_keypoints_100k_v2 \
  --job_name=act_pusht_keypoints_100k_v2 \
  --policy.device=cuda \
  --policy.chunk_size=64 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --policy.optimizer_lr=3e-5 \
  --policy.push_to_hub=false \
  --steps=100000 \
  --batch_size=64 \
  --env.type=pusht \
  --env.obs_type=environment_state_agent_pos \
  --eval_freq=10000 \
  --save_freq=10000 \
  --eval.n_episodes=50 \
  --wandb.enable=false
~~~

### Evaluation

For a comparable evaluation, record:

- checkpoint identity and checksum;
- dataset version;
- policy configuration;
- inference seed;
- initial-condition definition;
- success threshold;
- per-trial outcome and final distance.

~~~bash
lerobot-eval \
  --policy.path=outputs/train/<experiment>/checkpoints/last/pretrained_model \
  --env.type=pusht \
  --eval.n_episodes=50 \
  --device=cuda
~~~

### FR3 workflow

See [projects/fr3_block_push/README.md](projects/fr3_block_push/README.md) for task-specific collection, training, evaluation, and safety instructions.

## Troubleshooting

- **A reported value cannot be reproduced:** first confirm whether it is marked raw-retained or historical. The Push-T DP 100k result is historical only.
- **Repeated DP evaluations differ:** check whether the inference path is seeded. The legacy FR3 DP evaluations were unseeded.
- **Success counts and distances appear inconsistent:** verify that evaluation uses the 0.035 m threshold; 0.025 m is collection-only.
- **Checkpoint comparisons appear confounded:** confirm batch size as well as training steps. The FR3 DP 20k run used batch size 4, while 5k and 10k used batch size 8.
- **ACT changes are difficult to attribute:** temporal ensembling and <code>n_action_steps</code> were changed together in the exploratory Push-T run.

Common LeRobot setup issues retained from the experiments:

| Symptom | Resolution used in this study |
|---|---|
| <code>gym_pusht</code> not found | Install <code>gym-pusht</code> |
| <code>pymunk</code> incompatibility | Use <code>pymunk==6.4.0</code> for the historical environment |
| Missing <code>policy.repo_id</code> during local training | Set <code>--policy.push_to_hub=false</code> |
| <code>KeyError: observation.environment_state</code> in keypoint evaluation | Set <code>--env.obs_type=environment_state_agent_pos</code> |
| Unrecognized top-level <code>--optimizer_lr</code> | Use <code>--policy.optimizer_lr</code> |
| Missing <code>accelerate</code> | Install <code>lerobot[training]</code> |
| Output directory already exists | Choose a new output directory or intentionally manage the existing run; do not delete results blindly |

## References

- [Action Chunking with Transformers project page](https://tonyzhaozh.github.io/aloha/)
- [ACT paper: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy project page](https://diffusion-policy.cs.columbia.edu/)
- [Diffusion Policy paper: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [LeRobot](https://github.com/huggingface/lerobot)
- [Reference Diffusion Policy implementation](https://github.com/real-stanford/diffusion_policy)
- [LeRobot PR #314: ACT environment-state support](https://github.com/huggingface/lerobot/pull/314)
