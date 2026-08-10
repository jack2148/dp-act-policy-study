# Table 1. ACT vs Diffusion Policy under the same 5k training budget

## Verified experiment setup

Both checkpoints were trained from the same `datasets/data_act_balanced_140` dataset:

- 140 episodes / 37,478 frames / 20Hz
- fixed initial block position around `(0.620, 0.000)`
- 5,000 optimizer steps
- batch size 8
- 5 evaluation episodes
- maximum evaluation length 600 steps
- success threshold used by the analysis: goal distance `<= 0.035m`

Checkpoint configuration was read from each `config.json` and `train_config.json`.

| Setting | ACT | Diffusion Policy |
|---|---:|---:|
| Dataset | balanced-140 | balanced-140 |
| Training steps | 5,000 | 5,000 |
| Batch size | 8 | 8 |
| Observation state | 7D FR3 joint | 7D FR3 joint |
| Action | 7D absolute joint target | 7D absolute joint target |
| Observation images | `observation.images.top` | `observation.images.top` |
| `n_obs_steps` | 1 | 2 |
| Chunk/horizon | chunk 64 | horizon 64 |
| `n_action_steps` | 32 | 32 |
| Normalization | MEAN_STD state/action | MIN_MAX state/action |
| Temporal ensemble | off (`null`) | not used |
| Diffusion scheduler | N/A | DDPM |

## Evaluation results

| Metric | ACT | Diffusion Policy |
|---|---:|---:|
| Evaluation episodes | 5 | 5 |
| Success episodes | 5/5 | 1/5 |
| Success rate | 100% | 20% |
| First success step, successful episodes | 241 ± 0 | 194 ± 0 |
| Success time | 12.05 ± 0.00s | 9.70s (one episode) |
| Final goal distance, mean | 0.0312m | 0.0778m |
| Final goal distance, range | 0.0312–0.0312m | 0.0344–0.1087m |
| Maximum lateral deviation, mean | 0.0270m | 0.0795m |
| Block path length, mean | 0.1791m | 0.2603m |
| Mean action difference | N/A | N/A |
| Max action difference | N/A | N/A |

Action smoothness was not calculated because the evaluation logs contain only
`action_min`, `action_max`, and `action_mean`, not all seven action dimensions
for every step. The analysis does not infer smoothness from those summaries.

## Episode-level behavior

ACT crossed the success distance in all five rollouts. The first crossing was
at step 241; the CSV contains 255 rows per episode because logging continues for
the final transition after the threshold crossing.

DP succeeded only in episode 2, crossing the threshold at step 194. The other
four episodes did not cross the threshold within 600 steps:

- episode 0: final distance `0.0994m`, y deviation about `+0.0929m`
- episode 1: final distance `0.0580m`, overshot to `x≈0.835`, y≈`+0.0463m`
- episode 3: final distance `0.1087m`, y deviation about `+0.1042m`
- episode 4: final distance `0.0883m`, y deviation about `-0.0840m`

The dominant DP failure mode is lateral drift and unstable approach, not a
complete actuator or simulator stop. The successful DP episode shows that the
task is reachable by this checkpoint, but the policy is not reliable under the
same fixed rollout condition.

## Source files

- ACT log: `outputs/eval/act_fr3_balanced_140/debug_steps.csv`
- DP log: `outputs/eval/dp_fr3_balanced_140/debug_steps.csv`
- ACT checkpoint: `outputs/train/act_fr3_balanced_140/checkpoints/005000/pretrained_model`
- DP checkpoint: `outputs/train/dp_fr3_balanced_140/checkpoints/005000/pretrained_model`
- Generated summary: `projects/fr3_block_push/analysis/outputs/act_dp_5k_summary.csv`
- Generated episode metrics: `projects/fr3_block_push/analysis/outputs/act_dp_5k_episode_metrics.csv`
- Config report: `projects/fr3_block_push/analysis/outputs/act_dp_5k_configs.json`

Re-run the analysis with:

```bash
python3 projects/fr3_block_push/analysis/compare_act_dp_5k.py
```

The script accepts `--act-eval`, `--dp-eval`, `--act-checkpoint`,
`--dp-checkpoint`, and `--output-dir` overrides and does not hard-code the
metric values above.

## Limitations

This is a deterministic fixed-condition comparison: all five episodes use the
same initial block/robot state. It is not a generalization evaluation over
randomized initial conditions. The comparison also uses an equal 5,000-step
budget; it does not establish that ACT is intrinsically better than DP at each
policy's optimal training duration. DP requires a training-step ablation before
that conclusion can be made.

## Follow-up DP ablation plan

### Phase 1 — equal-budget baseline

Completed above: ACT-5k vs DP-5k on balanced-140.

### Phase 2 — training-step ablation

Train DP with only `steps` changed:

```bash
STEPS=10000 OUTPUT_DIR=outputs/train/dp_fr3_balanced_140_10k ./train_dp_balanced
STEPS=15000 OUTPUT_DIR=outputs/train/dp_fr3_balanced_140_15k ./train_dp_balanced
```

Evaluate each checkpoint with the same five fixed initial-condition episodes and
the same success evaluator. Interpret the results as follows:

1. Success rate increasing with steps suggests optimization insufficiency at 5k.
2. Improvement at 10k followed by degradation at 15k suggests overfitting or checkpoint-selection issues.
3. No improvement through 15k points to execution horizon, observation horizon, normalization, or diffusion inference configuration.

### Phase 3 — execution ablation

After selecting one DP checkpoint, compare only `n_action_steps` values 2, 4,
and 8. Do not change training steps and execution horizon in the same experiment.
