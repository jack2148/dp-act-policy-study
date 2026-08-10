# Diffusion Policy 10k evaluation

- Checkpoint: `outputs/train/dp_fr3_balanced_140_10k/checkpoints/010000/pretrained_model`
- Dataset: `datasets/data_act_balanced_140`
- Evaluation: 5 fixed-initial-condition episodes, 600-step maximum, 20Hz
- Success threshold: goal distance `<= 0.035m`

## Result

| Metric | DP 5k | DP 10k |
|---|---:|---:|
| Success episodes | 1/5 | 2/5 |
| Success rate | 20% | 40% |
| Final goal distance mean | 0.0778m | 0.0578m |
| Final goal distance range | 0.0344–0.1087m | 0.0326–0.1115m |
| Maximum lateral deviation mean | 0.0795m | 0.0520m |
| Block path length mean | 0.2603m | 0.2048m |

The 10k checkpoint succeeded in episodes 0 and 4 at steps 62 and 149
(3.10s and 7.45s). The other episodes timed out at 600 steps:

- episode 1: final distance `0.0510m`, y=`+0.0490m`
- episode 2: final distance `0.1115m`, y=`+0.1101m`
- episode 3: final distance `0.0604m`, x=`0.8181`, y=`+0.0576m`

## Interpretation

Increasing training from 5k to 10k improved success rate from 20% to 40%,
reduced average lateral deviation, and shortened the average path. This is
evidence that the 5k DP model was partly optimization-limited. It is not yet a
reliable policy: three of five fixed-condition rollouts still fail, mainly from
lateral drift and occasional x overshoot.

The comparison remains a deterministic fixed-initial-condition evaluation, not
a generalization benchmark. DP 15k is the next planned ablation; keep all other
training and evaluation settings unchanged.

Source log: `outputs/eval/dp_fr3_balanced_140_10k/debug_steps.csv`.
