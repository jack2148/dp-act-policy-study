# Diffusion Policy 20k evaluation

The DP 20k checkpoint was evaluated on the same balanced-140 dataset and the
same five fixed initial conditions as the 5k and 10k checkpoints.

| Metric | DP 5k | DP 10k | DP 20k |
|---|---:|---:|---:|
| Success episodes | 1/5 | 2/5 | 4/5 |
| Success rate | 20% | 40% | 80% |
| Final goal distance mean | 0.0778m | 0.0578m | 0.0406m |
| Maximum lateral deviation mean | 0.0795m | 0.0520m | 0.0526m |
| Block path length mean | 0.2603m | 0.2048m | 0.2204m |

Successful episodes reached the threshold at steps 152, 226, 275, and 277
(7.60s, 11.30s, 13.75s, and 13.85s). Episode 0 was the only failure: it
overshot to approximately `(0.804, 0.064)` and timed out with a final distance
of `0.0642m`.

## Interpretation

DP improves substantially with more training: success rises from 20% at 5k to
40% at 10k and 80% at 20k. This makes optimization insufficiency the strongest
current explanation for the 5k failure. The remaining failure is an overshoot
and lateral-drift case, so the next execution-horizon ablation should use this
20k checkpoint while keeping training fixed.

The evaluation is still a fixed-initial-condition rollout, not a generalization
benchmark. Action smoothness remains N/A because the logs do not contain all
seven action dimensions per step.

Source log: `outputs/eval/dp_fr3_balanced_140_20k/debug_steps.csv`.
