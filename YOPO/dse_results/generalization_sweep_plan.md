# Generalization Sweep Plan (deferred to follow-up training round)

Round 1 reviewer ranked this the second-most-important fix. It needs GPU
training time we did not have this session. The plan below is ready to
launch when GPU time is available.

## Goal

Test whether the physics-informed loss generalizes across cable lengths
and payload masses without per-config retraining, and across one new
environment family.

## Held-out grid

| Axis | Train values | Test values |
|------|--------------|-------------|
| Cable length L (m) | 0.8 (current) | 0.5, 1.0, 1.2 |
| Payload mass m_L (kg) | 0.3 (current) | 0.15, 0.45 |
| Environment family | forest | forest (held-out maps) + indoor-pillars |

Held-out combinations: 9 (3 lengths × 3 masses) per env family × 2 env
families = 18 settings. Each setting evaluated zero-shot from the existing
P1_wd75_wa20 checkpoint, no retraining.

## Two retraining variants for ablating "physics generalization"

1. **Single-config baseline (already trained)**: P1_wd75_wa20, trained on
   L=0.8, m=0.3 only. Tests whether the network has memorized that pair.
2. **Multi-config training**: re-run train_yopo.py with cable length and
   payload mass *sampled* per minibatch from the train grid above.
   Hypothesis: this should generalize better, especially because the
   pendulum loss makes the physical parameters explicit to the gradient.

Compute budget: 2 × ~290 min on RTX 4060 Ti = ~10 GPU-hours.

## Metrics to log

Per setting (L, m, env):
- Success rate (collision-free goal reach), n=200 episodes
- Mean / P90 peak swing
- Mean RMS swing
- Mean time-to-goal
- Distribution of segment safety failures (peak_swing > 60°)

## Expected reviewer-aimed plot

Two curves on one chart: peak-swing vs cable length, one for the
single-config model and one for the multi-config model, with the
swing-aware MPC baseline as a third curve. If the multi-config model is
flat across L while the single-config curve degrades, we have a clean
"physics-informed loss generalizes" story.

## Indoor-pillars environment

Reuse `Simulator/src/dataset_generator.cpp` with pillar_radius = 0.4m,
spacing 2.5–4.0 m, height 4 m, density 0.15 obstacles/m². This puts the
scene into a regime where the bubble-chain safety term must do real work
(narrow gaps).

## Launch script (placeholder)

Add to `run_all_experiments.py` once GPU time is allocated:

```bash
# Multi-config training
python train_yopo.py --trial 70 --multi_config_sampling 1
# Held-out evaluation
python evaluate_success_rate.py --trial 70 --eval_grid generalization_grid.yaml
```
