"""
Iterative swing-optimization training for YOPO-Payload.

Strategy:
  Iter 1 (YOPO_44): wd 8→16, score_dyn_boost 2→3       (stronger swing penalty)
  Iter 2 (YOPO_45): wd 16→25, jerk_coeff 0.15→0.30     (heavier penalty + jerk)
  Iter 3 (YOPO_46): wd 25→35, final_coeff 0.3→0.5      (aggressive swing suppression)

Each iteration:
  1. Modify config weights
  2. Fine-tune 30 epochs from previous best checkpoint
  3. Save results

Usage:
  python train_iterative.py --iter 1       # Run iteration 1 only
  python train_iterative.py --iter 1 2 3   # Run all 3 iterations
  python train_iterative.py --iter 2 --from_trial 44  # Iter 2 from YOPO_44
"""

import os
import sys
import yaml
import argparse
import random
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import cfg


def configure_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─── Iteration configs ───
ITERATIONS = {
    1: {
        "trial": 44,
        "from_trial": 43,
        "from_epoch": 50,
        "epochs": 30,
        "desc": "Increase dynamics weight: wd 8→16, score_dyn_boost 2→3",
        "config_changes": {
            "wd": 16.0,
        },
        "trainer_changes": {
            "score_dyn_boost": 3.0,
        },
    },
    2: {
        "trial": 45,
        "from_trial": 44,
        "from_epoch": 10,
        "epochs": 30,
        "desc": "Further increase: wd 16→25, jerk_coeff 0.15→0.30",
        "config_changes": {
            "wd": 25.0,
        },
        "trainer_changes": {
            "score_dyn_boost": 3.0,
            "jerk_coeff": 0.30,
        },
    },
    3: {
        "trial": 46,
        "from_trial": 45,
        "from_epoch": 30,
        "epochs": 30,
        "desc": "Aggressive: wd 25→35, final_coeff 0.3→0.5, jerk 0.30→0.40",
        "config_changes": {
            "wd": 35.0,
        },
        "trainer_changes": {
            "score_dyn_boost": 4.0,
            "jerk_coeff": 0.40,
            "final_coeff": 0.5,
        },
    },
}


def apply_config_changes(changes):
    """Apply config changes to the global cfg object and update YAML."""
    yaml_path = os.path.join(os.path.dirname(__file__), "config", "traj_opt.yaml")

    with open(yaml_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)

    for key, value in changes.items():
        yaml_cfg[key] = value
        cfg[key] = value
        print(f"  Config: {key} = {value}")

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_cfg, f, default_flow_style=False, sort_keys=False)


def apply_trainer_changes(trainer, changes):
    """Apply trainer-level changes (score_dyn_boost, loss formula coefficients)."""
    # These changes need to be applied in the trainer's compute_loss method
    # We'll monkey-patch the relevant values
    for key, value in changes.items():
        print(f"  Trainer: {key} = {value}")


def run_iteration(iter_id, from_trial_override=None):
    iter_cfg = ITERATIONS[iter_id]

    from_trial = from_trial_override or iter_cfg["from_trial"]
    from_epoch = iter_cfg["from_epoch"]
    trial = iter_cfg["trial"]
    epochs = iter_cfg["epochs"]

    print(f"\n{'='*70}")
    print(f"ITERATION {iter_id}: {iter_cfg['desc']}")
    print(f"  Fine-tune from YOPO_{from_trial}/epoch{from_epoch}")
    print(f"  → Save as YOPO_{trial} ({epochs} epochs)")
    print(f"{'='*70}")

    # 1. Apply config changes
    print("\nApplying config changes:")
    apply_config_changes(iter_cfg["config_changes"])

    # 2. Reload cfg to pick up changes
    # cfg is already updated in apply_config_changes

    # 3. Apply trainer changes by modifying the source files temporarily
    trainer_changes = iter_cfg.get("trainer_changes", {})

    # Modify score_dyn_boost in trainer
    if "score_dyn_boost" in trainer_changes:
        modify_score_dyn_boost(trainer_changes["score_dyn_boost"])

    # Modify pendulum loss coefficients
    if "jerk_coeff" in trainer_changes:
        modify_pendulum_coeff("jerk", trainer_changes["jerk_coeff"])
    if "final_coeff" in trainer_changes:
        modify_pendulum_coeff("final", trainer_changes["final_coeff"])

    # 4. Import trainer (after config changes)
    # Need to reload modules to pick up changes
    import importlib
    import policy.yopo_trainer
    importlib.reload(policy.yopo_trainer)
    from policy.yopo_trainer import YopoTrainer

    configure_random_seed(iter_id)

    log_dir = os.path.dirname(os.path.abspath(__file__)) + "/saved"
    checkpoint_path = f"{log_dir}/YOPO_{from_trial}/epoch{from_epoch}.pth"

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    trainer = YopoTrainer(
        learning_rate=1.0e-4,  # slightly lower LR for fine-tuning
        batch_size=16,
        loss_weight=[1.0, 1.0, 1.0],
        tensorboard_path=log_dir,
        checkpoint_path=checkpoint_path,
        save_on_exit=True,
    )

    print(f"\nStarting training: {epochs} epochs")
    print(f"  Dynamics weight (normalized): {trainer.yopo_loss.dynamics_weight:.4f}")
    print(f"  Safety weight: {trainer.yopo_loss.safety_weight:.4f}")
    print(f"  Goal weight: {trainer.yopo_loss.goal_weight:.4f}")

    trainer.train(epoch=epochs, save_interval=10)

    print(f"\nIteration {iter_id} complete! Model saved as YOPO_{trial}")


def modify_score_dyn_boost(new_value):
    """Modify score_dyn_boost in yopo_trainer.py."""
    trainer_path = os.path.join(os.path.dirname(__file__), "policy", "yopo_trainer.py")
    with open(trainer_path, 'r') as f:
        content = f.read()

    import re
    content = re.sub(
        r'score_dyn_boost\s*=\s*[\d.]+',
        f'score_dyn_boost = {new_value}',
        content
    )

    with open(trainer_path, 'w') as f:
        f.write(content)
    print(f"  Modified score_dyn_boost → {new_value}")


def modify_pendulum_coeff(coeff_type, new_value):
    """Modify coefficients in differentiable_pendulum.py."""
    pend_path = os.path.join(os.path.dirname(__file__), "loss", "differentiable_pendulum.py")
    with open(pend_path, 'r') as f:
        content = f.read()

    import re
    if coeff_type == "jerk":
        content = re.sub(
            r'swing_loss\s*=\s*swing_loss\s*\+\s*[\d.]+\s*\*\s*jerk_penalty',
            f'swing_loss = swing_loss + {new_value} * jerk_penalty',
            content
        )
        print(f"  Modified jerk_coeff → {new_value}")
    elif coeff_type == "final":
        content = re.sub(
            r'swing_loss\s*=\s*max_peak\s*\+\s*[\d.]+\s*\*\s*final_theta',
            f'swing_loss = max_peak + {new_value} * final_theta',
            content
        )
        print(f"  Modified final_coeff → {new_value}")

    with open(pend_path, 'w') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, nargs="+", default=[1],
                        help="Which iterations to run (1, 2, 3)")
    parser.add_argument("--from_trial", type=int, default=None,
                        help="Override source trial for fine-tuning")
    args = parser.parse_args()

    for it in args.iter:
        if it not in ITERATIONS:
            print(f"Unknown iteration {it}, skipping")
            continue
        run_iteration(it, args.from_trial)

    print("\n" + "="*70)
    print("All iterations complete!")
    print("="*70)


if __name__ == "__main__":
    main()
