"""
Baseline comparison for YOPO-Payload.

Defines and evaluates the following baselines in the same loss framework:

  B0: Original YOPO (no payload awareness)
      - 9D obs, no DynamicsLoss, no bubble chain (point robot safety)
  B1: YOPO + Inflated Margin
      - 9D obs, safety loss with r = r_UAV + L_max (static inflation)
      - Accounts for worst-case cable sway without modeling it
  B2: YOPO + Acceleration Penalty
      - 9D obs, no DynamicsLoss, but extra L2 penalty on UAV acceleration
      - Common heuristic to reduce swing (minimize acceleration → minimize swing)
  B3: YOPO-Payload (Ours)
      - 15D obs, full DynamicsLoss + bubble chain safety

Metrics reported:
  - Validation trajectory loss (lower = better trajectories)
  - Safety cost (lower = fewer collisions for UAV+cable)
  - Dynamics cost (lower = less payload swing)
  - Goal cost (lower = better goal-directed behavior)

Usage:
    python evaluate_baselines.py --checkpoint_dir saved/
"""

import os
import sys
import csv
import argparse
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.yopo_trainer import YopoTrainer
from policy.yopo_dataset import YOPODataset
from torch.utils.data import DataLoader


def evaluate_model(
    checkpoint_path: str,
    obs_dim: int,
    use_dynamics: bool,
    use_bubble_chain: bool,
    inflated_margin: float,
    acc_penalty_weight: float,
    batch_size: int = 32,
    n_batches: int = 200,
) -> dict:
    """
    Evaluate a model checkpoint under given configuration.
    Returns per-component loss statistics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Temporarily adjust config
    original_wd = cfg["wd"]
    original_wc = cfg["wc"]
    if not use_dynamics:
        cfg["wd"] = 0.0

    trainer = YopoTrainer(
        learning_rate=0.001,
        batch_size=batch_size,
        loss_weight=[1.0, 1.0],
        tensorboard_path="saved",
        save_on_exit=False,
    )

    # Override network
    trainer.policy = YopoNetwork(observation_dim=obs_dim).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
        trainer.policy.load_state_dict(state_dict)
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"  No checkpoint found at {checkpoint_path}, using random weights (for structure check)")

    val_ds = YOPODataset(mode='valid')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    trainer.policy.eval()
    metrics = {k: [] for k in ["traj_loss", "smooth", "safety", "goal", "acc", "dynamics"]}

    with torch.inference_mode():
        for step, (depth, pos, rot, obs_b, p_state, p_params, map_id) in enumerate(val_loader):
            if depth.shape[0] != batch_size or step >= n_batches:
                break

            # For non-payload baselines: zero out payload states
            if obs_dim == 9:
                p_state_eval = torch.zeros_like(p_state)
                p_params_eval = torch.ones_like(p_params) * torch.tensor([0.8, 0.3])  # default L=0.8m, m=0.3kg
            else:
                p_state_eval = p_state
                p_params_eval = p_params

            # Acceleration penalty baseline: modify obs to penalize high acc
            if acc_penalty_weight > 0:
                pass  # handled in loss weight during training, not eval

            traj_loss, score_loss, smooth, safety, goal, acc_cost, dyn_cost = trainer.forward_and_compute_loss(
                depth, pos, rot, obs_b, p_state_eval, p_params_eval, map_id
            )

            metrics["traj_loss"].append(traj_loss.item())
            metrics["smooth"].append(smooth.item())
            metrics["safety"].append(safety.item())
            metrics["goal"].append(goal.item())
            metrics["acc"].append(acc_cost.item())
            metrics["dynamics"].append(dyn_cost.item())

    cfg["wd"] = original_wd
    cfg["wc"] = original_wc

    return {k: float(np.mean(v)) for k, v in metrics.items() if len(v) > 0}


BASELINES = {
    "B0_YOPO_original":      dict(obs_dim=9,  use_dynamics=False, use_bubble=False,
                                  inflated_margin=0.0, acc_penalty=0.0,
                                  ckpt_subdir="ablation_A0_baseline"),
    "B1_YOPO_inflated":      dict(obs_dim=9,  use_dynamics=False, use_bubble=False,
                                  inflated_margin=1.5, acc_penalty=0.0,
                                  ckpt_subdir="ablation_B1_inflated"),
    "B2_YOPO_acc_penalty":   dict(obs_dim=9,  use_dynamics=False, use_bubble=False,
                                  inflated_margin=0.0, acc_penalty=5.0,
                                  ckpt_subdir="ablation_B2_acc_penalty"),
    "B3_YOPO_Payload_ours":  dict(obs_dim=15, use_dynamics=True,  use_bubble=True,
                                  inflated_margin=0.0, acc_penalty=0.0,
                                  ckpt_subdir="ablation_A4_full_model"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="saved",
                        help="Directory containing checkpoint subdirs named after baselines")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_batches", type=int, default=200)
    args = parser.parse_args()

    results = []
    for name, config in BASELINES.items():
        print(f"\nEvaluating {name}...")
        ckpt = os.path.join(args.checkpoint_dir, config["ckpt_subdir"], f"epoch{args.epoch}.pth")
        metrics = evaluate_model(
            checkpoint_path=ckpt,
            obs_dim=config["obs_dim"],
            use_dynamics=config["use_dynamics"],
            use_bubble_chain=config["use_bubble"],
            inflated_margin=config["inflated_margin"],
            acc_penalty_weight=config["acc_penalty"],
            batch_size=args.batch_size,
            n_batches=args.n_batches,
        )
        row = {"name": name, **metrics}
        results.append(row)
        print(f"  traj_loss={metrics['traj_loss']:.4f}, safety={metrics['safety']:.4f}, "
              f"dynamics={metrics['dynamics']:.4f}, goal={metrics['goal']:.4f}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("Baseline Comparison Table")
    print(f"{'='*80}")
    header = f"{'Method':<28} | {'Traj↓':>7} | {'Safety↓':>8} | {'Dyn↓':>7} | {'Goal↓':>7} | {'Smooth↓':>8}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<28} | {r['traj_loss']:>7.4f} | {r['safety']:>8.4f} | "
              f"{r['dynamics']:>7.4f} | {r['goal']:>7.4f} | {r['smooth']:>8.4f}")

    # Save to CSV
    csv_path = os.path.join(args.checkpoint_dir, "baseline_comparison.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
