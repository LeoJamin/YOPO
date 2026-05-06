"""
Ablation study runner for YOPO-Payload.

Runs the following ablations to justify each component of the 15D state extension:
  A0: Baseline YOPO (no payload, 9D obs, no DynamicsLoss)
  A1: +Cable length L only (10D obs)
  A2: +Swing angles theta,phi (12D obs)
  A3: +Angular velocities dtheta,dphi (14D obs)
  A4: Full model (15D obs with L,m, DynamicsLoss) — default

Each ablation is trained for the same number of epochs and evaluated on the same
validation set. Results are written to saved/ablation_results.csv.

Usage:
    python train_ablation.py --epochs 30 --batch_size 32
"""

import os
import sys
import csv
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.yopo_trainer import YopoTrainer
from policy.state_transform import state_body2world

# Ablation configuration: name -> (obs_dim, use_dynamics_loss, state_mask)
# state_mask: boolean mask over [theta, phi, dtheta, dphi, L, m] (6 extra dims)
ABLATIONS = {
    # --- Baselines (B0-B3) ---
    "A0_baseline":    dict(obs_dim=9,  use_dynamics=False, mask=[False]*6, detach_qvec=True,
                           wa_override=None, inflated_r=None),
    "B1_inflated":    dict(obs_dim=9,  use_dynamics=False, mask=[False]*6, detach_qvec=True,
                           wa_override=None, inflated_r=1.5),  # r += L_max
    "B2_acc_penalty": dict(obs_dim=9,  use_dynamics=False, mask=[False]*6, detach_qvec=True,
                           wa_override=5.0, inflated_r=None),  # increased acc penalty
    # --- State dimension ablations ---
    "A1_cable_L":     dict(obs_dim=10, use_dynamics=False, mask=[False, False, False, False, True, False], detach_qvec=True,
                           wa_override=None, inflated_r=None),
    "A2_swing_ang":   dict(obs_dim=12, use_dynamics=False, mask=[True, True, False, False, True, False], detach_qvec=True,
                           wa_override=None, inflated_r=None),
    "A3_swing_full":  dict(obs_dim=14, use_dynamics=False, mask=[True, True, True, True, True, False], detach_qvec=True,
                           wa_override=None, inflated_r=None),
    "A4_full_model":  dict(obs_dim=15, use_dynamics=True,  mask=[True, True, True, True, True, True], detach_qvec=True,
                           wa_override=None, inflated_r=None),
    # --- Critical ablations from reviewer feedback ---
    "A4_no_mass":     dict(obs_dim=14, use_dynamics=True,  mask=[True, True, True, True, True, False], detach_qvec=True,
                           wa_override=None, inflated_r=None),
    "D1_no_detach":   dict(obs_dim=15, use_dynamics=True,  mask=[True, True, True, True, True, True], detach_qvec=False,
                           wa_override=None, inflated_r=None),
    # --- DynamicsLoss isolation ablations (Round 1 review) ---
    "A1_cable_L_dyn": dict(obs_dim=10, use_dynamics=True,  mask=[False, False, False, False, True, False], detach_qvec=True,
                           wa_override=None, inflated_r=None),  # L + DynamicsLoss
    "A4_no_dyn":      dict(obs_dim=15, use_dynamics=False, mask=[True, True, True, True, True, True], detach_qvec=True,
                           wa_override=None, inflated_r=None),  # Full 15D, NO DynamicsLoss
    # 13D obs (theta, phi, dtheta, dphi only — no L, no m), DynamicsLoss OFF.
    # Matched-architecture ablation for YOPO_3 (the canonical 13D paper
    # model). Used in B2 of EXPERIMENT_PLAN to isolate the effect of the
    # dynamics loss while holding network architecture fixed.
    "A_yopo3_no_dyn": dict(obs_dim=13, use_dynamics=False, mask=[True, True, True, True, False, False], detach_qvec=True,
                           wa_override=None, inflated_r=None),
}


class AblationForwardMixin:
    """Override forward_and_compute_loss to support variable obs_dim via masking.

    The mask selects which of the 6 payload dims [theta, phi, dtheta, dphi, L, m]
    to include in the observation. The full p_state/p_params are always passed to
    the loss functions (they need L for bubble chain, etc.), but only masked dims
    enter the network input.
    """
    payload_mask = None  # Set by run_ablation()

    def forward_and_compute_loss(self, depth, pos, rot, obs_b, p_state, p_params, map_id):
        depth, pos, rot, obs_b, p_state, p_params, map_id = [
            x.to(self.device) for x in [depth, pos, rot, obs_b, p_state, p_params, map_id]]

        goal_w, start_vel_w, start_acc_w = state_body2world(
            pos, rot, obs_b[:, 6:9], obs_b[:, 0:3], obs_b[:, 3:6])
        start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)

        # Build obs_augmented with only the masked payload dims
        full_payload = torch.cat([p_state, p_params], dim=-1)  # [B, 6]
        if self.payload_mask is not None:
            mask_t = torch.tensor(self.payload_mask, dtype=torch.bool, device=self.device)
            masked_payload = full_payload[:, mask_t]  # [B, payload_dim]
        else:
            masked_payload = full_payload  # [B, 6]
        obs_augmented = torch.cat([obs_b, masked_payload], dim=-1)  # [B, 9+payload_dim]

        endstate, score = self.policy.inference(depth, obs_augmented)

        endstate_flat = endstate.permute(0, 2, 3, 1).reshape(self.batch_size * self.traj_num, 9)
        score_flat = score.reshape(self.batch_size * self.traj_num)

        pos_expanded = pos.repeat_interleave(self.traj_num, dim=0)
        rot_expanded = rot.repeat_interleave(self.traj_num, dim=0)
        start_state_w = start_state_w.repeat_interleave(self.traj_num, dim=0)
        goal_w = goal_w.repeat_interleave(self.traj_num, dim=0)
        p_state_expanded = p_state.repeat_interleave(self.traj_num, dim=0)
        p_params_expanded = p_params.repeat_interleave(self.traj_num, dim=0)

        end_pos_w, end_vel_w, end_acc_w = state_body2world(
            pos_expanded, rot_expanded,
            endstate_flat[:, 0:3], endstate_flat[:, 3:6], endstate_flat[:, 6:9])
        end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)

        smooth_cost, safety_cost, goal_cost, acc_cost, dyn_cost = self.yopo_loss(
            start_state_w, end_state_w, goal_w, map_id, p_state_expanded, p_params_expanded)

        w_smooth = self.yopo_loss.smoothness_weight
        w_safe = self.yopo_loss.safety_weight
        w_goal = self.yopo_loss.goal_weight
        w_acc = self.yopo_loss.accele_weight
        w_dyn = self.yopo_loss.dynamics_weight

        smooth_loss_scaled = w_smooth * smooth_cost.mean()
        safety_loss_scaled = w_safe * safety_cost.mean()
        goal_loss_scaled = w_goal * goal_cost.mean()
        acc_loss_scaled = w_acc * acc_cost.mean()
        dyn_loss_scaled = w_dyn * dyn_cost.mean()

        trajectory_loss = smooth_loss_scaled + safety_loss_scaled + goal_loss_scaled + acc_loss_scaled + dyn_loss_scaled

        score_label = (w_smooth * smooth_cost + w_safe * safety_cost +
                       w_goal * goal_cost + w_acc * acc_cost + w_dyn * dyn_cost).clone().detach()
        score_loss = torch.nn.functional.smooth_l1_loss(score_flat, score_label)

        return trajectory_loss, score_loss, smooth_loss_scaled, safety_loss_scaled, goal_loss_scaled, acc_loss_scaled, dyn_loss_scaled


def run_ablation(name: str, config: dict, epochs: int, batch_size: int, save_dir: str,
                  learning_rate: float = 1.5e-4, seed: int = 0):
    """Train and evaluate one ablation configuration."""
    print(f"\n{'='*60}")
    print(f"Running ablation: {name} | obs_dim={config['obs_dim']} | dynamics={config['use_dynamics']} | detach={config.get('detach_qvec', True)}")
    print(f"{'='*60}")

    # Set seed for reproducibility
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    obs_dim = config["obs_dim"]
    use_dynamics = config["use_dynamics"]
    mask = config["mask"]
    detach_qvec = config.get("detach_qvec", True)

    wa_override = config.get("wa_override")
    inflated_r = config.get("inflated_r")

    # Temporarily override config
    original_wd = cfg["wd"]
    original_wa = cfg["wa"]
    original_r = cfg["r"]
    original_detach = cfg._data.get("detach_qvec", True)
    if not use_dynamics:
        cfg["wd"] = 0.0
    cfg._data["detach_qvec"] = detach_qvec
    if wa_override is not None:
        cfg["wa"] = wa_override
    if inflated_r is not None:
        cfg["r"] = cfg["r"] + inflated_r  # inflate safety radius by cable length

    checkpoint_dir = os.path.join(save_dir, f"ablation_{name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        # Create a dynamic subclass that uses AblationForwardMixin
        class AblationTrainer(AblationForwardMixin, YopoTrainer):
            pass

        # Build trainer with correct obs_dim
        trainer = AblationTrainer(
            learning_rate=learning_rate,
            batch_size=batch_size,
            loss_weight=[1.0, 1.0, 1.0],
            tensorboard_path=save_dir,
            save_on_exit=False,
        )

        # Set the payload mask for the mixin
        trainer.payload_mask = mask

        # Override network observation dim
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer.policy = YopoNetwork(observation_dim=obs_dim).to(device)
        trainer.optimizer = torch.optim.AdamW(trainer.policy.parameters(), lr=learning_rate, fused=True)

        # Stronger gradient clipping for no-detach to prevent adversarial training
        if not detach_qvec:
            trainer.max_grad_norm = 1.0

        trainer.train(epochs, save_interval=epochs)

        # Evaluate final validation loss
        trainer.policy.eval()
        val_losses = []
        with torch.inference_mode():
            for step, (depth, pos, rot, obs_b, p_state, p_params, map_id) in enumerate(trainer.val_dataloader):
                if depth.shape[0] != batch_size:
                    continue
                traj_loss, score_loss, _, _, _, _, dyn_cost = trainer.forward_and_compute_loss(
                    depth, pos, rot, obs_b, p_state, p_params, map_id
                )
                val_losses.append((traj_loss + score_loss).item())
                if step >= 100:
                    break

        mean_val_loss = np.mean(val_losses)
        print(f"  Final val loss ({name}): {mean_val_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"epoch{epochs}.pth")
        torch.save(trainer.policy.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

    finally:
        # Restore config even if training crashes
        cfg["wd"] = original_wd
        cfg["wa"] = original_wa
        cfg["r"] = original_r
        cfg._data["detach_qvec"] = original_detach

    return {"name": name, "obs_dim": obs_dim, "use_dynamics": use_dynamics,
            "detach_qvec": detach_qvec, "final_val_loss": mean_val_loss,
            "checkpoint": ckpt_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1.5e-4)
    parser.add_argument("--save_dir", type=str, default="saved")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS.keys()),
                        help="Which ablations to run (default: all)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    results = []

    for name in args.ablations:
        if name not in ABLATIONS:
            print(f"Unknown ablation: {name}, skipping")
            continue
        result = run_ablation(name, ABLATIONS[name], args.epochs, args.batch_size,
                              args.save_dir, args.learning_rate, args.seed)
        results.append(result)

    # Save results table
    csv_path = os.path.join(args.save_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "obs_dim", "use_dynamics", "detach_qvec", "final_val_loss", "checkpoint"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print("Ablation Results Summary")
    print(f"{'='*60}")
    print(f"{'Ablation':<22} | {'Obs':>3} | {'Dyn':>5} | {'Detach':>6} | {'Val Loss':>9}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: x["final_val_loss"]):
        print(f"{r['name']:<22} | {r['obs_dim']:>3} | {str(r['use_dynamics']):>5} | {str(r['detach_qvec']):>6} | {r['final_val_loss']:>9.4f}")
    print(f"\nFull results saved to: {csv_path}")


if __name__ == "__main__":
    main()
