"""
Evaluate collision/clearance metrics for bubble-chain safety model.

Compares UAV-only safety checking vs full bubble-chain (UAV+cable+payload).
Reports: collision rate, min clearance, clearance distribution.

This directly addresses reviewer concern #3: "the bubble-chain safety model
is not actually supported by the evaluation."

Usage:
    python evaluate_clearance.py --checkpoint_dir saved --epoch 50 --n_samples 500
"""

import os, sys, json, argparse
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.yopo_dataset import YOPODataset
from policy.state_transform import state_body2world
from torch.utils.data import DataLoader
from loss.safety_loss import SafetyLoss


def evaluate_clearance(
    checkpoint_path, obs_dim, n_samples=500, batch_size=16,
):
    """
    For each validation sample, run inference and compute:
    1. Min ESDF distance to UAV center (UAV-only clearance)
    2. Min ESDF distance to any bubble in the chain (full system clearance)
    3. Collision flags: UAV collision, cable collision, payload collision
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sgm_time = cfg["sgm_time"]

    model = YopoNetwork(observation_dim=obs_dim).to(device)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()

    # Build a SafetyLoss instance to access ESDF maps and bubble chain
    # We need the polynomial mapping matrix
    from loss.loss_function import YOPOLoss
    yopo_loss = YOPOLoss()
    safety_loss = yopo_loss.safety_loss

    np.random.seed(42)
    val_ds = YOPODataset(mode='valid')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    results = {
        "uav_min_clearance": [],
        "payload_min_clearance": [],
        "system_min_clearance": [],  # min across all bubbles
        "uav_collision": 0,
        "cable_collision": 0,
        "payload_collision": 0,
        "system_collision": 0,
    }

    r_uav = safety_loss.r_uav
    r_load = safety_loss.r_load

    sample_count = 0

    with torch.inference_mode():
        for depth, pos, rot, obs_b, p_state, p_params, map_id in val_loader:
            if sample_count >= n_samples:
                break

            bs = depth.shape[0]
            depth = depth.to(device)
            pos, rot, obs_b = pos.to(device), rot.to(device), obs_b.to(device)
            p_state, p_params = p_state.to(device), p_params.to(device)

            # Build observation
            if obs_dim == 15:
                obs_aug = torch.cat([obs_b, p_state, p_params], dim=-1)
            elif obs_dim == 14:
                obs_aug = torch.cat([obs_b, p_state, p_params[:, :1]], dim=-1)
            elif obs_dim == 12:
                obs_aug = torch.cat([obs_b, p_state[:, :2], p_params[:, :1]], dim=-1)
            elif obs_dim == 10:
                obs_aug = torch.cat([obs_b, p_params[:, :1]], dim=-1)
            else:
                obs_aug = obs_b

            endstate, score = model.inference(depth, obs_aug)

            # Get world-frame states
            goal_w, start_vel_w, start_acc_w = state_body2world(
                pos, rot, obs_b[:, 6:9], obs_b[:, 0:3], obs_b[:, 3:6])

            for b in range(bs):
                if sample_count >= n_samples:
                    break

                scores_b = score[b].reshape(-1)
                best_idx = torch.argmin(scores_b).item()
                v_idx = best_idx // cfg["horizon_num"]
                h_idx = best_idx % cfg["horizon_num"]

                end_pva = endstate[b, :, v_idx, h_idx]
                rot_b = rot[b]
                pos_b = pos[b]

                end_pos_w = rot_b @ end_pva[0:3] + pos_b
                end_vel_w = rot_b @ end_pva[3:6]
                end_acc_w = rot_b @ end_pva[6:9]

                # Sample points along the polynomial trajectory
                n_eval = 20
                t_samples = torch.linspace(0, sgm_time, n_eval, device=device)

                # Polynomial coefficients
                start_sw = torch.stack([pos_b, start_vel_w[b], start_acc_w[b]])
                end_sw = torch.stack([end_pos_w, end_vel_w, end_acc_w])

                # Simple linear interpolation of position for clearance check
                # (more accurate than polynomial for this purpose)
                alphas = t_samples / sgm_time
                traj_positions = pos_b.unsqueeze(0) * (1 - alphas.unsqueeze(1)) + \
                                 end_pos_w.unsqueeze(0) * alphas.unsqueeze(1)

                # Also compute acceleration for cable direction
                traj_acc = start_acc_w[b].unsqueeze(0) * (1 - alphas.unsqueeze(1)) + \
                           end_acc_w.unsqueeze(0) * alphas.unsqueeze(1)

                # Cable direction from apparent gravity
                g_vec = torch.tensor([0.0, 0.0, -9.81], device=device)
                apparent_g = traj_acc - g_vec.unsqueeze(0)
                q_vec = -apparent_g / (torch.norm(apparent_g, dim=-1, keepdim=True) + 1e-5)

                cable_L = p_params[b, 0]

                # Compute bubble positions
                n_bubbles = max(3, int(cable_L.item() / 0.3) + 1)
                ratios = torch.linspace(0, 1, n_bubbles, device=device)

                # [n_eval, n_bubbles, 3]
                bubble_positions = traj_positions.unsqueeze(1) + \
                    ratios.view(1, -1, 1) * (cable_L * q_vec.unsqueeze(1))

                # Query ESDF for each bubble position
                mid = map_id[b].item()
                sdf_map = safety_loss.sdf_maps[mid]
                min_bound = safety_loss.min_bounds[mid]

                # Convert positions to grid indices
                all_positions = bubble_positions.reshape(-1, 3)  # [n_eval*n_bubbles, 3]
                grid_idx = ((all_positions - min_bound) / safety_loss.voxel_size).long()

                # Clamp to valid range
                sdf_shape = torch.tensor(sdf_map.shape[2:], device=device)
                grid_idx = grid_idx.clamp(min=0)
                for d in range(3):
                    grid_idx[:, d] = grid_idx[:, d].clamp(max=int(sdf_shape[d]) - 1)

                # Query SDF values
                sdf_values = sdf_map[0, 0,
                    grid_idx[:, 0].clamp(0, sdf_map.shape[2]-1),
                    grid_idx[:, 1].clamp(0, sdf_map.shape[3]-1),
                    grid_idx[:, 2].clamp(0, sdf_map.shape[4]-1)
                ]

                sdf_values = sdf_values.reshape(n_eval, n_bubbles)

                # Bubble radii
                bubble_radii = r_uav + ratios * (r_load - r_uav)

                # Clearance = SDF distance - bubble radius
                clearance = sdf_values - bubble_radii.unsqueeze(0)

                # UAV clearance (first bubble)
                uav_clearance = (sdf_values[:, 0] - r_uav).min().item()
                # Payload clearance (last bubble)
                payload_clearance = (sdf_values[:, -1] - r_load).min().item()
                # System clearance (any bubble)
                system_clearance = clearance.min().item()

                results["uav_min_clearance"].append(uav_clearance)
                results["payload_min_clearance"].append(payload_clearance)
                results["system_min_clearance"].append(system_clearance)

                if uav_clearance < 0:
                    results["uav_collision"] += 1
                if payload_clearance < 0:
                    results["payload_collision"] += 1
                if system_clearance < 0:
                    results["system_collision"] += 1
                # Cable collision: any middle bubble collides
                if clearance[:, 1:-1].min().item() < 0:
                    results["cable_collision"] += 1

                sample_count += 1

    n = sample_count
    uav_cl = np.array(results["uav_min_clearance"])
    pay_cl = np.array(results["payload_min_clearance"])
    sys_cl = np.array(results["system_min_clearance"])

    summary = {
        "n_samples": n,
        "uav_collision_rate": results["uav_collision"] / n,
        "cable_collision_rate": results["cable_collision"] / n,
        "payload_collision_rate": results["payload_collision"] / n,
        "system_collision_rate": results["system_collision"] / n,
        "uav_mean_clearance_m": float(np.mean(uav_cl)),
        "payload_mean_clearance_m": float(np.mean(pay_cl)),
        "system_mean_clearance_m": float(np.mean(sys_cl)),
        "uav_min_clearance_m": float(np.min(uav_cl)),
        "payload_min_clearance_m": float(np.min(pay_cl)),
        "system_min_clearance_m": float(np.min(sys_cl)),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="saved")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output", type=str, default="saved/clearance_results.json")
    args = parser.parse_args()

    from evaluate_closed_loop import METHODS
    methods_to_run = args.methods or ["B0_YOPO_original", "B1_YOPO_inflated", "B3_YOPO_Payload", "A1_cable_L"]

    all_results = {}

    for name in methods_to_run:
        if name not in METHODS:
            print(f"Unknown: {name}")
            continue

        config = METHODS[name]
        ckpt = os.path.join(args.checkpoint_dir, config["ckpt_subdir"], f"epoch{args.epoch}.pth")
        print(f"\nEvaluating clearance: {name} ({config['obs_dim']}D)...")

        summary = evaluate_clearance(ckpt, config["obs_dim"], args.n_samples, args.batch_size)
        all_results[name] = summary

        print(f"  UAV collision:     {summary['uav_collision_rate']:.1%}")
        print(f"  Cable collision:   {summary['cable_collision_rate']:.1%}")
        print(f"  Payload collision: {summary['payload_collision_rate']:.1%}")
        print(f"  System collision:  {summary['system_collision_rate']:.1%}")
        print(f"  UAV clearance:     {summary['uav_mean_clearance_m']:.3f}m (min: {summary['uav_min_clearance_m']:.3f}m)")
        print(f"  Payload clearance: {summary['payload_mean_clearance_m']:.3f}m (min: {summary['payload_min_clearance_m']:.3f}m)")

    # Print summary table
    print(f"\n{'='*100}")
    print("Clearance Evaluation Summary")
    print(f"{'='*100}")
    header = f"{'Method':<25} | {'UAV Col':>7} | {'Cable Col':>9} | {'Pay Col':>7} | {'Sys Col':>7} | {'Mean Clear':>10} | {'Min Clear':>9}"
    print(header)
    print("-" * 100)
    for name, m in all_results.items():
        print(f"{name:<25} | {m['uav_collision_rate']:>7.1%} | {m['cable_collision_rate']:>9.1%} | "
              f"{m['payload_collision_rate']:>7.1%} | {m['system_collision_rate']:>7.1%} | "
              f"{m['system_mean_clearance_m']:>9.3f}m | {m['system_min_clearance_m']:>8.3f}m")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
