"""
Success rate evaluation for YOPO-Payload.
Success = reach goal AND no collision (UAV+cable+payload).

Combines:
1. Multi-segment episode simulation (does the UAV reach the goal?)
2. ESDF collision checking along each segment (bubble-chain model)

Usage:
    python evaluate_success_rate.py --checkpoint saved/YOPO_43/epoch50.pth --n_episodes 200
"""

import os, sys, json, argparse
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.yopo_dataset import YOPODataset
from policy.state_transform import state_body2world
from policy.pendulum_simulator import simulate_pendulum
from evaluate_closed_loop import extract_acc_from_polynomial
from loss.loss_function import YOPOLoss
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader


def check_collision_along_segment(
    pos_start, pos_end, acc_start, acc_end, p_params,
    safety_loss, map_id, device, n_eval=20,
):
    """
    Check collision along a trajectory segment using the bubble-chain model.
    Returns: (has_collision, min_clearance)
    """
    sgm_time = cfg["sgm_time"]
    r_uav = safety_loss.r_uav
    r_load = safety_loss.r_load

    pos_start_t = torch.tensor(pos_start, dtype=torch.float32, device=device)
    pos_end_t = torch.tensor(pos_end, dtype=torch.float32, device=device)
    acc_start_t = torch.tensor(acc_start, dtype=torch.float32, device=device)
    acc_end_t = torch.tensor(acc_end, dtype=torch.float32, device=device)

    t_samples = torch.linspace(0, 1, n_eval, device=device)

    # Linear interpolation of position and acceleration
    traj_pos = pos_start_t.unsqueeze(0) * (1 - t_samples.unsqueeze(1)) + \
               pos_end_t.unsqueeze(0) * t_samples.unsqueeze(1)
    traj_acc = acc_start_t.unsqueeze(0) * (1 - t_samples.unsqueeze(1)) + \
               acc_end_t.unsqueeze(0) * t_samples.unsqueeze(1)

    # Cable direction from apparent gravity
    g_vec = torch.tensor([0.0, 0.0, -9.81], device=device)
    apparent_g = traj_acc - g_vec.unsqueeze(0)
    q_vec = -apparent_g / (torch.norm(apparent_g, dim=-1, keepdim=True) + 1e-5)

    cable_L = float(p_params[0])
    n_bubbles = max(3, int(cable_L / 0.3) + 1)
    ratios = torch.linspace(0, 1, n_bubbles, device=device)

    # Bubble positions [n_eval, n_bubbles, 3]
    bubble_positions = traj_pos.unsqueeze(1) + \
        ratios.view(1, -1, 1) * (cable_L * q_vec.unsqueeze(1))

    # Query ESDF
    sdf_map = safety_loss.sdf_maps[map_id]
    min_bound = safety_loss.min_bounds[map_id]

    all_pos = bubble_positions.reshape(-1, 3)
    grid_idx = ((all_pos - min_bound) / safety_loss.voxel_size).long()

    for d in range(3):
        grid_idx[:, d] = grid_idx[:, d].clamp(0, int(sdf_map.shape[d + 2]) - 1)

    sdf_values = sdf_map[0, 0,
        grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]]
    sdf_values = sdf_values.reshape(n_eval, n_bubbles)

    bubble_radii = r_uav + ratios * (r_load - r_uav)
    clearance = sdf_values - bubble_radii.unsqueeze(0)

    min_clearance = clearance.min().item()
    has_collision = min_clearance < 0

    return has_collision, min_clearance


def run_episode(
    model, device, obs_dim, val_ds, safety_loss,
    start_idx, n_segments=12, goal_dist_threshold=2.0,
):
    """
    Run a multi-segment episode.
    Returns dict with: goal_reached, any_collision, peak_swing, etc.
    """
    sgm_time = cfg["sgm_time"]
    acc_max = cfg["acc_max_train"]

    # Get starting sample
    depth, pos_t, rot_t, obs_b_t, p_state_t, p_params_t, map_id_t = val_ds[start_idx]

    if isinstance(pos_t, np.ndarray):
        pos_t = torch.from_numpy(pos_t)
        rot_t = torch.from_numpy(rot_t)
        obs_b_t = torch.from_numpy(obs_b_t)
        p_state_t = torch.from_numpy(p_state_t)
        p_params_t = torch.from_numpy(p_params_t)
        map_id_t = torch.tensor(map_id_t) if isinstance(map_id_t, (int, float)) else torch.from_numpy(map_id_t)
        depth = torch.from_numpy(depth)

    pos_w = pos_t.numpy().copy()
    rot_mat = rot_t.numpy().copy()
    p_state = p_state_t.numpy().copy()
    p_params = p_params_t.numpy().copy()
    map_id = int(map_id_t) if isinstance(map_id_t, torch.Tensor) else int(map_id_t)

    # Goal from observation (body frame → world)
    goal_b = obs_b_t.numpy()[6:9]
    goal_w = rot_mat @ goal_b + pos_w

    # Initial velocity from obs
    vel_b = obs_b_t.numpy()[0:3]
    vel_w = rot_mat @ vel_b
    acc_b = obs_b_t.numpy()[3:6]
    acc_w = rot_mat @ acc_b

    all_peak_swings = []
    any_collision = False
    collision_count = 0
    total_segments = 0

    for seg in range(n_segments):
        # Find nearest depth image (reuse from dataset for now)
        # In real closed-loop, this comes from the sensor
        nearest_idx = start_idx  # simplification: reuse same depth

        depth_seg, _, _, _, _, _, mid_seg = val_ds[nearest_idx]
        if isinstance(depth_seg, np.ndarray):
            depth_seg = torch.from_numpy(depth_seg)

        # Build body-frame observation
        rot_inv = rot_mat.T  # orthogonal, so inverse = transpose
        vel_b = (rot_inv @ vel_w).astype(np.float32)
        acc_b = (rot_inv @ acc_w).astype(np.float32)
        goal_dir = goal_w - pos_w
        goal_b = (rot_inv @ goal_dir).astype(np.float32)

        obs_9d = np.concatenate([vel_b, acc_b, goal_b]).astype(np.float32)

        if obs_dim == 13:
            obs = np.concatenate([obs_9d, p_state.astype(np.float32)])
        elif obs_dim == 15:
            obs = np.concatenate([obs_9d, p_state.astype(np.float32), p_params.astype(np.float32)])
        elif obs_dim == 9:
            obs = obs_9d
        else:
            obs = obs_9d

        # Inference
        depth_t = depth_seg.unsqueeze(0).to(device)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)

        with torch.inference_mode():
            endstate, score = model.inference(depth_t, obs_t)

        # Pick best trajectory
        scores = score[0].reshape(-1).cpu().numpy()
        best_idx = np.argmin(scores)

        endstate_np = endstate[0].permute(1, 2, 0).reshape(cfg["traj_num"], 9).cpu().numpy()
        best_end = endstate_np[best_idx]

        # Transform to world
        end_pos_w = rot_mat @ best_end[0:3] + pos_w
        end_vel_w = rot_mat @ best_end[3:6]
        end_acc_w = rot_mat @ best_end[6:9]

        # Check collision along this segment
        has_collision, min_cl = check_collision_along_segment(
            pos_w, end_pos_w, acc_w, end_acc_w,
            p_params, safety_loss, map_id, device)

        if has_collision:
            any_collision = True
            collision_count += 1

        # Compute swing for this segment
        start_sw = np.array([pos_w, vel_w, acc_w])
        end_sw = np.array([end_pos_w, end_vel_w, end_acc_w])
        t_arr, acc_profile = extract_acc_from_polynomial(start_sw, end_sw, sgm_time)
        acc_clamped = np.clip(acc_profile, -acc_max, acc_max)

        L = p_params[0]
        try:
            acc_interp = interp1d(t_arr, acc_clamped, axis=0, bounds_error=False,
                                  fill_value=(acc_clamped[0], acc_clamped[-1]))
            t_ode, y_ode = simulate_pendulum(
                y0=list(p_state), L=L, T=sgm_time,
                acc_func=lambda t: acc_interp(t),
                dt=max(sgm_time / 200, 0.005))
            seg_peak = float(np.degrees(np.max(np.abs(y_ode[:, 0]))))
            all_peak_swings.append(seg_peak)
            p_state = y_ode[-1].copy()  # propagate pendulum state
        except Exception:
            all_peak_swings.append(float('nan'))

        total_segments += 1

        # Update UAV state
        pos_w = end_pos_w.copy()
        vel_w = end_vel_w.copy()
        acc_w = end_acc_w.copy()

        # Update orientation (align with velocity)
        vel_norm_xy = np.linalg.norm(vel_w[:2])
        if vel_norm_xy > 0.5:
            yaw = np.arctan2(vel_w[1], vel_w[0])
            rot_mat = R.from_euler('z', yaw).as_matrix().astype(np.float32)

        # Check if goal reached
        if np.linalg.norm(goal_w - pos_w) < goal_dist_threshold:
            break

    goal_reached = np.linalg.norm(goal_w - pos_w) < goal_dist_threshold
    peaks_valid = [p for p in all_peak_swings if not np.isnan(p)]

    return {
        "goal_reached": goal_reached,
        "any_collision": any_collision,
        "success": goal_reached and not any_collision,
        "n_segments": total_segments,
        "collision_count": collision_count,
        "goal_distance_remaining": float(np.linalg.norm(goal_w - pos_w)),
        "episode_peak_swing_deg": float(np.max(peaks_valid)) if peaks_valid else float('nan'),
        "episode_mean_peak_swing_deg": float(np.mean(peaks_valid)) if peaks_valid else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="saved/YOPO_43/epoch50.pth")
    parser.add_argument("--obs_dim", type=int, default=13)
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--n_segments", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="saved/YOPO_43/success_rate_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = YopoNetwork(observation_dim=args.obs_dim).to(device)
    sd = torch.load(args.checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    # Load safety loss for ESDF access
    yopo_loss = YOPOLoss()
    safety_loss = yopo_loss.safety_loss
    print("ESDF maps loaded")

    # Load validation dataset
    np.random.seed(args.seed)
    val_ds = YOPODataset(mode='valid')
    print(f"Validation set: {len(val_ds)} samples")

    # Sample starting indices spread across dataset
    n_val = len(val_ds)
    start_indices = np.random.choice(n_val, size=args.n_episodes, replace=False)

    episodes = []
    for i, idx in enumerate(start_indices):
        result = run_episode(
            model, device, args.obs_dim, val_ds, safety_loss,
            start_idx=idx, n_segments=args.n_segments)
        episodes.append(result)

        if (i + 1) % 50 == 0:
            successes = [e["success"] for e in episodes]
            goals = [e["goal_reached"] for e in episodes]
            collisions = [e["any_collision"] for e in episodes]
            print(f"  [{i+1}/{args.n_episodes}] "
                  f"success={np.mean(successes):.1%}, "
                  f"goal_reached={np.mean(goals):.1%}, "
                  f"collision_free={1-np.mean(collisions):.1%}")

    # Aggregate
    successes = [e["success"] for e in episodes]
    goals_reached = [e["goal_reached"] for e in episodes]
    collision_free = [not e["any_collision"] for e in episodes]
    peaks = [e["episode_peak_swing_deg"] for e in episodes if not np.isnan(e["episode_peak_swing_deg"])]
    mean_peaks = [e["episode_mean_peak_swing_deg"] for e in episodes if not np.isnan(e["episode_mean_peak_swing_deg"])]
    remaining_dists = [e["goal_distance_remaining"] for e in episodes]
    seg_counts = [e["n_segments"] for e in episodes]

    summary = {
        "n_episodes": len(episodes),
        "success_rate": float(np.mean(successes)),
        "goal_reached_rate": float(np.mean(goals_reached)),
        "collision_free_rate": float(np.mean(collision_free)),
        "mean_episode_peak_swing_deg": float(np.mean(peaks)) if peaks else float('nan'),
        "std_episode_peak_swing_deg": float(np.std(peaks)) if peaks else float('nan'),
        "mean_episode_mean_peak_swing_deg": float(np.mean(mean_peaks)) if mean_peaks else float('nan'),
        "mean_goal_distance_remaining_m": float(np.mean(remaining_dists)),
        "mean_segments_used": float(np.mean(seg_counts)),
        "episodes": episodes,
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"Success Rate Evaluation — YOPO_43 ({args.obs_dim}D)")
    print(f"{'='*60}")
    print(f"Episodes:           {len(episodes)}")
    print(f"Segments per ep:    {args.n_segments} max")
    print(f"{'─'*60}")
    print(f"Goal reached:       {np.mean(goals_reached):.1%}  ({sum(goals_reached)}/{len(episodes)})")
    print(f"Collision free:     {np.mean(collision_free):.1%}  ({sum(collision_free)}/{len(episodes)})")
    print(f"SUCCESS RATE:       {np.mean(successes):.1%}  ({sum(successes)}/{len(episodes)})")
    print(f"{'─'*60}")
    print(f"Mean peak swing:    {np.mean(peaks):.1f}° ± {np.std(peaks):.1f}°")
    print(f"Mean avg swing:     {np.mean(mean_peaks):.1f}°")
    print(f"Avg segments used:  {np.mean(seg_counts):.1f}")
    print(f"Avg remaining dist: {np.mean(remaining_dists):.1f} m")
    print(f"{'='*60}")

    # Breakdown: why failures happen
    failed = [e for e in episodes if not e["success"]]
    if failed:
        fail_collision_only = sum(1 for e in failed if e["goal_reached"] and e["any_collision"])
        fail_goal_only = sum(1 for e in failed if not e["goal_reached"] and not e["any_collision"])
        fail_both = sum(1 for e in failed if not e["goal_reached"] and e["any_collision"])
        print(f"\nFailure breakdown ({len(failed)} failures):")
        print(f"  Collision only (reached goal but hit obstacle): {fail_collision_only}")
        print(f"  Goal only (no collision but didn't reach goal): {fail_goal_only}")
        print(f"  Both (collision + didn't reach goal):           {fail_both}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
