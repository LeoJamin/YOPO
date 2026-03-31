"""
Multi-segment closed-loop evaluation for YOPO-Payload.

Chains multiple trajectory segments together to simulate realistic flight,
propagating UAV state AND pendulum state across segments. This addresses
reviewer concerns about single-segment evaluation being too local.

Usage:
    python evaluate_multiseg.py --checkpoint_dir saved --epoch 50 --n_episodes 100
"""

import os, sys, argparse, json, csv
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.yopo_dataset import YOPODataset
from policy.state_transform import StateTransform, state_body2world
from policy.pendulum_simulator import simulate_pendulum
from policy.poly_solver import Poly5Solver
from evaluate_closed_loop import extract_acc_from_polynomial, METHODS
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R


def run_multiseg_episode(
    model, device, obs_dim, depth_images, poses, map_id,
    n_segments=10, sgm_time=None, acc_max=6.0,
):
    """Run a multi-segment episode, chaining trajectory segments with pendulum ODE."""
    if sgm_time is None:
        sgm_time = cfg["sgm_time"]

    n_poses = len(poses)
    rng = np.random.default_rng()

    # Pick starting pose
    start_idx = rng.integers(0, n_poses)
    state = poses[start_idx]
    pos_w = state[0:3].copy()
    q_wxyz = state[3:7].copy()
    p_state = state[7:11].copy()  # [theta, phi, dtheta, dphi]
    p_params = state[11:13].copy()  # [L, m]

    # Set goal ahead
    rot_wb = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    forward = rot_wb.as_matrix()[:, 0]
    goal_w = pos_w + forward * 40.0

    # Initial velocity
    vel_w = forward * 3.0
    acc_w = np.zeros(3, dtype=np.float32)

    all_peak_swings = []
    all_rms_swings = []
    cumulative_distance = 0.0

    # Pre-sort available image keys for fast nearest lookup
    available_keys = sorted(depth_images.keys())
    available_positions = poses[available_keys, 0:3] if max(available_keys) < len(poses) else None

    for seg in range(n_segments):
        # Find nearest depth image from available ones
        if available_positions is not None:
            dists = np.linalg.norm(available_positions - pos_w[None, :], axis=1)
            best_avail = np.argmin(dists)
            nearest_idx = available_keys[best_avail]
        else:
            dists = np.linalg.norm(poses[:, 0:3] - pos_w[None, :], axis=1)
            nearest_idx = int(np.argmin(dists))

        # Load depth image
        depth = depth_images.get(nearest_idx)
        if depth is None:
            break

        # Update orientation from nearest pose for depth image alignment
        q_nearest = poses[nearest_idx, 3:7]
        rot_wb = R.from_quat([q_nearest[1], q_nearest[2], q_nearest[3], q_nearest[0]])

        # Build body-frame observation
        rot_inv = rot_wb.inv()
        vel_b = rot_inv.apply(vel_w).astype(np.float32)
        acc_b = rot_inv.apply(acc_w).astype(np.float32)
        goal_dir = goal_w - pos_w
        goal_b = rot_inv.apply(goal_dir).astype(np.float32)

        obs_9d = np.concatenate([vel_b, acc_b, goal_b])
        full_payload = np.concatenate([p_state, p_params])
        if obs_dim >= 10:
            payload_obs = full_payload[:obs_dim - 9]
            obs = np.concatenate([obs_9d, payload_obs]).astype(np.float32)
        else:
            obs = obs_9d.astype(np.float32)

        # Inference
        depth_t = torch.from_numpy(depth).unsqueeze(0).to(device)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            endstate, score = model.inference(depth_t, obs_t)

        endstate_np = endstate[0].permute(1, 2, 0).reshape(15, 9).cpu().numpy()
        score_np = score[0].reshape(15).cpu().numpy()
        best_idx = np.argmin(score_np)
        best_end = endstate_np[best_idx]

        # Transform to world
        rot_mat = rot_wb.as_matrix().astype(np.float32)
        end_pos_w = rot_mat @ best_end[0:3] + pos_w
        end_vel_w = rot_mat @ best_end[3:6]
        end_acc_w = rot_mat @ best_end[6:9]

        # Extract acceleration profile from polynomial
        start_sw = np.array([pos_w, vel_w, acc_w])
        end_sw = np.array([end_pos_w, end_vel_w, end_acc_w])
        t_arr, acc_profile = extract_acc_from_polynomial(start_sw, end_sw, sgm_time)

        # Clamp to physical limits
        acc_profile = np.clip(acc_profile, -acc_max, acc_max)

        # Integrate pendulum ODE for this segment
        L = p_params[0]
        try:
            from scipy.interpolate import interp1d
            acc_interp = interp1d(t_arr, acc_profile, axis=0, bounds_error=False,
                                  fill_value=(acc_profile[0], acc_profile[-1]))

            t_out, y_out = simulate_pendulum(
                y0=list(p_state), L=L, T=sgm_time,
                acc_func=lambda t: acc_interp(t),
                dt=max(sgm_time / 200, 0.005),
            )

            thetas = y_out[:, 0]
            seg_peak = float(np.degrees(np.max(np.abs(thetas))))
            seg_rms = float(np.degrees(np.sqrt(np.mean(thetas**2))))
            all_peak_swings.append(seg_peak)
            all_rms_swings.append(seg_rms)

            # Propagate pendulum state to next segment
            p_state = y_out[-1].copy()
        except Exception:
            all_peak_swings.append(float('nan'))
            all_rms_swings.append(float('nan'))

        # Update UAV state for next segment
        cumulative_distance += np.linalg.norm(end_pos_w - pos_w)
        pos_w = end_pos_w.copy()
        vel_w = end_vel_w.copy()
        acc_w = end_acc_w.copy()

        # Update orientation
        vel_norm = np.linalg.norm(vel_w[:2])
        if vel_norm > 0.5:
            yaw = np.arctan2(vel_w[1], vel_w[0])
            pitch = np.arctan2(-vel_w[2], vel_norm) * 0.1
            rot_wb = R.from_euler('ZYX', [yaw, pitch, 0])

        # Check goal reached
        if np.linalg.norm(goal_w - pos_w) < 2.0:
            break

    peaks = [p for p in all_peak_swings if not np.isnan(p)]
    rms_vals = [r for r in all_rms_swings if not np.isnan(r)]

    return {
        "n_segments": len(peaks),
        "episode_peak_swing_deg": float(np.max(peaks)) if peaks else float('nan'),
        "episode_mean_peak_swing_deg": float(np.mean(peaks)) if peaks else float('nan'),
        "episode_rms_swing_deg": float(np.sqrt(np.mean([r**2 for r in rms_vals]))) if rms_vals else float('nan'),
        "total_distance_m": float(cumulative_distance),
        "goal_reached": np.linalg.norm(goal_w - pos_w) < 2.0,
    }


def preload_depth_images(map_id, max_images=2000):
    """Preload depth images into memory for fast access."""
    import cv2
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
    img_dir = os.path.join(dataset_dir, str(map_id))
    images = {}
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith('.png'):
            continue
        idx = int(fname.split('_')[1].split('.')[0])
        if len(images) >= max_images:
            break
        img = cv2.imread(os.path.join(img_dir, fname), -1).astype(np.float32)
        img = cv2.resize(img, (int(cfg["image_width"]), int(cfg["image_height"])),
                         interpolation=cv2.INTER_NEAREST)
        images[idx] = np.expand_dims(img / 65535.0, axis=0)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="saved")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--n_segments", type=int, default=10)
    parser.add_argument("--map_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output", type=str, default="saved/multiseg_results.csv")
    args = parser.parse_args()

    import pandas as pd
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
    df = pd.read_csv(os.path.join(dataset_dir, f"pose-{args.map_id}.csv"))
    while df.shape[1] < 13:
        df[f'_col{df.shape[1]}'] = 0.3 if df.shape[1] == 12 else 0.8
    poses = df.iloc[:, :13].values.astype(np.float32)

    print(f"Preloading depth images for map {args.map_id}...")
    depth_images = preload_depth_images(args.map_id)
    print(f"  Loaded {len(depth_images)} images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods_to_run = args.methods or ["B0_YOPO_original", "B3_YOPO_Payload", "A1_cable_L"]

    all_results = {}

    for name in methods_to_run:
        if name not in METHODS:
            print(f"Unknown: {name}, skipping")
            continue

        config = METHODS[name]
        ckpt = os.path.join(args.checkpoint_dir, config["ckpt_subdir"], f"epoch{args.epoch}.pth")
        if not os.path.exists(ckpt):
            print(f"  Checkpoint not found: {ckpt}, skipping")
            continue

        model = YopoNetwork(observation_dim=config["obs_dim"]).to(device)
        state_dict = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"\nEvaluating {name} ({config['obs_dim']}D), {args.n_episodes} episodes × {args.n_segments} segments...")

        np.random.seed(args.seed)
        episodes = []
        for ep in range(args.n_episodes):
            result = run_multiseg_episode(
                model, device, config["obs_dim"],
                depth_images, poses, args.map_id,
                n_segments=args.n_segments,
            )
            episodes.append(result)

        # Aggregate
        peaks = [e["episode_mean_peak_swing_deg"] for e in episodes if not np.isnan(e["episode_mean_peak_swing_deg"])]
        rms_vals = [e["episode_rms_swing_deg"] for e in episodes if not np.isnan(e["episode_rms_swing_deg"])]
        max_peaks = [e["episode_peak_swing_deg"] for e in episodes if not np.isnan(e["episode_peak_swing_deg"])]

        summary = {
            "n_episodes": len(episodes),
            "mean_peak_swing_deg": float(np.mean(peaks)) if peaks else float('nan'),
            "std_peak_swing_deg": float(np.std(peaks)) if peaks else float('nan'),
            "max_peak_swing_deg": float(np.mean(max_peaks)) if max_peaks else float('nan'),
            "mean_rms_swing_deg": float(np.mean(rms_vals)) if rms_vals else float('nan'),
            "std_rms_swing_deg": float(np.std(rms_vals)) if rms_vals else float('nan'),
        }
        all_results[name] = summary
        print(f"  {name}: mean_peak={summary['mean_peak_swing_deg']:.1f}° "
              f"rms={summary['mean_rms_swing_deg']:.1f}°")

    # Print summary
    print(f"\n{'='*90}")
    print("Multi-Segment Evaluation Summary")
    print(f"{'='*90}")
    for name, m in all_results.items():
        print(f"{name:<25} | peak={m['mean_peak_swing_deg']:>5.1f}±{m['std_peak_swing_deg']:.1f}° "
              f"| rms={m['mean_rms_swing_deg']:>5.1f}±{m['std_rms_swing_deg']:.1f}°")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output.replace(".csv", ".json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output.replace('.csv', '.json')}")


if __name__ == "__main__":
    main()
