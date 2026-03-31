"""
Visualize YOPO-Payload policy flying through a forest environment.

Loads a trained checkpoint, runs multi-step inference on real dataset samples,
and plots the UAV trajectory with forest obstacles, candidate trajectories,
and payload cable.

Usage:
    python visualize_forest_flight.py                          # B3 (ours, 15D)
    python visualize_forest_flight.py --ablation A0_baseline   # B0 (9D)
    python visualize_forest_flight.py --map_id 0 --n_steps 30  # more steps
"""

import os, sys, argparse
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.state_transform import StateTransform, state_body2world
from policy.poly_solver import Poly5Solver

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "..", "dataset")
SAVE_DIR = os.path.join(SCRIPT_DIR, "saved")


def load_pointcloud(map_id):
    """Load forest point cloud and subsample for plotting."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(os.path.join(DATASET_DIR, f"pointcloud-{map_id}.ply"))
        pts = np.asarray(pcd.points)
    except ImportError:
        print("open3d not available, skipping point cloud")
        return None
    return pts


def load_depth_image(map_id, img_idx):
    """Load and normalize a depth image."""
    img_path = os.path.join(DATASET_DIR, str(map_id), f"img_{img_idx}.png")
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path, -1).astype(np.float32)
    img = cv2.resize(img, (int(cfg["image_width"]), int(cfg["image_height"])),
                     interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(img / 65535.0, axis=0)


def load_poses(map_id):
    """Load all poses and payload states from CSV."""
    import pandas as pd
    csv_path = os.path.join(DATASET_DIR, f"pose-{map_id}.csv")
    df = pd.read_csv(csv_path)
    while df.shape[1] < 13:
        df[f'_col{df.shape[1]}'] = 0.3 if df.shape[1] == 12 else 0.8
    return df.iloc[:, :13].values.astype(np.float32)


def quat_to_rotmat(q_wxyz):
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    rot = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return rot.as_matrix().astype(np.float32)


def get_body_frame_rotation(q_wxyz):
    """Get rotation from world to body-level frame (yaw-removed, for training).
    NOT suitable for closed-loop simulation — use get_full_body_rotation instead.
    """
    rot = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    euler = rot.as_euler('ZYX', degrees=False)
    R_Bw = R.from_euler('ZYX', [0, euler[1], euler[2]], degrees=False).inv()
    return R_Bw


def get_full_body_rotation(q_wxyz):
    """Get full world-to-body rotation (including yaw).
    For closed-loop simulation: goal/vel/acc must be in the same frame as the depth image.
    """
    rot_WB = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return rot_WB.inv()  # world → body


def run_multistep_flight(policy, device, map_id, obs_dim, n_steps=20, seed=42):
    """Run multi-step closed-loop simulation through the forest."""
    np.random.seed(seed)
    poses = load_poses(map_id)
    n_poses = len(poses)

    # Pick a starting pose
    start_idx = np.random.randint(0, n_poses)
    state = poses[start_idx]
    pos_w = state[0:3].copy()
    q_wxyz = state[3:7].copy()

    # Set a goal ~40m ahead in the forward direction
    rot_wb = quat_to_rotmat(q_wxyz)
    forward_dir = rot_wb[:, 0]  # x-axis of body frame in world
    goal_w = pos_w + forward_dir * 40.0

    vel_max = cfg["vel_max_train"]
    acc_max = cfg["acc_max_train"]
    sgm_time = cfg["sgm_time"]

    # Current velocity & acceleration (start from moderate forward flight)
    vel_w = forward_dir * 3.0  # 3 m/s forward
    acc_w = np.zeros(3, dtype=np.float32)

    # Payload state
    p_state = state[7:11].copy()  # theta, phi, dtheta, dphi
    p_params = state[11:13].copy()  # L, m

    trajectory_positions = [pos_w.copy()]
    trajectory_velocities = [vel_w.copy()]
    all_candidates = []  # store candidate trajectories at each step
    best_indices = []
    depth_images = []

    for step in range(n_steps):
        # Find nearest dataset sample for depth image
        dists = np.linalg.norm(poses[:, 0:3] - pos_w[None, :], axis=1)
        nearest_idx = np.argmin(dists)

        # Load depth image from nearest pose
        depth = load_depth_image(map_id, nearest_idx)
        if depth is None:
            # Try nearby indices
            for offset in range(-5, 6):
                depth = load_depth_image(map_id, nearest_idx + offset)
                if depth is not None:
                    break
        if depth is None:
            print(f"Step {step}: no depth image found near idx {nearest_idx}, stopping")
            break

        depth_images.append(depth[0, :, :])  # store for visualization

        # Build observation in body frame (full rotation including yaw)
        R_WB_inv = get_full_body_rotation(q_wxyz)  # world → body
        vel_b = R_WB_inv.apply(vel_w).astype(np.float32)
        acc_b = R_WB_inv.apply(acc_w).astype(np.float32)
        goal_dir_w = goal_w - pos_w
        goal_b = R_WB_inv.apply(goal_dir_w).astype(np.float32)

        # Build obs vector
        obs_9d = np.concatenate([vel_b, acc_b, goal_b])
        if obs_dim >= 10:
            payload_obs = np.concatenate([p_state, p_params])[:obs_dim - 9]
            obs = np.concatenate([obs_9d, payload_obs]).astype(np.float32)
        else:
            obs = obs_9d.astype(np.float32)

        # Run inference
        depth_t = torch.from_numpy(depth).unsqueeze(0).to(device)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            endstate, score = policy.inference(depth_t, obs_t)

        # endstate: [1, 9, 3, 5] -> [15, 9] in body frame
        endstate_np = endstate[0].permute(1, 2, 0).reshape(15, 9).cpu().numpy()
        score_np = score[0].reshape(15).cpu().numpy()

        # Select best trajectory (lowest score)
        best_idx = np.argmin(score_np)
        best_indices.append(best_idx)

        # Convert all candidate end states to world frame for visualization
        candidates_world = []
        for i in range(15):
            end_pos_b = endstate_np[i, 0:3]
            end_vel_b = endstate_np[i, 3:6]
            end_acc_b = endstate_np[i, 6:9]

            rot_mat = quat_to_rotmat(q_wxyz)
            end_pos_w = rot_mat @ end_pos_b + pos_w
            end_vel_w = rot_mat @ end_vel_b
            end_acc_w = rot_mat @ end_acc_b

            # Generate polynomial trajectory
            traj_pts = []
            t_samples = np.linspace(0, sgm_time, 20)
            poly_x = Poly5Solver(pos_w[0], vel_w[0], acc_w[0], end_pos_w[0], end_vel_w[0], end_acc_w[0], sgm_time)
            poly_y = Poly5Solver(pos_w[1], vel_w[1], acc_w[1], end_pos_w[1], end_vel_w[1], end_acc_w[1], sgm_time)
            poly_z = Poly5Solver(pos_w[2], vel_w[2], acc_w[2], end_pos_w[2], end_vel_w[2], end_acc_w[2], sgm_time)
            for t in t_samples:
                traj_pts.append([poly_x.get_position(t), poly_y.get_position(t), poly_z.get_position(t)])
            candidates_world.append(np.array(traj_pts))

        all_candidates.append((candidates_world, best_idx, score_np))

        # Advance state along best trajectory
        best_end = endstate_np[best_idx]
        rot_mat = quat_to_rotmat(q_wxyz)
        new_pos_w = rot_mat @ best_end[0:3] + pos_w
        new_vel_w = rot_mat @ best_end[3:6]
        new_acc_w = rot_mat @ best_end[6:9]

        pos_w = new_pos_w
        vel_w = new_vel_w
        acc_w = new_acc_w

        # Update orientation to face velocity direction
        vel_norm = np.linalg.norm(vel_w[:2])
        if vel_norm > 0.5:
            yaw = np.arctan2(vel_w[1], vel_w[0])
            pitch = np.arctan2(-vel_w[2], vel_norm) * 0.1  # slight pitch
            rot_new = R.from_euler('ZYX', [yaw, pitch, 0])
            q_xyzw = rot_new.as_quat()
            q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

        trajectory_positions.append(pos_w.copy())
        trajectory_velocities.append(vel_w.copy())

        # Simple payload dynamics update
        g = 9.81
        L = p_params[0]
        a_horiz = np.sqrt(acc_w[0]**2 + acc_w[1]**2)
        eq_theta = np.arctan2(a_horiz, g)
        p_state[0] = 0.8 * p_state[0] + 0.2 * eq_theta  # smooth towards equilibrium
        p_state[2] *= 0.9  # damp angular velocity
        p_state[3] *= 0.9

        # Check if reached goal
        if np.linalg.norm(goal_w - pos_w) < 2.0:
            print(f"Reached goal at step {step}!")
            break

    return {
        'trajectory': np.array(trajectory_positions),
        'velocities': np.array(trajectory_velocities),
        'candidates': all_candidates,
        'goal': goal_w,
        'start': trajectory_positions[0],
        'depth_images': depth_images,
    }


def plot_forest_flight(result, pts, map_id, obs_dim, ablation_name, output_path):
    """Create the multi-panel forest flight visualization."""
    traj = result['trajectory']
    candidates = result['candidates']
    goal = result['goal']
    start = result['start']
    depths = result['depth_images']

    fig = plt.figure(figsize=(20, 14))

    # --- Panel 1: 3D bird's eye view ---
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Plot forest obstacles (subsample for speed)
    if pts is not None:
        # Crop point cloud around trajectory
        traj_center = traj.mean(axis=0)
        traj_range = max(np.ptp(traj[:, 0]), np.ptp(traj[:, 1])) / 2 + 15
        mask = (np.abs(pts[:, 0] - traj_center[0]) < traj_range) & \
               (np.abs(pts[:, 1] - traj_center[1]) < traj_range) & \
               (pts[:, 2] > 0.1)  # filter ground
        local_pts = pts[mask]
        if len(local_pts) > 50000:
            idx = np.random.choice(len(local_pts), 50000, replace=False)
            local_pts = local_pts[idx]
        if len(local_pts) > 0:
            # Color by height
            z_norm = (local_pts[:, 2] - local_pts[:, 2].min()) / (local_pts[:, 2].max() - local_pts[:, 2].min() + 1e-6)
            colors = plt.cm.YlGn(0.3 + 0.7 * z_norm)
            colors[:, 3] = 0.15  # transparency
            ax1.scatter(local_pts[:, 0], local_pts[:, 1], local_pts[:, 2],
                       c=colors, s=0.3, rasterized=True)

    # Plot candidate trajectories at selected steps
    viz_steps = np.linspace(0, len(candidates) - 1, min(6, len(candidates))).astype(int)
    for step_idx in viz_steps:
        cands, best_idx, scores = candidates[step_idx]
        for i, cand in enumerate(cands):
            if i == best_idx:
                continue
            alpha = max(0.05, 0.3 - scores[i] / (scores.max() + 1e-6) * 0.25)
            ax1.plot(cand[:, 0], cand[:, 1], cand[:, 2],
                    color='steelblue', alpha=alpha, linewidth=0.5)

    # Plot best trajectory segments
    for step_idx in range(len(candidates)):
        cands, best_idx, scores = candidates[step_idx]
        best_traj = cands[best_idx]
        ax1.plot(best_traj[:, 0], best_traj[:, 1], best_traj[:, 2],
                color='orangered', alpha=0.6, linewidth=1.5)

    # Plot main trajectory
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2],
            color='red', linewidth=3, label='Selected path', zorder=10)
    ax1.scatter(*start, color='lime', s=100, marker='o', zorder=15, label='Start', edgecolors='black')
    ax1.scatter(*goal, color='gold', s=150, marker='*', zorder=15, label='Goal', edgecolors='black')

    # Draw payload cable at a few positions
    if obs_dim >= 13:
        for i in range(0, len(traj), max(1, len(traj) // 8)):
            cable_len = 0.8  # approximate
            payload_pos = traj[i].copy()
            payload_pos[2] -= cable_len
            ax1.plot([traj[i, 0], payload_pos[0]],
                    [traj[i, 1], payload_pos[1]],
                    [traj[i, 2], payload_pos[2]],
                    color='purple', linewidth=1.5, alpha=0.7)
            ax1.scatter(*payload_pos, color='purple', s=30, alpha=0.7)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Forest Flight — {ablation_name} ({obs_dim}D)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)

    # --- Panel 2: Top-down view ---
    ax2 = fig.add_subplot(2, 2, 2)
    if pts is not None and len(local_pts) > 0:
        # Draw tree trunks as circles (points near ground level)
        trunk_mask = local_pts[:, 2] < 2.0
        trunk_pts = local_pts[trunk_mask]
        if len(trunk_pts) > 20000:
            idx = np.random.choice(len(trunk_pts), 20000, replace=False)
            trunk_pts = trunk_pts[idx]
        ax2.scatter(trunk_pts[:, 0], trunk_pts[:, 1], c='forestgreen', s=1, alpha=0.3, rasterized=True)

    # Candidate trajectories (top-down)
    for step_idx in viz_steps:
        cands, best_idx, scores = candidates[step_idx]
        for i, cand in enumerate(cands):
            if i == best_idx:
                continue
            ax2.plot(cand[:, 0], cand[:, 1], color='steelblue', alpha=0.15, linewidth=0.5)
        # Best
        best_traj = cands[best_idx]
        ax2.plot(best_traj[:, 0], best_traj[:, 1], color='orangered', alpha=0.5, linewidth=1.2)

    ax2.plot(traj[:, 0], traj[:, 1], color='red', linewidth=2.5, label='Path')

    # Velocity arrows
    vels = result['velocities']
    arrow_steps = np.linspace(0, len(traj) - 1, min(12, len(traj))).astype(int)
    for i in arrow_steps:
        if i < len(vels):
            v = vels[i]
            vnorm = np.linalg.norm(v[:2])
            if vnorm > 0.3:
                ax2.annotate('', xy=(traj[i, 0] + v[0] * 0.3, traj[i, 1] + v[1] * 0.3),
                           xytext=(traj[i, 0], traj[i, 1]),
                           arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

    ax2.plot(*start[:2], 'go', markersize=10, label='Start', zorder=10)
    ax2.plot(*goal[:2], 'y*', markersize=15, label='Goal', zorder=10)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View', fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Velocity and height profile ---
    ax3 = fig.add_subplot(2, 2, 3)
    vels = result['velocities']
    t_axis = np.arange(len(traj))
    speeds = np.linalg.norm(vels[:len(traj)], axis=1) if len(vels) >= len(traj) else np.linalg.norm(vels, axis=1)

    ax3_z = ax3.twinx()
    ax3.plot(t_axis[:len(speeds)], speeds[:len(t_axis)], 'b-', linewidth=2, label='Speed (m/s)')
    ax3_z.plot(t_axis, traj[:, 2], 'r--', linewidth=2, label='Altitude (m)')

    ax3.set_xlabel('Step')
    ax3.set_ylabel('Speed (m/s)', color='blue')
    ax3_z.set_ylabel('Altitude (m)', color='red')
    ax3.set_title('Speed & Altitude Profile', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    ax3_z.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Sample depth images ---
    n_depth = min(4, len(depths))
    if n_depth > 0:
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        ax4.set_title('Depth Images (sampled steps)', fontsize=13, fontweight='bold')

        depth_indices = np.linspace(0, len(depths) - 1, n_depth).astype(int)
        for j, di in enumerate(depth_indices):
            inset = fig.add_axes([0.55 + (j % 2) * 0.2, 0.05 + (1 - j // 2) * 0.18, 0.18, 0.15])
            inset.imshow(depths[di], cmap='inferno', vmin=0, vmax=1)
            inset.set_title(f'Step {di}', fontsize=8)
            inset.axis('off')

    plt.suptitle(f'YOPO-Payload Forest Flight Visualization\n'
                 f'Map {map_id} | {ablation_name} | {obs_dim}D obs | {len(traj)-1} steps',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default="A4_full_model",
                        help="Ablation name (matches saved/ablation_<name>/ directory)")
    parser.add_argument("--obs_dim", type=int, default=None,
                        help="Override observation dimension")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--map_id", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Determine obs_dim from ablation name
    OBS_DIMS = {
        "A0_baseline": 9, "B1_inflated": 9, "B2_acc_penalty": 9,
        "A1_cable_L": 10, "A2_swing_ang": 12, "A3_swing_full": 14,
        "A4_full_model": 15, "A4_no_mass": 14, "D1_no_detach": 15,
    }
    obs_dim = args.obs_dim or OBS_DIMS.get(args.ablation, 15)

    # Load checkpoint
    ckpt_path = os.path.join(SAVE_DIR, f"ablation_{args.ablation}", f"epoch{args.epoch}.pth")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = YopoNetwork(observation_dim=obs_dim)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy = policy.to(device)
    policy.eval()
    print(f"Loaded {args.ablation} ({obs_dim}D) from {ckpt_path}")

    # Load forest point cloud
    print(f"Loading point cloud for map {args.map_id}...")
    pts = load_pointcloud(args.map_id)
    if pts is not None:
        print(f"  {len(pts)} points loaded")

    # Run simulation
    print(f"Running {args.n_steps}-step flight simulation...")
    result = run_multistep_flight(policy, device, args.map_id, obs_dim,
                                  n_steps=args.n_steps, seed=args.seed)
    print(f"  Flight: {len(result['trajectory'])} waypoints, "
          f"distance: {np.linalg.norm(result['trajectory'][-1] - result['trajectory'][0]):.1f}m")

    # Plot
    output_path = args.output or os.path.join(
        SCRIPT_DIR, "..", "..", "refine-logs",
        f"forest_flight_{args.ablation}_map{args.map_id}.png")
    plot_forest_flight(result, pts, args.map_id, obs_dim, args.ablation, output_path)


if __name__ == "__main__":
    main()
