"""
Comprehensive YOPO-Payload policy visualization.
Shows: trajectory, swing time-series, acceleration profile, and aggregate stats.

Usage:
    python visualize_policy.py --checkpoint saved/YOPO_43/epoch50.pth --n_samples 200
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.yopo_dataset import YOPODataset
from policy.state_transform import state_body2world
from policy.pendulum_simulator import simulate_pendulum
from evaluate_closed_loop import extract_acc_from_polynomial, evaluate_trajectory_with_pendulum
from torch.utils.data import DataLoader


def evaluate_single_sample(model, depth, pos, rot, obs_b, p_state, p_params, device, obs_dim=13):
    """Run inference on a single sample, return full trajectory + swing details."""
    sgm_time = cfg["sgm_time"]

    # Convert numpy arrays to tensors if needed
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth)
        pos = torch.from_numpy(pos)
        rot = torch.from_numpy(rot)
        obs_b = torch.from_numpy(obs_b)
        p_state = torch.from_numpy(p_state)
        p_params = torch.from_numpy(p_params)

    depth = depth.unsqueeze(0).to(device)
    pos = pos.unsqueeze(0).to(device)
    rot = rot.unsqueeze(0).to(device)
    obs_b = obs_b.unsqueeze(0).to(device)
    p_state_t = p_state.unsqueeze(0).to(device)
    p_params_t = p_params.unsqueeze(0).to(device)

    if obs_dim == 13:
        obs_aug = torch.cat([obs_b, p_state_t], dim=-1)
    elif obs_dim == 15:
        obs_aug = torch.cat([obs_b, p_state_t, p_params_t], dim=-1)
    else:
        obs_aug = obs_b

    with torch.inference_mode():
        endstate, score = model.inference(depth, obs_aug)

    # Pick best trajectory (lowest score = lowest cost)
    scores = score[0].reshape(-1)
    best_idx = torch.argmin(scores).item()
    v_idx = best_idx // cfg["horizon_num"]
    h_idx = best_idx % cfg["horizon_num"]

    end_pva = endstate[0, :, v_idx, h_idx].cpu().numpy()  # [9]

    # World frame transform
    goal_w, start_vel_w, start_acc_w = state_body2world(
        pos, rot, obs_b[:, 6:9], obs_b[:, 0:3], obs_b[:, 3:6])

    rot_np = rot[0].cpu().numpy()
    pos_np = pos[0].cpu().numpy()
    start_vel_np = start_vel_w[0].cpu().numpy()
    start_acc_np = start_acc_w[0].cpu().numpy()
    goal_np = goal_w[0].cpu().numpy()

    end_pos_w = rot_np @ end_pva[0:3] + pos_np
    end_vel_w = rot_np @ end_pva[3:6]
    end_acc_w = rot_np @ end_pva[6:9]

    start_sw = np.array([pos_np, start_vel_np, start_acc_np])
    end_sw = np.array([end_pos_w, end_vel_w, end_acc_w])

    # Extract polynomial trajectory
    n_traj_pts = 100
    t_arr, acc_profile = extract_acc_from_polynomial(start_sw, end_sw, sgm_time, n_points=n_traj_pts)

    # Compute full position trajectory from polynomial
    pos_traj = np.zeros((n_traj_pts, 3))
    vel_traj = np.zeros((n_traj_pts, 3))
    for axis in range(3):
        p0 = start_sw[0, axis]
        v0 = start_sw[1, axis]
        a0 = start_sw[2, axis]
        pf = end_sw[0, axis]
        vf = end_sw[1, axis]
        af = end_sw[2, axis]
        T = sgm_time
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3],
        ])
        b = np.array([p0, v0, a0, pf, vf, af])
        try:
            coeff = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coeff = np.zeros(6)
        for i, t in enumerate(t_arr):
            pos_traj[i, axis] = coeff[0] + coeff[1]*t + coeff[2]*t**2 + coeff[3]*t**3 + coeff[4]*t**4 + coeff[5]*t**5
            vel_traj[i, axis] = coeff[1] + 2*coeff[2]*t + 3*coeff[3]*t**2 + 4*coeff[4]*t**3 + 5*coeff[5]*t**4

    # Integrate pendulum ODE for swing
    L = float(p_params[0]) if not isinstance(p_params, torch.Tensor) else p_params[0].item()
    p_st = p_state if isinstance(p_state, np.ndarray) else p_state.numpy()
    acc_clamped = np.clip(acc_profile, -cfg["acc_max_train"], cfg["acc_max_train"])

    from scipy.interpolate import interp1d
    acc_interp = interp1d(t_arr, acc_clamped, axis=0, bounds_error=False,
                          fill_value=(acc_clamped[0], acc_clamped[-1]))
    ode_dt = max(sgm_time / 500, 0.002)
    try:
        t_ode, y_ode = simulate_pendulum(
            y0=list(p_st), L=L, T=sgm_time,
            acc_func=lambda t: acc_interp(t), dt=ode_dt)
        theta_series = np.degrees(y_ode[:, 0])
        phi_series = np.degrees(y_ode[:, 1])
        dtheta_series = np.degrees(y_ode[:, 2])
        dphi_series = np.degrees(y_ode[:, 3])
    except Exception:
        t_ode = t_arr
        theta_series = np.zeros_like(t_arr)
        phi_series = np.zeros_like(t_arr)
        dtheta_series = np.zeros_like(t_arr)
        dphi_series = np.zeros_like(t_arr)

    # Compute payload position (for visualization)
    # payload_pos = uav_pos + L * [sin(theta)cos(phi), sin(theta)sin(phi), -cos(theta)]
    # We interpolate UAV position to ODE time
    pos_interp = interp1d(t_arr, pos_traj, axis=0, bounds_error=False,
                          fill_value=(pos_traj[0], pos_traj[-1]))
    uav_pos_at_ode = pos_interp(t_ode)

    theta_rad = np.radians(theta_series)
    phi_rad = np.radians(phi_series)
    payload_offset = L * np.column_stack([
        np.sin(theta_rad) * np.cos(phi_rad),
        np.sin(theta_rad) * np.sin(phi_rad),
        -np.cos(theta_rad)
    ])
    payload_pos = uav_pos_at_ode + payload_offset

    # All candidate trajectories (for visualization)
    all_end_positions = []
    all_scores_val = []
    for vi in range(cfg["vertical_num"]):
        for hi in range(cfg["horizon_num"]):
            ep = endstate[0, :, vi, hi].cpu().numpy()
            ep_w = rot_np @ ep[0:3] + pos_np
            all_end_positions.append(ep_w)
            all_scores_val.append(score[0, vi, hi].item())
    all_end_positions = np.array(all_end_positions)
    all_scores_val = np.array(all_scores_val)

    # Speed profile
    speed = np.linalg.norm(vel_traj, axis=1)
    acc_mag = np.linalg.norm(acc_profile, axis=1)

    return {
        "t_arr": t_arr,
        "t_ode": t_ode,
        "pos_traj": pos_traj,
        "vel_traj": vel_traj,
        "acc_profile": acc_profile,
        "speed": speed,
        "acc_mag": acc_mag,
        "theta": theta_series,
        "phi": phi_series,
        "dtheta": dtheta_series,
        "dphi": dphi_series,
        "payload_pos": payload_pos,
        "uav_pos_ode": uav_pos_at_ode,
        "start_pos": pos_np,
        "goal_pos": goal_np,
        "end_pos": end_pos_w,
        "all_end_positions": all_end_positions,
        "all_scores": all_scores_val,
        "best_idx": best_idx,
        "L": L,
        "init_theta_deg": np.degrees(p_st[0]),
        "init_dtheta_deg": np.degrees(p_st[2]),
        "peak_swing_deg": float(np.max(np.abs(theta_series))),
        "final_swing_deg": float(abs(theta_series[-1])),
        "rms_swing_deg": float(np.sqrt(np.mean(theta_series**2))),
    }


def plot_single_sample(result, idx, save_dir):
    """Generate a comprehensive 6-panel figure for one sample."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"Sample #{idx}  |  Init swing: {result['init_theta_deg']:.1f}°  |  "
        f"Peak: {result['peak_swing_deg']:.1f}°  |  Final: {result['final_swing_deg']:.1f}°  |  "
        f"L={result['L']:.2f}m",
        fontsize=14, fontweight='bold')

    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel 1: 3D trajectory (UAV + payload) ---
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.plot(*result["pos_traj"].T, 'b-', linewidth=2, label='UAV trajectory')
    ax1.plot(*result["payload_pos"].T, 'r-', linewidth=1.5, alpha=0.7, label='Payload trajectory')
    # Draw cable at a few time points
    n_cables = 8
    cable_indices = np.linspace(0, len(result["t_ode"])-1, n_cables, dtype=int)
    for ci in cable_indices:
        ax1.plot([result["uav_pos_ode"][ci, 0], result["payload_pos"][ci, 0]],
                 [result["uav_pos_ode"][ci, 1], result["payload_pos"][ci, 1]],
                 [result["uav_pos_ode"][ci, 2], result["payload_pos"][ci, 2]],
                 'k-', linewidth=0.5, alpha=0.4)
    ax1.scatter(*result["start_pos"], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(*result["goal_pos"], c='red', s=100, marker='*', label='Goal', zorder=5)
    ax1.scatter(*result["end_pos"], c='blue', s=80, marker='^', label='End', zorder=5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory (UAV + Payload)')
    ax1.legend(fontsize=7, loc='upper left')

    # --- Panel 2: All candidate endpoints ---
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    sc = ax2.scatter(*result["all_end_positions"].T,
                     c=result["all_scores"], cmap='RdYlGn_r', s=40, alpha=0.8)
    ax2.scatter(*result["end_pos"], c='blue', s=150, marker='*', label='Selected', zorder=5, edgecolors='black')
    ax2.scatter(*result["start_pos"], c='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(*result["goal_pos"], c='red', s=100, marker='*', label='Goal', zorder=5)
    plt.colorbar(sc, ax=ax2, shrink=0.6, label='Score (lower=better)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'15 Candidate Endpoints (best=#{result["best_idx"]})')
    ax2.legend(fontsize=7)

    # --- Panel 3: XY top-down view ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(result["pos_traj"][:, 0], result["pos_traj"][:, 1], 'b-', linewidth=2, label='UAV')
    ax3.plot(result["payload_pos"][:, 0], result["payload_pos"][:, 1], 'r-', linewidth=1.5, alpha=0.7, label='Payload')
    ax3.scatter(result["start_pos"][0], result["start_pos"][1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.scatter(result["goal_pos"][0], result["goal_pos"][1], c='red', s=100, marker='*', label='Goal', zorder=5)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Top-Down View (XY)')
    ax3.legend(fontsize=8)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Swing angle over time ---
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(result["t_ode"], result["theta"], 'r-', linewidth=2, label='θ (polar swing)')
    ax4.plot(result["t_ode"], result["phi"], 'b-', linewidth=1.5, alpha=0.6, label='φ (azimuth)')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.axhline(y=result["peak_swing_deg"], color='r', linestyle=':', alpha=0.5, label=f'Peak θ = {result["peak_swing_deg"]:.1f}°')
    ax4.axhline(y=-result["peak_swing_deg"], color='r', linestyle=':', alpha=0.5)
    ax4.fill_between(result["t_ode"], -15, 15, alpha=0.1, color='green', label='Safe zone (<15°)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angle (deg)')
    ax4.set_title('Pendulum Swing Over Time')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: Angular velocity ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(result["t_ode"], result["dtheta"], 'r-', linewidth=1.5, label='dθ/dt')
    ax5.plot(result["t_ode"], result["dphi"], 'b-', linewidth=1.5, alpha=0.6, label='dφ/dt')
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angular velocity (deg/s)')
    ax5.set_title('Angular Velocity')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # --- Panel 6: Speed and acceleration ---
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(result["t_arr"], result["speed"], 'b-', linewidth=2, label='Speed')
    ax6.axhline(y=cfg["vel_max_train"], color='r', linestyle='--', alpha=0.5, label=f'v_max={cfg["vel_max_train"]}')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Speed (m/s)')
    ax6.set_title('UAV Speed Profile')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # --- Panel 7: Acceleration components ---
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(result["t_arr"], result["acc_profile"][:, 0], label='ax', alpha=0.8)
    ax7.plot(result["t_arr"], result["acc_profile"][:, 1], label='ay', alpha=0.8)
    ax7.plot(result["t_arr"], result["acc_profile"][:, 2], label='az', alpha=0.8)
    ax7.plot(result["t_arr"], result["acc_mag"], 'k--', linewidth=1.5, label='|a|')
    ax7.axhline(y=cfg["acc_max_train"], color='r', linestyle=':', alpha=0.5)
    ax7.axhline(y=-cfg["acc_max_train"], color='r', linestyle=':', alpha=0.5)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Acceleration (m/s²)')
    ax7.set_title('Acceleration Profile')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # --- Panel 8: Summary stats text ---
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    stats_text = (
        f"Trajectory Summary\n"
        f"{'─'*30}\n"
        f"Init swing (θ₀):  {result['init_theta_deg']:.1f}°\n"
        f"Init dθ/dt:       {result['init_dtheta_deg']:.1f}°/s\n"
        f"Cable length:     {result['L']:.2f} m\n"
        f"{'─'*30}\n"
        f"Peak swing:       {result['peak_swing_deg']:.1f}°\n"
        f"Final swing:      {result['final_swing_deg']:.1f}°\n"
        f"RMS swing:        {result['rms_swing_deg']:.1f}°\n"
        f"{'─'*30}\n"
        f"Max speed:        {np.max(result['speed']):.2f} m/s\n"
        f"Max |accel|:      {np.max(result['acc_mag']):.2f} m/s²\n"
        f"Duration:         {result['t_arr'][-1]:.2f} s\n"
        f"Goal dist:        {np.linalg.norm(result['goal_pos']-result['start_pos']):.2f} m\n"
        f"End-to-goal:      {np.linalg.norm(result['end_pos']-result['goal_pos']):.2f} m\n"
        f"Selected traj:    #{result['best_idx']}\n"
    )
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(os.path.join(save_dir, f"sample_{idx:03d}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate(all_results, save_dir):
    """Generate aggregate analysis plots."""
    peaks = [r["peak_swing_deg"] for r in all_results]
    rms_vals = [r["rms_swing_deg"] for r in all_results]
    finals = [r["final_swing_deg"] for r in all_results]
    init_thetas = [r["init_theta_deg"] for r in all_results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'YOPO_43 Aggregate Analysis ({len(all_results)} samples)', fontsize=14, fontweight='bold')

    # 1. Peak swing distribution
    ax = axes[0, 0]
    ax.hist(peaks, bins=40, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(peaks), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(peaks):.1f}°')
    ax.axvline(np.median(peaks), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(peaks):.1f}°')
    ax.axvline(19.2, color='green', linestyle=':', linewidth=2, label='MPC baseline: 19.2°')
    ax.axvline(55.3, color='gray', linestyle=':', linewidth=2, label='B3 (old): 55.3°')
    ax.set_xlabel('Peak Swing (deg)')
    ax.set_ylabel('Count')
    ax.set_title('Peak Swing Distribution')
    ax.legend(fontsize=8)

    # 2. Initial swing vs peak swing
    ax = axes[0, 1]
    sc = ax.scatter(init_thetas, peaks, c=finals, cmap='RdYlGn_r', s=20, alpha=0.6)
    ax.plot([0, max(init_thetas)], [0, max(init_thetas)], 'k--', alpha=0.3, label='θ_peak = θ_init')
    plt.colorbar(sc, ax=ax, label='Final swing (deg)')
    ax.set_xlabel('Initial Swing θ₀ (deg)')
    ax.set_ylabel('Peak Swing (deg)')
    ax.set_title('Init vs Peak Swing')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. RMS swing distribution
    ax = axes[0, 2]
    ax.hist(rms_vals, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(rms_vals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rms_vals):.1f}°')
    ax.set_xlabel('RMS Swing (deg)')
    ax.set_ylabel('Count')
    ax.set_title('RMS Swing Distribution')
    ax.legend(fontsize=8)

    # 4. CDF of peak swing
    ax = axes[1, 0]
    sorted_peaks = np.sort(peaks)
    cdf = np.arange(1, len(sorted_peaks)+1) / len(sorted_peaks)
    ax.plot(sorted_peaks, cdf, 'b-', linewidth=2)
    ax.axvline(19.2, color='green', linestyle=':', linewidth=2, label='MPC: 19.2°')
    ax.axvline(55.3, color='gray', linestyle=':', linewidth=2, label='B3 (old): 55.3°')
    # Mark percentiles
    for pct in [50, 75, 90, 95]:
        val = np.percentile(peaks, pct)
        ax.axhline(pct/100, color='gray', linestyle=':', alpha=0.3)
        ax.annotate(f'P{pct}: {val:.1f}°', xy=(val, pct/100), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Peak Swing (deg)')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution of Peak Swing')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Swing categories
    ax = axes[1, 1]
    categories = {
        '<10°': sum(1 for p in peaks if p < 10),
        '10-20°': sum(1 for p in peaks if 10 <= p < 20),
        '20-30°': sum(1 for p in peaks if 20 <= p < 30),
        '30-45°': sum(1 for p in peaks if 30 <= p < 45),
        '45-60°': sum(1 for p in peaks if 45 <= p < 60),
        '>60°': sum(1 for p in peaks if p >= 60),
    }
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
    bars = ax.bar(categories.keys(), categories.values(), color=colors, edgecolor='black')
    for bar, count in zip(bars, categories.values()):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{count}\n({count/len(peaks)*100:.0f}%)', ha='center', fontsize=9)
    ax.set_xlabel('Peak Swing Range')
    ax.set_ylabel('Count')
    ax.set_title('Swing Category Breakdown')

    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    summary = (
        f"{'='*35}\n"
        f"  YOPO_43 (13D, v4.0, analytical)\n"
        f"{'='*35}\n"
        f"  Samples:         {len(all_results)}\n"
        f"{'─'*35}\n"
        f"  Peak Swing:\n"
        f"    Mean:          {np.mean(peaks):.1f}°\n"
        f"    Median:        {np.median(peaks):.1f}°\n"
        f"    Std:           {np.std(peaks):.1f}°\n"
        f"    Min:           {np.min(peaks):.1f}°\n"
        f"    Max:           {np.max(peaks):.1f}°\n"
        f"    P90:           {np.percentile(peaks, 90):.1f}°\n"
        f"    P95:           {np.percentile(peaks, 95):.1f}°\n"
        f"{'─'*35}\n"
        f"  RMS Swing:\n"
        f"    Mean:          {np.mean(rms_vals):.1f}°\n"
        f"{'─'*35}\n"
        f"  Baselines:\n"
        f"    B3 (old):      55.3°\n"
        f"    MPC:           19.2°\n"
        f"{'─'*35}\n"
        f"  % below 20°:    {sum(1 for p in peaks if p<20)/len(peaks)*100:.1f}%\n"
        f"  % below 30°:    {sum(1 for p in peaks if p<30)/len(peaks)*100:.1f}%\n"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(os.path.join(save_dir, "aggregate_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved aggregate_analysis.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="saved/YOPO_43/epoch50.pth")
    parser.add_argument("--obs_dim", type=int, default=13)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_detail_plots", type=int, default=10,
                        help="Number of individual sample plots (picked to span swing range)")
    parser.add_argument("--output_dir", type=str, default="saved/YOPO_43/eval_plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YopoNetwork(observation_dim=args.obs_dim).to(device)
    state_dict = torch.load(args.checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    np.random.seed(42)
    val_ds = YOPODataset(mode='valid')
    print(f"Validation set: {len(val_ds)} samples")

    all_results = []
    for i in range(min(args.n_samples, len(val_ds))):
        depth, pos, rot, obs_b, p_state, p_params, map_id = val_ds[i]
        result = evaluate_single_sample(model, depth, pos, rot, obs_b, p_state, p_params, device, args.obs_dim)
        all_results.append(result)

        if (i+1) % 50 == 0:
            peaks_so_far = [r["peak_swing_deg"] for r in all_results]
            print(f"  [{i+1}/{args.n_samples}] mean peak: {np.mean(peaks_so_far):.1f}°, "
                  f"median: {np.median(peaks_so_far):.1f}°")

    # Sort by peak swing for selecting representative samples
    indexed = [(i, r["peak_swing_deg"]) for i, r in enumerate(all_results)]
    indexed.sort(key=lambda x: x[1])

    # Pick samples spanning the swing range: best, worst, and evenly spaced
    n_det = min(args.n_detail_plots, len(all_results))
    if n_det > 0:
        detail_indices = [indexed[int(i * (len(indexed)-1) / (n_det-1))][0] for i in range(n_det)]
        print(f"\nGenerating {n_det} detailed plots...")
        for rank, idx in enumerate(detail_indices):
            plot_single_sample(all_results[idx], idx, args.output_dir)
            print(f"  Saved sample_{idx:03d}.png (peak={all_results[idx]['peak_swing_deg']:.1f}°)")

    # Aggregate analysis
    print("\nGenerating aggregate analysis...")
    plot_aggregate(all_results, args.output_dir)

    # Print summary
    peaks = [r["peak_swing_deg"] for r in all_results]
    print(f"\n{'='*50}")
    print(f"YOPO_43 Evaluation Complete ({len(all_results)} samples)")
    print(f"{'='*50}")
    print(f"Peak swing: {np.mean(peaks):.1f}° ± {np.std(peaks):.1f}° (mean ± std)")
    print(f"Median: {np.median(peaks):.1f}°, P90: {np.percentile(peaks, 90):.1f}°, P95: {np.percentile(peaks, 95):.1f}°")
    print(f"Plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
