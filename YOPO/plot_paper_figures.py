"""
Publication-quality figures for RA-L paper.
IEEE RA-L: single column = 3.5in, double column = 7.16in
Font: Times New Roman / serif, min 8pt
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.yopo_dataset import YOPODataset
from policy.state_transform import state_body2world
from policy.pendulum_simulator import simulate_pendulum
from evaluate_closed_loop import extract_acc_from_polynomial
from scipy.interpolate import interp1d

# ── IEEE style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.7',
    'grid.linewidth': 0.3,
    'grid.alpha': 0.4,
    'text.usetex': False,
})

# Color palette (colorblind-friendly)
C_OURS = '#2166AC'       # blue
C_BASELINE = '#B2182B'   # red
C_MPC = '#4DAF4A'        # green
C_PAYLOAD = '#E66101'    # orange
C_SAFE = '#92C5DE'       # light blue
C_CABLE = '#666666'      # gray


def get_eval_data(checkpoint, obs_dim=13, n_samples=300):
    """Run evaluation and return per-sample data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sgm_time = cfg["sgm_time"]

    model = YopoNetwork(observation_dim=obs_dim).to(device)
    sd = torch.load(checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    np.random.seed(42)
    val_ds = YOPODataset(mode='valid')

    results = []
    for i in range(min(n_samples, len(val_ds))):
        depth, pos, rot, obs_b, p_state, p_params, map_id = val_ds[i]
        # to tensor
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth)
            pos = torch.from_numpy(pos)
            rot = torch.from_numpy(rot)
            obs_b = torch.from_numpy(obs_b)
            p_state_t = torch.from_numpy(p_state)
            p_params_t = torch.from_numpy(p_params)
        else:
            p_state_t = p_state
            p_params_t = p_params

        d = depth.unsqueeze(0).to(device)
        p = pos.unsqueeze(0).to(device)
        r = rot.unsqueeze(0).to(device)
        o = obs_b.unsqueeze(0).to(device)
        ps = p_state_t.unsqueeze(0).to(device)

        if obs_dim == 13:
            obs_aug = torch.cat([o, ps], dim=-1)
        else:
            obs_aug = o

        with torch.inference_mode():
            endstate, score = model.inference(d, obs_aug)

        scores = score[0].reshape(-1)
        best_idx = torch.argmin(scores).item()
        v_idx = best_idx // cfg["horizon_num"]
        h_idx = best_idx % cfg["horizon_num"]

        end_pva = endstate[0, :, v_idx, h_idx].cpu().numpy()
        goal_w, start_vel_w, start_acc_w = state_body2world(
            p, r, o[:, 6:9], o[:, 0:3], o[:, 3:6])

        rot_np = rot if isinstance(rot, np.ndarray) else rot.numpy()
        pos_np = pos if isinstance(pos, np.ndarray) else pos.numpy()
        p_st = p_state if isinstance(p_state, np.ndarray) else p_state.numpy()
        L = float(p_params[0]) if not isinstance(p_params, torch.Tensor) else p_params[0].item()

        start_vel_np = start_vel_w[0].cpu().numpy()
        start_acc_np = start_acc_w[0].cpu().numpy()
        goal_np = goal_w[0].cpu().numpy()

        end_pos_w = rot_np @ end_pva[0:3] + pos_np
        end_vel_w = rot_np @ end_pva[3:6]
        end_acc_w = rot_np @ end_pva[6:9]

        start_sw = np.array([pos_np, start_vel_np, start_acc_np])
        end_sw = np.array([end_pos_w, end_vel_w, end_acc_w])

        n_pts = 100
        t_arr, acc_profile = extract_acc_from_polynomial(start_sw, end_sw, sgm_time, n_points=n_pts)

        # position trajectory
        pos_traj = np.zeros((n_pts, 3))
        vel_traj = np.zeros((n_pts, 3))
        for axis in range(3):
            T = sgm_time
            A = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, T, T**2, T**3, T**4, T**5],
                [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                [0, 0, 2, 6*T, 12*T**2, 20*T**3],
            ])
            b = np.array([start_sw[0, axis], start_sw[1, axis], start_sw[2, axis],
                          end_sw[0, axis], end_sw[1, axis], end_sw[2, axis]])
            try:
                coeff = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                coeff = np.zeros(6)
            for k, t in enumerate(t_arr):
                pos_traj[k, axis] = sum(coeff[j]*t**j for j in range(6))
                vel_traj[k, axis] = sum(j*coeff[j]*t**(j-1) for j in range(1, 6))

        # pendulum ODE
        acc_clamped = np.clip(acc_profile, -cfg["acc_max_train"], cfg["acc_max_train"])
        acc_interp = interp1d(t_arr, acc_clamped, axis=0, bounds_error=False,
                              fill_value=(acc_clamped[0], acc_clamped[-1]))
        ode_dt = max(sgm_time / 500, 0.002)
        try:
            t_ode, y_ode = simulate_pendulum(
                y0=list(p_st), L=L, T=sgm_time,
                acc_func=lambda t: acc_interp(t), dt=ode_dt)
            theta_deg = np.degrees(y_ode[:, 0])
        except Exception:
            t_ode = t_arr
            theta_deg = np.zeros_like(t_arr)

        # payload position
        pos_interp_f = interp1d(t_arr, pos_traj, axis=0, bounds_error=False,
                                fill_value=(pos_traj[0], pos_traj[-1]))
        uav_at_ode = pos_interp_f(t_ode)
        theta_rad = np.radians(theta_deg)
        phi_rad = y_ode[:, 1] if len(y_ode) > 0 else np.zeros_like(t_ode)
        payload_offset = L * np.column_stack([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            -np.cos(theta_rad)])
        payload_pos = uav_at_ode + payload_offset

        # all candidates
        all_ends = []
        all_scores = []
        for vi in range(cfg["vertical_num"]):
            for hi in range(cfg["horizon_num"]):
                ep = endstate[0, :, vi, hi].cpu().numpy()
                ep_w = rot_np @ ep[0:3] + pos_np
                all_ends.append(ep_w)
                all_scores.append(score[0, vi, hi].item())

        results.append({
            "t_arr": t_arr, "t_ode": t_ode,
            "pos_traj": pos_traj, "vel_traj": vel_traj,
            "acc_profile": acc_profile, "acc_clamped": acc_clamped,
            "theta_deg": theta_deg,
            "payload_pos": payload_pos, "uav_at_ode": uav_at_ode,
            "start_pos": pos_np, "goal_pos": goal_np, "end_pos": end_pos_w,
            "all_ends": np.array(all_ends), "all_scores": np.array(all_scores),
            "best_idx": best_idx, "L": L,
            "init_theta_deg": np.degrees(p_st[0]),
            "peak_swing_deg": float(np.max(np.abs(theta_deg))),
            "rms_swing_deg": float(np.sqrt(np.mean(theta_deg**2))),
            "final_swing_deg": float(abs(theta_deg[-1])),
            "speed": np.linalg.norm(vel_traj, axis=1),
            "acc_mag": np.linalg.norm(acc_profile, axis=1),
        })

        if (i+1) % 100 == 0:
            peaks = [r["peak_swing_deg"] for r in results]
            print(f"  [{i+1}/{n_samples}] mean peak: {np.mean(peaks):.1f}")

    return results


# =====================================================================
# Figure 1: Trajectory visualization (double-column, 3 representative)
# =====================================================================
def fig_trajectory_showcase(results, save_dir):
    # Pick 3 samples: good (<15°), typical (20-28°), challenging (35-50°)
    good = [r for r in results if r["peak_swing_deg"] < 15]
    typical = [r for r in results if 20 < r["peak_swing_deg"] < 28]
    hard = [r for r in results if 35 < r["peak_swing_deg"] < 50]

    picks = []
    if good: picks.append(good[len(good)//2])
    else: picks.append(min(results, key=lambda r: r["peak_swing_deg"]))
    if typical: picks.append(typical[len(typical)//2])
    else: picks.append(results[len(results)//2])
    if hard: picks.append(hard[len(hard)//2])
    else: picks.append(max(results, key=lambda r: r["peak_swing_deg"]))

    labels = ["Low swing", "Moderate swing", "High swing"]

    fig = plt.figure(figsize=(7.16, 5.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                  top=0.93, bottom=0.08, left=0.06, right=0.97)

    for col, (res, label) in enumerate(zip(picks, labels)):
        # Top row: 3D trajectory
        ax = fig.add_subplot(gs[0, col], projection='3d')
        ax.plot(*res["pos_traj"].T, color=C_OURS, linewidth=1.2, label='UAV', zorder=3)
        ax.plot(*res["payload_pos"].T, color=C_PAYLOAD, linewidth=0.9, alpha=0.8, label='Payload', zorder=2)
        # cables
        n_c = 6
        cidx = np.linspace(0, len(res["t_ode"])-1, n_c, dtype=int)
        for ci in cidx:
            ax.plot([res["uav_at_ode"][ci, 0], res["payload_pos"][ci, 0]],
                    [res["uav_at_ode"][ci, 1], res["payload_pos"][ci, 1]],
                    [res["uav_at_ode"][ci, 2], res["payload_pos"][ci, 2]],
                    color=C_CABLE, linewidth=0.4, alpha=0.5)
        ax.scatter(*res["start_pos"], c='green', s=30, marker='o', zorder=5, edgecolors='k', linewidths=0.3, label='Start')
        ax.scatter(*res["end_pos"], c='red', s=50, marker='^', zorder=5, edgecolors='k', linewidths=0.3, label='End')
        # Draw arrow indicating goal direction from end position
        goal_dir = res["goal_pos"] - res["end_pos"]
        goal_dist = np.linalg.norm(goal_dir)
        if goal_dist > 0.1:
            goal_dir_norm = goal_dir / goal_dist
            arrow_len = min(goal_dist, 1.5)  # cap arrow length for readability
            arrow_end = res["end_pos"] + goal_dir_norm * arrow_len
            ax.plot([res["end_pos"][0], arrow_end[0]],
                    [res["end_pos"][1], arrow_end[1]],
                    [res["end_pos"][2], arrow_end[2]],
                    color='red', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_xlabel('X (m)', labelpad=1)
        ax.set_ylabel('Y (m)', labelpad=1)
        ax.set_zlabel('Z (m)', labelpad=1)
        ax.tick_params(pad=0)
        peak = res["peak_swing_deg"]
        ax.set_title(f'{label}\n(peak $\\theta$ = {peak:.1f}$^\\circ$)', fontsize=9, pad=2)
        if col == 0:
            ax.legend(fontsize=6.5, loc='upper left', handlelength=1.2)

        # Bottom row: swing time-series
        ax2 = fig.add_subplot(gs[1, col])
        ax2.fill_between(res["t_ode"], -15, 15, alpha=0.15, color=C_SAFE, label='Safe ($<$15$^\\circ$)')
        ax2.plot(res["t_ode"], res["theta_deg"], color=C_BASELINE, linewidth=1.0, label='$\\theta(t)$')
        ax2.axhline(0, color='k', linewidth=0.3, alpha=0.5)
        ax2.axhline(res["peak_swing_deg"], color=C_BASELINE, linewidth=0.5, linestyle=':', alpha=0.5)
        ax2.axhline(-res["peak_swing_deg"], color=C_BASELINE, linewidth=0.5, linestyle=':', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        if col == 0:
            ax2.set_ylabel('Swing angle $\\theta$ ($^\\circ$)')
        ax2.set_xlim(0, cfg["sgm_time"])
        ax2.grid(True)
        if col == 0:
            ax2.legend(fontsize=7, loc='upper right')

    plt.savefig(os.path.join(save_dir, "fig_trajectory_showcase.pdf"))
    plt.savefig(os.path.join(save_dir, "fig_trajectory_showcase.png"), dpi=300)
    plt.close()
    print("Saved fig_trajectory_showcase.pdf/png")


# =====================================================================
# Figure 2: Swing distribution + CDF comparison (single-column)
# =====================================================================
def fig_swing_distribution(results, save_dir):
    peaks = [r["peak_swing_deg"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.2),
                                    gridspec_kw={'hspace': 0.4})

    # (a) Histogram
    bins = np.arange(0, max(peaks)+5, 3)
    ax1.hist(peaks, bins=bins, color=C_OURS, edgecolor='white', linewidth=0.3, alpha=0.85)
    ax1.axvline(np.mean(peaks), color=C_BASELINE, linewidth=1.0, linestyle='--',
                label=f'Mean: {np.mean(peaks):.1f}$^\\circ$')
    ax1.axvline(np.median(peaks), color=C_MPC, linewidth=1.0, linestyle='-.',
                label=f'Median: {np.median(peaks):.1f}$^\\circ$')
    ax1.set_xlabel('Peak swing angle ($^\\circ$)')
    ax1.set_ylabel('Count')
    ax1.set_title('(a) Peak swing distribution', fontsize=9)
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, None)

    # (b) CDF with baseline references
    sorted_p = np.sort(peaks)
    cdf = np.arange(1, len(sorted_p)+1) / len(sorted_p)
    ax2.plot(sorted_p, cdf * 100, color=C_OURS, linewidth=1.2, label='YOPO-Payload (ours)')
    # Reference lines for baselines
    ax2.axvline(55.3, color=C_BASELINE, linewidth=0.8, linestyle='--', label='YOPO-Original (55.3$^\\circ$)')
    ax2.axvline(19.2, color=C_MPC, linewidth=0.8, linestyle=':', label='Swing-aware MPC (19.2$^\\circ$)')

    # Percentile annotations
    for pct in [50, 90]:
        val = np.percentile(peaks, pct)
        y = pct
        ax2.plot(val, y, 'o', color=C_OURS, markersize=3)
        ax2.annotate(f'P{pct}: {val:.0f}$^\\circ$', xy=(val, y),
                     xytext=(6, -2), textcoords='offset points', fontsize=7)

    ax2.set_xlabel('Peak swing angle ($^\\circ$)')
    ax2.set_ylabel('Cumulative percentage (%)')
    ax2.set_title('(b) CDF of peak swing', fontsize=9)
    ax2.legend(fontsize=6.5, loc='lower right')
    ax2.set_xlim(0, max(peaks) * 1.05)
    ax2.set_ylim(0, 105)
    ax2.grid(True)

    plt.savefig(os.path.join(save_dir, "fig_swing_distribution.pdf"))
    plt.savefig(os.path.join(save_dir, "fig_swing_distribution.png"), dpi=300)
    plt.close()
    print("Saved fig_swing_distribution.pdf/png")


# =====================================================================
# Figure 3: Detailed single-trajectory analysis (double-column)
# =====================================================================
def fig_detailed_trajectory(results, save_dir):
    # Pick a representative moderate-swing case
    target = sorted(results, key=lambda r: abs(r["peak_swing_deg"] - 25))[0]

    fig = plt.figure(figsize=(7.16, 3.8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.45,
                  top=0.92, bottom=0.12, left=0.07, right=0.97)

    # (a) 3D trajectory
    ax = fig.add_subplot(gs[:, 0], projection='3d')
    ax.plot(*target["pos_traj"].T, color=C_OURS, linewidth=1.2, label='UAV')
    ax.plot(*target["payload_pos"].T, color=C_PAYLOAD, linewidth=0.9, alpha=0.8, label='Payload')
    n_c = 8
    cidx = np.linspace(0, len(target["t_ode"])-1, n_c, dtype=int)
    for ci in cidx:
        ax.plot([target["uav_at_ode"][ci, 0], target["payload_pos"][ci, 0]],
                [target["uav_at_ode"][ci, 1], target["payload_pos"][ci, 1]],
                [target["uav_at_ode"][ci, 2], target["payload_pos"][ci, 2]],
                color=C_CABLE, linewidth=0.4, alpha=0.5)
    ax.scatter(*target["start_pos"], c='green', s=25, marker='o', zorder=5, edgecolors='k', linewidths=0.3, label='Start')
    ax.scatter(*target["end_pos"], c='red', s=40, marker='^', zorder=5, edgecolors='k', linewidths=0.3, label='End')
    # Goal direction arrow
    goal_dir = target["goal_pos"] - target["end_pos"]
    goal_dist = np.linalg.norm(goal_dir)
    if goal_dist > 0.1:
        gdn = goal_dir / goal_dist
        arrow_end = target["end_pos"] + gdn * min(goal_dist, 1.5)
        ax.plot([target["end_pos"][0], arrow_end[0]],
                [target["end_pos"][1], arrow_end[1]],
                [target["end_pos"][2], arrow_end[2]],
                color='red', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xlabel('X (m)', labelpad=1)
    ax.set_ylabel('Y (m)', labelpad=1)
    ax.set_zlabel('Z (m)', labelpad=1)
    ax.set_title('(a) 3D Trajectory', fontsize=9)
    ax.legend(fontsize=6, loc='upper left')

    # (b) Swing angle
    ax2 = fig.add_subplot(gs[0, 1:3])
    ax2.fill_between(target["t_ode"], -15, 15, alpha=0.12, color=C_SAFE)
    ax2.plot(target["t_ode"], target["theta_deg"], color=C_BASELINE, linewidth=1.0)
    ax2.axhline(0, color='k', linewidth=0.3)
    ax2.set_ylabel('$\\theta$ ($^\\circ$)')
    ax2.set_title('(b) Pendulum swing angle', fontsize=9)
    ax2.set_xlim(0, cfg["sgm_time"])
    ax2.grid(True)
    # annotate peak
    peak_idx = np.argmax(np.abs(target["theta_deg"]))
    ax2.annotate(f'Peak: {target["peak_swing_deg"]:.1f}$^\\circ$',
                 xy=(target["t_ode"][peak_idx], target["theta_deg"][peak_idx]),
                 xytext=(0.3, target["theta_deg"][peak_idx] + 8),
                 fontsize=7, arrowprops=dict(arrowstyle='->', color='k', lw=0.5))

    # (c) Acceleration
    ax3 = fig.add_subplot(gs[1, 1:3])
    ax3.plot(target["t_arr"], target["acc_profile"][:, 0], color='#e41a1c', linewidth=0.8, label='$a_x$')
    ax3.plot(target["t_arr"], target["acc_profile"][:, 1], color='#377eb8', linewidth=0.8, label='$a_y$')
    ax3.plot(target["t_arr"], target["acc_profile"][:, 2], color='#4daf4a', linewidth=0.8, label='$a_z$')
    ax3.plot(target["t_arr"], target["acc_mag"], color='k', linewidth=0.8, linestyle='--', label='$|\\mathbf{a}|$')
    ax3.axhline(cfg["acc_max_train"], color='gray', linewidth=0.5, linestyle=':')
    ax3.axhline(-cfg["acc_max_train"], color='gray', linewidth=0.5, linestyle=':')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Accel. (m/s$^2$)')
    ax3.set_title('(c) Acceleration profile', fontsize=9)
    ax3.legend(fontsize=6.5, ncol=4, loc='upper right')
    ax3.set_xlim(0, cfg["sgm_time"])
    ax3.grid(True)

    # (d) Speed
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(target["t_arr"], target["speed"], color=C_OURS, linewidth=1.0)
    ax4.axhline(cfg["vel_max_train"], color='gray', linewidth=0.5, linestyle=':', label=f'$v_{{max}}$={cfg["vel_max_train"]}')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('(d) Speed', fontsize=9)
    ax4.set_xlim(0, cfg["sgm_time"])
    ax4.grid(True)
    ax4.legend(fontsize=6.5)

    # (e) Top-down XY
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.plot(target["pos_traj"][:, 0], target["pos_traj"][:, 1], color=C_OURS, linewidth=1.0, label='UAV')
    ax5.plot(target["payload_pos"][:, 0], target["payload_pos"][:, 1],
             color=C_PAYLOAD, linewidth=0.8, alpha=0.7, label='Payload')
    ax5.scatter(target["start_pos"][0], target["start_pos"][1], c='green', s=25, marker='o',
                zorder=5, edgecolors='k', linewidths=0.3)
    ax5.scatter(target["end_pos"][0], target["end_pos"][1], c='red', s=35, marker='^',
                zorder=5, edgecolors='k', linewidths=0.3)
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('(e) Top view', fontsize=9)
    ax5.set_aspect('equal')
    ax5.grid(True)
    ax5.legend(fontsize=6.5)

    plt.savefig(os.path.join(save_dir, "fig_detailed_trajectory.pdf"))
    plt.savefig(os.path.join(save_dir, "fig_detailed_trajectory.png"), dpi=300)
    plt.close()
    print("Saved fig_detailed_trajectory.pdf/png")


# =====================================================================
# Figure 4: Candidate trajectory selection (single-column)
# =====================================================================
def fig_trajectory_selection(results, save_dir):
    # Pick a sample with spread-out candidates
    res = sorted(results, key=lambda r: abs(r["peak_swing_deg"] - 22))[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0),
                                    gridspec_kw={'wspace': 0.35},
                                    subplot_kw={})
    # Override ax1 to 3D
    ax1.remove()
    ax1 = fig.add_subplot(121, projection='3d')

    # (a) All candidates in 3D
    sc = ax1.scatter(*res["all_ends"].T, c=res["all_scores"], cmap='RdYlGn_r',
                     s=25, alpha=0.8, edgecolors='k', linewidths=0.2, zorder=3)
    best_end = res["all_ends"][res["best_idx"]]
    ax1.scatter(*best_end, c=C_OURS, s=80, marker='*', zorder=5,
                edgecolors='k', linewidths=0.5, label='Selected')
    ax1.scatter(*res["start_pos"], c='green', s=30, marker='o', zorder=5,
                edgecolors='k', linewidths=0.3, label='Start')
    ax1.plot(*res["pos_traj"].T, color=C_OURS, linewidth=1.0, alpha=0.7)
    cb = plt.colorbar(sc, ax=ax1, shrink=0.55, pad=0.08, aspect=15)
    cb.set_label('Score (lower = better)', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax1.set_xlabel('X (m)', labelpad=1)
    ax1.set_ylabel('Y (m)', labelpad=1)
    ax1.set_zlabel('Z (m)', labelpad=1)
    ax1.set_title('(a) Candidate endpoints & scores', fontsize=9)
    ax1.legend(fontsize=6.5, loc='upper left')

    # (b) Score bar chart
    n_cand = len(res["all_scores"])
    colors = [C_OURS if i == res["best_idx"] else '#AAAAAA' for i in range(n_cand)]
    ax2.bar(range(n_cand), res["all_scores"], color=colors, edgecolor='k', linewidth=0.3)
    ax2.set_xlabel('Candidate index')
    ax2.set_ylabel('Predicted cost score')
    ax2.set_title('(b) Score network output', fontsize=9)
    ax2.set_xticks(range(n_cand))
    ax2.set_xticklabels(range(n_cand), fontsize=6)
    ax2.grid(True, axis='y')
    # Highlight selected
    ax2.annotate('Selected', xy=(res["best_idx"], res["all_scores"][res["best_idx"]]),
                 xytext=(res["best_idx"]+2, res["all_scores"][res["best_idx"]]+0.3),
                 fontsize=7, arrowprops=dict(arrowstyle='->', color=C_OURS, lw=0.8),
                 color=C_OURS, fontweight='bold')

    plt.savefig(os.path.join(save_dir, "fig_trajectory_selection.pdf"))
    plt.savefig(os.path.join(save_dir, "fig_trajectory_selection.png"), dpi=300)
    plt.close()
    print("Saved fig_trajectory_selection.pdf/png")


# =====================================================================
# Figure 5: Init swing vs peak swing scatter (single-column)
# =====================================================================
def fig_init_vs_peak(results, save_dir):
    peaks = [r["peak_swing_deg"] for r in results]
    inits = [r["init_theta_deg"] for r in results]
    finals = [r["final_swing_deg"] for r in results]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    sc = ax.scatter(inits, peaks, c=finals, cmap='RdYlGn_r', s=12, alpha=0.7,
                    edgecolors='k', linewidths=0.15)
    # Reference lines
    max_val = max(max(peaks), max(inits)) * 1.05
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.5, alpha=0.4, label='$\\theta_{peak}=\\theta_0$')
    ax.axhline(19.2, color=C_MPC, linewidth=0.7, linestyle=':', label='MPC (19.2$^\\circ$)')
    ax.axhline(55.3, color=C_BASELINE, linewidth=0.7, linestyle=':', alpha=0.5, label='YOPO-Orig. (55.3$^\\circ$)')

    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label('Final swing ($^\\circ$)', fontsize=7)
    cb.ax.tick_params(labelsize=6)

    ax.set_xlabel('Initial swing $\\theta_0$ ($^\\circ$)')
    ax.set_ylabel('Peak swing $\\theta_{peak}$ ($^\\circ$)')
    ax.legend(fontsize=6.5, loc='upper left')
    ax.grid(True)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.savefig(os.path.join(save_dir, "fig_init_vs_peak.pdf"))
    plt.savefig(os.path.join(save_dir, "fig_init_vs_peak.png"), dpi=300)
    plt.close()
    print("Saved fig_init_vs_peak.pdf/png")


# =====================================================================
# Figure 6: Swing comparison bar chart (single-column)
# =====================================================================
def fig_method_comparison(results, save_dir):
    peaks = [r["peak_swing_deg"] for r in results]

    methods = ['YOPO\n(original)', 'Swing-aware\nMPC', 'YOPO-Payload\n(ours)']
    means = [55.3, 19.2, np.mean(peaks)]
    stds = [None, None, np.std(peaks)]  # only have std for ours
    colors = [C_BASELINE, C_MPC, C_OURS]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    bars = ax.bar(methods, means, color=colors, edgecolor='k', linewidth=0.5, width=0.55)

    # Error bar for ours only
    ax.errorbar(2, means[2], yerr=stds[2], fmt='none', color='k', capsize=4, linewidth=0.8)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}$^\\circ$', ha='center', fontsize=8, fontweight='bold')

    # Reduction annotation
    reduction = (55.3 - np.mean(peaks)) / 55.3 * 100
    ax.annotate('', xy=(2, means[2]+stds[2]+3), xytext=(0, means[0]+3),
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=0.8))
    mid_y = (means[0] + means[2]) / 2
    ax.text(1, mid_y + 3, f'{reduction:.0f}% reduction',
            ha='center', fontsize=7.5, color='#333333', fontweight='bold')

    ax.set_ylabel('Mean peak swing ($^\\circ$)')
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(True, axis='y')

    plt.savefig(os.path.join(save_dir, "fig_method_comparison.pdf"))
    plt.savefig(os.path.join(save_dir, "fig_method_comparison.png"), dpi=300)
    plt.close()
    print("Saved fig_method_comparison.pdf/png")


# =====================================================================
# Figure 7: Multi-panel swing time series comparison (double-column)
# =====================================================================
def fig_swing_timeseries_grid(results, save_dir):
    # Sort by peak swing, pick 6 evenly spaced
    sorted_res = sorted(results, key=lambda r: r["peak_swing_deg"])
    n = len(sorted_res)
    indices = [int(i * (n-1) / 5) for i in range(6)]
    picks = [sorted_res[i] for i in indices]

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.2),
                              gridspec_kw={'hspace': 0.5, 'wspace': 0.3})

    for idx, (ax, res) in enumerate(zip(axes.flat, picks)):
        ax.fill_between(res["t_ode"], -15, 15, alpha=0.12, color=C_SAFE)
        ax.plot(res["t_ode"], res["theta_deg"], color=C_BASELINE, linewidth=0.8)
        ax.axhline(0, color='k', linewidth=0.3)
        peak = res["peak_swing_deg"]
        init = res["init_theta_deg"]
        ax.set_title(f'$\\theta_0$={init:.0f}$^\\circ$ | peak={peak:.0f}$^\\circ$', fontsize=7.5)
        ax.set_xlim(0, cfg["sgm_time"])
        ax.grid(True)
        if idx >= 3:
            ax.set_xlabel('Time (s)')
        if idx % 3 == 0:
            ax.set_ylabel('$\\theta$ ($^\\circ$)')

    plt.savefig(os.path.join(save_dir, "fig_swing_timeseries_grid.pdf"))
    plt.savefig(os.path.join(save_dir, "fig_swing_timeseries_grid.png"), dpi=300)
    plt.close()
    print("Saved fig_swing_timeseries_grid.pdf/png")


# =====================================================================
# Main
# =====================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="saved/YOPO_43/epoch50.pth")
    parser.add_argument("--obs_dim", type=int, default=13)
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--output_dir", default="saved/YOPO_43/paper_figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Evaluating model...")
    results = get_eval_data(args.checkpoint, args.obs_dim, args.n_samples)

    peaks = [r["peak_swing_deg"] for r in results]
    print(f"\n--- {len(results)} samples ---")
    print(f"Mean peak: {np.mean(peaks):.1f}°, Median: {np.median(peaks):.1f}°, "
          f"P90: {np.percentile(peaks, 90):.1f}°, P95: {np.percentile(peaks, 95):.1f}°\n")

    print("Generating paper figures...")
    fig_trajectory_showcase(results, args.output_dir)
    fig_swing_distribution(results, args.output_dir)
    fig_detailed_trajectory(results, args.output_dir)
    fig_trajectory_selection(results, args.output_dir)
    fig_init_vs_peak(results, args.output_dir)
    fig_method_comparison(results, args.output_dir)
    fig_swing_timeseries_grid(results, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")
    print("Formats: PDF (vector) + PNG (300 dpi)")


if __name__ == "__main__":
    main()
