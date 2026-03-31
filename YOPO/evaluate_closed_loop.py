"""
Closed-loop evaluation for YOPO-Payload vs baselines.

For each method, this script:
1. Loads a trained checkpoint (or runs MPC online)
2. Generates trajectories on validation dataset scenes
3. Along each trajectory, integrates the pendulum ODE to get true swing
4. Reports: peak/RMS swing, dynamics cost, safety cost

YOPO models are evaluated dataset-driven (using real depth images from validation set).
MPC baseline is evaluated on equivalent kinematic conditions.

Usage:
    python evaluate_closed_loop.py --checkpoint_dir saved/ --n_samples 500
    python evaluate_closed_loop.py --methods B0 B3 B_MPC --n_samples 200
"""

import os
import sys
import csv
import json
import argparse
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.yopo_dataset import YOPODataset
from policy.state_transform import StateTransform, state_body2world
from policy.pendulum_simulator import simulate_pendulum
from loss.loss_function import YOPOLoss
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# MPC baseline (unchanged)
# ---------------------------------------------------------------------------

class SlungLoadMPC:
    """
    Simple model-predictive baseline for slung-load UAV transportation.
    Uses a receding-horizon polynomial planner with swing-aware cost:
      cost = w_goal * ||p_end - p_goal|| + w_swing * theta_eq + w_smooth * jerk²
    """

    def __init__(self, L=0.8, m=0.3, dt=0.1, horizon=2.0, vel_max=4.0, acc_max=4.0):
        self.L = L
        self.m = m
        self.dt = dt
        self.horizon = horizon
        self.vel_max = vel_max
        self.acc_max = acc_max
        self.G = 9.81

    def compute_eq_swing(self, acc):
        ax, ay, az = acc
        acc_xy = np.sqrt(ax**2 + ay**2)
        return np.arctan(acc_xy / max(az + self.G, 1e-3))

    def plan(self, pos, vel, acc, goal_pos, pendulum_state):
        goal_dir = goal_pos - pos
        goal_dist = np.linalg.norm(goal_dir)
        if goal_dist > 1e-3:
            goal_dir /= goal_dist

        best_cost = np.inf
        best_acc = np.zeros(3)

        for ax in np.linspace(-self.acc_max, self.acc_max, 7):
            for ay in np.linspace(-self.acc_max, self.acc_max, 7):
                az = 0.0
                T = self.horizon
                end_pos = pos + vel * T + 0.5 * np.array([ax, ay, az]) * T**2
                end_vel = vel + np.array([ax, ay, az]) * T

                goal_cost = np.linalg.norm(end_pos - goal_pos)
                swing_cost = self.compute_eq_swing(np.array([ax, ay, az]))
                smooth_cost = (ax**2 + ay**2 + az**2) * 0.1
                vel_penalty = max(0, np.linalg.norm(end_vel) - self.vel_max) * 5.0

                total = goal_cost + 3.0 * swing_cost + smooth_cost + vel_penalty
                if total < best_cost:
                    best_cost = total
                    best_acc = np.array([ax, ay, az])

        return best_acc, self.compute_eq_swing(best_acc)


# ---------------------------------------------------------------------------
# Pendulum evaluation helper
# ---------------------------------------------------------------------------

def evaluate_trajectory_with_pendulum(
    trajectory_acc: np.ndarray,  # (T, 3) UAV accelerations
    L: float,
    initial_pendulum_state: np.ndarray,  # [theta, phi, dtheta, dphi]
    t_traj: np.ndarray = None,  # (T,) time stamps; if None, use uniform dt
    dt: float = None,           # uniform dt (used only if t_traj is None)
    acc_max: float = 6.0,       # physical acceleration limit of the UAV
) -> dict:
    """
    Integrate pendulum ODE along a UAV acceleration profile.
    Clamps acceleration to physical UAV limits before simulation.
    Returns swing statistics.
    """
    from scipy.interpolate import interp1d

    if t_traj is None:
        assert dt is not None, "Must provide either t_traj or dt"
        t_traj = np.arange(len(trajectory_acc)) * dt

    T_total = t_traj[-1]
    if T_total < 0.01:
        return {"peak_swing_deg": 0.0, "rms_swing_deg": 0.0, "final_swing_deg": 0.0}

    # Clamp polynomial acceleration to physical UAV limits.
    # The 5th-order polynomial can produce mid-trajectory accelerations exceeding
    # the UAV's physical capability, which would unrealistically inflate swing.
    trajectory_acc_clamped = np.clip(trajectory_acc, -acc_max, acc_max)

    acc_interp = interp1d(t_traj, trajectory_acc_clamped, axis=0, bounds_error=False,
                          fill_value=(trajectory_acc_clamped[0], trajectory_acc_clamped[-1]))

    ode_dt = max(T_total / 200, 0.005)  # at least 200 steps or 5ms
    try:
        t_out, y_out = simulate_pendulum(
            y0=list(initial_pendulum_state),
            L=L, T=T_total,
            acc_func=lambda t: acc_interp(t),
            dt=ode_dt,
        )
        thetas = y_out[:, 0]
        return {
            "peak_swing_deg": float(np.degrees(np.max(np.abs(thetas)))),
            "rms_swing_deg":  float(np.degrees(np.sqrt(np.mean(thetas**2)))),
            "final_swing_deg": float(np.degrees(abs(thetas[-1]))),
        }
    except Exception as e:
        return {"peak_swing_deg": float('nan'), "rms_swing_deg": float('nan'),
                "final_swing_deg": float('nan')}


# ---------------------------------------------------------------------------
# YOPO trajectory extraction: get acceleration profile from polynomial
# ---------------------------------------------------------------------------

def extract_acc_from_polynomial(start_state_w, end_state_w, sgm_time, n_points=40):
    """
    Given start and end states in world frame, compute polynomial coefficients
    and extract the acceleration profile along the trajectory.

    start_state_w: (3, 3) [pos, vel, acc]
    end_state_w:   (3, 3) [pos, vel, acc]
    sgm_time: trajectory duration

    Returns: (t_arr, acc_profile) where t_arr is (n_points,) and acc_profile is (n_points, 3)
    """
    accs = []
    t_arr = np.linspace(0, sgm_time, n_points)  # Include t=0 for correct ODE alignment

    for axis in range(3):
        # Boundary conditions: p0, v0, a0, pf, vf, af
        p0 = start_state_w[0, axis]
        v0 = start_state_w[1, axis]
        a0 = start_state_w[2, axis]
        pf = end_state_w[0, axis]
        vf = end_state_w[1, axis]
        af = end_state_w[2, axis]

        # Solve 5th-order polynomial: p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
        # Boundary conditions give us 6 equations for 6 unknowns
        T = sgm_time
        A = np.array([
            [1, 0, 0, 0, 0, 0],           # p(0) = p0
            [0, 1, 0, 0, 0, 0],           # v(0) = v0
            [0, 0, 2, 0, 0, 0],           # a(0) = a0
            [1, T, T**2, T**3, T**4, T**5],       # p(T) = pf
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],  # v(T) = vf
            [0, 0, 2, 6*T, 12*T**2, 20*T**3],      # a(T) = af
        ])
        b = np.array([p0, v0, a0, pf, vf, af])

        try:
            coeff = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coeff = np.zeros(6)

        # Acceleration: a(t) = 2*c2 + 6*c3*t + 12*c4*t^2 + 20*c5*t^3
        axis_acc = (2 * coeff[2] + 6 * coeff[3] * t_arr
                    + 12 * coeff[4] * t_arr**2 + 20 * coeff[5] * t_arr**3)
        accs.append(axis_acc)

    return t_arr, np.stack(accs, axis=-1)  # (n_points,), (n_points, 3)


# ---------------------------------------------------------------------------
# YOPO model evaluation (dataset-driven)
# ---------------------------------------------------------------------------

def evaluate_yopo_model(
    checkpoint_path: str,
    obs_dim: int,
    n_samples: int = 500,
    batch_size: int = 16,
) -> list:
    """
    Evaluate a trained YOPO model on the validation dataset.
    For each sample:
    1. Run inference to get best trajectory
    2. Extract acceleration profile from polynomial
    3. Integrate pendulum ODE
    4. Report swing metrics

    Returns list of per-sample results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sgm_time = cfg["sgm_time"]
    traj_num = cfg["traj_num"]

    # Load model
    model = YopoNetwork(observation_dim=obs_dim).to(device)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  Loaded: {checkpoint_path}")
    else:
        print(f"  WARNING: {checkpoint_path} not found, using random weights")
    model.eval()

    # Load validation dataset with fixed seed for deterministic evaluation
    np.random.seed(42)
    val_ds = YOPODataset(mode='valid')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # num_workers=0 ensures deterministic ordering (no multiprocess randomness)

    results = []
    sample_count = 0

    with torch.inference_mode():
        for depth, pos, rot, obs_b, p_state, p_params, map_id in val_loader:
            if sample_count >= n_samples:
                break

            actual_bs = depth.shape[0]
            depth = depth.to(device)
            pos = pos.to(device)
            rot = rot.to(device)
            obs_b = obs_b.to(device)
            p_state = p_state.to(device)
            p_params = p_params.to(device)

            # Build augmented observation
            if obs_dim == 15:
                obs_augmented = torch.cat([obs_b, p_state, p_params], dim=-1)
            elif obs_dim == 14:
                obs_augmented = torch.cat([obs_b, p_state, p_params[:, :1]], dim=-1)
            elif obs_dim == 13:
                obs_augmented = torch.cat([obs_b, p_state], dim=-1)
            elif obs_dim == 12:
                obs_augmented = torch.cat([obs_b, p_state[:, :2], p_params[:, :1]], dim=-1)
            elif obs_dim == 10:
                obs_augmented = torch.cat([obs_b, p_params[:, :1]], dim=-1)
            else:
                obs_augmented = obs_b

            # Forward pass
            endstate, score = model.inference(depth, obs_augmented)

            # Get world-frame states
            goal_w, start_vel_w, start_acc_w = state_body2world(
                pos, rot, obs_b[:, 6:9], obs_b[:, 0:3], obs_b[:, 3:6])
            start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)

            # For each sample in batch, find best trajectory and evaluate
            for b in range(actual_bs):
                if sample_count >= n_samples:
                    break

                # Get scores for all 15 primitives, pick best (lowest cost)
                scores_b = score[b]  # [V, H]
                flat_scores = scores_b.reshape(-1)
                best_idx = torch.argmin(flat_scores).item()

                v_idx = best_idx // cfg["horizon_num"]
                h_idx = best_idx % cfg["horizon_num"]

                # Extract end state for best primitive
                end_pva = endstate[b, :, v_idx, h_idx].cpu().numpy()  # [9]

                # Transform to world frame
                rot_b = rot[b].cpu().numpy()
                pos_b = pos[b].cpu().numpy()

                end_pos_w = rot_b @ end_pva[0:3] + pos_b
                end_vel_w = rot_b @ end_pva[3:6]
                end_acc_w = rot_b @ end_pva[6:9]

                start_sw = start_state_w[b].cpu().numpy()  # [3, 3]: [pos, vel, acc]

                end_sw = np.array([end_pos_w, end_vel_w, end_acc_w])

                # Extract acceleration profile from polynomial
                t_arr, acc_profile = extract_acc_from_polynomial(start_sw, end_sw, sgm_time)

                # Evaluate with pendulum ODE
                L = p_params[b, 0].item()
                p_st = p_state[b].cpu().numpy()

                swing = evaluate_trajectory_with_pendulum(
                    acc_profile, L, p_st, t_traj=t_arr)

                results.append({
                    "success": True,  # Per-segment eval: always True (trajectory was generated)
                    "time_to_goal": float(sgm_time),  # Segment duration (not episode time)
                    **swing,
                })
                sample_count += 1

    return results


# ---------------------------------------------------------------------------
# MPC baseline on generated episodes
# ---------------------------------------------------------------------------

def generate_benchmark_episodes(n_episodes: int = 200, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(n_episodes):
        start_pos = rng.uniform([-5, -5, 1.5], [5, 5, 3.0])
        goal_pos  = start_pos + rng.uniform([-10, -10, -1], [10, 10, 1])
        start_vel = rng.uniform([-2, -2, -0.5], [2, 2, 0.5])
        L = rng.uniform(0.4, 1.2)
        m = rng.uniform(0.1, 0.8)
        init_theta = rng.uniform(0.0, 0.25)
        init_phi   = rng.uniform(0, 2 * np.pi)
        init_dtheta = rng.uniform(-0.3, 0.3)
        init_dphi   = rng.uniform(-0.3, 0.3)
        episodes.append({
            "start_pos": start_pos, "goal_pos": goal_pos,
            "start_vel": start_vel, "L": L, "m": m,
            "init_pendulum": np.array([init_theta, init_phi, init_dtheta, init_dphi]),
        })
    return episodes


def run_mpc_baseline(episodes: list, n_steps: int = 80, dt: float = 0.1) -> list:
    results = []
    for ep in episodes:
        mpc = SlungLoadMPC(L=ep["L"], m=ep["m"], dt=dt)
        pos = ep["start_pos"].copy()
        vel = ep["start_vel"].copy()
        acc = np.zeros(3)
        goal = ep["goal_pos"]

        traj_acc = []
        success = False

        for step in range(n_steps):
            cmd_acc, _ = mpc.plan(pos, vel, acc, goal, ep["init_pendulum"])
            pos = pos + vel * dt + 0.5 * cmd_acc * dt**2
            vel = np.clip(vel + cmd_acc * dt, -mpc.vel_max, mpc.vel_max)
            acc = cmd_acc
            traj_acc.append(cmd_acc.copy())

            if np.linalg.norm(pos - goal) < 1.0:
                success = True
                break

        traj_acc = np.array(traj_acc)

        swing = evaluate_trajectory_with_pendulum(
            traj_acc, ep["L"], ep["init_pendulum"], dt=dt)
        results.append({
            "success": success,
            "time_to_goal": (len(traj_acc) * dt) if success else float('nan'),
            **swing,
        })
    return results


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def compute_metrics_summary(results: list) -> dict:
    successes = [r["success"] for r in results]
    peaks = [r["peak_swing_deg"] for r in results if not np.isnan(r["peak_swing_deg"])]
    rms   = [r["rms_swing_deg"] for r in results if not np.isnan(r["rms_swing_deg"])]
    ttgs  = [r["time_to_goal"] for r in results if r["success"] and not np.isnan(r.get("time_to_goal", float('nan')))]

    return {
        "n_samples": len(results),
        "success_rate": float(np.mean(successes)),
        "mean_peak_swing_deg": float(np.mean(peaks)) if peaks else float('nan'),
        "std_peak_swing_deg":  float(np.std(peaks)) if peaks else float('nan'),
        "mean_rms_swing_deg":  float(np.mean(rms)) if rms else float('nan'),
        "std_rms_swing_deg":   float(np.std(rms)) if rms else float('nan'),
        "mean_time_to_goal_s": float(np.mean(ttgs)) if ttgs else float('nan'),
    }


# ---------------------------------------------------------------------------
# Method configurations
# ---------------------------------------------------------------------------

METHODS = {
    "B0_YOPO_original":     dict(obs_dim=9,  ckpt_subdir="ablation_A0_baseline"),
    "B1_YOPO_inflated":     dict(obs_dim=9,  ckpt_subdir="ablation_B1_inflated"),
    "B2_YOPO_acc_penalty":  dict(obs_dim=9,  ckpt_subdir="ablation_B2_acc_penalty"),
    "B3_YOPO_Payload":      dict(obs_dim=15, ckpt_subdir="ablation_A4_full_model"),
    "A1_cable_L":           dict(obs_dim=10, ckpt_subdir="ablation_A1_cable_L"),
    "A2_swing_ang":         dict(obs_dim=12, ckpt_subdir="ablation_A2_swing_ang"),
    "A3_swing_full":        dict(obs_dim=14, ckpt_subdir="ablation_A3_swing_full"),
    "A4_no_mass":           dict(obs_dim=14, ckpt_subdir="ablation_A4_no_mass"),
    "D1_no_detach":         dict(obs_dim=15, ckpt_subdir="ablation_D1_no_detach"),
    "A1_cable_L_dyn":       dict(obs_dim=10, ckpt_subdir="ablation_A1_cable_L_dyn"),
    "A4_no_dyn":            dict(obs_dim=15, ckpt_subdir="ablation_A4_no_dyn"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="saved")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to evaluate (default: all). Use 'B_MPC' for MPC baseline.")
    parser.add_argument("--output", type=str, default="saved/closed_loop_results.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    methods_to_run = args.methods or (list(METHODS.keys()) + ["B_MPC"])

    all_results = {}

    # Evaluate YOPO methods
    for name in methods_to_run:
        if name == "B_MPC":
            continue  # handled separately

        if name not in METHODS:
            print(f"Unknown method: {name}, skipping")
            continue

        config = METHODS[name]
        ckpt = os.path.join(args.checkpoint_dir, config["ckpt_subdir"], f"epoch{args.epoch}.pth")
        print(f"\nEvaluating {name} (obs_dim={config['obs_dim']})...")

        per_sample = evaluate_yopo_model(
            checkpoint_path=ckpt,
            obs_dim=config["obs_dim"],
            n_samples=args.n_samples,
            batch_size=args.batch_size,
        )
        all_results[name] = compute_metrics_summary(per_sample)
        print(f"  {name}: peak={all_results[name]['mean_peak_swing_deg']:.2f}° "
              f"rms={all_results[name]['mean_rms_swing_deg']:.2f}°")

    # Evaluate MPC baseline
    if "B_MPC" in methods_to_run:
        print(f"\nEvaluating B_MPC baseline ({args.n_samples} episodes)...")
        episodes = generate_benchmark_episodes(args.n_samples, seed=args.seed)
        mpc_per_sample = run_mpc_baseline(episodes)
        all_results["B_MPC"] = compute_metrics_summary(mpc_per_sample)
        print(f"  B_MPC: success={all_results['B_MPC']['success_rate']:.1%} "
              f"peak={all_results['B_MPC']['mean_peak_swing_deg']:.2f}°")

    # Print comparison table
    print(f"\n{'='*100}")
    print("Closed-Loop Evaluation Summary")
    print(f"{'='*100}")
    header = (f"{'Method':<25} | {'N':>5} | {'Success':>7} | "
              f"{'Peak Swing':>14} | {'RMS Swing':>14} | {'Time(s)':>8}")
    print(header)
    print("-" * 100)
    for name, m in all_results.items():
        peak_str = f"{m['mean_peak_swing_deg']:>6.2f}±{m['std_peak_swing_deg']:.2f}°"
        rms_str  = f"{m['mean_rms_swing_deg']:>6.2f}±{m['std_rms_swing_deg']:.2f}°"
        print(f"{name:<25} | {m['n_samples']:>5} | {m['success_rate']:>7.1%} | "
              f"{peak_str:>14} | {rms_str:>14} | {m['mean_time_to_goal_s']:>8.2f}s")

    # Save CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        fieldnames = ["method", "n_samples", "success_rate", "mean_peak_swing_deg",
                      "std_peak_swing_deg", "mean_rms_swing_deg", "std_rms_swing_deg",
                      "mean_time_to_goal_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, m in all_results.items():
            writer.writerow({"method": name, **m})
    print(f"\nSaved to {args.output}")

    # Save JSON for programmatic access
    json_output = args.output.replace(".csv", ".json")
    with open(json_output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {json_output}")


if __name__ == "__main__":
    main()
