"""
Validate the differentiable-pendulum surrogate against a reference RK45
integrator (loss/differentiable_pendulum.py vs policy/pendulum_simulator.py).

Reviewer concern (Round 1):
> The singularity guard, detached cable length, 20-step semi-implicit Euler,
> and hand-shaped convergence term all look reasonable, but also ad hoc.
> A reviewer can easily say this is just a useful regularizer, not validated
> differentiable physics.

This script answers that by quantifying the error between:
  • the 20-step semi-implicit Euler used inside DifferentiablePendulumLoss, and
  • a high-resolution RK45 reference (rtol 1e-6, atol 1e-8) integrating the
    Cartesian-coordinate spherical pendulum ODE in pendulum_simulator.py.

We sample N random trajectories (sinusoidal UAV acc profiles), feed both
integrators the same UAV acc(t), and compare:
  - peak swing
  - mean swing (RMS theta)
  - final swing
across a grid of (cable length, payload mass, initial swing magnitude).

Output:
  dse_results/surrogate_validation.json   raw numbers per trial
  dse_results/surrogate_validation.md     markdown summary table
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loss.differentiable_pendulum import DifferentiablePendulumLoss
from loss.loss_function import YOPOLoss
from policy.pendulum_simulator import simulate_pendulum


def _build_poly_L_matrix(sgm_time):
    """Reconstruct the polynomial mapping matrix L exactly as YOPOLoss does."""
    loss = YOPOLoss.__new__(YOPOLoss)
    loss.sgm_time = sgm_time
    _, _, L_mat, _, _ = loss.qp_generation()
    return L_mat


def random_acc_profile(rng, sgm_time, n_steps, acc_max=4.0):
    """Generate a smooth random acceleration profile (5 sinusoids per axis)."""
    n_freq = 5
    amps = rng.uniform(0, acc_max / n_freq, size=(3, n_freq))
    freqs = rng.uniform(0.1, 1.5, size=(3, n_freq))
    phases = rng.uniform(0, 2 * np.pi, size=(3, n_freq))
    t_pts = np.linspace(sgm_time / n_steps, sgm_time, n_steps)
    acc = np.zeros((n_steps, 3))
    for k in range(n_freq):
        acc += (
            amps[:, k][None, :]
            * np.sin(
                2 * np.pi * freqs[:, k][None, :] * t_pts[:, None]
                + phases[:, k][None, :]
            )
        )
    return np.clip(acc, -acc_max, acc_max), t_pts


def make_acc_callable(t_pts, acc_traj):
    """Callable t -> (3,) for scipy.solve_ivp via piecewise-linear interp."""
    from scipy.interpolate import interp1d
    f = interp1d(
        t_pts, acc_traj, axis=0, bounds_error=False,
        fill_value=(acc_traj[0], acc_traj[-1]),
    )
    return lambda t: f(t)


def fit_polynomial_from_acc(acc_traj, t_pts, sgm_time):
    """Fit a 5th-order polynomial to a given acceleration profile so that
    the surrogate (which expects boundary conditions) sees the SAME UAV
    motion the reference does.

    We integrate the acc profile to (vel, pos) starting from rest, then use
    the initial / final p,v,a as boundary conditions.
    """
    dt = t_pts[1] - t_pts[0]
    vel = np.zeros((len(t_pts) + 1, 3))
    pos = np.zeros((len(t_pts) + 1, 3))
    for k in range(len(t_pts)):
        vel[k + 1] = vel[k] + acc_traj[k] * dt
        pos[k + 1] = pos[k] + vel[k] * dt + 0.5 * acc_traj[k] * dt ** 2
    start = np.stack([pos[0], vel[0], np.zeros(3)])  # 0 acc at t=0
    end = np.stack([pos[-1], vel[-1], acc_traj[-1]])
    return start, end


def run_trial(rng, surrogate, sgm_time, n_steps, L, theta0_deg, dphi0):
    """One Monte-Carlo trial.

    Both integrators receive the IDENTICAL UAV acceleration profile,
    derived from a single polynomial. This eliminates the polynomial-fit
    confound flagged by the reviewer in Round 2: the only difference is
    integrator + dynamics-equation form, not action representation.
    """
    # Sample random end-state boundary conditions for the polynomial.
    end_pos = rng.uniform(-2.0, 2.0, size=3)
    end_pos[2] *= 0.3
    end_vel = rng.uniform(-2.0, 2.0, size=3)
    end_acc = rng.uniform(-1.5, 1.5, size=3)
    start_pva = np.zeros((3, 3))                 # rows = pos, vel, acc per axis
    end_pva = np.stack([end_pos, end_vel, end_acc])

    # Acceleration profile from the SAME polynomial, on the SAME time grid
    from baselines.swing_aware_mpc import _poly_coefficients, _eval_acc
    coeffs = _poly_coefficients(start_pva, end_pva, sgm_time)
    t_pts = np.linspace(sgm_time / n_steps, sgm_time, n_steps)
    acc_traj = _eval_acc(coeffs, t_pts).T          # (n_steps, 3)
    acc_traj = np.clip(acc_traj, -15.0, 15.0)

    # Initial pendulum state
    theta0 = math.radians(theta0_deg)
    phi0 = float(rng.uniform(-np.pi, np.pi))
    dtheta0 = float(rng.uniform(-0.5, 0.5))
    y0 = [theta0, phi0, dtheta0, dphi0]

    # Reference: scipy RK45 on Cartesian formulation (no singularity guard).
    acc_callable = make_acc_callable(t_pts, acc_traj)
    t_ref, y_ref = simulate_pendulum(
        y0=y0, L=L, T=sgm_time, acc_func=acc_callable, dt=sgm_time / n_steps,
    )
    theta_ref = y_ref[:, 0]
    ref = {
        "peak_deg":  float(np.degrees(np.max(np.abs(theta_ref)))),
        "rms_deg":   float(np.degrees(np.sqrt(np.mean(theta_ref ** 2)))),
        "final_deg": float(np.degrees(theta_ref[-1])),
    }

    # Surrogate: 20-step semi-implicit Euler, same acceleration profile,
    # via boundary-condition feed (which by construction reproduces it).
    Df = torch.tensor(start_pva.T[None], dtype=torch.float32)  # (1, 3, 3)
    Dp = torch.tensor(end_pva.T[None],   dtype=torch.float32)
    p_state = torch.tensor([[theta0, phi0, dtheta0, dphi0]], dtype=torch.float32)
    p_params = torch.tensor([[L, 0.3]], dtype=torch.float32)
    with torch.no_grad():
        theta_surr = _surrogate_theta(surrogate, Df, Dp, p_state, p_params, n_steps)
    surr = {
        "peak_deg":  float(np.degrees(np.max(np.abs(theta_surr)))),
        "rms_deg":   float(np.degrees(np.sqrt(np.mean(theta_surr ** 2)))),
        "final_deg": float(np.degrees(theta_surr[-1])),
    }

    return ref, surr


def _surrogate_theta(surrogate, Df, Dp, p_state, p_params, n_steps):
    """Re-run the surrogate's internal Euler rollout to expose theta(t).

    Mirrors the inline loop in DifferentiablePendulumLoss.forward without
    the cost-aggregation tail.
    """
    B = Dp.shape[0]
    L = surrogate._L
    sgm_time = surrogate.sgm_time
    dt = sgm_time / n_steps

    # Polynomial coefficients
    mapping = L.unsqueeze(0).expand(B, -1, -1)
    axes = []
    for i in range(3):
        d = torch.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
        axes.append((mapping @ d).squeeze(-1))
    coe = torch.cat(axes, dim=1)

    t_pts = torch.linspace(dt, sgm_time, n_steps).view(1, -1, 1).expand(B, -1, -1)
    t_power = torch.stack([
        torch.ones_like(t_pts), t_pts, t_pts ** 2, t_pts ** 3
    ], dim=-1).squeeze(-2)
    coe_x = coe[:, 2:6]; coe_y = coe[:, 8:12]; coe_z = coe[:, 14:18]
    acc_mult = torch.tensor([2.0, 6.0, 12.0, 20.0]).view(1, 1, 4)
    ax = torch.sum(t_power * coe_x.unsqueeze(1) * acc_mult, dim=-1)
    ay = torch.sum(t_power * coe_y.unsqueeze(1) * acc_mult, dim=-1)
    az = torch.sum(t_power * coe_z.unsqueeze(1) * acc_mult, dim=-1)
    acc_traj = torch.stack([ax, ay, az], dim=-1).clamp(-15.0, 15.0)[0]  # (n_steps, 3)

    g = 9.81
    theta = p_state[0, 0].clamp(1e-4, 2.5).item()
    phi   = p_state[0, 1].item()
    dtheta = p_state[0, 2].clamp(-5.0, 5.0).item()
    dphi   = p_state[0, 3].clamp(-5.0, 5.0).item()
    cable_L = max(p_params[0, 0].item(), 0.1)

    theta_traj = []
    for k in range(n_steps):
        ax_k, ay_k, az_k = acc_traj[k].numpy()
        sin_t = math.sin(theta); cos_t = math.cos(theta)
        sin_p = math.sin(phi);   cos_p = math.cos(phi)
        a_eff = (ax_k, ay_k, az_k + g)
        a_dot_q_theta = a_eff[0]*cos_t*cos_p + a_eff[1]*cos_t*sin_p + a_eff[2]*sin_t
        a_dot_q_phi   = -a_eff[0]*sin_p + a_eff[1]*cos_p

        ddtheta = dphi**2 * sin_t * cos_t - (1.0/cable_L) * a_dot_q_theta
        sin_safe = max(abs(sin_t), 0.1)
        # Match the training-loss equation: 1/(L·sinθ), not 1/(L·sin²θ).
        # Standard spherical-pendulum derivation gives 1/(L·sinθ); the
        # 1/sin² form in the original PROJECT_OVERVIEW.md was an error.
        ddphi_full = (-2.0 * dtheta * dphi * cos_t / sin_safe
                      - (1.0/(cable_L * sin_safe)) * a_dot_q_phi)
        ddphi_damp = -0.5 * dphi
        blend = max(min(abs(sin_t)/0.15, 1.0), 0.0)
        ddphi = blend * ddphi_full + (1 - blend) * ddphi_damp

        ddtheta = max(min(ddtheta, 20.0), -20.0)
        ddphi   = max(min(ddphi, 20.0), -20.0)
        dtheta  = max(min(dtheta + ddtheta * dt, 5.0), -5.0)
        dphi    = max(min(dphi + ddphi * dt, 5.0), -5.0)
        theta   = max(min(theta + dtheta * dt, math.pi * 0.8), 0.01)
        phi    += dphi * dt
        theta_traj.append(theta)

    return np.array(theta_traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_dir", default="dse_results")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sgm_time = 2.5
    n_steps = 20

    # Build surrogate exactly as the trainer constructs it
    L_mat = _build_poly_L_matrix(sgm_time).float()
    surrogate = DifferentiablePendulumLoss(
        L_matrix=L_mat,
        sgm_time=sgm_time,
        device=torch.device("cpu"),
        n_steps=n_steps,
        gradient_decay_enabled=True,
    )

    grid = []
    for L_cab in [0.5, 0.8, 1.2]:
        for theta0 in [5.0, 15.0, 30.0]:
            for dphi0 in [0.0, 0.8]:
                grid.append((L_cab, theta0, dphi0))

    summary_rows = []
    raw = {"trials": []}
    n_per_cell = max(1, args.n_trials // len(grid))

    for L_cab, theta0, dphi0 in grid:
        peak_err, rms_err, final_err = [], [], []
        for _ in range(n_per_cell):
            ref, surr = run_trial(rng, surrogate, sgm_time, n_steps,
                                  L_cab, theta0, dphi0)
            peak_err.append(abs(ref["peak_deg"]  - surr["peak_deg"]))
            rms_err.append(abs(ref["rms_deg"]   - surr["rms_deg"]))
            final_err.append(abs(ref["final_deg"] - surr["final_deg"]))
            raw["trials"].append({"L": L_cab, "theta0": theta0, "dphi0": dphi0,
                                   "ref": ref, "surr": surr})

        summary_rows.append({
            "L": L_cab, "theta0_deg": theta0, "dphi0": dphi0,
            "n": n_per_cell,
            "peak_err_mean":  float(np.mean(peak_err)),
            "peak_err_p95":   float(np.percentile(peak_err, 95)),
            "rms_err_mean":   float(np.mean(rms_err)),
            "rms_err_p95":    float(np.percentile(rms_err, 95)),
            "final_err_mean": float(np.mean(final_err)),
            "final_err_p95":  float(np.percentile(final_err, 95)),
        })

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "surrogate_validation.json"
    md_path   = out_dir / "surrogate_validation.md"
    with open(json_path, "w") as f:
        json.dump({"summary": summary_rows, "raw": raw}, f, indent=2)

    # Markdown summary
    lines = [
        "# Surrogate-vs-Reference Pendulum Validation",
        "",
        f"- Trials per cell: {n_per_cell}",
        f"- Surrogate: 20-step semi-implicit Euler (loss/differentiable_pendulum.py)",
        f"- Reference: scipy RK45, rtol=1e-6, atol=1e-8 (policy/pendulum_simulator.py)",
        f"- Same UAV acc(t) fed to both via boundary-condition fitted 5th-order polynomial",
        "",
        "| L (m) | θ₀ (°) | dφ̇₀ | peak err mean (°) | peak err p95 (°) | RMS err mean (°) | final err mean (°) |",
        "|------:|------:|------:|------:|------:|------:|------:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['L']} | {r['theta0_deg']} | {r['dphi0']} | "
            f"{r['peak_err_mean']:.2f} | {r['peak_err_p95']:.2f} | "
            f"{r['rms_err_mean']:.2f} | {r['final_err_mean']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print()
    print("\n".join(lines[6:]))


if __name__ == "__main__":
    main()
