"""
A/B compare the spherical vs Cartesian differentiable-pendulum surrogate
against a high-resolution RK45 reference.

Same protocol as `validate_surrogate.py`, but runs both surrogates side by
side and reports the peak-swing-error delta per cell. Use this to decide
whether the Cartesian rewrite is worth a training run.

Output:
  dse_results/surrogate_comparison.json
  dse_results/surrogate_comparison.md
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loss.differentiable_pendulum import DifferentiablePendulumLoss
from loss.cartesian_pendulum import CartesianPendulumLoss
from policy.pendulum_simulator import simulate_pendulum
from validate_surrogate import (
    _build_poly_L_matrix,
    _surrogate_theta as _spherical_theta,
    make_acc_callable,
)


def _cartesian_theta(surrogate, Df, Dp, p_state, p_params, n_steps):
    """Mirror CartesianPendulumLoss.forward but expose theta_k(t)."""
    L_mat = surrogate._L
    sgm_time = surrogate.sgm_time
    dt = sgm_time / n_steps

    # Polynomial coefficients (same shape as the spherical helper).
    B = Dp.shape[0]
    mapping = L_mat.unsqueeze(0).expand(B, -1, -1)
    axes = []
    for i in range(3):
        d = torch.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
        axes.append((mapping @ d).squeeze(-1))
    coe = torch.cat(axes, dim=1)

    t_pts = torch.linspace(dt, sgm_time, n_steps).view(1, -1, 1).expand(B, -1, -1)
    t_power = torch.stack(
        [torch.ones_like(t_pts), t_pts, t_pts ** 2, t_pts ** 3], dim=-1
    ).squeeze(-2)
    coe_x = coe[:, 2:6]; coe_y = coe[:, 8:12]; coe_z = coe[:, 14:18]
    acc_mult = torch.tensor([2.0, 6.0, 12.0, 20.0]).view(1, 1, 4)
    ax = torch.sum(t_power * coe_x.unsqueeze(1) * acc_mult, dim=-1)
    ay = torch.sum(t_power * coe_y.unsqueeze(1) * acc_mult, dim=-1)
    az = torch.sum(t_power * coe_z.unsqueeze(1) * acc_mult, dim=-1)
    acc_traj = torch.stack([ax, ay, az], dim=-1).clamp(-15.0, 15.0)[0].numpy()

    g = 9.81
    theta0 = max(p_state[0, 0].item(), 1e-4)
    phi0   = p_state[0, 1].item()
    dth0   = max(min(p_state[0, 2].item(), 5.0), -5.0)
    dph0   = max(min(p_state[0, 3].item(), 5.0), -5.0)
    cable_L = max(p_params[0, 0].item(), 0.1)

    st, ct = math.sin(theta0), math.cos(theta0)
    sp, cp = math.sin(phi0),   math.cos(phi0)
    qx, qy, qz = st * cp, st * sp, -ct
    dqx = ct * cp * dth0 - st * sp * dph0
    dqy = ct * sp * dth0 + st * cp * dph0
    dqz = st * dth0

    theta_traj = []
    for k in range(n_steps):
        ax_k, ay_k, az_k = acc_traj[k]
        # g_eff in pivot non-inertial frame (z-up world, gravity along -z).
        gex, gey, gez = -ax_k, -ay_k, -(g + az_k)
        gdq = gex * qx + gey * qy + gez * qz
        gtx = gex - gdq * qx
        gty = gey - gdq * qy
        gtz = gez - gdq * qz
        dq2 = dqx * dqx + dqy * dqy + dqz * dqz
        ddqx = max(min(gtx / cable_L - dq2 * qx, 50.0), -50.0)
        ddqy = max(min(gty / cable_L - dq2 * qy, 50.0), -50.0)
        ddqz = max(min(gtz / cable_L - dq2 * qz, 50.0), -50.0)
        dqx = max(min(dqx + ddqx * dt, 10.0), -10.0)
        dqy = max(min(dqy + ddqy * dt, 10.0), -10.0)
        dqz = max(min(dqz + ddqz * dt, 10.0), -10.0)
        qx = qx + dqx * dt
        qy = qy + dqy * dt
        qz = qz + dqz * dt
        qn = max(math.sqrt(qx * qx + qy * qy + qz * qz), 1e-6)
        qx, qy, qz = qx / qn, qy / qn, qz / qn
        # theta from -qz (q_z = -cos theta).
        theta = math.acos(max(min(-qz, 1.0 - 1e-9), -1.0 + 1e-9))
        theta_traj.append(theta)

    return np.array(theta_traj)


def run_trial(rng, surr_sph, surr_cart, sgm_time, n_steps, L_cab, theta0_deg, dphi0):
    """Same trajectory fed to RK45, spherical surrogate, and Cartesian surrogate."""
    end_pos = rng.uniform(-2.0, 2.0, size=3)
    end_pos[2] *= 0.3
    end_vel = rng.uniform(-2.0, 2.0, size=3)
    end_acc = rng.uniform(-1.5, 1.5, size=3)
    start_pva = np.zeros((3, 3))
    end_pva = np.stack([end_pos, end_vel, end_acc])

    from baselines.swing_aware_mpc import _poly_coefficients, _eval_acc
    coeffs = _poly_coefficients(start_pva, end_pva, sgm_time)
    t_pts = np.linspace(sgm_time / n_steps, sgm_time, n_steps)
    acc_traj = _eval_acc(coeffs, t_pts).T
    acc_traj = np.clip(acc_traj, -15.0, 15.0)

    theta0 = math.radians(theta0_deg)
    phi0 = float(rng.uniform(-np.pi, np.pi))
    dtheta0 = float(rng.uniform(-0.5, 0.5))
    y0 = [theta0, phi0, dtheta0, dphi0]

    # RK45 reference.
    acc_callable = make_acc_callable(t_pts, acc_traj)
    _, y_ref = simulate_pendulum(
        y0=y0, L=L_cab, T=sgm_time, acc_func=acc_callable, dt=sgm_time / n_steps,
    )
    theta_ref = y_ref[:, 0]
    ref_peak = float(np.degrees(np.max(np.abs(theta_ref))))

    Df = torch.tensor(start_pva.T[None], dtype=torch.float32)
    Dp = torch.tensor(end_pva.T[None],   dtype=torch.float32)
    p_state = torch.tensor([[theta0, phi0, dtheta0, dphi0]], dtype=torch.float32)
    p_params = torch.tensor([[L_cab, 0.3]], dtype=torch.float32)

    with torch.no_grad():
        theta_sph = _spherical_theta(surr_sph, Df, Dp, p_state, p_params, n_steps)
        theta_cart = _cartesian_theta(surr_cart, Df, Dp, p_state, p_params, n_steps)

    sph_peak = float(np.degrees(np.max(np.abs(theta_sph))))
    cart_peak = float(np.degrees(np.max(np.abs(theta_cart))))
    return ref_peak, sph_peak, cart_peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=900)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_dir", default="dse_results")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sgm_time = 2.5
    n_steps = 20

    L_mat = _build_poly_L_matrix(sgm_time).float()
    surr_sph  = DifferentiablePendulumLoss(L_mat, sgm_time, torch.device("cpu"), n_steps, True)
    surr_cart = CartesianPendulumLoss(L_mat, sgm_time, torch.device("cpu"), n_steps, True)

    grid = []
    for L_cab in [0.5, 0.8, 1.2]:
        for theta0 in [5.0, 15.0, 30.0]:
            for dphi0 in [0.0, 0.8]:
                grid.append((L_cab, theta0, dphi0))
    n_per_cell = max(1, args.n_trials // len(grid))

    rows = []
    raw_trials = []
    for L_cab, theta0, dphi0 in grid:
        sph_errs, cart_errs = [], []
        for _ in range(n_per_cell):
            ref, sph, cart = run_trial(
                rng, surr_sph, surr_cart, sgm_time, n_steps, L_cab, theta0, dphi0,
            )
            sph_errs.append(abs(sph - ref))
            cart_errs.append(abs(cart - ref))
            raw_trials.append({
                "L": L_cab, "theta0": theta0, "dphi0": dphi0,
                "ref_peak": ref, "sph_peak": sph, "cart_peak": cart,
            })
        rows.append({
            "L": L_cab, "theta0": theta0, "dphi0": dphi0, "n": n_per_cell,
            "sph_peak_err_mean":  float(np.mean(sph_errs)),
            "sph_peak_err_p95":   float(np.percentile(sph_errs, 95)),
            "cart_peak_err_mean": float(np.mean(cart_errs)),
            "cart_peak_err_p95":  float(np.percentile(cart_errs, 95)),
            "improvement_deg":    float(np.mean(sph_errs) - np.mean(cart_errs)),
        })

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "surrogate_comparison.json"
    md_path   = out_dir / "surrogate_comparison.md"
    with open(json_path, "w") as f:
        json.dump({"summary": rows, "raw": raw_trials}, f, indent=2)

    overall_sph  = float(np.mean([r["sph_peak_err_mean"]  for r in rows]))
    overall_cart = float(np.mean([r["cart_peak_err_mean"] for r in rows]))

    lines = [
        "# Surrogate A/B: Spherical vs Cartesian",
        "",
        f"- Trials per cell: {n_per_cell}    (total: {n_per_cell * len(grid)})",
        "- Reference: scipy RK45 rtol 1e-6, atol 1e-8 (`policy/pendulum_simulator.py`)",
        "- Spherical surrogate: `loss/differentiable_pendulum.py` (clamp 0.01 ≤ θ ≤ 0.8π)",
        "- Cartesian surrogate: `loss/cartesian_pendulum.py` (no clamp; explicit |q|=1 projection)",
        "- Same polynomial-derived UAV acc(t) fed to all three integrators.",
        "",
        "| L (m) | θ₀ (°) | dφ̇₀ | sph peak err mean (°) | cart peak err mean (°) | Δ (°) |",
        "|------:|------:|------:|------:|------:|------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['L']} | {r['theta0']} | {r['dphi0']} | "
            f"{r['sph_peak_err_mean']:.2f} | {r['cart_peak_err_mean']:.2f} | "
            f"{r['improvement_deg']:+.2f} |"
        )
    lines += [
        "",
        f"**Overall mean peak error:** "
        f"spherical {overall_sph:.2f}° → cartesian {overall_cart:.2f}° "
        f"(Δ = {overall_sph - overall_cart:+.2f}°). "
        f"Positive Δ means Cartesian closer to RK45.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print()
    print("\n".join(lines[7:]))


if __name__ == "__main__":
    main()
