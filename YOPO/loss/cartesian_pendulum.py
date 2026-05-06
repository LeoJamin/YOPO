"""
Cartesian-formulation differentiable pendulum rollout loss.

Drop-in replacement for `DifferentiablePendulumLoss` (spherical formulation).
Same constructor and forward signature, so callers don't change.

Why this exists
---------------
The spherical-coord version `clamp(theta, 0.01, pi * 0.8)` after every
integration step. When the true dynamics carry the polar angle through
zero (e.g. swing reverses), the clamp saturates the gradient at exactly
the moment the physics matters most. The auto-review-loop's surrogate-vs-RK45
validation found this drives a 9-29 deg systematic under-prediction bias
in peak swing.

This module integrates the same Lagrangian in Cartesian unit-direction form
q = [qx, qy, qz] (|q| = 1, q points from pivot to payload). The constraint
is enforced by explicit normalization after each step — no clamp, no
singularity.

Cost shaping
------------
Step cost     : (qx^2 + qy^2) + 0.1 * |dq|^2
                (== sin^2(theta) + 0.1 * (theta_dot^2 + sin^2(theta) * phi_dot^2))
Aggregation   : same mean + soft-max-peak + final + convergence as spherical.
Length scale  : 1/L, identical to spherical.

Validated against scipy RK45 reference; see `validate_surrogate.py`.
"""

import torch
import torch.nn as nn

G = 9.81


class CartesianPendulumLoss(nn.Module):
    """Cartesian-coord differentiable pendulum rollout. Drop-in for DifferentiablePendulumLoss."""

    def __init__(self, L_matrix: torch.Tensor, sgm_time: float, device: torch.device,
                 n_steps: int = 20, gradient_decay_enabled: bool = True):
        super().__init__()
        self._L = L_matrix
        self.sgm_time = sgm_time
        self.device = device
        self.n_steps = n_steps
        self.dt = sgm_time / n_steps
        self.g = G
        self.gradient_decay_enabled = gradient_decay_enabled

    def forward(self, Df: torch.Tensor, Dp: torch.Tensor,
                p_state: torch.Tensor, p_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Df:       (B, 3, 3) initial state [pos, vel, acc] per axis
            Dp:       (B, 3, 3) predicted end state per axis
            p_state:  (B, 4) [theta, phi, dtheta, dphi] (spherical, for API compat)
            p_params: (B, 2) [length, mass]

        Returns:
            dynamics_cost: (B,)
        """
        B = Dp.shape[0]
        mapping_matrix = self._L.unsqueeze(0).expand(B, -1, -1)

        # 1. Polynomial coefficients (same as spherical version).
        coe = self._get_coefficients(Dp, Df, mapping_matrix)

        # 2. Dense time points.
        t_points = torch.linspace(self.dt, self.sgm_time, self.n_steps, device=self.device)
        t_list = t_points.view(1, -1, 1).expand(B, -1, -1)

        # 3. UAV acceleration along the trajectory (B, n_steps, 3).
        acc_traj = self._get_acceleration(coe, t_list).clamp(-15.0, 15.0)

        # 4. Initial pendulum state — convert spherical -> Cartesian. Detached.
        p_state_d = p_state.detach()
        theta0 = p_state_d[:, 0].clamp(min=1e-4)  # avoid the exact singularity at theta=0
        phi0   = p_state_d[:, 1]
        dth0   = p_state_d[:, 2].clamp(-5.0, 5.0)
        dph0   = p_state_d[:, 3].clamp(-5.0, 5.0)

        # q = [sin th cos ph, sin th sin ph, -cos th]
        st, ct = torch.sin(theta0), torch.cos(theta0)
        sp, cp = torch.sin(phi0),   torch.cos(phi0)
        qx = st * cp
        qy = st * sp
        qz = -ct
        # dq/dt from (dtheta, dphi)
        dqx = ct * cp * dth0 - st * sp * dph0
        dqy = ct * sp * dth0 + st * cp * dph0
        dqz = st * dth0

        # Initial swing energy proxy for the convergence term.
        init_swing_sq = qx * qx + qy * qy
        init_dq_sq    = dqx * dqx + dqy * dqy + dqz * dqz
        initial_energy = init_swing_sq + 0.1 * init_dq_sq

        # 5. Cable length (detached).
        cable_L = p_params[:, 0].detach().clamp(min=0.1)

        # 6. Temporal gradient decay matched to natural frequency.
        omega = torch.sqrt(torch.tensor(self.g, device=self.device) / cable_L)
        decay_per_step = (torch.exp(-omega * self.dt)
                          if self.gradient_decay_enabled
                          else torch.ones(B, device=self.device))

        # 7. Cartesian semi-implicit Euler with explicit normalization.
        swing_costs = []
        cumulative_decay = torch.ones(B, device=self.device)

        for k in range(self.n_steps):
            ax = acc_traj[:, k, 0]
            ay = acc_traj[:, k, 1]
            az = acc_traj[:, k, 2]

            # Effective gravity in pivot non-inertial frame (z-up world, g along -z).
            #   g_eff = -a_uav - g * z_hat
            ge_x = -ax
            ge_y = -ay
            ge_z = -(self.g + az)

            # Tangent component: g_eff - (g_eff . q) q.
            g_dot_q = ge_x * qx + ge_y * qy + ge_z * qz
            gt_x = ge_x - g_dot_q * qx
            gt_y = ge_y - g_dot_q * qy
            gt_z = ge_z - g_dot_q * qz

            # Centripetal: (dq . dq) q.
            dq_sq = dqx * dqx + dqy * dqy + dqz * dqz
            cx = dq_sq * qx
            cy = dq_sq * qy
            cz = dq_sq * qz

            # Cartesian angular acceleration.
            ddqx = (gt_x / cable_L - cx).clamp(-50.0, 50.0)
            ddqy = (gt_y / cable_L - cy).clamp(-50.0, 50.0)
            ddqz = (gt_z / cable_L - cz).clamp(-50.0, 50.0)

            # Semi-implicit Euler: dq first, then q.
            dqx = (dqx + ddqx * self.dt).clamp(-10.0, 10.0)
            dqy = (dqy + ddqy * self.dt).clamp(-10.0, 10.0)
            dqz = (dqz + ddqz * self.dt).clamp(-10.0, 10.0)
            qx = qx + dqx * self.dt
            qy = qy + dqy * self.dt
            qz = qz + dqz * self.dt

            # Project back onto |q| = 1. Replaces the spherical theta.clamp.
            q_norm = torch.sqrt(qx * qx + qy * qy + qz * qz).clamp(min=1e-6)
            qx = qx / q_norm
            qy = qy / q_norm
            qz = qz / q_norm

            # Step cost: sin^2(theta) + 0.1 * |dq|^2.
            swing_sq    = qx * qx + qy * qy
            dq_sq_post  = dqx * dqx + dqy * dqy + dqz * dqz
            step_cost   = swing_sq + 0.1 * dq_sq_post

            cumulative_decay = cumulative_decay * decay_per_step
            swing_costs.append(step_cost * cumulative_decay)

        # 8. Aggregate (same shape as spherical version).
        swing_stack = torch.stack(swing_costs, dim=1)  # (B, n_steps)
        mean_cost   = swing_stack.mean(dim=1)
        alpha_lse   = 5.0
        peak_cost   = torch.logsumexp(alpha_lse * swing_stack, dim=1) / alpha_lse

        final_swing_sq = qx * qx + qy * qy
        final_dq_sq    = dqx * dqx + dqy * dqy + dqz * dqz
        final_energy   = final_swing_sq + 0.1 * final_dq_sq

        convergence_cost = torch.relu(final_energy - initial_energy)
        final_cost       = final_swing_sq

        length_scale = 1.0 / cable_L.clamp(min=0.1)

        return (1.0 * mean_cost
                + 0.5 * peak_cost
                + 0.3 * final_cost
                + 0.8 * convergence_cost) * length_scale

    def _get_coefficients(self, Dp: torch.Tensor, Df: torch.Tensor,
                          L: torch.Tensor) -> torch.Tensor:
        axes = []
        for i in range(3):
            d = torch.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
            axes.append((L @ d).squeeze(-1))
        return torch.cat(axes, dim=1)  # (B, 18)

    def _get_acceleration(self, coe: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_power = torch.stack([torch.ones_like(t), t, t ** 2, t ** 3], dim=-1).squeeze(-2)

        coe_x = coe[:, 2:6]
        coe_y = coe[:, 8:12]
        coe_z = coe[:, 14:18]

        acc_mult = torch.tensor([2.0, 6.0, 12.0, 20.0], device=self.device).view(1, 1, 4)

        ax = torch.sum(t_power * coe_x.unsqueeze(1) * acc_mult, dim=-1)
        ay = torch.sum(t_power * coe_y.unsqueeze(1) * acc_mult, dim=-1)
        az = torch.sum(t_power * coe_z.unsqueeze(1) * acc_mult, dim=-1)

        return torch.stack([ax, ay, az], dim=-1)
