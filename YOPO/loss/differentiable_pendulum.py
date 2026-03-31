"""
Differentiable Pendulum Rollout Loss.

Replaces the linearized harmonic oscillator with a full nonlinear spherical
pendulum ODE integration, fully differentiable via torch.autograd.

Key innovations over the linearized version:
1. Nonlinear ODE: valid for ALL swing angles, not just <30 deg
2. Temporal gradient decay: alpha matched to pendulum natural frequency
   - Physical interpretation: gradient window = one swing period
   - Prevents over-penalization of far-future swing
3. Semi-implicit Euler integration: stable and differentiable

Reference: Zhang et al., "Back to Newton's Laws", Nature Machine Intelligence, 2025
  - We extend their temporal gradient decay from single rigid body to
    coupled multi-body oscillatory dynamics (UAV + spherical pendulum).
"""

import torch
import torch.nn as nn
import math

G = 9.81


class DifferentiablePendulumLoss(nn.Module):
    """Nonlinear differentiable pendulum rollout with temporal gradient decay.

    For each polynomial trajectory, evaluates UAV acceleration at dense
    timesteps, then integrates the full nonlinear spherical pendulum ODE.
    Temporal gradient decay (e^{-alpha*dt}) is applied per integration step,
    where alpha is matched to the pendulum's natural frequency.
    """

    def __init__(self, L_matrix: torch.Tensor, sgm_time: float, device: torch.device,
                 n_steps: int = 20, gradient_decay_enabled: bool = True):
        """
        Args:
            L_matrix: polynomial mapping matrix (6, 6)
            sgm_time: segment time duration (s)
            device: torch device
            n_steps: number of ODE integration steps
            gradient_decay_enabled: whether to apply temporal gradient decay
        """
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
            Df: (B, 3, 3) initial state [pos, vel, acc] per axis
            Dp: (B, 3, 3) predicted end state per axis
            p_state: (B, 4) [theta, phi, dtheta, dphi]
            p_params: (B, 2) [length, mass]

        Returns:
            dynamics_cost: (B,) scalar cost per sample
        """
        B = Dp.shape[0]
        mapping_matrix = self._L.unsqueeze(0).expand(B, -1, -1)

        # 1. Get polynomial coefficients from boundary conditions
        coe = self._get_coefficients(Dp, Df, mapping_matrix)

        # 2. Generate dense time points for ODE integration
        t_points = torch.linspace(
            self.dt, self.sgm_time, self.n_steps, device=self.device
        )
        t_list = t_points.view(1, -1, 1).expand(B, -1, -1)

        # 3. Evaluate trajectory acceleration at each timestep
        acc_traj = self._get_acceleration(coe, t_list)  # (B, n_steps, 3)
        acc_traj = acc_traj.clamp(-15.0, 15.0)

        # 4. Initial pendulum state (detached — no gradient through IC)
        p_state_d = p_state.detach()
        theta = p_state_d[:, 0].clamp(1e-4, 2.5)   # (B,)
        phi = p_state_d[:, 1]
        dtheta = p_state_d[:, 2].clamp(-5.0, 5.0)
        dphi = p_state_d[:, 3].clamp(-5.0, 5.0)

        # Record initial swing energy for convergence incentive
        initial_theta = theta.clone()  # (B,)
        initial_energy = (theta ** 2
                          + 0.1 * (dtheta ** 2 + dphi ** 2 * torch.sin(theta) ** 2))  # (B,)

        # 5. Cable length (detached)
        cable_L = p_params[:, 0].detach().clamp(min=0.1)  # (B,)

        # 6. Compute physics-matched gradient decay rate
        #    alpha = natural frequency omega = sqrt(g/L)
        #    This means gradients decay over ~1 swing period
        omega_natural = torch.sqrt(torch.tensor(self.g, device=self.device) / cable_L)  # (B,)
        if self.gradient_decay_enabled:
            decay_per_step = torch.exp(-omega_natural * self.dt)  # (B,)
        else:
            decay_per_step = torch.ones(B, device=self.device)

        # 7. Nonlinear ODE integration with temporal gradient decay
        swing_costs = []
        cumulative_decay = torch.ones(B, device=self.device)

        for k in range(self.n_steps):
            ax = acc_traj[:, k, 0]  # (B,)
            ay = acc_traj[:, k, 1]
            az = acc_traj[:, k, 2]

            # --- Nonlinear spherical pendulum dynamics ---
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            sin_phi = torch.sin(phi)
            cos_phi = torch.cos(phi)

            # Effective acceleration vector: a_eff = a_uav + g*e_z
            # Projected onto spherical coordinate basis vectors
            # q_theta = [cos(th)cos(ph), cos(th)sin(ph), sin(th)]
            # q_phi   = [-sin(ph), cos(ph), 0]
            a_eff_x = ax
            a_eff_y = ay
            a_eff_z = az + self.g

            # Projection onto theta direction
            a_dot_q_theta = (a_eff_x * cos_theta * cos_phi
                             + a_eff_y * cos_theta * sin_phi
                             + a_eff_z * sin_theta)

            # Projection onto phi direction
            a_dot_q_phi = (-a_eff_x * sin_phi
                           + a_eff_y * cos_phi)

            # Angular accelerations (full nonlinear equations)
            ddtheta = (dphi ** 2 * sin_theta * cos_theta
                       - (1.0 / cable_L) * a_dot_q_theta)

            # Singularity protection at theta ~ 0
            sin_theta_safe = sin_theta.clamp(min=0.02)
            ddphi = (-2.0 * dtheta * dphi * cos_theta / sin_theta_safe
                     - (1.0 / (cable_L * sin_theta_safe ** 2)) * a_dot_q_phi)

            # Clamp angular accelerations for numerical stability
            ddtheta = ddtheta.clamp(-50.0, 50.0)
            ddphi = ddphi.clamp(-50.0, 50.0)

            # Semi-implicit Euler integration (more stable than explicit)
            dtheta = dtheta + ddtheta * self.dt
            dphi = dphi + ddphi * self.dt
            theta = theta + dtheta * self.dt
            phi = phi + dphi * self.dt

            # Clamp states for stability
            theta = theta.clamp(1e-4, math.pi - 0.01)
            dtheta = dtheta.clamp(-10.0, 10.0)
            dphi = dphi.clamp(-10.0, 10.0)

            # --- Temporal gradient decay ---
            # Apply decay to the cost contribution of this timestep
            cumulative_decay = cumulative_decay * decay_per_step

            # Instantaneous swing cost at this timestep
            step_cost = theta ** 2 + 0.1 * (dtheta ** 2 + dphi ** 2 * sin_theta ** 2)

            # Weight by temporal decay: near-future emphasized, far-future decayed
            swing_costs.append(step_cost * cumulative_decay)

        # 8. Aggregate: mean of decay-weighted costs + peak penalty
        swing_stack = torch.stack(swing_costs, dim=1)  # (B, n_steps)

        # Decay-weighted mean cost
        mean_cost = swing_stack.mean(dim=1)  # (B,)

        # Soft-max peak (LogSumExp): worst-case swing with gradients to all steps
        alpha_lse = 5.0
        peak_cost = torch.logsumexp(alpha_lse * swing_stack, dim=1) / alpha_lse  # (B,)

        # Final swing state
        final_theta = theta  # theta at end of rollout
        final_energy = (theta ** 2
                        + 0.1 * (dtheta ** 2 + dphi ** 2 * torch.sin(theta) ** 2))  # (B,)

        # Swing convergence incentive:
        # If the payload is already swinging (initial_theta > threshold),
        # penalize failure to reduce swing. This teaches the network to
        # choose counter-acceleration trajectories after sharp turns.
        #   convergence_cost > 0  when final_energy >= initial_energy (swing not reduced)
        #   convergence_cost = 0  when final_energy < initial_energy  (swing reduced, good)
        convergence_cost = torch.relu(final_energy - initial_energy)  # (B,)

        # Also keep absolute final swing penalty (prevents low-swing trajectories
        # from being "rewarded" for increasing from 0 to 0.01)
        final_cost = final_theta ** 2  # (B,)

        # Cable length sensitivity: shorter cables -> higher penalty
        length_scale = 1.0 / cable_L.clamp(min=0.1)

        # Combined loss
        dynamics_cost = (1.0 * mean_cost
                         + 0.5 * peak_cost
                         + 0.3 * final_cost
                         + 0.8 * convergence_cost) * length_scale

        return dynamics_cost

    def _get_coefficients(self, Dp: torch.Tensor, Df: torch.Tensor,
                          L: torch.Tensor) -> torch.Tensor:
        """Compute polynomial coefficients from boundary conditions."""
        axes = []
        for i in range(3):
            d = torch.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
            axes.append((L @ d).squeeze(-1))
        return torch.cat(axes, dim=1)  # (B, 18)

    def _get_acceleration(self, coe: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate acceleration from 5th-order polynomial at given times."""
        t_power = torch.stack([
            torch.ones_like(t), t, t ** 2, t ** 3
        ], dim=-1).squeeze(-2)

        coe_x = coe[:, 2:6]
        coe_y = coe[:, 8:12]
        coe_z = coe[:, 14:18]

        acc_mult = torch.tensor([2.0, 6.0, 12.0, 20.0], device=self.device).view(1, 1, 4)

        ax = torch.sum(t_power * coe_x.unsqueeze(1) * acc_mult, dim=-1)
        ay = torch.sum(t_power * coe_y.unsqueeze(1) * acc_mult, dim=-1)
        az = torch.sum(t_power * coe_z.unsqueeze(1) * acc_mult, dim=-1)

        return torch.stack([ax, ay, az], dim=-1)
