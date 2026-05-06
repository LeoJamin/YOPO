"""
Pendulum State Encoder.

Learns a compact latent representation of the pendulum swing state
[theta, phi, dtheta, dphi] -> latent(8D).

Compared to passing raw angles directly to the planning head, this encoder:
1. Learns nonlinear features (e.g., energy = dtheta^2 + dphi^2*sin^2(theta))
2. Enables the network to discover swing-relevant features automatically
3. Provides a physics-informed bottleneck between sensing and planning

Two angular feature encodings are supported via the `encoding` argument:
  "raw"     — pass [theta, phi, dtheta, dphi] unchanged. Backward-compatible.
  "sin_cos" — pass [sin theta, cos theta, sin phi, cos phi, dtheta, dphi].
              Makes phi naturally periodic and removes the discontinuity at
              ±pi that the network would otherwise have to learn around.
              Only applies when input_dim == 4 (the full pendulum state); for
              ablations with reduced input_dim it silently falls back to raw.
"""

import torch
import torch.nn as nn


class PendulumStateEncoder(nn.Module):
    """Encode raw pendulum state into a learned latent representation."""

    def __init__(self, input_dim: int = 4, latent_dim: int = 8,
                 encoding: str = "raw"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # sin_cos encoding only well-defined for the full 4D pendulum layout.
        if encoding == "sin_cos" and input_dim == 4:
            self.encoding = "sin_cos"
            effective_dim = 6
        else:
            self.encoding = "raw"
            effective_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(effective_dim, 16),
            nn.GELU(),
            nn.Linear(16, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.residual_proj = nn.Linear(effective_dim, latent_dim, bias=False)

    def _featurize(self, p_state: torch.Tensor) -> torch.Tensor:
        if self.encoding == "sin_cos":
            theta  = p_state[..., 0:1]
            phi    = p_state[..., 1:2]
            dtheta = p_state[..., 2:3]
            dphi   = p_state[..., 3:4]
            return torch.cat(
                [torch.sin(theta), torch.cos(theta),
                 torch.sin(phi),   torch.cos(phi),
                 dtheta, dphi], dim=-1
            )
        return p_state

    def forward(self, p_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_state: (..., input_dim) — typically [theta, phi, dtheta, dphi]
        Returns:
            latent: (..., latent_dim)
        """
        x = self._featurize(p_state)
        return self.encoder(x) + self.residual_proj(x)
