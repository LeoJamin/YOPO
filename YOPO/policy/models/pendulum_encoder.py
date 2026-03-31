"""
Pendulum State Encoder.

Learns a compact latent representation of the pendulum swing state
[theta, phi, dtheta, dphi] -> latent(8D).

Compared to passing raw angles directly to the planning head, this encoder:
1. Learns nonlinear features (e.g., energy = dtheta^2 + dphi^2*sin^2(theta))
2. Enables the network to discover swing-relevant features automatically
3. Provides a physics-informed bottleneck between sensing and planning
"""

import torch
import torch.nn as nn


class PendulumStateEncoder(nn.Module):
    """Encode raw pendulum state [theta, phi, dtheta, dphi] into a
    learned latent representation of dimension `latent_dim`.

    Architecture: 2-layer MLP with residual connection and LayerNorm.
    The residual connection preserves the raw signal while allowing the
    network to learn additional nonlinear features.
    """

    def __init__(self, input_dim: int = 4, latent_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.GELU(),
            nn.Linear(16, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Residual projection: match input_dim -> latent_dim
        self.residual_proj = nn.Linear(input_dim, latent_dim, bias=False)

    def forward(self, p_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_state: (B, 4) or (B, N, 4) raw pendulum state
                     [theta, phi, dtheta, dphi]

        Returns:
            latent: (B, latent_dim) or (B, N, latent_dim)
        """
        return self.encoder(p_state) + self.residual_proj(p_state)
