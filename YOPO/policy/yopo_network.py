"""
YOPO Network with Pendulum State Encoder.

Architecture:
  depth -> ResNet18 -> depth_feat (64D)
  spatial_obs [vel, acc, goal] (9D) -> pass through
  pendulum_state [theta, phi, dtheta, dphi] (4D) -> PendulumEncoder -> latent (8D)
  concat [depth_feat(64) + spatial(9) + pendulum_latent(8)] = 81D -> YopoHead -> output (10D)
"""

import torch
from torch import nn
from policy.models.backbone import YopoBackbone
from policy.models.head import YopoHead
from policy.models.pendulum_encoder import PendulumStateEncoder
from policy.state_transform import *


class YopoNetwork(nn.Module):

    def __init__(
            self,
            observation_dim: int = 13,
            output_dim: int = 10,
            hidden_state: int = 64, # depth feature dimension from ResNet18
            pendulum_latent_dim: int = 8,
    ):
        super(YopoNetwork, self).__init__()
        self.state_transform = StateTransform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.observation_dim = observation_dim

        # Spatial observation dimension (always 9: vel + acc + goal)
        self.spatial_dim = 9
        # Pendulum state dimension (4: theta, phi, dtheta, dphi)
        self.pendulum_dim = max(0, observation_dim - self.spatial_dim)
        self.pendulum_latent_dim = pendulum_latent_dim if self.pendulum_dim > 0 else 0

        # Image backbone: depth -> 64D feature
        self.image_backbone = YopoBackbone(hidden_state)

        # Pendulum state encoder: 4D -> 8D learned latent.
        # Encoding ("raw" | "sin_cos") read from cfg if available, else "raw".
        try:
            from config.config import cfg as _cfg
            encoding = _cfg._data.get("pendulum_encoder", {}).get("encoding", "raw")
        except Exception:
            encoding = "raw"
        if self.pendulum_dim > 0:
            self.pendulum_encoder = PendulumStateEncoder(
                input_dim=self.pendulum_dim,
                latent_dim=self.pendulum_latent_dim,
                encoding=encoding,
            )
        else:
            self.pendulum_encoder = None

        # Head input: depth_feat + spatial_obs + pendulum_latent
        head_input_dim = hidden_state + self.spatial_dim + self.pendulum_latent_dim
        self.yopo_head = YopoHead(head_input_dim, output_dim)

    def forward(self, depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            obs: (B, head_input_dim, V, H) — already in primitive frame grid format
                 Channels: [spatial(9) + pendulum_latent(8), V, H]
        """
        depth_feature = self.image_backbone(depth)
        input_tensor = torch.cat((obs, depth_feature), dim=1)
        output = self.yopo_head(input_tensor)
        endstate = torch.tanh(output[:, :9])
        score = torch.nn.functional.softplus(output[:, 9])
        return endstate, score

    def inference(self, depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
        For network training:
        (1) normalize the input state
        (2) encode pendulum state through PendulumStateEncoder
        (3) transform spatial obs to primitive frame
        (4) forward propagation
        (5) convert the prediction to endstate in body frame.

        Args:
            obs: (B, observation_dim) raw observation in body frame
                 [vel(3), acc(3), goal(3), theta, phi, dtheta, dphi]

        Returns:
            endstate: (B, 9, V, H) in body frame
            score: (B, V, H)
        """
        obs = self.state_transform.normalize_obs(obs)

        # Split spatial and pendulum components
        spatial_obs = obs[:, :self.spatial_dim]  # (B, 9)

        if self.pendulum_encoder is not None and obs.shape[1] > self.spatial_dim:
            pendulum_raw = obs[:, self.spatial_dim:]  # (B, 4)
            pendulum_latent = self.pendulum_encoder(pendulum_raw)  # (B, 8)
            # Recombine: spatial + encoded pendulum
            obs_encoded = torch.cat([spatial_obs, pendulum_latent], dim=1)  # (B, 17)
        else:
            obs_encoded = spatial_obs  # (B, 9)

        obs_grid = self._prepare_input_with_encoder(obs_encoded)
        endstate_pred, score_pred = self.forward(depth, obs_grid)
        endstate = self.state_transform.pred_to_endstate(endstate_pred)
        return endstate, score_pred

    def _prepare_input_with_encoder(self, obs_encoded: torch.Tensor) -> torch.Tensor:
        """Transform encoded observation to primitive frame grid format.

        Similar to StateTransform.prepare_input, but handles the encoded
        dimension (9+8=17) instead of raw (9+4=13).

        The spatial 9D rotates through primitive frames.
        The pendulum latent 8D is broadcast unchanged (rotation-invariant).
        """
        B = obs_encoded.shape[0]
        obs_dim = obs_encoded.shape[1]
        N = self.state_transform.lattice_primitive.traj_num
        payload_dim = obs_dim - self.spatial_dim  # 8 (encoded) or 0

        spatial = obs_encoded[:, :self.spatial_dim].reshape(B, 3, 3)  # (B, 3, 3)

        Rbp_all = self.state_transform.lattice_primitive.getRotation().flip(0)  # (N, 3, 3)

        obs_exp = spatial[:, None, :, :].expand(B, N, 3, 3)
        Rbp_exp = Rbp_all[None, :, :, :].expand(B, N, 3, 3)

        transformed = torch.matmul(obs_exp, Rbp_exp)  # (B, N, 3, 3)
        transformed_flat = transformed.reshape(B, N, 9)  # (B, N, 9)

        if payload_dim > 0:
            payload_feat = obs_encoded[:, self.spatial_dim:]  # (B, payload_dim)
            payload_exp = payload_feat[:, None, :].expand(B, N, payload_dim)
            combined = torch.cat([transformed_flat, payload_exp], dim=-1)  # (B, N, 9+8)
        else:
            combined = transformed_flat

        V = self.state_transform.lattice_primitive.vertical_num
        H = self.state_transform.lattice_primitive.horizon_num
        out = combined.permute(0, 2, 1).contiguous()  # (B, obs_dim, N)
        out = out.view(B, obs_dim, V, H)
        return out
