import torch
import numpy as np
from config.config import cfg
from policy.primitive import LatticePrimitive


class StateTransform:
    def __init__(self):
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.goal_length = cfg['goal_length']

    def pred_to_endstate(self, endstate_pred: torch.Tensor) -> torch.Tensor:
        """
            Transform the predicted state to the body frame (Original prediction → Primitive frame → Body frame).
            endstate_pred: [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h]
            :return [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h] in body frame
        """
        B, V, H = endstate_pred.shape[0], endstate_pred.shape[2], endstate_pred.shape[3]

        # [B, 9, 3, 5] -> [B, 3, 5, 9] -> [B, 15, 9]
        endstate_pred = endstate_pred.permute(0, 2, 3, 1).reshape(B, V * H, 9)

        # 获取 lattice angle 和 rotation (.flip: 由于lattice和grid的顺序相反)
        yaw, pitch = self.lattice_primitive.getAngleLattice()  # [15]
        yaw = yaw.flip(0)[None, :].expand(B, -1)  # [B, 15]
        pitch = pitch.flip(0)[None, :].expand(B, -1)  # [B, 15]
        Rbp = self.lattice_primitive.getRotation().flip(0)  # [15, 3, 3]
        Rbp = Rbp[None, :, :, :].expand(B, -1, -1, -1)  # [B, 15, 3, 3]

        delta_yaw = endstate_pred[:, :, 0] * self.lattice_primitive.yaw_diff  # [B, 15]
        delta_pitch = endstate_pred[:, :, 1] * self.lattice_primitive.pitch_diff
        radio = (endstate_pred[:, :, 2] + 1.0) * self.lattice_primitive.radio_range

        cos_pitch = torch.cos(pitch + delta_pitch)
        endstate_x = cos_pitch * torch.cos(yaw + delta_yaw) * radio
        endstate_y = cos_pitch * torch.sin(yaw + delta_yaw) * radio
        endstate_z = torch.sin(pitch + delta_pitch) * radio
        endstate_p = torch.stack([endstate_x, endstate_y, endstate_z], dim=-1)  # [B, 15, 3]

        # vel / acc
        endstate_vp = endstate_pred[:, :, 3:6] * self.lattice_primitive.vel_max  # [B, 15, 3]
        endstate_ap = endstate_pred[:, :, 6:9] * self.lattice_primitive.acc_max  # [B, 15, 3]

        # v/a 变换到 body frame
        endstate_vb = torch.matmul(Rbp, endstate_vp.unsqueeze(-1)).squeeze(-1)  # [B, 15, 3]
        endstate_ab = torch.matmul(Rbp, endstate_ap.unsqueeze(-1)).squeeze(-1)

        endstate = torch.cat([endstate_p, endstate_vb, endstate_ab], dim=-1)  # [B, 15, 9]

        endstate = endstate.permute(0, 2, 1).reshape(B, 9, V, H)  # [B, 9, 3, 5]
        return endstate

    def pred_to_endstate_cpu(self, endstate_pred: np.ndarray, lattice_id: torch.Tensor) -> np.ndarray:
        """
            Used during test:
            Numpy version of pred_to_endstate() on CPU (used in test, x10 times faster than torch on CUDA)
            :return [B; px py pz vx vy vz ax ay az] in body frame
        """
        delta_yaw = endstate_pred[:, 0] * self.lattice_primitive.yaw_diff
        delta_pitch = endstate_pred[:, 1] * self.lattice_primitive.pitch_diff
        radio = (endstate_pred[:, 2] + 1.0) * self.lattice_primitive.radio_range

        yaw, pitch = self.lattice_primitive.getAngleLattice(lattice_id)
        yaw, pitch = yaw.cpu().numpy(), pitch.cpu().numpy()
        endstate_x = np.cos(pitch + delta_pitch) * np.cos(yaw + delta_yaw) * radio
        endstate_y = np.cos(pitch + delta_pitch) * np.sin(yaw + delta_yaw) * radio
        endstate_z = np.sin(pitch + delta_pitch) * radio
        endstate_p = np.stack((endstate_x, endstate_y, endstate_z), axis=1)

        endstate_vp = endstate_pred[:, 3:6] * self.lattice_primitive.vel_max
        endstate_ap = endstate_pred[:, 6:9] * self.lattice_primitive.acc_max

        Rpb = self.lattice_primitive.getRotation(lattice_id).cpu().numpy()
        endstate_vb = np.matmul(Rpb, endstate_vp[:, :, np.newaxis]).squeeze(-1)
        endstate_ab = np.matmul(Rpb, endstate_ap[:, :, np.newaxis]).squeeze(-1)

        return np.concatenate((endstate_p, endstate_vb, endstate_ab), axis=1)

    def prepare_input(self, obs):
        """
            Transform the observation to the primitive frame (Body frame → Primitive frame → Body frame).
            obs: [batch; vx, vy, yz, ax, ay, az, gx, gy, gz, ...] in body frame
            Supports variable obs_dim: 9 (no payload), 10-15 (with payload dims).
            :return [batch; obs_dim; primitive_v; primitive_h] in primitive frame
        """
        B, N = obs.shape[0], self.lattice_primitive.traj_num
        obs_dim = obs.shape[1]
        payload_dim = obs_dim - 9  # 0 for 9D, up to 6 for 15D

        # Split spatial (9D) and payload (0-6D) components
        spatial_obs = obs[:, :9]  # [B, 9]

        # 获取所有 Rbp 并倒序排列 (由于lattice和grid的顺序相反)
        Rbp_all = self.lattice_primitive.getRotation().flip(0)  # shape: [N, 3, 3]

        # 仅对 9 维空间观测进行 view 转换
        spatial_obs = spatial_obs.reshape(B, 3, 3)  # [B, 3, 3]

        # 扩展 obs 和 Rbp 到 [B, N, 3, 3]
        obs_exp = spatial_obs[:, None, :, :].expand(B, N, 3, 3)
        Rbp_exp = Rbp_all[None, :, :, :].expand(B, N, 3, 3)

        # 执行批量坐标变换 (仅作用于空间向量)
        transformed = torch.matmul(obs_exp, Rbp_exp)  # [B, N, 3, 3]
        transformed_flat = transformed.reshape(B, N, 9)  # [B, N, 9]

        if payload_dim > 0:
            # 负载状态在各 primitive 下保持一致(它是机体相关的标量或角度)
            payload_obs = obs[:, 9:]  # [B, payload_dim]
            payload_exp = payload_obs[:, None, :].expand(B, N, payload_dim)  # [B, N, payload_dim]
            combined = torch.cat([transformed_flat, payload_exp], dim=-1)  # [B, N, obs_dim]
        else:
            combined = transformed_flat  # [B, N, 9]

        out = combined.permute(0, 2, 1).contiguous()  # [B, obs_dim, N]
        out = out.view(B, obs_dim, self.lattice_primitive.vertical_num, self.lattice_primitive.horizon_num)
        return out

    def unnormalize_obs(self, vel_acc):
        out = vel_acc.clone()
        out[:, 0:3] = out[:, 0:3] * self.lattice_primitive.vel_max
        out[:, 3:6] = out[:, 3:6] * self.lattice_primitive.acc_max
        return out

    def normalize_obs(self, obs):
        out = obs.clone()
        out[:, 0:3] = out[:, 0:3] / self.lattice_primitive.vel_max
        out[:, 3:6] = out[:, 3:6] / self.lattice_primitive.acc_max

        # Clamp the goal direction to unit length
        goal_norm = out[:, 6:9].norm(dim=1, keepdim=True)
        out[:, 6:9] = out[:, 6:9] / goal_norm.clamp(min=self.goal_length)

        # NOTE: Payload dims 9-14 are kept in their natural units (radians / m / kg).
        # Do NOT add normalization here unless retraining from scratch with new weights.
        return out


def rotate_body2world(rot_wb, pos_b):
    """
    Rotate pos_b from body frame to world frame using quaternion q_wb.
    rot_wb: (..., 3, 3)
    pos_b: (..., 3)
    """
    pos_w = torch.matmul(rot_wb, pos_b.unsqueeze(-1)).squeeze(-1)
    return pos_w


def transform_body2world(rot_wb, t_w, pos_b):
    """
    Transform pos_b from body frame to world frame using quaternion q_wb and t_w.
    rot_wb: (..., 3, 3)
    t_w: (..., 3)
    pos_b: (..., 3)
    """
    return rotate_body2world(rot_wb, pos_b) + t_w


def state_body2world(pos_w, rot_wb, pos_b, vel_b, acc_b):
    pos_b = transform_body2world(rot_wb, pos_w, pos_b)
    vel_b = rotate_body2world(rot_wb, vel_b)
    acc_b = rotate_body2world(rot_wb, acc_b)
    return pos_b, vel_b, acc_b


if __name__ == '__main__':
    CoordTransform = StateTransform()