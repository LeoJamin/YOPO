import torch as th
import torch.nn as nn


class DynamicsLoss(nn.Module):
    def __init__(self, L, sgm_time, device):
        super(DynamicsLoss, self).__init__()
        self._L = L
        self.sgm_time = sgm_time
        self.eval_points = 20  # 在一条轨迹上采样20个点来检查摆角
        self.device = device
        self.g = 9.81  # 重力加速度

    def forward(self, Df, Dp, p_state, p_params):
        """
        计算轨迹中每一刻的负载摆角，并返回其惩罚代价
        Df: 初始状态 (batch, 3, 3)
        Dp: 预测参数 (batch, 3, 3)
        p_state: 负载初始状态 (batch, 4) -> [theta, phi, d_theta, d_phi]
        p_params: 负载物理参数 (batch, 2) -> [length, mass]
        """
        batch_size = Dp.shape[0]
        mapping_matrix = self._L.unsqueeze(0).expand(batch_size, -1, -1)

        # 1. 求解多项式系数
        coe = self.get_coefficient_from_derivative(Dp, Df, mapping_matrix) 

        # 2. 生成离散的时间序列
        t = th.linspace(0, self.sgm_time, self.eval_points, device=self.device)
        t_list = t.view(1, -1, 1).expand(batch_size, -1, -1)

        # 3. 获取每个时间点的真实加速度张量 [Batch, Points, 3]
        acc = self.get_acceleration_from_coeff(coe, t_list)

        # 4. 提取物理参数
        cable_L = p_params[:, 0].view(-1, 1)  # 获取当前样本的真实绳长
        init_theta = p_state[:, 0].view(-1, 1)  # 获取初始时刻的摆角

        # 5. 可微动力学评估
        # 物理逻辑：无人机的加速度 acc 会改变摆角的平衡点。
        # 我们计算“期望平衡摆角”与“当前实际摆动”的偏差。
        acc_xy_norm = th.norm(acc[:, :, :2], dim=-1)  # 水平加速度模长
        acc_z = acc[:, :, 2]  # 垂直加速度

        denominator = th.clamp(acc_z + self.g, min=1e-3, max=self.g + 5.0)
        # 计算由当前轨迹加速度产生的“准静态平衡摆角” (Equilibrium angle)
        target_theta_rad = th.atan(acc_xy_norm / denominator)

        # 核心改进：引入初始摆角约束
        # 如果初始摆角 init_theta 很大，且当前加速度方向加剧了摆动，则惩罚加大
        # 这里使用简化模型：惩罚 (当前轨迹诱导摆角 + 初始摆角偏差)
        total_swing = target_theta_rad + 0.2 * th.abs(init_theta - target_theta_rad)

        # 6. 计算动力学代价值
        # 考虑绳长 cable_L 的影响：绳子越短，摆动频率越高，对加速度越敏感
        length_penalty = 1.0 / th.clamp(cable_L, min=0.1)

        mean_swing_loss = total_swing.mean(dim=1)
        max_swing_loss = total_swing.max(dim=1).values

        # 综合代价：根据绳长加权
        dynamics_cost = (mean_swing_loss + 0.5 * max_swing_loss) * length_penalty.squeeze()

        return dynamics_cost

    def get_coefficient_from_derivative(self, Dp, Df, L):
        coefficient = th.zeros(Dp.shape[0], 18, device=self.device)
        for i in range(3):
            d = th.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
            coe = (L @ d).squeeze(-1)
            coefficient[:, 6 * i: 6 * (i + 1)] = coe
        return coefficient

    def get_acceleration_from_coeff(self, coe, t):
        """
        对五次多项式求二阶导，获取加速度 a(t)
        """
        t_power = th.stack([
            th.ones_like(t),
            t,
            t ** 2,
            t ** 3
        ], dim=-1).squeeze(-2)

        coe_x = coe[:, 2:6]
        coe_y = coe[:, 8:12]
        coe_z = coe[:, 14:18]

        # 对应的常数项系数 [2, 6, 12, 20]
        acc_mult = th.tensor([2.0, 6.0, 12.0, 20.0], device=self.device).view(1, 1, 4)

        ax = th.sum(t_power * coe_x.unsqueeze(1) * acc_mult, dim=-1)
        ay = th.sum(t_power * coe_y.unsqueeze(1) * acc_mult, dim=-1)
        az = th.sum(t_power * coe_z.unsqueeze(1) * acc_mult, dim=-1)

        acc = th.stack([ax, ay, az], dim=-1)
        return acc