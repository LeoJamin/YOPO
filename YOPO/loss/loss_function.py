import math
import torch as th
import torch.nn as nn
from config.config import cfg
from loss.safety_loss import SafetyLoss
from loss.smoothness_loss import SmoothnessLoss
from loss.guidance_loss import GuidanceLoss
from loss.differentiable_pendulum import DifferentiablePendulumLoss

class YOPOLoss(nn.Module):
    def __init__(self):
        """
        Compute the cost: including smoothness, safety, guidance, goal cost, etc.
        Currently, keeping multi-segment polynomial support (not yet verified), but only using a single-segment polynomial (m = 1) for now.
        dp: decision parameters
        df: fixed parameters
        """
        super(YOPOLoss, self).__init__()
        self.sgm_time = cfg["sgm_time"] # time duration of each segment, currently set to 2s (the time to traverse the radio range at max speed)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self._C, self._B, self._L, self._RJ, self._RA = self.qp_generation()# 论文中的映射矩阵、约束矩阵、线性项矩阵、Jerk海森矩阵、Accel海森矩阵
        self._RJ = self._RJ.to(self.device)# 将矩阵移动到正确的设备上（GPU或CPU）
        self._RA = self._RA.to(self.device)
        self._L = self._L.to(self.device)
        self.denormalize_weight()
        self.smoothness_loss = SmoothnessLoss(self._RJ, self._RA)
        # detach_qvec=True by default; set False only for ablation study
        detach_qvec = cfg._data.get("detach_qvec", True)
        self.safety_loss = SafetyLoss(self._L, detach_qvec=detach_qvec)
        self.goal_loss = GuidanceLoss()

        gradient_decay = cfg._data.get("gradient_decay", True)
        self.dynamics_loss = DifferentiablePendulumLoss(
            self._L, self.sgm_time, self.device,
            gradient_decay_enabled=gradient_decay,
        )
        # dynamics_weight is already set by denormalize_weight() above — do NOT overwrite


        print("------ Actual Loss ------")
        print(f"| {'smooth':<12} = {self.smoothness_weight:6.4f} |")
        print(f"| {'safety':<12} = {self.safety_weight:6.4f} |")
        print(f"| {'goal':<12} = {self.goal_weight:6.4f} |")
        print(f"| {'dynamics':<12} = {self.dynamics_weight:6.4f} |")
        print("-------------------------")

    def qp_generation(self):
        # 论文中的映射矩阵
        A = th.zeros((6, 6))# 6个约束条件（初始位置、速度、加速度）对应6个决策参数（每段的多项式系数）
        for i in range(3):
            A[2 * i, i] = math.factorial(i)
            for j in range(i, 6):
                A[2 * i + 1, j] = math.factorial(j) / math.factorial(j - i) * (self.sgm_time ** (j - i))

        # H海森矩阵，对应Jerk
        H = th.zeros((6, 6))
        for i in range(3, 6):
            for j in range(3, 6):
                H[i, j] = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * (self.sgm_time ** (i + j - 5))

        # Q海森矩阵，对应Accel
        Q = th.zeros((6, 6))
        for i in range(2, 6):
            for j in range(2, 6):
                Q[i, j] = (i * (i - 1)) * (j * (j - 1)) / (i + j - 3) * (self.sgm_time ** (i + j - 3))

        return self.stack_opt_dep(A, H, Q)
    #stack_opt_dep函数的作用是根据论文中的映射矩阵A和海森矩阵H、Q，计算出优化问题中需要的矩阵C、B、L、R_Jerk、R_Acc。这些矩阵在后续的损失计算中会被用到，特别是在计算平滑性损失时。
    def stack_opt_dep(self, A, H, Q):
        Ct = th.zeros((6, 6))
        Ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1
        # Ct是一个6x6的矩阵，通过这种方式将其设置为一个特定的排列矩阵（Permutation Matrix）。
        # #具体来说，这行代码将Ct的第0行第0列、第2行第1列、第4行第2列、第1行第3列、第3行第4列、第5行第5列设置为1，其余元素保持为0。这样，Ct就成为了一个将决策参数从一个顺序映射到另一个顺序的矩阵。

        _C = th.transpose(Ct, 0, 1)
        # _C是Ct的转置矩阵。转置操作将矩阵的行和列进行交换，因此_C[i, j] = Ct[j, i]。在这个上下文中，_C可能被用来将决策参数从一个顺序映射到另一个顺序，或者在计算损失时进行某种变换。

        B = th.inverse(A)
        # B是A矩阵的逆矩阵。由于A是一个6x6的矩阵，如果A是可逆的，那么B就是满足A @ B = I（单位矩阵）的矩阵。在优化问题中，B可能被用来将约束条件映射到决策参数空间，或者在计算损失时进行某种变换。

        B_T = th.transpose(B, 0, 1)

        _L = B @ Ct
        # _L是一个矩阵，计算方式是将B矩阵与Ct矩阵相乘。这个矩阵可能在后续的损失计算中被用来将约束条件映射到决策参数空间，或者在计算安全性损失时进行某种变换。

        _R_Jerk = _C @ (B_T) @ H @ B @ Ct

        _R_Acc = _C @ (B_T) @ Q @ B @ Ct

        return _C, B, _L, _R_Jerk, _R_Acc

    def denormalize_weight(self):
        """
        Denormalize the cost weight to ensure consistency across different speeds to simplify parameter tuning.
        smoothness cost: time integral of jerk² is used as a smoothness cost.
                         If the speed is scaled by n, the cost is scaled by n⁵ (because jerk * n⁶ and time * 1/n).
        safety cost:     time integral of the distance from trajectory to obstacles.
                         If the speed is scaled by n, the cost is scaled by 1/n (because time * 1/n).
        goal cost:       projection of the trajectory onto goal direction.
                         Independent of speed.
        """
        vel_scale = cfg["vel_max_train"] / 1.0
        self.smoothness_weight = cfg["ws"] / vel_scale ** 5
        self.accele_weight = cfg["wa"] / vel_scale ** 3
        self.safety_weight = cfg["wc"]
        self.goal_weight = cfg["wg"]
        # --- 修改位置：使用 wd 键名并根据速度缩放 ---
        self.dynamics_weight = cfg["wd"] / vel_scale ** 2
        # 因为dynamics loss是基于加速度的，所以它的权重也需要根据速度进行调整，具体来说，如果速度增加n倍，那么加速度会增加n²倍，因此dynamics loss会增加n³倍（因为时间缩短了1/n），所以我们需要将dynamics_weight除以vel_scale的三次方来保持损失的一致性。

    # --- 修改位置：更新参数列表以接收 p_state 和 p_params ---
    def forward(self, state, prediction, goal, map_id, p_state, p_params):
        """
        Args:
            prediction: (batch_size, 3, 3) → [px, py, pz; vx, vy, vz; ax, ay, az] in world frame
            state: (batch_size, 3, 3) → [px, py, pz; vx, vy, vz; ax, ay, az] in world frame
            goal: (batch_size, 3) → target direction
            map_id: (batch_size) which ESDF map to query
            p_state: (batch_size, 4) → [theta, phi, d_theta, d_phi]
            p_params: (batch_size, 2) → [length, mass]

        Returns:
            cost: (batch_size) → weighted cost
        """
        # Fixed part: initial pos, vel, acc → (batch_size, 3, 3) [px, vx, ax; py, vy, ay; pz, vz, az]
        Df = state.permute(0, 2, 1)

        # Decision parameters (local frame) → (batch_size, 3, 3) [px, vx, ax; py, vy, ay; pz, vz, az]
        Dp = prediction.permute(0, 2, 1)

        smoothness_cost, acceleration_cost = self.smoothness_loss(Df, Dp)
        safety_cost = self.safety_loss(Df, Dp, map_id,p_state, p_params)
        goal_cost = self.goal_loss(Df, Dp, goal)

        # --- 修改位置：将负载状态和物理参数传给 dynamics_loss ---
        dynamics_cost = self.dynamics_loss(Df, Dp, p_state, p_params)

        return (smoothness_cost,
                safety_cost,
                goal_cost,
                acceleration_cost,
                dynamics_cost)