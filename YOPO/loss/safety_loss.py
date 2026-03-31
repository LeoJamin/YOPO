import os
import glob
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from scipy.ndimage import distance_transform_edt
from config.config import cfg


class SafetyLoss(nn.Module):
    def __init__(self, L, detach_qvec: bool = True):
        """
        Args:
            L: polynomial mapping matrix
            detach_qvec: If True (default), cut gradient through q_vec to prevent
                         the network from learning to use extreme accelerations to
                         move the bubble chain away from obstacles (adversarial shortcut).
                         Set False for ablation study (requires gradient clipping to stabilize).
        """
        super(SafetyLoss, self).__init__()
        self.traj_num = cfg['traj_num']
        self.map_expand_min = np.array(cfg['map_expand_min'])
        self.map_expand_max = np.array(cfg['map_expand_max'])
        self.detach_qvec = detach_qvec

        # 气泡包络物理参数
        self.r_uav = cfg["bubble"]["r_uav"]
        self.r_load = cfg["bubble"]["r_load"]
        self.r_safe = cfg["bubble"]["r_safe"]

        self._L = L
        self.sgm_time = cfg["sgm_time"]
        self.eval_points = 30  # 轨迹时间采样点
        self.device = self._L.device
        self.time_integral = True

        # SDF 地图初始化
        self.voxel_size = 0.2
        self.min_bounds = None
        self.max_bounds = None
        self.sdf_shapes = None
        print("Building ESDF map for Slung-load Bubble Chain...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "../", cfg["dataset_path"])
        self.sdf_maps = self.get_sdf_from_ply(data_dir)
        print("ESDF Maps Built Successfully!")

    def forward(self, Df, Dp, map_id, p_state, p_params):
        batch_size = Dp.shape[0]
        mapping_matrix = self._L.unsqueeze(0).expand(batch_size, -1, -1)
        coe = self.get_coefficient_from_derivative(Dp, Df, mapping_matrix)

        dt = self.sgm_time / self.eval_points
        t_list = th.linspace(dt, self.sgm_time, self.eval_points, device=self.device)
        t_list = t_list.view(1, -1, 1).expand(batch_size, -1, -1)

        # 1. 轨迹与真实加速度计算
        pos_uav = self.get_position_from_coeff(coe, t_list)
        acc_uav = self.get_acceleration_from_coeff(coe, t_list)  # [batch, eval_points, 3]

        length = p_params[:, 0]

        # 【核心修复1】：根据表观重力，动态实时计算绳索的物理朝向 q_vec
        g_vec = th.tensor([0.0, 0.0, -9.81], device=self.device).view(1, 1, 3)
        apparent_g = acc_uav - g_vec
        q_vec = -apparent_g / (th.norm(apparent_g, dim=-1, keepdim=True) + 1e-5)  # [B, eval_points, 3]

        # Gradient detachment for q_vec (ablatable):
        # When True: prevents the network from learning adversarial accelerations that
        # deliberately tilt the bubble chain away from obstacles to reduce safety cost.
        # When False (ablation): allows gradient flow but requires stronger clipping.
        if self.detach_qvec:
            q_vec = q_vec.detach()

        # 2. 动态气泡链生成
        r_min = min(self.r_uav, self.r_load)
        d_max = 2.0 * np.sqrt(max(r_min ** 2 - self.r_safe ** 2, 1e-4))

        max_L = length.max().item()
        curr_N = int(np.ceil(max_L / d_max) + 1)
        curr_N = max(curr_N, 2)

        r_ratios = th.linspace(0, 1, curr_N, device=self.device)

        # 将动态的 q_vec (携带 eval_points 维度) 融入计算
        pos_bubbles = pos_uav.unsqueeze(2) + \
                      r_ratios.view(1, 1, -1, 1) * \
                      (length.view(-1, 1, 1, 1) * q_vec.unsqueeze(2))

        r_bubbles = self.r_uav + r_ratios * (self.r_load - self.r_uav)

        # 3. ESDF 采样
        actual_batch = map_id.shape[0]
        pos_for_sdf = pos_bubbles.reshape(actual_batch, -1, 3)
        cost, dist = self.get_distance_cost(pos_for_sdf, map_id, r_bubbles, batch_size, curr_N)

        # 4. 损失聚合 (Max-Pooling)
        max_cost_per_time = cost.max(dim=-1).values
        return max_cost_per_time.mean(dim=-1)

    def get_distance_cost(self, pos, map_id, r_bubbles, orig_batch_size, curr_N):
        B, N_total, _ = pos.shape
        sdf_maps, local_origin, local_shape = self.get_batch_sdf(pos, map_id)

        if sdf_maps.shape[0] == 1 and B > 1:
            sdf_maps = sdf_maps.expand(B, -1, -1, -1, -1)

        # 坐标归一化到 [-1, 1] 供 grid_sample 使用
        grid = (pos - local_origin.unsqueeze(1)) / self.voxel_size
        grid_point = 2.0 * grid / (local_shape - 1).unsqueeze(1) - 1.0
        grid_point = grid_point.view(B, 1, 1, N_total, 3)
        grid_point = th.clamp(grid_point, min=-0.99, max=0.99)

        dist_query = F.grid_sample(sdf_maps, grid_point, mode='bilinear', padding_mode='zeros', align_corners=True)
        dist_query = dist_query.view(B, N_total)

        dist_reshaped = dist_query.view(orig_batch_size, self.eval_points, curr_N)

        # 🚨【核心修复3】：分段线性-指数惩罚，彻底消灭指数爆炸，保证 Lipschitz 连续！
        penetration = th.clamp(r_bubbles - dist_reshaped, min=0.0)
        threshold = 0.2  # 临界穿透深度设置为 0.2m

        # 指数部分 (浅穿透)
        exp_part = th.exp(penetration / 0.1) - 1.0

        # 线性部分 (深穿透)
        linear_k = 10.0 * th.exp(th.tensor(threshold / 0.1, device=self.device))
        linear_y0 = th.exp(th.tensor(threshold / 0.1, device=self.device)) - 1.0
        linear_part = linear_k * (penetration - threshold) + linear_y0

        # 掩码合并
        mask = (penetration <= threshold).float()
        cost = mask * exp_part + (1.0 - mask) * linear_part

        return cost, dist_reshaped

    def get_batch_sdf(self, pos, map_id):
        min_bounds = self.min_bounds[map_id]
        sdf_shapes = self.sdf_shapes[map_id]

        min_pos = pos.amin(dim=1)
        max_pos = pos.amax(dim=1)
        min_indices = ((min_pos - min_bounds) / self.voxel_size).int()
        max_indices = ((max_pos - min_bounds) / self.voxel_size).int()

        spans = max_indices - min_indices
        max_spans = spans.amax(dim=0)

        centers = (min_indices + max_indices) // 2
        target_shape = max_spans + 10
        ts_list = target_shape.tolist()

        min_indices = centers - target_shape // 2
        max_indices = min_indices + target_shape

        cropped_maps = []
        for i, map_idx in enumerate(map_id.tolist()):
            sdf = self.sdf_maps[map_idx]
            shape_x, shape_y, shape_z = sdf.shape[2], sdf.shape[3], sdf.shape[4]

            mx, my, mz = min_indices[i].tolist()
            Mx, My, Mz = max_indices[i].tolist()

            # 生成绝对统一尺寸的安全画布
            canvas = th.full((1, ts_list[0], ts_list[1], ts_list[2]),
                             10.0, device=self.device, dtype=sdf.dtype)

            cmx, cMx = max(0, mx), min(shape_x, Mx)
            cmy, cMy = max(0, my), min(shape_y, My)
            cmz, cMz = max(0, mz), min(shape_z, Mz)

            if cmx < cMx and cmy < cMy and cmz < cMz:
                valid_sdf = sdf[0, :, cmx:cMx, cmy:cMy, cmz:cMz]
                pmx, pMx = cmx - mx, cMx - mx
                pmy, pMy = cmy - my, cMy - my
                pmz, pMz = cmz - mz, cMz - mz
                canvas[:, pmx:pMx, pmy:pMy, pmz:pMz] = valid_sdf

            cropped_maps.append(canvas)

        # 🚨【核心修复4】：将以下代码撤出 for 循环！否则 batch 处理会彻底失效！
        # 裁剪出来的地图拼接
        sdf_maps = th.cat(cropped_maps, dim=0).unsqueeze(1)  # 此时形状是 [Batch, 1, X, Y, Z]

        # 专为 grid_sample 定制的维度反转 (x->W, y->H, z->D)
        sdf_maps = sdf_maps.permute(0, 1, 4, 3, 2)  # 形状变成 [Batch, 1, Z, Y, X]

        local_origin = min_indices * self.voxel_size + min_bounds
        local_shape = target_shape.unsqueeze(0).expand(map_id.shape[0], 3)

        return sdf_maps, local_origin, local_shape

    def get_acceleration_from_coeff(self, coe, t):
        t_power = th.stack([th.ones_like(t), t, t ** 2, t ** 3], dim=-1).squeeze(-2)
        coe_x, coe_y, coe_z = coe[:, 2:6], coe[:, 8:12], coe[:, 14:18]
        acc_mult = th.tensor([2.0, 6.0, 12.0, 20.0], device=self.device).view(1, 1, 4)
        ax = th.sum(t_power * coe_x.unsqueeze(1) * acc_mult, dim=-1)
        ay = th.sum(t_power * coe_y.unsqueeze(1) * acc_mult, dim=-1)
        az = th.sum(t_power * coe_z.unsqueeze(1) * acc_mult, dim=-1)
        return th.stack([ax, ay, az], dim=-1)

    def get_coefficient_from_derivative(self, Dp, Df, L):
        coefficient = th.zeros(Dp.shape[0], 18, device=self.device)
        for i in range(3):
            d = th.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
            coe = (L @ d).squeeze(-1)
            coefficient[:, 6 * i: 6 * (i + 1)] = coe
        return coefficient

    def get_position_from_coeff(self, coe, t):
        t_power = th.stack([th.ones_like(t), t, t ** 2, t ** 3, t ** 4, t ** 5], dim=-1).squeeze(-2)
        coe_x, coe_y, coe_z = coe[:, 0:6], coe[:, 6:12], coe[:, 12:18]
        x = th.sum(t_power * coe_x.unsqueeze(1), dim=-1)
        y = th.sum(t_power * coe_y.unsqueeze(1), dim=-1)
        z = th.sum(t_power * coe_z.unsqueeze(1), dim=-1)
        return th.stack([x, y, z], dim=-1)

    def get_sdf_from_ply(self, path):
        sorted_files = self.read_sorted_ply_files(path)
        sdf_maps, min_bounds, max_bounds, sdf_shapes = [], [], [], []
        for file in sorted_files:
            pcd = o3d.io.read_point_cloud(file)
            min_bound = np.array(pcd.get_min_bound()) - self.map_expand_min
            max_bound = np.array(pcd.get_max_bound()) + self.map_expand_max
            points = np.asarray(pcd.points)
            sdf_shape = np.ceil((max_bound - min_bound) / self.voxel_size).astype(int)
            voxel_indices = ((points - min_bound) / self.voxel_size).astype(int)
            valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < sdf_shape), axis=1)
            voxel_indices = voxel_indices[valid_mask]
            occupancy = np.zeros(sdf_shape, dtype=np.uint8)
            occupancy[tuple(voxel_indices.T)] = 1
            dist_to_obstacle = distance_transform_edt(occupancy == 0) * self.voxel_size
            dist_inside_obstacle = distance_transform_edt(occupancy == 1) * self.voxel_size
            dist_to_obstacle[occupancy == 1] = -dist_inside_obstacle[occupancy == 1]

            sdf_tensor = th.from_numpy(dist_to_obstacle).float().unsqueeze(0).unsqueeze(0).to(self.device)
            sdf_maps.append(sdf_tensor)
            sdf_shapes.append(sdf_tensor.shape[-3:][::-1])
            min_bounds.append(min_bound)
            max_bounds.append(max_bound)
        self.min_bounds = th.tensor(np.array(min_bounds), device=self.device).float()
        self.max_bounds = th.tensor(np.array(max_bounds), device=self.device).float()
        self.sdf_shapes = th.tensor(np.array(sdf_shapes), device=self.device).float()
        return sdf_maps

    def read_sorted_ply_files(self, path):
        ply_files = glob.glob(os.path.join(path, 'pointcloud-*.ply'))

        def extract_index(filename):
            return int(os.path.basename(filename).replace('pointcloud-', '').replace('.ply', ''))

        return sorted(ply_files, key=extract_index)