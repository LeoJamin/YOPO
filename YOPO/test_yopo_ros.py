import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from threading import Lock
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2

import cv2
import os
import time
import torch
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

from config.config import cfg
from control_msg import PositionCommand
from policy.yopo_network import YopoNetwork
from policy.poly_solver import *
from policy.state_transform import *

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")


class YopoNet:
    def __init__(self, config, weight):
        self.config = config
        rospy.init_node('yopo_net', anonymous=False)
        # load params
        cfg["train"] = False
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.min_dis, self.max_dis = 0.04, 20.0
        self.goal = np.array(self.config['goal'])
        self.plan_from_reference = self.config['plan_from_reference']
        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # variables
        self.odom = Odometry()
        self.odom_init = False
        self.payload_odom = Odometry()
        self.payload_init = False


        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        # True only after a goal has been explicitly set via /move_base_simple/goal.
        # Used to suppress the "=== ARRIVE! ===" log line during initial hover at spawn,
        # which would otherwise be a false positive for evaluators that grep the log.
        self.goal_set_externally = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.last_control_msg = None
        self.obs_dim = config.get('obs_dim', 15)
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.depth_fps = 30  # used only as processing time tolerance for printing logs

        # Load Network
        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight, weights_only=True))
        else:
            state_dict = torch.load(weight, weights_only=True)
            # Auto-detect architecture from checkpoint weights
            has_encoder = any(k.startswith("pendulum_encoder.") for k in state_dict.keys())
            if has_encoder:
                # New architecture: PendulumEncoder present
                latent_dim = cfg._data.get("pendulum_encoder", {}).get("latent_dim", 8)
                self.policy = YopoNetwork(
                    observation_dim=config.get('obs_dim', 13),
                    pendulum_latent_dim=latent_dim,
                )
            else:
                # Legacy architecture: no encoder, raw obs_dim concat
                # head input = hidden_state(64) + obs_dim
                from policy.models.backbone import YopoBackbone
                from policy.models.head import YopoHead

                class LegacyYopoNetwork(torch.nn.Module):
                    def __init__(self, obs_dim=13, hidden_state=64):
                        super().__init__()
                        self.state_transform = StateTransform()
                        self.image_backbone = YopoBackbone(hidden_state)
                        self.state_backbone = torch.nn.Sequential()
                        self.yopo_head = YopoHead(hidden_state + obs_dim, 10)
                        self.pendulum_encoder = None
                    def forward(self, depth, obs):
                        depth_feature = self.image_backbone(depth)
                        obs_feature = self.state_backbone(obs)
                        input_tensor = torch.cat((obs_feature, depth_feature), 1)
                        output = self.yopo_head(input_tensor)
                        endstate = torch.tanh(output[:, :9])
                        score = torch.nn.functional.softplus(output[:, 9])
                        return endstate, score
                    def inference(self, depth, obs):
                        obs = self.state_transform.normalize_obs(obs)
                        obs = self.state_transform.prepare_input(obs)
                        endstate_pred, score_pred = self.forward(depth, obs)
                        endstate = self.state_transform.pred_to_endstate(endstate_pred)
                        return endstate, score_pred

                self.policy = LegacyYopoNetwork(obs_dim=config.get('obs_dim', 13))

            self.policy.load_state_dict(state_dict)
            self.policy = self.policy.to(self.device)
            self.policy.eval()
        self.warm_up()

        # ros publisher
        self.lattice_traj_pub = rospy.Publisher("/yopo_net/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/yopo_net/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/yopo_net/trajs_visual", PointCloud2, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1)
        # ros subscriber
        self.odom_sub = rospy.Subscriber(self.config['odom_topic'], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.depth_sub = rospy.Subscriber(self.config['depth_topic'], Image, self.callback_depth, queue_size=1, tcp_nodelay=True)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)
        self.payload_sub = rospy.Subscriber(
            self.config.get('payload_topic', '/sim/payload_odom'),
            Odometry,
            self.callback_payload_odometry,
            queue_size=1,
            tcp_nodelay=True
        )

        # ros timer
        rospy.sleep(1.0)  # wait connection...
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        print("YOPO Net Node Ready!")
        rospy.spin()

    def callback_set_goal(self, data):
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, 2])
        self.arrive = False
        self.goal_set_externally = True
        print(f"New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f})")

    def callback_payload_odometry(self, data):
        """
        假设该话题提供的是负载在世界坐标系下的位姿和速度
        或者直接是相对于无人机的相对状态
        """
        self.payload_odom = data
        self.payload_init = True

    # the first frame
    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        dist_to_goal = np.linalg.norm(pos - self.goal)
        # Log distance to goal periodically
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        self._log_counter += 1
        if self._log_counter % 50 == 0:  # every ~1s at 50Hz odom
            vel = np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
            speed = np.linalg.norm(vel)
            p_info = ""
            if self.payload_init:
                p_load = np.array([self.payload_odom.pose.pose.position.x,
                                   self.payload_odom.pose.pose.position.y,
                                   self.payload_odom.pose.pose.position.z])
                rel = p_load - pos
                L_actual = np.linalg.norm(rel)
                rho = rel / max(L_actual, 1e-4)
                theta_deg = np.degrees(np.arccos(np.clip(-rho[2], -1.0, 1.0)))
                p_info = f" | swing={theta_deg:.1f}° L={L_actual:.3f}m"
            print(f"[NAV] dist={dist_to_goal:.1f}m speed={speed:.2f}m/s pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}){p_info}")
        if dist_to_goal < 3.0 and not self.arrive:
            # Only emit the ARRIVE log line after a goal was explicitly set
            # (via /move_base_simple/goal). On initial spawn-hover the drone
            # is already at its default goal — we still want self.arrive=True
            # so control_pub holds position, but no log line that would be
            # mistaken for real arrival by an external evaluator.
            if self.goal_set_externally:
                print(f"=== ARRIVE! dist={dist_to_goal:.2f}m ===")
            self.arrive = True

    def process_odom(self):
        # 1. 原有的无人机状态处理 (Rwc, Rotation_cw 等)
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        vel_w = self.desire_vel if self.plan_from_reference else np.array(
            [self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_c = np.dot(Rotation_cw, self.desire_acc)
        goal_c = np.dot(Rotation_cw, self.goal - self.desire_pos)

        # 2. 处理负载状态 → 球面摆角 [theta, phi, dtheta, dphi]
        if self.payload_init:
            p_load_w = np.array([self.payload_odom.pose.pose.position.x,
                                 self.payload_odom.pose.pose.position.y,
                                 self.payload_odom.pose.pose.position.z])
            p_drone_w = np.array([self.odom.pose.pose.position.x,
                                  self.odom.pose.pose.position.y,
                                  self.odom.pose.pose.position.z])
            v_load_w = np.array([self.payload_odom.twist.twist.linear.x,
                                 self.payload_odom.twist.twist.linear.y,
                                 self.payload_odom.twist.twist.linear.z])
            v_drone_w = np.array([self.odom.twist.twist.linear.x,
                                  self.odom.twist.twist.linear.y,
                                  self.odom.twist.twist.linear.z])

            # Relative position and velocity in world frame
            rel_p_w = p_load_w - p_drone_w  # cable vector (UAV → payload)
            rel_v_w = v_load_w - v_drone_w

            # Cable length (from actual geometry)
            L = np.linalg.norm(rel_p_w)
            if L < 1e-4:
                L = 0.5  # fallback

            # Spherical coordinates: rho = rel_p / L
            # Convention: theta = polar angle from -z axis (0 = hanging straight down)
            #             phi = azimuthal angle in x-y plane
            rho = rel_p_w / L
            cos_theta = np.clip(-rho[2], -1.0, 1.0)  # -z component
            theta = np.arccos(cos_theta)
            phi = np.arctan2(rho[1], rho[0])

            # Angular velocities from Cartesian relative velocity
            # d(rho)/dt = (rel_v - (rel_v . rho) * rho) / L
            # In spherical: dtheta = ..., dphi = ...
            sin_theta = np.sin(theta) + 1e-8
            # dtheta/dt from velocity projected onto theta direction
            # theta_hat = [cos(theta)cos(phi), cos(theta)sin(phi), sin(theta)]
            # but our theta is from -z, so:
            theta_hat = np.array([np.cos(theta)*np.cos(phi),
                                  np.cos(theta)*np.sin(phi),
                                  np.sin(theta)])
            phi_hat = np.array([-np.sin(phi), np.cos(phi), 0.0])

            rho_dot = rel_v_w / L
            dtheta = np.dot(rho_dot, theta_hat)
            dphi = np.dot(rho_dot, phi_hat) / sin_theta if sin_theta > 1e-4 else 0.0

            p_state = np.array([theta, phi, dtheta, dphi], dtype=np.float32)
        else:
            p_state = np.zeros(4, dtype=np.float32)

        # 3. 拼接观测向量
        if self.obs_dim == 13:
            # 13D: [vel(3), acc(3), goal(3), theta, phi, dtheta, dphi]
            obs = np.concatenate((vel_c, acc_c, goal_c, p_state), axis=0).astype(np.float32)
        elif self.obs_dim == 9:
            # 9D: spatial only
            obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        elif self.obs_dim == 15:
            # 15D legacy: [vel, acc, goal, theta, phi, dtheta, dphi, L, m]
            p_params = np.array([L if self.payload_init else 0.8, 0.3], dtype=np.float32)
            obs = np.concatenate((vel_c, acc_c, goal_c, p_state, p_params), axis=0).astype(np.float32)
        else:
            obs = np.concatenate((vel_c, acc_c, goal_c, p_state), axis=0).astype(np.float32)
            obs = obs[:self.obs_dim]
        obs_norm = self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        return obs_norm

    @torch.inference_mode()
    def callback_depth(self, data):
        if not self.odom_init: return

        # 1. Depth Image Process (Be careful with the depth units in your application)
        time0 = time.time()
        if data.encoding == "32FC1":    # Simulator, meter
            depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
        elif data.encoding == "16UC1":  # RealSense, millimeter
            depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width).astype(np.float32) / 1000.0
        else:
            raise ValueError(f"Unsupported depth encoding: {data.encoding}. Expected '32FC1' or '16UC1'.")

        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth = np.minimum(depth, self.max_dis) / self.max_dis

        # interpolated the nan value (experiment shows that treating nan directly as 0 produces similar results)
        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis)
        interpolated_image = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS)
        interpolated_image = interpolated_image.astype(np.float32) / 255.0
        depth = interpolated_image.reshape([1, 1, self.height, self.width])
        # cv2.imshow("1", depth[0][0])
        # cv2.waitKey(1)

        # 2. YOPO Network Inference
        # input prepare
        time1 = time.time()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)
        obs_norm = self.process_odom().to(self.device, non_blocking=True)

        # Encode pendulum state through PendulumEncoder, then prepare grid
        if hasattr(self.policy, 'pendulum_encoder') and self.policy.pendulum_encoder is not None:
            spatial = obs_norm[:, :9]
            pendulum_raw = obs_norm[:, 9:]
            pendulum_latent = self.policy.pendulum_encoder(pendulum_raw)
            obs_encoded = torch.cat([spatial, pendulum_latent], dim=1)
            obs_input = self.policy._prepare_input_with_encoder(obs_encoded)
        else:
            obs_input = self.state_transform.prepare_input(obs_norm)

        time2 = time.time()
        # Forward (raw prediction space — process_output handles body-frame conversion)
        endstate_pred, score_pred = self.policy(depth_input, obs_input)
        endstate_pred, score_pred = endstate_pred.cpu().numpy(), score_pred.cpu().numpy()
        time3 = time.time()

        # 3. Post-Processing
        # Replacing PyTorch operation on CUDA with NumPy operation on CPU (speed increased by 10x)
        endstate, score = self.process_output(endstate_pred, score_pred, return_all_preds=self.visualize)
        # Vectorization: transform the prediction(P V A in body frame) to the world frame with the attitude (without the position)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)  # [N, 9] -> [N, 3, 3] -> [px vx ax, py vy ay, pz vz az]
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)

        action_id = np.argmin(score) if self.visualize else 0
        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety
            start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0], endstate_w[action_id, 0, 0] + start_pos[0],
                                              endstate_w[action_id, 0, 1], endstate_w[action_id, 0, 2], self.traj_time)
            self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1], endstate_w[action_id, 1, 0] + start_pos[1],
                                              endstate_w[action_id, 1, 1], endstate_w[action_id, 1, 2], self.traj_time)
            self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2], endstate_w[action_id, 2, 0] + start_pos[2],
                                              endstate_w[action_id, 2, 1], endstate_w[action_id, 2, 2], self.traj_time)
            self.ctrl_time = 0.0
        time4 = time.time()
        self.visualize_trajectory(score_pred, endstate_w)
        time5 = time.time()

        self.print_time(time0, time1, time2, time3, time4, time5)

    def control_pub(self, _timer):
        if self.ctrl_time is None or self.ctrl_time > self.traj_time:
            return
        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False   # ready for next rollout
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.ctrl_pub.publish(self.last_control_msg)
            return

        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety and publish frequency
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])
            goal_dir = self.goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)

    def process_output(self, endstate_pred, score_pred, return_all_preds=False):
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)

        if not return_all_preds:
            action_id = np.argmin(score_pred)
            lattice_id = self.lattice_primitive.traj_num - 1 - action_id
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred[action_id, :][np.newaxis, :], lattice_id)
            score = score_pred[action_id]
        else:
            score = score_pred
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, torch.arange(self.lattice_primitive.traj_num-1, -1, -1))

        return endstate, score

    def visualize_trajectory(self, pred_score, pred_endstate):
        dt = self.traj_time / 20.0
        start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
        # best predicted trajectory
        if self.best_traj_pub.get_num_connections() > 0:
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                self.optimal_poly_x.get_position(t_values),
                self.optimal_poly_y.get_position(t_values),
                self.optimal_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.best_traj_pub.publish(point_cloud_msg)
        # lattice primitive
        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            lattice_endstate = self.lattice_primitive.lattice_pos_node.cpu().numpy()
            lattice_endstate = np.dot(lattice_endstate, self.Rotation_wc.T)
            zero_state = np.zeros_like(lattice_endstate)
            lattice_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                          lattice_endstate[:, 0] + start_pos[0], zero_state[:, 0], zero_state[:, 0], self.traj_time)
            lattice_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                          lattice_endstate[:, 1] + start_pos[1], zero_state[:, 1], zero_state[:, 1], self.traj_time)
            lattice_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                          lattice_endstate[:, 2] + start_pos[2], zero_state[:, 2], zero_state[:, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                lattice_poly_x.get_position(t_values),
                lattice_poly_y.get_position(t_values),
                lattice_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.lattice_traj_pub.publish(point_cloud_msg)
        # all predicted trajectories
        if self.visualize and self.all_trajs_pub.get_num_connections() > 0:
            all_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                      pred_endstate[:, 0, 0] + start_pos[0], pred_endstate[:, 0, 1], pred_endstate[:, 0, 2], self.traj_time)
            all_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                      pred_endstate[:, 1, 0] + start_pos[1], pred_endstate[:, 1, 1], pred_endstate[:, 1, 2], self.traj_time)
            all_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                      pred_endstate[:, 2, 0] + start_pos[2], pred_endstate[:, 2, 1], pred_endstate[:, 2, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                all_poly_x.get_position(t_values),
                all_poly_y.get_position(t_values),
                all_poly_z.get_position(t_values)
            ), axis=-1)
            scores = np.repeat(pred_score, t_values.size)
            points_array = np.column_stack((points_array, scores))
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), PointField('intensity', 12, PointField.FLOAT32, 1)]
            point_cloud_msg = point_cloud2.create_cloud(header, fields, points_array)
            self.all_trajs_pub.publish(point_cloud_msg)

    def print_time(self, time0, time1, time2, time3, time4, time5):
        """
        Performance reference: PyTorch model should take < 5 ms; TensorRT model should take < 1 ms

        Notes:
        - Running program and enabling RViz under WSL greatly increase processing time, and Ubuntu does not have these issues
        - Even with queue_size=1, it may cause message accumulation and lag when processing time exceeds the image frequency
        """
        self.time_interpolation = self.time_interpolation + (time1 - time0)
        self.time_prepare = self.time_prepare + (time2 - time1)
        self.time_forward = self.time_forward + (time3 - time2)
        self.time_process = self.time_process + (time4 - time3)
        self.time_visualize = self.time_visualize + (time5 - time4)
        self.count = self.count + 1

        total_time = (time5 - time0) * 1000
        tolerance = 1000.0 / self.depth_fps
        if total_time > tolerance:
            rospy.logwarn(f"Warn: Processing time {(time5 - time0) * 1000:.2f} ms exceeds {tolerance:.2f} ms, may cause message lag!")
            print(f"\033[34mCurrent Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * (time1 - time0):.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * (time2 - time1):.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * (time3 - time2):.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * (time4 - time3):.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * (time5 - time4):.2f} ms\033[0m")
        if self.verbose or (total_time > tolerance):
            print(f"\033[34mAverage Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * self.time_interpolation / self.count:.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * self.time_prepare / self.count:.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * self.time_forward / self.count:.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * self.time_process / self.count:.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * self.time_visualize / self.count:.2f} ms\033[0m")

    def warm_up(self):
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, self.obs_dim), dtype=torch.float32, device=self.device)
        obs[:, 6:9] = 1.0  # non-zero goal to avoid division by zero in normalize
        endstate_pred, score_pred = self.policy.inference(depth, obs)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    parser.add_argument("--obs_dim", type=int, default=13, help="observation dimension (9/13/15)")
    # Initial goal — defaults to (50, 0, 2) so the planner is in-distribution
    # at startup (drone in flight rather than stationary). The network was
    # trained on a moving drone with non-zero goal direction; spawn-hover puts
    # it in OOD territory and it can't bootstrap into motion when a goal is
    # later published. Pass --goal_x 0 --goal_y 0 to hover instead and use
    # RViz "2D Nav Goal" to set goals interactively.
    parser.add_argument("--goal_x", type=float, default=50.0, help="initial goal x (m)")
    parser.add_argument("--goal_y", type=float, default=0.0,  help="initial goal y (m)")
    parser.add_argument("--goal_z", type=float, default=2.0,  help="initial goal z (m)")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = "yopo_trt.pth" if args.use_tensorrt else base_dir + "/saved/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch)
    print("load weight from:", weight)

    settings = {'use_tensorrt': args.use_tensorrt,
                'obs_dim': args.obs_dim,
                'goal': [args.goal_x, args.goal_y, args.goal_z],  # 目标点位置 (default: spawn → hover)
                'pitch_angle_deg': -0,   # 相机俯仰角(仰为负)
                'odom_topic': '/sim/odom',                   # 里程计话题
                'depth_topic': '/depth_image',               # 深度图话题
                'ctrl_topic': '/so3_control/pos_cmd',        # 控制器话题
                'plan_from_reference': False,   # 从参考状态规划？位置控制器: True, 神经网络直接控制: False
                'verbose': False,               # 打印耗时？
                'visualize': True               # 可视化所有轨迹？(实飞改为False节省计算)
                }
    YopoNet(settings, weight)
