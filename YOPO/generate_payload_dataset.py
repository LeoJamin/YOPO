"""
Payload-Aware Dataset Generator for YOPO-Payload.

Replaces the C++ dataset_generator for payload training.
Drives the UAV along smooth random trajectories, integrates the spherical
pendulum ODE simultaneously, and saves:
  - Depth images (same format as C++ generator)
  - pose-{i}.csv with physically consistent payload states

Run with:
    conda activate yopo
    python generate_payload_dataset.py

Requires the sensor simulator to be running:
    cd Simulator && source devel/setup.bash
    rosrun sensor_simulator sensor_simulator_cuda
"""

import os
import sys
import cv2
import csv
import time
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from threading import Lock, Event
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Quaternion, Point

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from policy.pendulum_simulator import (
    pendulum_ode,
    state_spherical_to_cartesian,
    state_cartesian_to_spherical,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_ENVS        = 10       # number of map environments (matches config.yaml env_num)
SAMPLES_PER_ENV = 10000    # depth images per environment
TRAJ_DT         = 0.05     # trajectory time step (s) — 20 Hz
SAVE_INTERVAL   = 5        # save one sample every N trajectory steps
ACC_MAX         = 4.0      # max UAV acceleration (m/s^2)
VEL_MAX         = 4.0      # max UAV velocity (m/s)
Z_RANGE         = (0.8, 3.5)
XY_RANGE        = 18.0
L_RANGE         = (0.5, 2.0)  # cable length range (m), matches paper §III-D
M_RANGE         = (0.1, 1.0)  # payload mass range (kg)
G               = 9.81

DATASET_PATH    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../dataset"))
ODOM_TOPIC      = "/sim/odom"
DEPTH_TOPIC     = "/depth_image"

# ---------------------------------------------------------------------------
# Pendulum ODE step (RK4, faster than scipy for single steps)
# ---------------------------------------------------------------------------

def rk4_pendulum_step(y, dt, acc, L, n_sub=5):
    """RK4 step for the spherical pendulum ODE with substeps.

    pendulum_ode uses Baumgarte stabilization with alpha=50 (timescale ~0.02s),
    so an outer dt of 0.05s sits at the explicit-RK4 stability boundary and
    can blow up under aggressive accelerations. Splitting into n_sub=5 substeps
    of ~0.01s puts us well inside the stable region. Renormalize q to unit
    length after each substep to keep the constraint satisfied numerically.
    """
    def f(yy):
        return pendulum_ode(0, yy, lambda t: acc, L)
    sub_dt = dt / n_sub
    for _ in range(n_sub):
        k1 = np.array(f(y))
        k2 = np.array(f(y + 0.5 * sub_dt * k1))
        k3 = np.array(f(y + 0.5 * sub_dt * k2))
        k4 = np.array(f(y + sub_dt * k3))
        y = y + (sub_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        q_norm = np.linalg.norm(y[:3])
        if q_norm > 1e-10:
            y[:3] = y[:3] / q_norm
    return y


# ---------------------------------------------------------------------------
# Random smooth trajectory generator
# ---------------------------------------------------------------------------

class SmoothTrajectory:
    """Generates smooth random UAV trajectories via random-walk acceleration."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, pos=None):
        self.pos = pos if pos is not None else np.array([
            self.rng.uniform(-XY_RANGE, XY_RANGE),
            self.rng.uniform(-XY_RANGE, XY_RANGE),
            self.rng.uniform(*Z_RANGE),
        ])
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self._acc_target = self.rng.uniform(-ACC_MAX, ACC_MAX, 3)
        self._steps_to_next = self.rng.integers(20, 60)
        self._step_count = 0

    def step(self, dt):
        """Advance trajectory by one dt, return (pos, vel, acc, yaw)."""
        self._step_count += 1
        if self._step_count >= self._steps_to_next:
            self._acc_target = self.rng.uniform(-ACC_MAX, ACC_MAX, 3)
            self._acc_target[2] *= 0.3  # reduce vertical acceleration
            self._steps_to_next = self.rng.integers(20, 60)
            self._step_count = 0

        # Smooth acceleration toward target
        self.acc += 0.1 * (self._acc_target - self.acc)
        self.vel += self.acc * dt
        speed = np.linalg.norm(self.vel)
        if speed > VEL_MAX:
            self.vel *= VEL_MAX / speed

        self.pos += self.vel * dt

        # Boundary clamp
        self.pos[:2] = np.clip(self.pos[:2], -XY_RANGE, XY_RANGE)
        self.pos[2]  = np.clip(self.pos[2], *Z_RANGE)

        # Yaw from velocity direction
        yaw = np.arctan2(self.vel[1], self.vel[0]) if speed > 0.1 else 0.0

        return self.pos.copy(), self.vel.copy(), self.acc.copy(), yaw


# ---------------------------------------------------------------------------
# ROS dataset generator node
# ---------------------------------------------------------------------------

class PayloadDatasetGenerator:

    def __init__(self):
        rospy.init_node('payload_dataset_generator', anonymous=False)
        self.bridge = CvBridge()
        self.odom_pub = rospy.Publisher(ODOM_TOPIC, Odometry, queue_size=1)

        self.latest_depth = None
        self.depth_lock = Lock()
        self.depth_event = Event()
        rospy.Subscriber(DEPTH_TOPIC, Image, self._depth_cb, queue_size=1)

        rospy.sleep(1.0)  # wait for sim to connect
        rospy.loginfo("PayloadDatasetGenerator ready.")

    def _depth_cb(self, msg):
        try:
            # Bypass cv_bridge to avoid the conda/system libgdal+libtiff
            # version conflict. The simulator publishes 32FC1 depth in metres.
            if msg.encoding == "32FC1":
                img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            elif msg.encoding == "16UC1":
                img = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
            else:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.depth_lock:
                self.latest_depth = img.copy()
            self.depth_event.set()
        except Exception as e:
            rospy.logwarn(f"Depth cb error: {e}")

    def _publish_odom(self, pos, vel, yaw, pitch=0.0, roll=0.0):
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.pose.pose.position = Point(*pos)
        q = R.from_euler('ZYX', [yaw, pitch, roll]).as_quat()  # xyzw
        msg.pose.pose.orientation = Quaternion(q[0], q[1], q[2], q[3])
        msg.twist.twist.linear.x = vel[0]
        msg.twist.twist.linear.y = vel[1]
        msg.twist.twist.linear.z = vel[2]
        self.odom_pub.publish(msg)

    def _get_depth(self, timeout=0.5):
        self.depth_event.clear()
        self.depth_event.wait(timeout)
        with self.depth_lock:
            return self.latest_depth.copy() if self.latest_depth is not None else None

    def _save_depth(self, img, save_dir, idx):
        """Save depth image as 16-bit PNG (same format as C++ generator)."""
        max_dist = 20.0
        scaled = np.clip(img / max_dist, 0, 1)
        img16 = (scaled * 65535).astype(np.uint16)
        path = os.path.join(save_dir, f"img_{idx}.png")
        cv2.imwrite(path, img16)

    def generate_env(self, env_idx, seed):
        """Generate SAMPLES_PER_ENV samples for one environment."""
        rng = np.random.default_rng(seed)
        save_dir = os.path.join(DATASET_PATH, str(env_idx))
        os.makedirs(save_dir, exist_ok=True)

        csv_path = os.path.join(DATASET_PATH, f"pose-{env_idx}.csv")

        if os.path.exists(csv_path):
            n_rows = sum(1 for _ in open(csv_path)) - 1
            n_imgs = sum(1 for f in os.listdir(save_dir) if f.endswith('.png'))
            if n_rows >= SAMPLES_PER_ENV and n_imgs >= SAMPLES_PER_ENV:
                rospy.loginfo(f"Env {env_idx}: already complete ({n_rows} rows, {n_imgs} images), skipping.")
                return

        csv_file = open(csv_path, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['px','py','pz','qw','qx','qy','qz',
                         'theta','phi','d_theta','d_phi','length','mass'])

        # Random cable params for this environment
        L = rng.uniform(*L_RANGE)
        m = rng.uniform(*M_RANGE)

        # Initial pendulum state — sample in spherical for interpretability,
        # then convert to Cartesian (qx,qy,qz,dqx,dqy,dqz) for ODE integration.
        p_state_sph = np.array([
            rng.uniform(0.0, 0.15),   # theta
            rng.uniform(0, 2*np.pi),   # phi
            rng.uniform(-0.1, 0.1),    # dtheta
            rng.uniform(-0.1, 0.1),    # dphi
        ])
        p_state = np.array(state_spherical_to_cartesian(p_state_sph))

        traj = SmoothTrajectory(seed=int(seed))
        collected = 0
        step = 0

        rospy.loginfo(f"Env {env_idx}: collecting {SAMPLES_PER_ENV} samples, L={L:.2f}m, m={m:.2f}kg")

        while collected < SAMPLES_PER_ENV and not rospy.is_shutdown():
            pos, vel, acc, yaw = traj.step(TRAJ_DT)

            # Integrate pendulum ODE (Cartesian state).
            p_state = rk4_pendulum_step(p_state, TRAJ_DT, acc, L)
            # Renormalize q to unit length. Baumgarte already does this inside
            # the ODE, but an explicit projection keeps the state numerically
            # stable across many steps.
            q_dir = p_state[:3]
            q_norm = np.linalg.norm(q_dir)
            if q_norm > 1e-10:
                p_state[:3] = q_dir / q_norm

            # Small random pitch/roll (realistic flight attitude)
            pitch = rng.uniform(-0.15, 0.15)
            roll  = rng.uniform(-0.15, 0.15)
            self._publish_odom(pos, vel, yaw, pitch, roll)

            step += 1
            if step % SAVE_INTERVAL != 0:
                continue

            # Get depth image
            depth = self._get_depth(timeout=0.3)
            if depth is None:
                continue

            # Save image
            self._save_depth(depth, save_dir, collected)

            # Save pose + payload state (convert pendulum back to spherical for CSV).
            q = R.from_euler('ZYX', [yaw, pitch, roll]).as_quat()  # xyzw → qw qx qy qz
            theta, phi, dtheta, dphi = state_cartesian_to_spherical(p_state)
            phi = (phi + np.pi) % (2*np.pi) - np.pi  # wrap to [-pi, pi]
            writer.writerow([
                f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}",
                f"{q[3]:.6f}", f"{q[0]:.6f}", f"{q[1]:.6f}", f"{q[2]:.6f}",
                f"{theta:.6f}", f"{phi:.6f}",
                f"{dtheta:.6f}", f"{dphi:.6f}",
                f"{L:.4f}", f"{m:.4f}",
            ])
            csv_file.flush()

            collected += 1
            if collected % 1000 == 0:
                rospy.loginfo(f"  Env {env_idx}: {collected}/{SAMPLES_PER_ENV}")

            # Occasionally reset trajectory and resample cable params
            if collected % 2000 == 0:
                traj.reset()
                L = rng.uniform(*L_RANGE)
                m = rng.uniform(*M_RANGE)
                p_state_sph = np.array([rng.uniform(0, 0.15), rng.uniform(0, 2*np.pi),
                                        rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)])
                p_state = np.array(state_spherical_to_cartesian(p_state_sph))

        csv_file.close()
        rospy.loginfo(f"Env {env_idx} done: {collected} samples saved.")

    def run(self, seed=42):
        rng = np.random.default_rng(seed)
        os.makedirs(DATASET_PATH, exist_ok=True)

        for env_idx in range(NUM_ENVS):
            if rospy.is_shutdown():
                break
            env_seed = int(rng.integers(0, 2**31))
            self.generate_env(env_idx, env_seed)

        rospy.loginfo(f"Dataset generation complete: {NUM_ENVS} envs × {SAMPLES_PER_ENV} = {NUM_ENVS*SAMPLES_PER_ENV} samples")
        rospy.loginfo(f"Saved to: {DATASET_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--envs", type=int, default=NUM_ENVS)
    args = parser.parse_args()

    NUM_ENVS = args.envs
    gen = PayloadDatasetGenerator()
    gen.run(seed=args.seed)
