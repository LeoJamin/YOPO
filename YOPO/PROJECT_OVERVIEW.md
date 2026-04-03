# YOPO: Physics-Informed Neural Trajectory Planning for UAV Slung-Load Transport

## 1. Project Overview

YOPO (Yet another Optimal Polynomial Online planner) is a deep learning-based real-time motion planning system for UAV slung-load operations. The system combines neural network inference with differentiable physics simulation to plan safe, smooth trajectories while actively managing swing dynamics of a suspended payload.

**Core Innovation**: The architecture replaces traditional imitation learning with **physics-informed trajectory optimization**, where gradients flow directly through a nonlinear spherical pendulum ODE. This enables the network to learn trajectory planning that minimizes payload swing implicitly, rather than explicitly mimicking expert demonstrations.

**Target Venues**: CoRL, RSS, RA-L + ICRA

---

## 2. System Architecture

```
+---------------------------+       +---------------------------+
|     Depth Camera (1ch)    |       |   UAV State (Odometry)    |
|     96 x 160 normalized   |       |  vel, acc, goal, swing    |
+-----------+---------------+       +-----------+---------------+
            |                                   |
            v                                   v
   +--------+---------+              +----------+------------+
   |  YopoBackbone    |              | PendulumStateEncoder   |
   |  ResNet18 -> 64D |              | 4D -> MLP -> 8D latent |
   +--------+---------+              +----------+------------+
            |                                   |
            +----------------+------------------+
                             |
                             v
                  +----------+----------+
                  |  Lattice Transform  |
                  |  15 primitive frames |
                  +----------+----------+
                             |
                             v
                  +----------+----------+
                  |     YopoHead        |
                  | Conv1D -> 9+1 output|
                  +----------+----------+
                             |
                  +----------+----------+
                  | endstate(9) score(1) |
                  +----------+----------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
   +----------+----------+       +----------+----------+
   | 5th-order Polynomial |      | Score-based Selector |
   | Trajectory Generation|      | argmin(score) -> best|
   +----------+----------+       +----------+----------+
              |                             |
              +----------------+------------+
                               |
                               v
                    +----------+----------+
                    |  SO3 Controller      |
                    |  pos, vel, acc cmds  |
                    +---------------------+
```

The project is organized into three subsystems:

| Subsystem | Path | Role |
|-----------|------|------|
| Controller | `/Controller/src/` | Real-time SO(3) attitude control, quadrotor simulator with payload physics |
| Simulator | `/Simulator/src/` | CUDA depth rendering, forest environment generation, dataset creation |
| YOPO | `/YOPO/` | Neural network, training pipeline, loss functions, ROS deployment |

---

## 3. Network Architecture

### 3.1 YopoNetwork (`policy/yopo_network.py`)

**Input**:
- Depth image: `(B, 1, 96, 160)` float32, normalized to [0, 1] from 0-20m range
- Observation: `(B, 13)` = `[vel(3), acc(3), goal(3), theta, phi, dtheta, dphi]`

**Forward Pass**:

```
1. YopoBackbone (ResNet18)
   depth (B, 1, 96, 160)
     -> Conv2d(1, 64, k=7, s=2) -> BN -> ReLU -> MaxPool
     -> ResBlock x4: 64 -> 64 -> 128 -> 256 -> 512
     -> Conv2d(512, 64, k=1)
     -> depth_feat (B, 64, 3, 5)

2. PendulumStateEncoder
   p_state (B, 4)  [theta, phi, dtheta, dphi]
     -> Linear(4, 16) -> GELU -> Linear(16, 8) -> LayerNorm
     + Linear(4, 8) residual skip
     -> p_latent (B, 8)

3. Lattice Transform (StateTransform)
   spatial_obs (B, 9) + p_latent (B, 8)
     -> Rotate spatial through 15 primitive frames
     -> Broadcast p_latent (rotation-invariant)
     -> obs_grid (B, 17, 3, 5)

4. Concatenate + YopoHead
   cat(obs_grid, depth_feat) -> (B, 81, 3, 5)
     -> Conv1D(81, 256) -> ReLU -> Conv1D(256, 256) -> ReLU -> Conv1D(256, 10)
     -> endstate: tanh(output[:, :9]) -> (B, 9, 3, 5)
        score: softplus(output[:, 9]) -> (B, 3, 5)
```

**Lattice Primitives**: 15 trajectory directions = 5 yaw x 3 pitch x 1 radius. Each primitive represents a candidate flight direction over a 5th-order polynomial trajectory segment.

### 3.2 PendulumStateEncoder (`policy/models/pendulum_encoder.py`)

Learns nonlinear features of swing state rather than passing raw angles directly:

```
Input: (B, 4)  [theta, phi, dtheta, dphi]
Main path:  Linear(4, 16) -> GELU -> Linear(16, 8) -> LayerNorm(8)
Skip path:  Linear(4, 8)
Output:     main + skip -> (B, 8)
```

The residual connection preserves raw signal while the main path can learn features like kinetic energy `dtheta^2 + dphi^2 * sin^2(theta)` or swing amplitude.

---

## 4. Loss Functions

### 4.1 Combined Loss (`loss/loss_function.py`)

```
L_total = w_s * L_smooth + w_c * L_safety + w_g * L_goal + w_a * L_accel + w_d * L_dynamics
```

All weights are velocity-normalized for consistent scaling across speed ranges:

| Loss | Config Key | Raw Weight | Velocity Scale | Effective Weight |
|------|-----------|------------|----------------|-----------------|
| Smoothness (jerk) | `ws` | 10.0 | v^5 = 1024 | 0.00977 |
| Safety (collision) | `wc` | 1.5 | 1 (none) | 1.5 |
| Goal (guidance) | `wg` | 0.15 | 1 (none) | 0.15 |
| Acceleration | `wa` | 0.3 | v^3 = 64 | 0.00469 |
| Dynamics (swing) | `wd` | 16.0 | v^2 = 16 | 1.0 |

### 4.2 Differentiable Pendulum Loss (`loss/differentiable_pendulum.py`)

The central innovation. For each candidate trajectory, evaluates UAV acceleration at dense timesteps, then integrates the full nonlinear spherical pendulum ODE:

**Spherical Pendulum Equations of Motion**:

```
theta_ddot = dphi^2 * sin(theta) * cos(theta) - (1/L) * a_eff . e_theta

phi_ddot = -2 * dtheta * dphi * (cos(theta) / sin(theta)) - (1 / (L * sin^2(theta))) * a_eff . e_phi
```

where:
- `a_eff = a_uav + g` (effective acceleration in non-inertial pivot frame)
- `e_theta = [cos(th)cos(ph), cos(th)sin(ph), sin(th)]`
- `e_phi = [-sin(ph), cos(ph), 0]`
- `L` = cable length (detached from gradient flow)

**Integration**: Semi-implicit Euler, 20 steps over segment time (dt = 0.1s). All operations are differentiable via `torch.autograd`.

**Temporal Gradient Decay**:

```
alpha = sqrt(g / L)           # natural pendulum frequency
decay_per_step = exp(-alpha * dt)
```

Physical interpretation: gradient window matches one swing period. Near-future swing is emphasized; far-future swing (which the network cannot directly control) receives exponentially decayed gradients.

**Cost Composition**:

```
cost = 1.0 * mean_swing       # average theta^2 over trajectory
     + 0.5 * peak_swing       # LogSumExp (soft worst-case)
     + 0.3 * final_swing      # theta^2 at trajectory end
     + 0.8 * convergence      # max(0, E_final - E_initial)
     * (1 / L)                # shorter cables -> higher sensitivity
```

The **convergence term** is key for post-turn oscillation damping: if the payload is already swinging (`E_initial > 0`), the network is penalized for failing to reduce swing energy by trajectory end.

**Gradient Flow Path**:

```
L_dynamics -> d(swing)/d(theta) -> d(theta)/d(a_uav) -> d(a_uav)/d(poly_coeff)
           -> d(coeff)/d(endstate) -> d(endstate)/d(network_weights)
```

### 4.3 Safety Loss with Bubble Chain (`loss/safety_loss.py`)

Models the UAV-cable-payload system as a chain of overlapping spheres:

```
UAV (r=0.35m) --- bubble_1 --- bubble_2 --- ... --- Payload (r=0.15m)
```

- Bubble positions interpolated along the cable direction
- Cable direction computed from apparent gravity: `g_app = a_uav - g_earth`
- Each bubble checked against ESDF (Euclidean Signed Distance Field)
- Piecewise cost: exponential near-contact, linear for deep penetration

The `detach_qvec` option (default: True) prevents the network from learning adversarial accelerations that trick the cable direction estimate.

### 4.4 Goal Loss (`loss/guidance_loss.py`)

Cosine similarity-based guidance toward the goal:

```
parallel_diff = |goal_length - projection_along_goal_direction|
perp_diff     = |perpendicular_component|
L_goal        = parallel_diff + 0.5 * perp_diff
```

Loose goal constraint (w=0.15) allows flexible navigation around obstacles.

### 4.5 Smoothness Loss (`loss/smoothness_loss.py`)

Penalizes jerk (3rd derivative) and acceleration (2nd derivative) of the polynomial trajectory:

```
L_smooth = integral(||d^3 x / dt^3||^2 dt)   via Hessian matrix RJ
L_accel  = integral(||d^2 x / dt^2||^2 dt)   via Hessian matrix RA
```

---

## 5. Training Pipeline

### 5.1 Dataset (`policy/yopo_dataset.py`)

**Sources**: 10 forest environments, each with ~10,000 depth images and corresponding poses.

**Per-sample output** (7 items):

| Item | Shape | Description |
|------|-------|-------------|
| depth | (1, 96, 160) | Normalized depth image |
| position | (3,) | World-frame UAV position |
| rotation | (3, 3) | World-to-body rotation matrix |
| obs_body | (9,) | [vel, acc, goal] in body frame |
| p_state | (4,) | [theta, phi, dtheta, dphi] |
| p_params | (2,) | [cable_length, mass] |
| map_id | scalar | Environment index |

**Payload State Sampling**: During training, pendulum states are sampled from a pre-generated ODE library (`payload_state_library.npy`, ~187k states) rather than random angles. This ensures physically consistent initial conditions -- states that could actually arise from real pendulum dynamics under plausible UAV accelerations.

The library is generated by `policy/pendulum_simulator.py`:
1. Simulate spherical pendulum under random sinusoidal UAV accelerations
2. Use Cartesian internal representation (avoids theta=0 singularity)
3. Baumgarte stabilization for constraint |q|=1
4. RK45 integration with rtol=1e-6, atol=1e-8
5. Sample states at 0.1s intervals, filter unrealistic states

### 5.2 Training Loop (`policy/yopo_trainer.py`)

```python
for epoch in range(50):
    for depth, pos, rot, obs_b, p_state, p_params, map_id in train_loader:
        # 1. Augment observation: concat spatial(9D) + pendulum(4D) = 13D
        obs_augmented = cat([obs_b, p_state], dim=-1)

        # 2. Network forward (inference path: normalize -> encode -> transform -> predict)
        endstate, score = policy.inference(depth, obs_augmented)

        # 3. Expand to all 15 primitives: (B, 9, 3, 5) -> (B*15, 9)
        #    Each primitive gets its own loss evaluation

        # 4. Differentiable physics loss
        smooth, safety, goal, accel, dynamics = yopo_loss(
            start_state, end_state, goal, map_id, p_state, p_params
        )

        # 5. Trajectory loss (gradients flow through ODE)
        L_traj = w_s*smooth + w_c*safety + w_g*goal + w_a*accel + w_d*dynamics

        # 6. Score loss (detached labels, no physics gradient)
        score_label = (costs + 5.0 * w_d * dynamics).detach()
        L_score = smooth_l1_loss(score, score_label)

        # 7. Combined loss + gradient step
        (L_traj + L_score).backward()
        clip_grad_norm_(parameters, max_norm=0.5)
        optimizer.step()
```

**Key Design Choices**:
- Score labels are **detached**: the trajectory selector learns to predict total cost without influencing trajectory optimization
- `score_dyn_boost = 5.0`: dynamics cost is amplified in the score label so the selector preferentially picks low-swing trajectories at inference time
- Gradient clipping at 0.5: prevents instability from stiff pendulum ODE coupling

### 5.3 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1.5e-4 | AdamW with fused CUDA |
| Batch size | 16 | Limited by RTX 2060 VRAM |
| Epochs | 50 | Checkpoints every 10 |
| Max grad norm | 0.5 | Prevents ODE gradient explosion |
| ODE steps | 20 | Semi-implicit Euler |
| Segment time | 2.5s | = 2 * radio_range / vel_max |
| Gradient decay | enabled | alpha = sqrt(g/L) |
| Pendulum latent dim | 8 | Encoder bottleneck |

---

## 6. Simulator and Controller (C++)

### 6.1 Quadrotor Simulator (`Controller/src/so3_quadrotor_simulator/`)

**Physics Engine**: 22-state ODE integrated via `boost::odeint`:
- States [0-2]: position (x, y, z)
- States [3-5]: velocity (vx, vy, vz)
- States [6-14]: rotation matrix R (column-major)
- States [15-17]: angular velocity (omega_x, omega_y, omega_z)
- States [18-21]: motor RPMs (with first-order lag)

**Payload Dynamics** (explicit Euler, separate from UAV ODE):

```
1. Compute total thrust from motor RPMs
2. Cable direction: rho = (pos_L - pos_Q) / rope_length
3. Tension: fc = (m_L * L * |rho_dot|^2 - m_L * rho . F_thrust) / (m_Q + m_L)
4. UAV external force: fc * rho (pulls UAV toward payload)
5. Payload acceleration: a_L = (-fc * rho) / m_L - g + aerodynamic_drag
6. Euler integration: vel_L += a_L * dt; pos_L += vel_L * dt
7. Baumgarte stabilization: project pos_L onto sphere of radius L
8. Velocity correction: remove radial component of relative velocity
```

**Design Decision**: Payload physics runs in the outer simulation loop (not inside the odeint ODE) to avoid double-counting cable forces. The internal state slots [22-25] are reserved but zeroed.

### 6.2 SO3 Controller (`Controller/src/so3_control/`)

Geometric controller on SO(3) manifold:
- Position PD control -> desired force vector
- Force decomposition -> desired attitude (quaternion)
- Attitude error on SO(3) -> moment commands
- Motor mixing -> individual RPM commands

Includes HGDO (Hierarchical Generalized Disturbance Observer) for robustness against unmodeled payload dynamics.

### 6.3 ROS Launch Configuration

```xml
<!-- simulator.launch -->
<param name="simulator/payload_mass" value="0.3"/>
<param name="simulator/cable_length" value="0.8"/>
<param name="mass" value="1.30"/>  <!-- 0.98 UAV + 0.30 payload + margin -->
```

Simulation rate: 1000 Hz (physics), 50 Hz (odometry), 30 Hz (depth).

---

## 7. ROS Deployment (`test_yopo_ros.py`)

Real-time inference pipeline at 30 Hz:

```
Depth Callback (30 Hz):
  1. Inpaint NaN -> normalize -> resize to (96, 160)
  2. Extract pendulum state from /sim/payload_odom:
       rel_pos = p_load - p_drone
       theta = arccos(-rel_pos_z / ||rel_pos||)
       phi = atan2(rel_pos_y, rel_pos_x)
       [dtheta, dphi from velocity projection]
  3. Compose observation: [vel_body, acc_body, goal_body, theta, phi, dtheta, dphi]
  4. Network inference (GPU, <5ms)
  5. Select best trajectory: argmin(score)
  6. Generate 5th-order polynomial from endstate

Control Timer (50 Hz):
  7. Evaluate polynomial at current time
  8. Publish PositionCommand (pos, vel, acc, yaw)
```

**Visualization** (RViz):
- Cable: LINE_STRIP marker (UAV to payload)
- Payload: SPHERE marker (orange, r=0.15m)
- Best trajectory: PointCloud2 (green)
- All candidate trajectories: PointCloud2 (colored by score)

---

## 8. Directory Structure

```
YOPO/
├── config/
│   ├── config.py                    # YAML config loader
│   └── traj_opt.yaml                # All hyperparameters
├── policy/
│   ├── models/
│   │   ├── backbone.py              # ResNet18 depth encoder -> 64D
│   │   ├── head.py                  # Conv1D decision head -> 9+1
│   │   ├── pendulum_encoder.py      # Swing state -> 8D latent
│   │   └── resnet.py                # ResNet18 implementation
│   ├── primitive.py                 # 15-direction lattice generation
│   ├── poly_solver.py               # 5th-order polynomial solver
│   ├── pendulum_simulator.py        # ODE state library generator
│   ├── state_transform.py           # Frame transforms, normalization
│   ├── yopo_dataset.py              # Dataset with payload augmentation
│   ├── yopo_network.py              # Main network module
│   └── yopo_trainer.py              # Training loop
├── loss/
│   ├── differentiable_pendulum.py   # Nonlinear pendulum ODE loss
│   ├── guidance_loss.py             # Goal alignment
│   ├── safety_loss.py               # Bubble chain collision avoidance
│   ├── smoothness_loss.py           # Jerk + acceleration penalty
│   └── loss_function.py             # Combined YOPOLoss
├── train_yopo.py                    # Training entry point
├── test_yopo_ros.py                 # ROS deployment node
├── generate_payload_dataset.py      # Payload state library creation
├── train_ablation.py                # Ablation study runner
├── evaluate_*.py                    # Evaluation scripts (10 variants)
└── saved/                           # Checkpoints and TensorBoard logs

Controller/src/
├── so3_control/                     # Geometric attitude controller
│   ├── SO3Control.{h,cpp}           # SO(3) PD controller
│   ├── NetworkControl.{h,cpp}       # ROS bridge to YOPO network
│   └── HGDO.h                       # Disturbance observer
├── so3_quadrotor_simulator/         # 6-DOF physics simulator
│   ├── Quadrotor.{h,cpp}            # 22-state ODE (odeint)
│   └── quadrotor_simulator_so3.cpp  # Main loop + payload integrator
└── utils/quadrotor_msgs/            # Custom ROS message definitions

Simulator/src/
├── sensor_simulator.{h,cpp}         # CUDA depth rendering
├── dataset_generator.cpp            # Forest environment + dataset
└── config/config.yaml               # Environment parameters
```

---

## 9. Key Algorithmic Contributions

### 9.1 Differentiable Pendulum Rollout

Prior work on UAV slung-load planning either:
- Uses linearized small-angle approximations (invalid above ~15 deg)
- Treats payload as a static disturbance (no swing prediction)
- Plans in a two-stage pipeline (trajectory then swing check)

YOPO integrates the **full nonlinear spherical pendulum ODE** directly into the loss function via `torch.autograd`. The gradient path is:

```
swing_cost -> d(theta)/d(ddtheta) -> d(ddtheta)/d(a_uav) -> d(a_uav)/d(poly_coeff)
           -> d(coeff)/d(boundary_conditions) -> d(BC)/d(network_output)
```

This is a single differentiable computation graph -- no REINFORCE, no reward shaping, no two-stage planning.

### 9.2 Temporal Gradient Decay

Adapted from Zhang et al. ("Back to Newton's Laws", Nature Machine Intelligence, 2025), extended from single rigid body to **coupled multi-body oscillatory dynamics**:

```
decay_rate = sqrt(g / L)     # pendulum natural frequency
weight(k) = exp(-decay_rate * k * dt)
```

The decay rate is physically motivated: it equals the pendulum's natural frequency, so the gradient window spans approximately one full swing period. This prevents the network from being penalized for unavoidable swing propagation beyond one period.

### 9.3 Dynamic Bubble Chain

Unlike point-robot collision avoidance, YOPO models the full UAV-cable-payload geometry:

```
For i = 0, 1, ..., N_bubbles:
    r_i = r_uav + (r_load - r_uav) * i / N
    p_i = p_uav + (i / N) * L * q_hat
    cost_i = penalty(SDF(p_i) - r_i - r_safe)
```

The cable direction `q_hat` is computed from the apparent gravity vector, making it differentiable through the trajectory acceleration.

### 9.4 Convergence Incentive

The convergence term explicitly teaches swing damping:

```
E_swing = theta^2 + 0.1 * (dtheta^2 + dphi^2 * sin^2(theta))
L_convergence = max(0, E_final - E_initial)
```

This activates only when the payload is already swinging (post-turn scenarios). The network learns to choose counter-acceleration trajectories that actively reduce oscillation -- a behavior that emerges purely from the physics loss without any expert demonstration.

---

## 10. Running the System

### Training

```bash
conda activate yopo
cd ~/yopo_ws/src/YOPO/YOPO

# Generate payload state library (once)
python policy/pendulum_simulator.py --mode generate --n_samples 200000

# Train from scratch
python train_yopo.py --trial 47

# Resume from checkpoint
python train_yopo.py --pretrained 1 --trial 46 --epoch 50

# Monitor training
tensorboard --logdir saved/
```

### ROS Simulation

```bash
# Terminal 1: Launch simulator + controller
source ~/yopo_ws/src/YOPO/Controller/devel/setup.bash
roslaunch so3_quadrotor_simulator simulator.launch

# Terminal 2: Run YOPO network node
conda activate yopo
cd ~/yopo_ws/src/YOPO/YOPO
python test_yopo_ros.py --trial 47 --epoch 50

# Terminal 3: Visualize
rviz -d ~/yopo_ws/src/YOPO/YOPO/yopo.rviz

# Terminal 4: Send goal
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped ...
```

### Ablation Studies

```bash
python train_ablation.py
# Runs: A0 (no payload), A1 (random swing), A2 (no encoder),
#        A3 (no decay), A4 (full model)
```
