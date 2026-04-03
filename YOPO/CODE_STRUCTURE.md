# YOPO Code Structure and File Roles

## Overview

| Subsystem | Path | Language | Purpose |
|-----------|------|----------|---------|
| **YOPO** | `YOPO/` | Python | Neural network, training, loss functions, ROS deployment |
| **Controller** | `Controller/src/` | C++ | SO(3) attitude control, quadrotor simulator, payload physics |
| **Simulator** | `Simulator/src/` | C++ | CUDA depth rendering, forest generation, dataset creation |

Total: ~40 Python files, ~15 C++ source files, ~10 config/launch/msg files.

---

## 1. YOPO (Python ML Pipeline)

### 1.1 Configuration

| File | Lines | Role |
|------|-------|------|
| `config/config.py` | 22 | YAML config loader. `Config` class reads `traj_opt.yaml`, computes derived params (`sgm_time`, `traj_num`, `goal_length`). Global singleton `cfg`. |
| `config/traj_opt.yaml` | 55 | All hyperparameters: speed limits, loss weights (`ws`, `wc`, `wg`, `wa`, `wd`), lattice grid (5x3x1), camera FOV, payload params (`cable_length=0.8`, `mass=0.3`), bubble radii, `score_dyn_boost`, `gradient_decay`. |

### 1.2 Neural Network (`policy/`)

| File | Lines | Classes/Functions | Role |
|------|-------|-------------------|------|
| `policy/yopo_network.py` | 140 | `YopoNetwork` | Main network module. `inference()`: normalize -> encode pendulum -> lattice transform -> forward -> decode. `forward()`: cat(depth_feat, obs_grid) -> head. `_prepare_input_with_encoder()`: rotate spatial 9D through 15 primitives, broadcast pendulum 8D latent. |
| `policy/models/backbone.py` | 41 | `YopoBackbone` | Image encoder. ResNet18 backbone -> `Conv2d(512, 64, k=1)`. Input: `(B,1,96,160)` -> Output: `(B,64,3,5)`. |
| `policy/models/resnet.py` | 391 | `ResNet`, `BasicBlock`, `Bottleneck` | Standard ResNet18 implementation. Modified first conv: `Conv2d(1, 64, k=7, s=2)` for single-channel depth input. |
| `policy/models/head.py` | 16 | `YopoHead` | Decision head. `Conv1d(81, 256) -> ReLU -> Conv1d(256, 256) -> ReLU -> Conv1d(256, 10)`. Outputs 9 endstate dims + 1 score dim per primitive. |
| `policy/models/pendulum_encoder.py` | 50 | `PendulumStateEncoder` | **New module.** Encodes `[theta, phi, dtheta, dphi]` (4D) -> latent (8D). Architecture: `Linear(4,16) -> GELU -> Linear(16,8) -> LayerNorm + Linear(4,8)` residual. 264 parameters. |
| `policy/primitive.py` | 110 | `LatticePrimitive` | Pre-computes 15 trajectory endpoints on a spherical lattice (5 yaw x 3 pitch x 1 radius). Provides rotation matrices `Rbp[i]` for coordinate transforms. |
| `policy/state_transform.py` | 166 | `StateTransform`, `state_body2world` | Coordinate frame conversions. `normalize_obs()`: vel/vel_max, acc/acc_max, goal/goal_length. `prepare_input()`: rotate 9D spatial through primitive frames, broadcast extra dims. `pred_to_endstate()`: decode tanh output back to body-frame endstate. |
| `policy/poly_solver.py` | 90 | `Poly5Solver` | 5th-order polynomial trajectory solver. Given boundary conditions (pos, vel, acc at start and end), computes coefficients. `get_position(t)`, `get_velocity(t)`, `get_acceleration(t)`. |

### 1.3 Loss Functions (`loss/`)

| File | Lines | Classes/Functions | Role |
|------|-------|-------------------|------|
| `loss/loss_function.py` | 141 | `YOPOLoss` | Combined loss coordinator. Instantiates all sub-losses. `forward()` returns 5 cost tensors: `(smooth, safety, goal, accel, dynamics)`. Computes velocity-normalized weights: `ws/v^5`, `wa/v^3`, `wd/v^2`. |
| `loss/differentiable_pendulum.py` | 235 | `DifferentiablePendulumLoss` | **Core contribution.** Nonlinear spherical pendulum ODE integration via torch.autograd. 20-step semi-implicit Euler. Temporal gradient decay: `alpha = sqrt(g/L)`. Cost = `mean_swing + 0.5*peak_swing + 0.3*final_swing + 0.8*convergence`. |
| `loss/safety_loss.py` | 253 | `SafetyLoss` | Bubble chain collision avoidance. Interpolates N spheres along UAV-to-payload cable. Checks each against ESDF map. Piecewise exponential/linear penalty. Cable direction from apparent gravity (`detach_qvec` option). |
| `loss/guidance_loss.py` | 87 | `GuidanceLoss` | Goal alignment via cosine similarity. `parallel_diff + 0.5 * perp_diff`. Loose constraint (w=0.15) allows flexible navigation. |
| `loss/smoothness_loss.py` | 30 | `SmoothnessLoss` | Jerk and acceleration penalty via Hessian matrices `RJ` and `RA`. Integrated `||d^3x/dt^3||^2` and `||d^2x/dt^2||^2` over polynomial trajectory. |
| `loss/dynamics_loss.py` | 96 | `DynamicsLoss` | **Legacy / dead code.** Linearized quasi-static equilibrium model. Superseded by `DifferentiablePendulumLoss`. Not imported anywhere. |

### 1.4 Dataset and Simulator (`policy/`)

| File | Lines | Classes/Functions | Role |
|------|-------|-------------------|------|
| `policy/yopo_dataset.py` | 291 | `YOPODataset` | PyTorch Dataset. Loads depth images + poses from 10 forest environments. Augments with random vel/acc/goal (body frame). Samples pendulum state from ODE library (training) or CSV (validation). Returns 7-tuple: `(depth, pos, rot, obs, p_state, p_params, map_id)`. |
| `policy/pendulum_simulator.py` | 430 | `pendulum_ode`, `simulate_pendulum`, `generate_payload_state_library`, `validate_dynamics_proxy` | Spherical pendulum ODE in Cartesian coords (avoids theta=0 singularity). Baumgarte stabilization. RK45 integration. Generates physically consistent state library (~200k states) under random sinusoidal accelerations. `validate_dynamics_proxy()`: correlates linearized proxy with true ODE swing (Pearson r). |

### 1.5 Training (`policy/` + root)

| File | Lines | Classes/Functions | Role |
|------|-------|-------------------|------|
| `policy/yopo_trainer.py` | 257 | `YopoTrainer` | Training loop. `train_one_epoch()`: forward pass -> 5 physics losses -> trajectory_loss + score_loss -> backward + clip_grad(0.5). Score labels are detached with `score_dyn_boost=5.0`. Logs 7 loss components to TensorBoard. Saves checkpoints every 10 epochs. |
| `train_yopo.py` | 46 | `configure_random_seed`, `parser` | Entry point. Sets deterministic seed, creates `YopoTrainer(lr=1.5e-4, batch=16)`, runs 50 epochs. |
| `train_ablation.py` | 272 | `AblationConfig`, `run_ablation` | Ablation study runner. 5 configs: A0 (no payload 9D), A1 (random swing), A2 (no encoder), A3 (no gradient decay), A4 (full model). Overrides `obs_dim`, `use_dynamics_loss`, `gradient_decay` per config. |
| `train_iterative.py` | 247 | `IterativeTrainer` | Multi-round training with dataset regeneration. Trains -> evaluates -> regenerates harder data -> repeats. For curriculum learning. |
| `yopo_trt_transfer.py` | 78 | `export_to_tensorrt` | TensorRT export for real-time deployment (<1ms inference). Converts PyTorch model to ONNX then TRT engine. |

### 1.6 ROS Deployment

| File | Lines | Classes/Functions | Role |
|------|-------|-------------------|------|
| `test_yopo_ros.py` | 499 | `YopoNet` | Real-time ROS inference node. Subscribes: `/sim/odom` (50Hz), depth image (30Hz), goal, `/sim/payload_odom`. Publishes: `/so3_control/pos_cmd` (50Hz), trajectory visualization (PointCloud2). Pipeline: inpaint depth -> extract pendulum state from payload odom -> network inference -> polynomial generation -> control commands. |

### 1.7 Evaluation Scripts

| File | Lines | Role |
|------|-------|------|
| `eval_ros_single.py` | 154 | Single-trial ROS evaluation. Launches sim, runs policy, measures success/collision/timeout. |
| `eval_ros_comparison.py` | 379 | Side-by-side comparison of two policies (e.g., with/without payload awareness). |
| `eval_ros_3way.py` | 186 | Three-way comparison (e.g., baseline vs ours vs ablation). |
| `eval_ros_hard.py` | 243 | Hard scenarios: high speed, tight corridors, large initial swing. |
| `eval_ros_success_rate.py` | 374 | Statistical success rate over N randomized trials. Reports collision rate, timeout rate, swing statistics. |
| `evaluate_baselines.py` | 188 | Offline baseline comparison (no ROS). Evaluates loss components on validation set across ablation configs. |
| `evaluate_clearance.py` | 277 | Analyzes minimum obstacle clearance along planned trajectories. |
| `evaluate_closed_loop.py` | 506 | Closed-loop simulation with physics. Steps policy + simulator together, records full state trajectory. |
| `evaluate_multiseg.py` | 287 | Multi-segment trajectory evaluation. Chains polynomial segments and checks continuity. |
| `evaluate_success_rate.py` | 350 | Offline success rate computation using pre-recorded trajectories. |

### 1.8 Visualization

| File | Lines | Role |
|------|-------|------|
| `visualize_policy.py` | 519 | Renders policy decisions: depth image + lattice endpoints + selected trajectory + swing state overlay. |
| `visualize_forest_flight.py` | 465 | 3D visualization of flight through forest with payload cable. Matplotlib animation. |
| `plot_paper_figures.py` | 652 | Publication-quality figures: loss curves, ablation comparisons, swing angle distributions, trajectory samples. |

### 1.9 Other

| File | Lines | Role |
|------|-------|------|
| `generate_payload_dataset.py` | 285 | Standalone script to generate `payload_state_library.npy`. Wraps `pendulum_simulator.py`. |
| `run_all_experiments.py` | 280 | Master orchestrator. Runs training + all evaluations sequentially. |
| `control_msg/_PositionCommand.py` | 312 | Auto-generated ROS message class for `PositionCommand`. Fields: position, velocity, acceleration, yaw, trajectory_id. |

---

## 2. Controller (C++ ROS)

### 2.1 SO3 Control

| File | Lines | Classes/Functions | Role |
|------|-------|-------------------|------|
| `so3_control/src/SO3Control.cpp` | 124 | `SO3Control` | Geometric attitude controller on SO(3). Position PD -> desired force -> desired quaternion -> moment commands. Gains: `kx`, `kv` (position/velocity), `kR`, `kOm` (attitude/angular rate). |
| `so3_control/include/SO3Control.h` | 42 | `SO3Control` | Header. Methods: `calculateControl()`, `setPosition()`, `setVelocity()`, `getComputedForce()`, `getComputedOrientation()`. |
| `so3_control/include/HGDO.h` | 109 | `HGDO` | Hierarchical Generalized Disturbance Observer. Estimates external disturbances (wind, payload tension) from position/velocity error. Used by NetworkControl for robustness. |
| `so3_control/include/NetworkControl.h` | 161 | `NetworkControl` | Main ROS node class. Bridges YOPO policy output (PositionCommand) to SO3Control input. Handles takeoff/land sequences, command interpolation, safety timeout. `simulateTakeoff()`, `simulateLanding()`, `process_command()`. |
| `so3_control/src/NetworkControl.cpp` | 454 | `NetworkControl` | Implementation. Subscribers: odom, imu, position_cmd. Publisher: SO3Command. Timer-based control loop at 50Hz. Integrates HGDO for disturbance compensation. |
| `so3_control/src/so3_control_nodelet.cpp` | 354 | `SO3ControlNodelet` | ROS nodelet wrapper. Allows zero-copy communication when loaded in same process as simulator. Alternative to standalone node. |
| `so3_control/src/control_example.cpp` | 78 | `main` | Minimal example: hardcoded waypoint following with SO3Control. For testing controller standalone. |
| `so3_control/src/network_control_node.cpp` | 10 | `main` | Standalone node entry point for NetworkControl. |
| `so3_control/include/mavros_interface.h` | 194 | `MavrosInterface` | PX4/ArduPilot bridge for real hardware. Converts SO3Command to MAVROS AttitudeTarget. Handles arming, mode switching, failsafe. |

### 2.2 Quadrotor Simulator

| File | Lines | Classes/Functions | Role |
|------|-------|-------------------|------|
| `so3_quadrotor_simulator/src/dynamics/Quadrotor.cpp` | 465 | `Quadrotor`, `operator()` | 6-DOF rigid body dynamics. 22-state ODE: position(3), velocity(3), rotation(9), angular velocity(3), motor RPMs(4). Integrated via `boost::odeint`. Includes aerodynamic drag, motor dynamics (first-order lag), thrust saturation. States [22-25] reserved for pendulum (zeroed -- payload handled externally). |
| `so3_quadrotor_simulator/include/Quadrotor.h` | 115 | `Quadrotor`, `State` | Header. `InternalState = std::array<double, 26>`. State struct: `x, v, R, omega, motor_rpm, swing_angle, swing_velocity`. Physical params: `mass_=0.98`, `payload_mass_=0.20`, `cable_length_=0.50`, `kf_`, `km_`, `arm_length_=0.26`. |
| `so3_quadrotor_simulator/src/quadrotor_simulator_so3.cpp` | 605 | `main`, `getControl`, `stateToOdomMsg`, `odomToTF`, `odomToMesh` | Main simulation loop (1000Hz). **Payload physics**: explicit Euler integration of payload position/velocity. Tension: `fc = (m_L*L*|rho_dot|^2 - m_L*rho.F_thrust) / (m_Q+m_L)`. Baumgarte stabilization (position projection + velocity correction). Aerodynamic drag on payload tangential velocity (`c_drag=0.3`). Publishes: odom (50Hz), IMU, payload markers (cable LINE_STRIP + SPHERE), payload odom, CSV log. |

### 2.3 Messages

| File | Lines | Role |
|------|-------|------|
| `quadrotor_msgs/msg/SO3Command.msg` | 6 | Force vector + desired orientation quaternion + gains `kR`, `kOm` + aux commands. |
| `quadrotor_msgs/msg/PositionCommand.msg` | 21 | Position + velocity + acceleration + jerk + yaw + yaw_dot + trajectory_id + header. Main interface between YOPO policy and controller. |
| `quadrotor_msgs/msg/AuxCommand.msg` | 5 | kf_correction, angle_corrections, current_yaw, use_external_yaw. |
| `quadrotor_msgs/msg/TRPYCommand.msg` | 6 | Thrust + roll/pitch/yaw commands (alternative to SO3). |
| `quadrotor_msgs/msg/Serial.msg` | 13 | Serial communication message for hardware interface. |

### 2.4 Launch Files

| File | Lines | Role |
|------|-------|------|
| `so3_quadrotor_simulator/launch/simulator.launch` | 40 | Main launch. Starts simulator node + SO3 control nodelet. Params: init position, payload mass/cable length, control gains, mass=1.30 (UAV+payload+margin). |
| `so3_control/launch/controller_network.launch` | 18 | Standalone controller launch (without simulator). For real hardware or external sim. |

### 2.5 Config

| File | Lines | Role |
|------|-------|------|
| `so3_control/config/gains_hummingbird.yaml` | 5 | PD gains for Hummingbird platform: `kx=[5.7, 5.7, 6.2]`, `kv=[3.4, 3.4, 4.0]`. |
| `so3_control/config/corrections_hummingbird.yaml` | 4 | Thrust/angle correction offsets. |

---

## 3. Simulator (C++ Dataset Generator)

| File | Lines | Classes/Functions | Role |
|------|-------|-------------------|------|
| `src/sensor_simulator.cpp` | 171 | `SensorSimulator` | CUDA-accelerated depth image rendering. Ray-casting against point cloud map. Outputs 16-bit PNG depth images (0-20m -> 0-65535). |
| `include/sensor_simulator.h` | 173 | `SensorSimulator` | Header. Params: `width`, `height`, `fov_h`, `fov_v`, `max_range`, `min_range`. Methods: `render()`, `setMap()`, `setPose()`. |
| `src/maps.cpp` | 1203 | `Maps` | Procedural environment generation. 7 map types: `generateForest()`, `generateMaze()`, `generateRoom()`, `generateWalls()`, `generatePerlinTerrain()`, `generateSparseForest()`, `generateDenseForest()`. Each outputs a PLY point cloud. Tree generation: cylinder model with configurable height/radius distributions. |
| `include/maps.hpp` | 112 | `Maps` | Header. Params: `map_size`, `resolution`, `tree_density`, `tree_height_range`, `tree_radius_range`. |
| `src/perlinnoise.cpp` | 123 | `PerlinNoise` | 2D/3D Perlin noise for terrain generation. |
| `include/perlinnoise.hpp` | 33 | `PerlinNoise` | Header. `noise(x, y)`, `noise(x, y, z)`. |
| `src/dataset_generator.cpp` | 318 | `main` | Dataset generation pipeline. For each environment: generate point cloud -> compute ESDF -> sample random poses -> render depth images -> save CSV (position, quaternion, payload state). Outputs: `pointcloud-N.ply`, `pose-N.csv`, `N/*.png`. Payload state columns: theta, phi, dtheta, dphi, cable_length, mass. |
| `src/test_simulator.cpp` | 10 | `main` | Minimal CPU test: loads map, renders one frame. |
| `src/test_simulator_cuda.cpp` | 248 | `main` | Full GPU test: renders sequence, measures throughput, validates depth accuracy. |
| `config/config.yaml` | 94 | -- | Camera intrinsics (`width=320, height=240, fov=90`), map params (`resolution=0.2, size=[80,80,6]`), rendering params, tree distributions. |
| `include/cuda_toolkit/helper_math.h` | 1508 | -- | NVIDIA CUDA math helpers (float3, float4 operations). Vendored from CUDA samples. |

---

## 4. Data Flow Between Subsystems

```
                    TRAINING TIME
                    =============

  Simulator (C++)                    YOPO (Python)
  +-----------------+                +------------------+
  | Maps.generate() |                |                  |
  | -> pointcloud   |--- PLY ------>| YOPODataset      |
  | -> ESDF         |--- NPY ------>|   load images    |
  | SensorSim.render|--- PNG ------>|   load poses     |
  | DatasetGen      |--- CSV ------>|   sample p_state |
  +-----------------+                |                  |
                                     | PendulumSim.py   |
                                     | -> state library --->  payload_state_library.npy
                                     |                  |
                                     | YopoTrainer      |
                                     |   network.inference()
                                     |   yopo_loss()    |
                                     |   backward()     |
                                     +------------------+
                                            |
                                       saved/epoch50.pth


                    INFERENCE TIME (ROS)
                    ====================

  Simulator (C++)                    YOPO (Python)
  +--------------------+             +-------------------+
  | QuadrotorSim       |<-- cmd ---- | test_yopo_ros.py  |
  |   1000Hz physics   |             |   30Hz inference  |
  |   payload dynamics |-- odom ---->|   depth callback  |
  |   Baumgarte stab.  |-- depth --->|   pendulum state  |
  |                    |-- payload -->|   network forward |
  +--------------------+ odom        |   poly_solver     |
        |                             |   publish cmd     |
        v                             +-------------------+
  Controller (C++)
  +--------------------+
  | NetworkControl     |
  |   pos_cmd -> SO3   |
  | SO3Control         |
  |   force -> quat    |
  |   -> motor RPMs    |
  +--------------------+
```

---

## 5. Key Dependencies

| Dependency | Used By | Purpose |
|------------|---------|---------|
| PyTorch | YOPO | Network, autograd, CUDA tensors |
| torchvision | YOPO | ResNet18 building blocks |
| OpenCV | YOPO | Depth image loading, inpainting |
| scipy | YOPO | `solve_ivp` for pendulum ODE (dataset generation) |
| pandas | YOPO | CSV pose file loading |
| scikit-learn | YOPO | `train_test_split` for val set |
| ruamel.yaml | YOPO | Config loading |
| rich | YOPO | Training progress bars |
| TensorBoard | YOPO | Loss logging |
| ROS Noetic | Controller, YOPO | Message passing, node lifecycle |
| Eigen3 | Controller | Linear algebra, SO(3) operations |
| boost::odeint | Controller | ODE integration (quadrotor dynamics) |
| CUDA | Simulator | GPU depth rendering |
