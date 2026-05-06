"""
ROS closed-loop success rate evaluation for YOPO-Payload.
Runs N independent episodes with VARIED GOALS, each with a fresh simulator reset.
Goals are published via /move_base_simple/goal after planner starts.
Records: goal reached, flight time, peak swing, final distance.
"""

import argparse
import subprocess
import signal
import time
import os
import sys
import json
import math
import re
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ─── Config ───
N_EPISODES = 10
TIMEOUT_SEC = 80          # max flight time per episode
ARRIVE_DIST = 3.0         # meters
SETTLE_SEC = 5            # wait after arrival for swing to settle
STARTUP_WAIT = 15         # seconds for sensor_simulator init (was 8 — too short, drone fell from init_z=2 to 1.5 before controller could hold)
PLANNER_WAIT = 20         # seconds for planner to load model + warm up + initial flight from spawn (was 12)

# Planner-checkpoint selection — overridden by argparse in main().
# Defaults match the canonical 13-D Cartesian-surrogate model (YOPO_3).
TRIAL = 3
EPOCH = 50
OBS_DIM = 13

WS_SIM = "/home/jamine/research/diff-slung/code/Simulator"
WS_CTRL = "/home/jamine/research/diff-slung/code/Controller"
YOPO_DIR = "/home/jamine/research/diff-slung/code/YOPO"
PYTHON = "/home/jamine/miniconda3/envs/yopo/bin/python"
SETUP = f"source {WS_SIM}/devel/setup.bash && source {WS_CTRL}/devel/setup.bash --extend"

# ─── Goal generation ───
# Start is always (0, 0, 2). Goals vary in distance and direction.
# sensor_simulator generates a NEW random map each restart, so obstacles differ too.
def generate_goals(n, seed=42):
    """Generate diverse goals at various distances and angles."""
    rng = np.random.RandomState(seed)
    goals = []
    # Mix of distances (20-60m) and angles (full 360°)
    distances = [20, 25, 30, 35, 40, 45, 50, 55, 60, 30,
                 40, 50, 35, 25, 45, 55, 20, 60, 35, 50]
    angles_deg = [0, 45, 90, -30, 180, -90, 15, 135, -60, -150,
                  60, -45, 120, -120, 30, 150, -15, 75, -75, 170]

    for i in range(n):
        idx = i % len(distances)
        d = distances[idx]
        a = math.radians(angles_deg[idx])
        gx = d * math.cos(a)
        gy = d * math.sin(a)
        gz = 2.0  # fixed altitude goal
        goals.append([round(gx, 1), round(gy, 1), gz])
    return goals


def kill_all():
    """Kill all ROS-related processes."""
    for proc in ["quadrotor_simulator_so3", "sensor_simulator_cuda",
                 "rosmaster", "roscore", "rosout", "roslaunch", "nodelet"]:
        subprocess.run(f"killall -9 {proc}", shell=True,
                       capture_output=True, timeout=5)
    subprocess.run("pkill -9 -f test_yopo_ros.py", shell=True,
                   capture_output=True, timeout=5)
    time.sleep(2)


def start_ros_stack():
    """Start roscore, simulator, and sensor simulator."""
    # roscore
    subprocess.Popen(
        f"bash -ic '{SETUP} && roscore'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    time.sleep(5)  # was 3 — give roscore time to fully bind on slow systems.

    # simulator + controller. Use simulator_attitude_control.launch
    # (network_control_node) — this is the controller stack the planner's
    # pos_cmd actually drives. simulator.launch (SO3ControlNodelet) accepts
    # pos_cmd but does not actuate the drone in our setup.
    subprocess.Popen(
        f"bash -ic '{SETUP} && roslaunch so3_quadrotor_simulator simulator_attitude_control.launch'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    time.sleep(8)  # was 3 — controller must converge to hold drone at init_z=2 before commands flow.

    # sensor simulator (generates a new random map each time)
    subprocess.Popen(
        f"bash -ic '{SETUP} && rosrun sensor_simulator sensor_simulator_cuda'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    time.sleep(STARTUP_WAIT)


def publish_goal(goal):
    """Publish a goal via /move_base_simple/goal topic."""
    gx, gy, gz = goal
    cmd = (
        f"bash -c \"{SETUP} && rostopic pub -1 /move_base_simple/goal "
        f"geometry_msgs/PoseStamped "
        f"'{{header: {{frame_id: world}}, "
        f"pose: {{position: {{x: {gx}, y: {gy}, z: {gz}}}, "
        f"orientation: {{w: 1.0}}}}}}'\""
    )
    subprocess.run(cmd, shell=True, capture_output=True, timeout=10)


def run_episode(episode_id, goal):
    """
    Run one episode: start planner, publish goal, monitor logs, return results.
    """
    log_file = f"/tmp/yopo_eval_ep{episode_id}.log"
    goal_dist = math.sqrt(goal[0]**2 + goal[1]**2)

    # Start planner (default goal doesn't matter, we override it)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    planner_proc = subprocess.Popen(
        f"bash -ic '{SETUP} && cd {YOPO_DIR} && {PYTHON} -u test_yopo_ros.py "
        f"--trial {TRIAL} --epoch {EPOCH} --obs_dim {OBS_DIM}'",
        shell=True, stdout=open(log_file, 'w'), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, env=env)

    time.sleep(PLANNER_WAIT)

    # Publish the actual goal
    publish_goal(goal)
    time.sleep(0.5)

    # Monitor the log for arrival or timeout
    start_time = time.time()
    arrived = False
    arrive_time = None
    first_nav_time = None

    while time.time() - start_time < TIMEOUT_SEC:
        time.sleep(1.0)

        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
        except:
            continue

        # Parse NAV lines
        nav_lines = [l.strip() for l in lines if l.startswith("[NAV]")]

        if nav_lines and first_nav_time is None:
            first_nav_time = time.time()

        # Check for arrival
        for line in lines:
            if "ARRIVE!" in line:
                arrived = True
                if arrive_time is None:
                    arrive_time = time.time()
                break

        # Also check: if latest NAV dist < ARRIVE_DIST (backup detection)
        if nav_lines and not arrived:
            last_nav = nav_lines[-1]
            m = re.search(r'dist=([\d.]+)m', last_nav)
            if m and float(m.group(1)) < ARRIVE_DIST:
                arrived = True
                arrive_time = time.time()

        if arrived and (time.time() - arrive_time > SETTLE_SEC):
            break

    # Read final log
    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
        nav_lines = [l.strip() for l in all_lines if l.startswith("[NAV]")]
    except:
        nav_lines = []

    # Kill planner
    try:
        os.killpg(os.getpgid(planner_proc.pid), signal.SIGKILL)
    except:
        pass
    try:
        planner_proc.wait(timeout=5)
    except:
        pass

    # Parse results
    flight_data = parse_nav_lines(nav_lines)

    if first_nav_time and arrive_time:
        flight_time = arrive_time - first_nav_time
    elif first_nav_time:
        flight_time = time.time() - first_nav_time
    else:
        flight_time = TIMEOUT_SEC

    result = {
        "episode": episode_id,
        "goal": goal,
        "goal_dist_m": round(goal_dist, 1),
        "arrived": arrived,
        "flight_time_s": round(max(flight_time, 0), 1),
        "final_dist_m": flight_data["final_dist"],
        "min_dist_m": flight_data["min_dist"],
        "peak_swing_deg": flight_data["peak_swing"],
        "mean_swing_deg": flight_data["mean_swing"],
        "final_swing_deg": flight_data["final_swing"],
        "mean_speed_ms": flight_data["mean_speed"],
        "peak_speed_ms": flight_data["peak_speed"],
        "min_z": flight_data["min_z"],
        "mean_z": flight_data["mean_z"],
        "n_nav_samples": len(nav_lines),
    }

    return result


def parse_nav_lines(nav_lines):
    """Parse [NAV] log lines to extract flight statistics."""
    dists = []
    swings = []
    speeds = []
    zs = []

    pattern = re.compile(
        r'\[NAV\] dist=([\d.]+)m speed=([\d.]+)m/s '
        r'pos=\(([-\d.]+),([-\d.]+),([-\d.]+)\)'
        r'(?: \| swing=([\d.]+)°)?')

    for line in nav_lines:
        m = pattern.search(line)
        if m:
            dists.append(float(m.group(1)))
            speeds.append(float(m.group(2)))
            zs.append(float(m.group(5)))
            if m.group(6):
                swings.append(float(m.group(6)))

    if not dists:
        return {
            "final_dist": 999, "min_dist": 999,
            "peak_swing": 0, "mean_swing": 0, "final_swing": 0,
            "mean_speed": 0, "peak_speed": 0,
            "min_z": 0, "mean_z": 0,
        }

    final_swings = swings[-5:] if len(swings) >= 5 else swings

    return {
        "final_dist": round(dists[-1], 2),
        "min_dist": round(min(dists), 2),
        "peak_swing": round(max(swings), 1) if swings else 0,
        "mean_swing": round(float(np.mean(swings)), 1) if swings else 0,
        "final_swing": round(float(np.mean(final_swings)), 1) if final_swings else 0,
        "mean_speed": round(float(np.mean(speeds)), 2),
        "peak_speed": round(max(speeds), 2),
        "min_z": round(min(zs), 2),
        "mean_z": round(float(np.mean(zs)), 2),
    }


def main():
    global TRIAL, EPOCH, OBS_DIM

    ap = argparse.ArgumentParser(description="YOPO-Payload ROS closed-loop eval")
    ap.add_argument("--trial",    type=int, default=TRIAL,    help="model: saved/YOPO_<trial>/")
    ap.add_argument("--epoch",    type=int, default=EPOCH,    help="checkpoint epoch number")
    ap.add_argument("--obs_dim",  type=int, default=OBS_DIM,  help="observation dimension (9/13/15)")
    ap.add_argument("--episodes", type=int, default=N_EPISODES, help="number of Monte-Carlo episodes")
    ap.add_argument("--seed",     type=int, default=42,        help="goal-generation seed")
    ap.add_argument("--output",   type=str, required=True,     help="output JSON path")
    ap.add_argument("--label",    type=str, default="",        help="optional cell label string")
    args = ap.parse_args()

    TRIAL   = args.trial
    EPOCH   = args.epoch
    OBS_DIM = args.obs_dim
    n_episodes = args.episodes

    goals = generate_goals(n_episodes, seed=args.seed)

    print(f"{'='*70}")
    print(f"YOPO-Payload ROS Closed-Loop Eval — YOPO_{TRIAL}/epoch{EPOCH} (obs_dim={OBS_DIM})")
    print(f"Episodes: {n_episodes} | Timeout: {TIMEOUT_SEC}s | Arrive: {ARRIVE_DIST}m | Seed: {args.seed}")
    if args.label:
        print(f"Label: {args.label}")
    print(f"Start: (0, 0, 2) | Goals: varied distances & directions")
    print(f"{'='*70}")
    for i, g in enumerate(goals):
        d = math.sqrt(g[0]**2 + g[1]**2)
        a = math.degrees(math.atan2(g[1], g[0]))
        print(f"  Ep {i+1}: goal=({g[0]:6.1f}, {g[1]:6.1f}, {g[2]:.0f})  dist={d:.0f}m  angle={a:.0f}°")
    print(f"{'='*70}")

    results = []

    for ep in range(n_episodes):
        goal = goals[ep]
        goal_dist = math.sqrt(goal[0]**2 + goal[1]**2)
        goal_angle = math.degrees(math.atan2(goal[1], goal[0]))

        print(f"\n--- Episode {ep+1}/{n_episodes} ---")
        print(f"  Goal: ({goal[0]:.1f}, {goal[1]:.1f})  dist={goal_dist:.0f}m  angle={goal_angle:.0f}°")

        # Fresh start each episode (new random map too)
        print("  Killing old processes...")
        kill_all()

        print("  Starting ROS stack (new random map)...")
        start_ros_stack()

        print("  Running episode...")
        result = run_episode(ep + 1, goal)
        results.append(result)

        status = "ARRIVED" if result["arrived"] else "TIMEOUT"
        print(f"  Result: {status} | final_dist={result['final_dist_m']:.1f}m "
              f"| min_dist={result['min_dist_m']:.1f}m "
              f"| peak_swing={result['peak_swing_deg']:.1f}° "
              f"| mean_swing={result['mean_swing_deg']:.1f}° "
              f"| time={result['flight_time_s']:.1f}s "
              f"| min_z={result['min_z']:.1f}m")

    # Clean up
    kill_all()

    # Aggregate
    arrived_count = sum(1 for r in results if r["arrived"])
    success_rate = arrived_count / n_episodes

    peak_swings = [r["peak_swing_deg"] for r in results]
    mean_swings = [r["mean_swing_deg"] for r in results]
    flight_times = [r["flight_time_s"] for r in results if r["arrived"]]
    min_dists = [r["min_dist_m"] for r in results]
    min_zs = [r["min_z"] for r in results]
    goal_dists = [r["goal_dist_m"] for r in results]

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY — YOPO_{TRIAL} ({OBS_DIM}D, payload-aware)")
    print(f"{'='*70}")
    print(f"Episodes:           {n_episodes}")
    print(f"Goal distances:     {min(goal_dists):.0f}m – {max(goal_dists):.0f}m")
    print(f"Arrive threshold:   {ARRIVE_DIST}m")
    print(f"{'─'*70}")
    print(f"SUCCESS RATE:       {success_rate:.0%}  ({arrived_count}/{n_episodes})")
    print(f"{'─'*70}")
    if flight_times:
        print(f"Avg flight time:    {np.mean(flight_times):.1f}s ± {np.std(flight_times):.1f}s")
    print(f"Mean peak swing:    {np.mean(peak_swings):.1f}° ± {np.std(peak_swings):.1f}°")
    print(f"Mean avg swing:     {np.mean(mean_swings):.1f}° ± {np.std(mean_swings):.1f}°")
    print(f"Mean min dist:      {np.mean(min_dists):.1f}m ± {np.std(min_dists):.1f}m")
    print(f"Mean min altitude:  {np.mean(min_zs):.2f}m ± {np.std(min_zs):.2f}m")
    print(f"{'='*70}")

    # Per-episode table
    print(f"\n{'─'*78}")
    print(f"{'Ep':>3} {'GoalD':>6} {'Angle':>6} {'Status':>7} {'FDist':>6} {'MinD':>5} "
          f"{'PkSw':>6} {'AvSw':>6} {'Time':>6} {'MinZ':>5}")
    print(f"{'─'*78}")
    for r in results:
        s = "OK" if r["arrived"] else "FAIL"
        a = math.degrees(math.atan2(r["goal"][1], r["goal"][0]))
        print(f"{r['episode']:>3} {r['goal_dist_m']:>5.0f}m {a:>5.0f}° {s:>7} "
              f"{r['final_dist_m']:>5.1f}m {r['min_dist_m']:>4.1f}m "
              f"{r['peak_swing_deg']:>5.1f}° {r['mean_swing_deg']:>5.1f}° "
              f"{r['flight_time_s']:>5.1f}s {r['min_z']:>4.1f}m")
    print(f"{'─'*78}")

    # Save
    summary = {
        "label": args.label,
        "trial": TRIAL,
        "epoch": EPOCH,
        "obs_dim": OBS_DIM,
        "seed": args.seed,
        "n_episodes": n_episodes,
        "arrive_threshold_m": ARRIVE_DIST,
        "timeout_s": TIMEOUT_SEC,
        "success_rate": success_rate,
        "arrived_count": arrived_count,
        "mean_peak_swing_deg": round(float(np.mean(peak_swings)), 1),
        "std_peak_swing_deg": round(float(np.std(peak_swings)), 1),
        "mean_avg_swing_deg": round(float(np.mean(mean_swings)), 1),
        "mean_flight_time_s": round(float(np.mean(flight_times)), 1) if flight_times else None,
        "episodes": results,
    }

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
