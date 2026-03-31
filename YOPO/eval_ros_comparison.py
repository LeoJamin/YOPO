"""
ROS closed-loop COMPARISON experiment.
Runs the same set of goals on multiple methods:
  - YOPO_1   (9D, original, no payload awareness)
  - YOPO_43  (13D, payload-aware with swing angles)

Each method × goal combination gets a fresh simulator (new random map).
"""

import subprocess
import signal
import time
import os
import sys
import json
import math
import re
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ─── Config ───
TIMEOUT_SEC = 80
ARRIVE_DIST = 3.0
SETTLE_SEC = 5
STARTUP_WAIT = 8
PLANNER_WAIT = 12

WS = "/home/jamine/yopo_ws"
YOPO_DIR = f"{WS}/src/YOPO/YOPO"
PYTHON = "/home/jamine/miniconda3/envs/yopo/bin/python"
SETUP = f"source {WS}/devel/setup.bash"

# Methods to compare
METHODS = [
    {"name": "YOPO-Original (9D)",  "trial": 1,  "obs_dim": 9,  "epoch": 50},
    {"name": "YOPO-Payload (13D)",  "trial": 43, "obs_dim": 13, "epoch": 50},
]

# Shared goal set — same goals for fair comparison
GOALS = [
    [20.0,   0.0, 2.0],    # 20m, forward
    [17.7,  17.7, 2.0],    # 25m, 45°
    [ 0.0,  30.0, 2.0],    # 30m, 90°
    [30.3, -17.5, 2.0],    # 35m, -30°
    [-40.0,  0.0, 2.0],    # 40m, backward
    [ 0.0, -45.0, 2.0],    # 45m, -90°
    [48.3,  12.9, 2.0],    # 50m, 15°
    [-38.9, 38.9, 2.0],    # 55m, 135°
    [30.0, -52.0, 2.0],    # 60m, -60°
    [-26.0, -15.0, 2.0],   # 30m, -150°
]


def kill_all():
    for proc in ["quadrotor_simulator_so3", "sensor_simulator_cuda",
                 "rosmaster", "roscore", "rosout", "roslaunch", "nodelet"]:
        subprocess.run(f"killall -9 {proc}", shell=True, capture_output=True, timeout=5)
    subprocess.run("pkill -9 -f test_yopo_ros.py", shell=True, capture_output=True, timeout=5)
    time.sleep(2)


def start_ros_stack():
    subprocess.Popen(
        f"bash -c '{SETUP} && roscore'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    time.sleep(3)

    subprocess.Popen(
        f"bash -c '{SETUP} && roslaunch so3_quadrotor_simulator simulator.launch'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    time.sleep(3)

    subprocess.Popen(
        f"bash -c '{SETUP} && rosrun sensor_simulator sensor_simulator_cuda'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    time.sleep(STARTUP_WAIT)


def publish_goal(goal):
    gx, gy, gz = goal
    cmd = (
        f"bash -c \"{SETUP} && rostopic pub -1 /move_base_simple/goal "
        f"geometry_msgs/PoseStamped "
        f"'{{header: {{frame_id: world}}, "
        f"pose: {{position: {{x: {gx}, y: {gy}, z: {gz}}}, "
        f"orientation: {{w: 1.0}}}}}}'\""
    )
    subprocess.run(cmd, shell=True, capture_output=True, timeout=10)


def run_episode(trial, obs_dim, epoch, goal, ep_id):
    log_file = f"/tmp/yopo_cmp_t{trial}_ep{ep_id}.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    planner_proc = subprocess.Popen(
        f"bash -c '{SETUP} && cd {YOPO_DIR} && {PYTHON} -u test_yopo_ros.py "
        f"--trial {trial} --epoch {epoch} --obs_dim {obs_dim}'",
        shell=True, stdout=open(log_file, 'w'), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, env=env)

    time.sleep(PLANNER_WAIT)

    # Publish goal
    publish_goal(goal)
    time.sleep(0.5)

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

        nav_lines = [l.strip() for l in lines if l.startswith("[NAV]")]

        if nav_lines and first_nav_time is None:
            first_nav_time = time.time()

        for line in lines:
            if "ARRIVE!" in line:
                arrived = True
                if arrive_time is None:
                    arrive_time = time.time()
                break

        if not arrived and nav_lines:
            m = re.search(r'dist=([\d.]+)m', nav_lines[-1])
            if m and float(m.group(1)) < ARRIVE_DIST:
                arrived = True
                arrive_time = time.time()

        if arrived and (time.time() - arrive_time > SETTLE_SEC):
            break

    # Final log read
    try:
        with open(log_file, 'r') as f:
            nav_lines = [l.strip() for l in f.readlines() if l.startswith("[NAV]")]
    except:
        nav_lines = []

    try:
        os.killpg(os.getpgid(planner_proc.pid), signal.SIGKILL)
    except:
        pass
    try:
        planner_proc.wait(timeout=5)
    except:
        pass

    flight_data = parse_nav_lines(nav_lines)

    if first_nav_time and arrive_time:
        flight_time = arrive_time - first_nav_time
    elif first_nav_time:
        flight_time = time.time() - first_nav_time
    else:
        flight_time = TIMEOUT_SEC

    return {
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


def parse_nav_lines(nav_lines):
    dists, swings, speeds, zs = [], [], [], []

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
        return {"final_dist": 999, "min_dist": 999,
                "peak_swing": 0, "mean_swing": 0, "final_swing": 0,
                "mean_speed": 0, "peak_speed": 0, "min_z": 0, "mean_z": 0}

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
    n_goals = len(GOALS)
    n_methods = len(METHODS)
    total_runs = n_goals * n_methods

    print(f"{'='*75}")
    print(f"YOPO Comparison Experiment — ROS Closed-Loop")
    print(f"{'='*75}")
    print(f"Methods:  {n_methods}")
    for m in METHODS:
        print(f"  - {m['name']} (trial={m['trial']}, obs_dim={m['obs_dim']})")
    print(f"Goals:    {n_goals} (distances 20-60m, varied angles)")
    print(f"Total:    {total_runs} runs")
    print(f"Timeout:  {TIMEOUT_SEC}s per run")
    print(f"{'='*75}")

    all_results = {}

    for method in METHODS:
        mname = method["name"]
        print(f"\n{'#'*75}")
        print(f"# Method: {mname}")
        print(f"{'#'*75}")

        method_results = []

        for gi, goal in enumerate(GOALS):
            goal_dist = math.sqrt(goal[0]**2 + goal[1]**2)
            goal_angle = math.degrees(math.atan2(goal[1], goal[0]))

            run_id = gi + 1
            print(f"\n  [{mname}] Goal {run_id}/{n_goals}: "
                  f"({goal[0]:.0f},{goal[1]:.0f}) dist={goal_dist:.0f}m angle={goal_angle:.0f}°")

            kill_all()
            start_ros_stack()

            result = run_episode(
                method["trial"], method["obs_dim"], method["epoch"],
                goal, run_id)

            result["goal"] = goal
            result["goal_dist_m"] = round(goal_dist, 1)
            result["goal_angle_deg"] = round(goal_angle, 1)
            method_results.append(result)

            status = "OK" if result["arrived"] else "FAIL"
            print(f"    {status} | min_dist={result['min_dist_m']:.1f}m "
                  f"| peak_swing={result['peak_swing_deg']:.1f}° "
                  f"| avg_swing={result['mean_swing_deg']:.1f}° "
                  f"| time={result['flight_time_s']:.1f}s "
                  f"| min_z={result['min_z']:.1f}m")

        all_results[mname] = method_results

    kill_all()

    # ─── Summary ───
    print(f"\n\n{'='*75}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*75}")

    header = f"{'Metric':<30}"
    for m in METHODS:
        header += f" {m['name']:>20}"
    print(header)
    print("─" * (30 + 21 * n_methods))

    metrics = []
    for m in METHODS:
        mname = m["name"]
        res = all_results[mname]
        arrived = sum(1 for r in res if r["arrived"])
        pk_sw = [r["peak_swing_deg"] for r in res]
        mn_sw = [r["mean_swing_deg"] for r in res]
        ft = [r["flight_time_s"] for r in res if r["arrived"]]
        md = [r["min_dist_m"] for r in res]
        mz = [r["min_z"] for r in res]

        metrics.append({
            "name": mname,
            "success_rate": arrived / len(res),
            "arrived": arrived,
            "total": len(res),
            "mean_peak_swing": float(np.mean(pk_sw)),
            "std_peak_swing": float(np.std(pk_sw)),
            "mean_avg_swing": float(np.mean(mn_sw)),
            "std_avg_swing": float(np.std(mn_sw)),
            "mean_flight_time": float(np.mean(ft)) if ft else float('nan'),
            "mean_min_dist": float(np.mean(md)),
            "mean_min_z": float(np.mean(mz)),
        })

    def row(label, key, fmt=".1f", suffix=""):
        line = f"{label:<30}"
        for met in metrics:
            v = met[key]
            if isinstance(v, float) and math.isnan(v):
                line += f" {'N/A':>20}"
            else:
                line += f" {f'{v:{fmt}}{suffix}':>20}"
        print(line)

    row("Success Rate", "success_rate", ".0%", "")
    row("  (arrived/total)", "arrived", ".0f", f"/{len(GOALS)}")
    print("─" * (30 + 21 * n_methods))
    row("Mean Peak Swing (°)", "mean_peak_swing", ".1f", "°")
    row("  ± Std", "std_peak_swing", ".1f", "°")
    row("Mean Avg Swing (°)", "mean_avg_swing", ".1f", "°")
    row("  ± Std", "std_avg_swing", ".1f", "°")
    print("─" * (30 + 21 * n_methods))
    row("Mean Flight Time (s)", "mean_flight_time", ".1f", "s")
    row("Mean Min Distance (m)", "mean_min_dist", ".1f", "m")
    row("Mean Min Altitude (m)", "mean_min_z", ".2f", "m")
    print("=" * (30 + 21 * n_methods))

    # Per-goal comparison table
    print(f"\n{'─'*80}")
    print(f"Per-Goal Comparison (Peak Swing / Avg Swing / Arrived)")
    print(f"{'─'*80}")
    header2 = f"{'Goal':>20}"
    for m in METHODS:
        header2 += f"  {m['name'][:18]:>22}"
    print(header2)
    print(f"{'─'*80}")

    for gi, goal in enumerate(GOALS):
        gd = math.sqrt(goal[0]**2 + goal[1]**2)
        ga = math.degrees(math.atan2(goal[1], goal[0]))
        line = f"{f'{gd:.0f}m / {ga:.0f}°':>20}"
        for m in METHODS:
            r = all_results[m["name"]][gi]
            s = "Y" if r["arrived"] else "N"
            line += f"  {r['peak_swing_deg']:>5.1f}° / {r['mean_swing_deg']:>4.1f}° / {s:>1}"
        print(line)
    print(f"{'─'*80}")

    # Save
    summary = {
        "methods": [m["name"] for m in METHODS],
        "n_goals": len(GOALS),
        "goals": GOALS,
        "metrics": metrics,
        "per_method_results": {k: v for k, v in all_results.items()},
    }

    out_path = f"{YOPO_DIR}/saved/YOPO_43/comparison_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
