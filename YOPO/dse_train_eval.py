"""
DSE Train-Eval Runner for YOPO Slung-Load.

Trains a model with given hyperparameters, then evaluates swing performance
in closed-loop ROS simulation. Returns a single objective scalar.

Usage:
    python dse_train_eval.py --wd 32 --wa 1.5 --wc 1.5 --ws 10 \
        --score_dyn_boost 5 --gradient_decay 1 --convergence 1 \
        --epochs 30 --run_id DSE_001

Output: writes results to dse_results/dse_log.csv
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import math
import re

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

WS = "/home/jamine/yopo_ws"
YOPO_DIR = f"{WS}/src/YOPO/YOPO"
PYTHON = "/home/jamine/miniconda3/envs/yopo/bin/python"
SETUP = f"source {WS}/devel/setup.bash"
DSE_DIR = f"{YOPO_DIR}/dse_results"

EVAL_GOALS = [
    # 5 representative goals (1 per category, faster than 15)
    [8.0, 12.0, 2.0],      # Sharp Turn
    [70.0, 0.0, 2.0],      # Long Straight
    [45.0, 30.0, 2.0],     # Diagonal Cut
    [-15.0, 3.0, 2.0],     # Near-Reversal
    [80.0, -25.0, 2.0],    # Very Long
]
EVAL_TIMEOUT = 90
ARRIVE_DIST = 3.0
SETTLE_SEC = 5
STARTUP_WAIT = 8
PLANNER_WAIT = 12


def patch_config(args):
    """Patch traj_opt.yaml with DSE parameters."""
    import ruamel.yaml
    yaml = ruamel.yaml.YAML()
    cfg_path = os.path.join(YOPO_DIR, "config", "traj_opt.yaml")
    with open(cfg_path, 'r') as f:
        data = yaml.load(f)

    data['wd'] = args.wd
    data['wa'] = args.wa
    data['wc'] = args.wc
    data['ws'] = args.ws
    data['score_dyn_boost'] = args.score_dyn_boost
    data['gradient_decay'] = bool(args.gradient_decay)

    with open(cfg_path, 'w') as f:
        yaml.dump(data, f)
    print(f"[DSE] Config patched: wd={args.wd} wa={args.wa} wc={args.wc} ws={args.ws} "
          f"score_dyn_boost={args.score_dyn_boost} gradient_decay={args.gradient_decay}")


def train(args):
    """Train the model and return the trial number."""
    cmd = f"cd {YOPO_DIR} && {PYTHON} -u train_yopo.py --pretrained 0"
    print(f"[DSE] Training {args.epochs} epochs...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Temporarily patch epochs in train_yopo.py via env var
    # We modify the train call directly
    train_script = f"""
import os, sys, random, numpy as np, torch
sys.path.insert(0, '{YOPO_DIR}')
os.chdir('{YOPO_DIR}')
random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.cuda.manual_seed(0); torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
from policy.yopo_trainer import YopoTrainer
log_dir = '{YOPO_DIR}/saved'
os.makedirs(log_dir, exist_ok=True)
trainer = YopoTrainer(
    learning_rate=1.5e-4, batch_size=16,
    loss_weight=[1.0, 1.0, 1.0],
    tensorboard_path=log_dir, save_on_exit=False,
)
trainer.train(epoch={args.epochs}, save_interval=10)
print("DSE_TRAIN_DONE")
print("DSE_TRIAL_PATH=" + trainer.tensorboard_path)
"""
    proc = subprocess.run(
        [PYTHON, "-u", "-c", train_script],
        capture_output=True, text=True, env=env, timeout=7200 * 5,  # 10h for 50-epoch runs
    )
    print(proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout)
    if proc.stderr:
        print("[DSE] STDERR:", proc.stderr[-1000:])

    # Extract trial path
    for line in proc.stdout.split('\n'):
        if line.startswith("DSE_TRIAL_PATH="):
            trial_path = line.split("=", 1)[1].strip()
            trial_num = int(os.path.basename(trial_path).split("_")[1])
            return trial_num, trial_path

    raise RuntimeError("Training failed — no DSE_TRIAL_PATH found")


def kill_ros():
    for p in ["quadrotor_simulator_so3", "sensor_simulator_cuda",
              "rosmaster", "roscore", "rosout", "roslaunch", "nodelet"]:
        subprocess.run(f"killall -9 {p}", shell=True, capture_output=True, timeout=5)
    subprocess.run("pkill -9 -f test_yopo_ros.py", shell=True, capture_output=True, timeout=5)
    time.sleep(2)


def start_ros():
    subprocess.Popen(f"bash -c '{SETUP} && roscore'", shell=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    time.sleep(3)
    subprocess.Popen(f"bash -c '{SETUP} && roslaunch so3_quadrotor_simulator simulator.launch'",
                     shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    time.sleep(3)
    subprocess.Popen(f"bash -c '{SETUP} && rosrun sensor_simulator sensor_simulator_cuda'",
                     shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    time.sleep(STARTUP_WAIT)


def publish_goal(g):
    cmd = (f"bash -c \"{SETUP} && rostopic pub -1 /move_base_simple/goal "
           f"geometry_msgs/PoseStamped '{{header: {{frame_id: world}}, "
           f"pose: {{position: {{x: {g[0]}, y: {g[1]}, z: {g[2]}}}, "
           f"orientation: {{w: 1.0}}}}}}'\""
           )
    subprocess.run(cmd, shell=True, capture_output=True, timeout=10)


def run_single_eval(trial, obs_dim, epoch, goal, ep_id):
    log_file = f"/tmp/yopo_dse_t{trial}_ep{ep_id}.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        f"bash -c '{SETUP} && cd {YOPO_DIR} && {PYTHON} -u test_yopo_ros.py "
        f"--trial {trial} --epoch {epoch} --obs_dim {obs_dim}'",
        shell=True, stdout=open(log_file, 'w'), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, env=env)
    time.sleep(PLANNER_WAIT)
    publish_goal(goal)
    time.sleep(0.5)

    start_time = time.time()
    arrived = False
    arrive_time = None
    while time.time() - start_time < EVAL_TIMEOUT:
        time.sleep(1.0)
        try:
            with open(log_file) as f:
                lines = f.readlines()
        except Exception:
            continue
        nav = [l.strip() for l in lines if l.startswith("[NAV]")]
        for l in lines:
            if "ARRIVE!" in l:
                arrived = True
                if arrive_time is None:
                    arrive_time = time.time()
                break
        if not arrived and nav:
            m = re.search(r'dist=([\d.]+)m', nav[-1])
            if m and float(m.group(1)) < ARRIVE_DIST:
                arrived = True
                arrive_time = time.time()
        if arrived and (time.time() - arrive_time > SETTLE_SEC):
            break

    try:
        with open(log_file) as f:
            nav = [l.strip() for l in f.readlines() if l.startswith("[NAV]")]
    except Exception:
        nav = []
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass

    # Parse swing
    swings = []
    pat = re.compile(r'swing=([\d.]+)')
    for l in nav:
        m = pat.search(l)
        if m:
            swings.append(float(m.group(1)))

    peak_swing = max(swings) if swings else 99.0
    mean_swing = float(np.mean(swings)) if swings else 99.0
    return arrived, peak_swing, mean_swing


def evaluate(trial, epoch, obs_dim=13):
    """Run 5 eval goals and return aggregate metrics."""
    results = []
    for gi, goal in enumerate(EVAL_GOALS):
        kill_ros()
        start_ros()
        arrived, peak, mean = run_single_eval(trial, obs_dim, epoch, goal, gi + 1)
        results.append({"arrived": arrived, "peak": peak, "mean": mean})
        status = "OK" if arrived else "FAIL"
        print(f"[DSE] Goal {gi+1}/5: {status} pk={peak:.1f} avg={mean:.1f}")
    kill_ros()

    n_success = sum(1 for r in results if r["arrived"])
    avg_peak = np.mean([r["peak"] for r in results])
    avg_mean = np.mean([r["mean"] for r in results])
    success_rate = n_success / len(results)

    return {
        "success_rate": success_rate,
        "avg_peak_swing": round(avg_peak, 2),
        "avg_mean_swing": round(avg_mean, 2),
        "n_success": n_success,
        "n_goals": len(results),
    }


def compute_objective(eval_result):
    """Single scalar objective to MINIMIZE.

    Combines swing reduction with success penalty.
    Lower is better.
    """
    # Penalize failure heavily
    fail_penalty = (1.0 - eval_result["success_rate"]) * 100.0
    # Primary objective: reduce peak swing
    swing_cost = eval_result["avg_peak_swing"]
    # Secondary: reduce mean swing
    mean_cost = eval_result["avg_mean_swing"] * 0.5

    return round(swing_cost + mean_cost + fail_penalty, 2)


def log_result(args, trial, eval_result, objective, elapsed_min):
    """Append result to CSV log."""
    os.makedirs(DSE_DIR, exist_ok=True)
    csv_path = os.path.join(DSE_DIR, "dse_log.csv")
    header = ["run_id", "wd", "wa", "wc", "ws", "score_dyn_boost",
              "gradient_decay", "convergence", "epochs",
              "trial", "success_rate", "avg_peak_swing", "avg_mean_swing",
              "objective", "elapsed_min"]
    row = [args.run_id, args.wd, args.wa, args.wc, args.ws,
           args.score_dyn_boost, args.gradient_decay, args.convergence,
           args.epochs, trial,
           eval_result["success_rate"], eval_result["avg_peak_swing"],
           eval_result["avg_mean_swing"], objective, round(elapsed_min, 1)]

    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    print(f"[DSE] Logged to {csv_path}")


def save_state(args, trial, eval_result, objective, iteration):
    """Save DSE state for crash recovery."""
    state = {
        "iteration": iteration,
        "run_id": args.run_id,
        "trial": trial,
        "params": {"wd": args.wd, "wa": args.wa, "wc": args.wc,
                    "ws": args.ws, "score_dyn_boost": args.score_dyn_boost,
                    "gradient_decay": args.gradient_decay},
        "eval_result": eval_result,
        "objective": objective,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    state_path = os.path.join(DSE_DIR, "DSE_STATE.json")
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="DSE Train-Eval for YOPO")
    parser.add_argument("--wd", type=float, default=16.0, help="dynamics loss weight (YAML)")
    parser.add_argument("--wa", type=float, default=0.3, help="acceleration loss weight (YAML)")
    parser.add_argument("--wc", type=float, default=1.5, help="safety loss weight (YAML)")
    parser.add_argument("--ws", type=float, default=10.0, help="smoothness loss weight (YAML)")
    parser.add_argument("--score_dyn_boost", type=float, default=5.0, help="dynamics boost in score label")
    parser.add_argument("--gradient_decay", type=int, default=1, help="enable temporal gradient decay")
    parser.add_argument("--convergence", type=int, default=1, help="enable convergence incentive")
    parser.add_argument("--epochs", type=int, default=30, help="training epochs")
    parser.add_argument("--run_id", type=str, default="DSE_001", help="run identifier")
    parser.add_argument("--obs_dim", type=int, default=13, help="observation dimension")
    args = parser.parse_args()

    start_time = time.time()
    print(f"[DSE] === Run {args.run_id} ===")
    print(f"[DSE] Params: wd={args.wd} wa={args.wa} wc={args.wc} ws={args.ws} "
          f"boost={args.score_dyn_boost} decay={args.gradient_decay} conv={args.convergence}")

    # 1. Patch config
    patch_config(args)

    # 2. Train
    trial, trial_path = train(args)
    print(f"[DSE] Trained: YOPO_{trial} at {trial_path}")

    # 3. Evaluate
    eval_result = evaluate(trial, args.epochs, args.obs_dim)
    print(f"[DSE] Eval: success={eval_result['success_rate']:.0%} "
          f"peak={eval_result['avg_peak_swing']:.1f} mean={eval_result['avg_mean_swing']:.1f}")

    # 4. Compute objective
    objective = compute_objective(eval_result)
    elapsed_min = (time.time() - start_time) / 60
    print(f"[DSE] Objective: {objective:.2f} (lower is better)")
    print(f"[DSE] Elapsed: {elapsed_min:.1f} min")

    # 5. Log and save state
    log_result(args, trial, eval_result, objective, elapsed_min)
    save_state(args, trial, eval_result, objective, iteration=0)

    print(f"[DSE] === Run {args.run_id} COMPLETE ===")
    return objective


if __name__ == "__main__":
    main()
