"""
Diagnostic: spawn the eval's exact ROS stack via subprocess.Popen and probe
which links of the chain are alive vs broken. Compares against what an
interactive shell sees.

Run with:
    python debug_subprocess.py

Output:
    dse_results/eval_results/debug_subprocess.log
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

WS_SIM   = "/home/jamine/research/diff-slung/code/Simulator"
WS_CTRL  = "/home/jamine/research/diff-slung/code/Controller"
YOPO_DIR = "/home/jamine/research/diff-slung/code/YOPO"
PYTHON   = "/home/jamine/miniconda3/envs/yopo/bin/python"
SETUP    = f"source {WS_SIM}/devel/setup.bash && source {WS_CTRL}/devel/setup.bash --extend"

OUT = Path(YOPO_DIR) / "dse_results/eval_results/debug_subprocess.log"
OUT.parent.mkdir(parents=True, exist_ok=True)


def log(msg, file_handle):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    file_handle.write(line + "\n")
    file_handle.flush()


def kill_all():
    for proc in ["quadrotor_simulator_so3", "sensor_simulator_cuda",
                 "rosmaster", "roscore", "rosout", "roslaunch", "nodelet"]:
        subprocess.run(f"killall -9 {proc}", shell=True,
                       capture_output=True, timeout=5)
    subprocess.run("pkill -9 -f test_yopo_ros.py", shell=True,
                   capture_output=True, timeout=5)
    time.sleep(2)


def run_probe(cmd, timeout=6):
    """Run a probe command in the SAME subprocess env as the eval uses."""
    try:
        r = subprocess.run(
            f"bash -ic '{SETUP} && {cmd}'",
            shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return f"<TIMEOUT after {timeout}s>"
    except Exception as e:
        return f"<ERROR: {e}>"


def main():
    f = open(OUT, "w")
    log("=== Diagnostic: eval subprocess stack ===", f)
    log(f"Output: {OUT}", f)
    log("", f)

    # Step 0: clean slate
    log("[0] kill_all", f)
    kill_all()

    # Step 1: parent env (this Python — inherited from interactive shell)
    log("[1] Parent env (this Python process)", f)
    parent_env = dict(os.environ)
    for k in ["PATH", "ROS_PACKAGE_PATH", "ROS_MASTER_URI", "LD_LIBRARY_PATH",
              "PYTHONPATH", "CONDA_DEFAULT_ENV", "CONDA_PREFIX",
              "CUDA_TOOLKIT_ROOT_DIR"]:
        log(f"   {k}={parent_env.get(k, '<unset>')[:200]}", f)
    log("", f)

    # Step 2: subprocess env via bash -c (non-interactive)
    log("[2] Subprocess env via 'bash -c' (non-interactive)", f)
    out = subprocess.run(
        "bash -c 'echo PATH=$PATH; echo ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH; "
        "echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH; echo PYTHONPATH=$PYTHONPATH'",
        shell=True, capture_output=True, text=True, timeout=5).stdout
    for line in out.splitlines():
        log(f"   {line[:200]}", f)
    log("", f)

    # Step 3: subprocess env via bash -ic (interactive — what eval uses now)
    log("[3] Subprocess env via 'bash -ic' (interactive)", f)
    out = subprocess.run(
        "bash -ic 'echo PATH=$PATH; echo ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH; "
        "echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH; echo PYTHONPATH=$PYTHONPATH'",
        shell=True, capture_output=True, text=True, timeout=5).stdout
    for line in out.splitlines():
        log(f"   {line[:200]}", f)
    log("", f)

    # Step 4: subprocess env after sourcing the workspace setup.bash files
    log("[4] After 'bash -ic && source ws_setups'", f)
    out = subprocess.run(
        f"bash -ic '{SETUP} && echo PATH=$PATH; echo ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH; "
        f"echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH; echo PYTHONPATH=$PYTHONPATH'",
        shell=True, capture_output=True, text=True, timeout=5).stdout
    for line in out.splitlines():
        log(f"   {line[:300]}", f)
    log("", f)

    # Step 5: Start the eval's ROS stack (mirrors eval_ros_success_rate.start_ros_stack)
    log("[5] Starting ROS stack", f)
    subprocess.Popen(
        f"bash -ic '{SETUP} && roscore'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    time.sleep(5)
    subprocess.Popen(
        f"bash -ic '{SETUP} && roslaunch so3_quadrotor_simulator simulator_attitude_control.launch'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    time.sleep(8)
    subprocess.Popen(
        f"bash -ic '{SETUP} && rosrun sensor_simulator sensor_simulator_cuda'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    time.sleep(15)
    log("    Stack should be up", f)
    log("", f)

    # Step 6: probes WITHOUT planner
    log("[6] Topics live (without planner)", f)
    log("--- rostopic list ---", f)
    log(run_probe("rostopic list", 5), f)
    log("--- rostopic hz /sim/odom (3s) ---", f)
    log(run_probe("timeout 3 rostopic hz /sim/odom", 5), f)
    log("--- rostopic hz /depth_image (3s) ---", f)
    log(run_probe("timeout 3 rostopic hz /depth_image", 5), f)
    log("--- /sim/odom position right now ---", f)
    log(run_probe("timeout 2 rostopic echo -n 1 /sim/odom | grep -A4 position:", 5), f)
    log("", f)

    # Step 7: launch planner
    log("[7] Launching planner — default goal=(50,0,2)", f)
    log_planner = "/tmp/yopo_debug_planner.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    planner_proc = subprocess.Popen(
        f"bash -ic '{SETUP} && cd {YOPO_DIR} && {PYTHON} -u test_yopo_ros.py "
        f"--trial 3 --epoch 50 --obs_dim 13'",
        shell=True, stdout=open(log_planner, 'w'), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, env=env)
    time.sleep(20)
    log("    Planner should be up", f)
    log("", f)

    # Step 8: probes WITH planner
    log("[8] Topics live (with planner)", f)
    log("--- rosnode list ---", f)
    log(run_probe("rosnode list", 5), f)
    log("--- rostopic hz /so3_control/pos_cmd (3s) ---", f)
    log(run_probe("timeout 3 rostopic hz /so3_control/pos_cmd", 5), f)
    log("--- rostopic hz /so3_cmd (3s) ---", f)
    log(run_probe("timeout 3 rostopic hz /so3_cmd", 5), f)
    log("--- /sim/odom position over 10 s (5 samples) ---", f)
    for i in range(5):
        time.sleep(2)
        out = run_probe(
            "timeout 2 rostopic echo -n 1 /sim/odom 2>/dev/null | "
            "grep -A2 'position:' | tr '\\n' ' '", 5)
        log(f"  t+{i*2}s: {out.strip()[:200]}", f)
    log("", f)

    # Step 9: planner stdout tail
    log("[9] Planner stdout (last 30 lines)", f)
    try:
        with open(log_planner) as g:
            lines = g.readlines()
        for line in lines[-30:]:
            log("   " + line.rstrip(), f)
    except Exception as e:
        log(f"   <couldn't read: {e}>", f)
    log("", f)

    # cleanup
    log("[10] cleanup", f)
    try:
        os.killpg(os.getpgid(planner_proc.pid), signal.SIGKILL)
    except Exception:
        pass
    kill_all()
    log("done.", f)
    f.close()


if __name__ == "__main__":
    main()
