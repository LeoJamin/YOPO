"""
4-way ROS comparison: YOPO_1 (9D) vs YOPO_43 (13D, wd=8) vs YOPO_44 (13D, wd=16) vs YOPO_45 (13D, wd=25).
Same 10 goals, same map (seed=3).
"""

import subprocess, signal, time, os, sys, json, math, re
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

TIMEOUT_SEC = 80
ARRIVE_DIST = 3.0
SETTLE_SEC = 5
STARTUP_WAIT = 8
PLANNER_WAIT = 12

WS = "/home/jamine/yopo_ws"
YOPO_DIR = f"{WS}/src/YOPO/YOPO"
PYTHON = "/home/jamine/miniconda3/envs/yopo/bin/python"
SETUP = f"source {WS}/devel/setup.bash"

METHODS = [
    {"name": "YOPO-Original(9D)",   "trial": 1,  "obs_dim": 9,  "epoch": 50},
    {"name": "YOPO-43(13D,wd=8)",   "trial": 43, "obs_dim": 13, "epoch": 50},
    {"name": "YOPO-44(13D,wd=16)",  "trial": 44, "obs_dim": 13, "epoch": 10},
    {"name": "YOPO-45(13D,wd=25)",  "trial": 45, "obs_dim": 13, "epoch": 20},
]

GOALS = [
    [20.0, 0.0, 2.0], [17.7, 17.7, 2.0], [0.0, 30.0, 2.0],
    [30.3, -17.5, 2.0], [-40.0, 0.0, 2.0], [0.0, -45.0, 2.0],
    [48.3, 12.9, 2.0], [-38.9, 38.9, 2.0], [30.0, -52.0, 2.0],
    [-26.0, -15.0, 2.0],
]


def kill_all():
    for p in ["quadrotor_simulator_so3","sensor_simulator_cuda","rosmaster","roscore","rosout","roslaunch","nodelet"]:
        subprocess.run(f"killall -9 {p}", shell=True, capture_output=True, timeout=5)
    subprocess.run("pkill -9 -f test_yopo_ros.py", shell=True, capture_output=True, timeout=5)
    time.sleep(2)

def start_ros():
    subprocess.Popen(f"bash -c '{SETUP} && roscore'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    time.sleep(3)
    subprocess.Popen(f"bash -c '{SETUP} && roslaunch so3_quadrotor_simulator simulator.launch'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    time.sleep(3)
    subprocess.Popen(f"bash -c '{SETUP} && rosrun sensor_simulator sensor_simulator_cuda'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    time.sleep(STARTUP_WAIT)

def publish_goal(g):
    cmd = f"bash -c \"{SETUP} && rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped '{{header: {{frame_id: world}}, pose: {{position: {{x: {g[0]}, y: {g[1]}, z: {g[2]}}}, orientation: {{w: 1.0}}}}}}'\""
    subprocess.run(cmd, shell=True, capture_output=True, timeout=10)

def run_ep(trial, obs_dim, epoch, goal, ep_id):
    log_file = f"/tmp/yopo_3way_t{trial}_ep{ep_id}.log"
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        f"bash -c '{SETUP} && cd {YOPO_DIR} && {PYTHON} -u test_yopo_ros.py --trial {trial} --epoch {epoch} --obs_dim {obs_dim}'",
        shell=True, stdout=open(log_file,'w'), stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env)
    time.sleep(PLANNER_WAIT)
    publish_goal(goal); time.sleep(0.5)

    start_time = time.time(); arrived = False; arrive_time = None; first_nav = None
    while time.time() - start_time < TIMEOUT_SEC:
        time.sleep(1.0)
        try:
            with open(log_file) as f: lines = f.readlines()
        except: continue
        nav = [l.strip() for l in lines if l.startswith("[NAV]")]
        if nav and first_nav is None: first_nav = time.time()
        for l in lines:
            if "ARRIVE!" in l:
                arrived = True
                if arrive_time is None: arrive_time = time.time()
                break
        if not arrived and nav:
            m = re.search(r'dist=([\d.]+)m', nav[-1])
            if m and float(m.group(1)) < ARRIVE_DIST: arrived = True; arrive_time = time.time()
        if arrived and (time.time() - arrive_time > SETTLE_SEC): break

    try:
        with open(log_file) as f: nav = [l.strip() for l in f.readlines() if l.startswith("[NAV]")]
    except: nav = []
    try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except: pass
    try: proc.wait(timeout=5)
    except: pass

    fd = parse_nav(nav)
    ft = (arrive_time - first_nav) if (first_nav and arrive_time) else (time.time() - first_nav if first_nav else TIMEOUT_SEC)
    return {"arrived": arrived, "flight_time_s": round(max(ft,0),1), **fd, "n_samples": len(nav)}

def parse_nav(nav):
    dists, swings, speeds, zs = [], [], [], []
    pat = re.compile(r'\[NAV\] dist=([\d.]+)m speed=([\d.]+)m/s pos=\(([-\d.]+),([-\d.]+),([-\d.]+)\)(?: \| swing=([\d.]+)°)?')
    for l in nav:
        m = pat.search(l)
        if m:
            dists.append(float(m.group(1))); speeds.append(float(m.group(2))); zs.append(float(m.group(5)))
            if m.group(6): swings.append(float(m.group(6)))
    if not dists: return {"final_dist":999,"min_dist":999,"peak_swing":0,"mean_swing":0,"cruise_swing":0,"min_z":0}
    cruise_sw = [s for s,sp in zip(swings,speeds[:len(swings)]) if sp > 2.5] if swings else []
    return {
        "final_dist": round(dists[-1],2), "min_dist": round(min(dists),2),
        "peak_swing": round(max(swings),1) if swings else 0,
        "mean_swing": round(float(np.mean(swings)),1) if swings else 0,
        "cruise_swing": round(float(np.mean(cruise_sw)),1) if cruise_sw else 0,
        "min_z": round(min(zs),2),
    }

def main():
    print(f"{'='*80}")
    print(f"3-Way YOPO Comparison — ROS Closed-Loop")
    print(f"{'='*80}")
    for m in METHODS: print(f"  - {m['name']} (trial={m['trial']}, epoch={m['epoch']})")
    print(f"Goals: {len(GOALS)} | Timeout: {TIMEOUT_SEC}s")
    print(f"{'='*80}")

    all_results = {}
    for method in METHODS:
        mn = method["name"]
        print(f"\n{'#'*80}\n# {mn}\n{'#'*80}")
        results = []
        for gi, goal in enumerate(GOALS):
            gd = math.sqrt(goal[0]**2+goal[1]**2); ga = math.degrees(math.atan2(goal[1],goal[0]))
            print(f"  [{mn}] Goal {gi+1}/10: ({goal[0]:.0f},{goal[1]:.0f}) {gd:.0f}m {ga:.0f}°")
            kill_all(); start_ros()
            r = run_ep(method["trial"], method["obs_dim"], method["epoch"], goal, gi+1)
            r["goal"] = goal; r["goal_dist"] = round(gd,1)
            results.append(r)
            s = "OK" if r["arrived"] else "FAIL"
            print(f"    {s} | min_d={r['min_dist']:.1f}m pk_sw={r['peak_swing']:.1f}° "
                  f"avg_sw={r['mean_swing']:.1f}° cruise_sw={r['cruise_swing']:.1f}° min_z={r['min_z']:.1f}m")
        all_results[mn] = results
    kill_all()

    # Summary
    print(f"\n\n{'='*85}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*85}")
    h = f"{'Metric':<25}";
    for m in METHODS: h += f" {m['name']:>18}"
    print(h); print("─"*85)

    metrics = []
    for m in METHODS:
        res = all_results[m["name"]]
        arr = sum(1 for r in res if r["arrived"])
        pk = [r["peak_swing"] for r in res]
        mn_sw = [r["mean_swing"] for r in res]
        cr = [r["cruise_swing"] for r in res if r["cruise_swing"]>0]
        metrics.append({"name":m["name"],"success":arr/len(res),"arrived":arr,
            "peak":np.mean(pk),"peak_std":np.std(pk),
            "avg":np.mean(mn_sw),"avg_std":np.std(mn_sw),
            "cruise":np.mean(cr) if cr else 0,"cruise_std":np.std(cr) if cr else 0,
            "min_z":np.mean([r["min_z"] for r in res])})

    def row(label, key, fmt=".1f", suf=""):
        ln = f"{label:<25}"
        for mt in metrics:
            v = mt[key]
            ln += f" {f'{v:{fmt}}{suf}':>18}"
        print(ln)

    row("Success Rate","success",".0%")
    print("─"*85)
    row("Peak Swing","peak",".1f","°")
    row("  ± Std","peak_std",".1f","°")
    row("Avg Swing","avg",".1f","°")
    row("  ± Std","avg_std",".1f","°")
    row("Cruise Swing (>2.5m/s)","cruise",".1f","°")
    row("  ± Std","cruise_std",".1f","°")
    print("─"*85)
    row("Mean Min Altitude","min_z",".2f","m")
    print("="*85)

    # Save
    out = f"{YOPO_DIR}/saved/YOPO_44/comparison_3way.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,'w') as f: json.dump({"methods":[m["name"] for m in METHODS],"metrics":metrics,"results":all_results},f,indent=2,default=str)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
