"""
DSE Loop Orchestrator for YOPO Slung-Load.

Explores the loss weight design space to find the best swing performance.
Runs overnight — trains + evaluates multiple configurations sequentially.

Design space:
  - wd (dynamics weight): [0, 16, 32, 48, 75]
  - wa (acceleration weight): [0.3, 1.0, 2.0, 4.0]
  - gradient_decay: [0, 1]
  - convergence: [0, 1]  (swing convergence incentive)
  - score_dyn_boost: [3, 5, 8]

Strategy:
  Phase 1: Sweep wd and wa (the two most impactful knobs) with 30-epoch quick trains
  Phase 2: Best wd/wa + ablations (gradient_decay, convergence)
  Phase 3: Full 50-epoch training of top-3 configs

Usage:
    python dse_loop.py                # full overnight run
    python dse_loop.py --phase 1      # phase 1 only
    python dse_loop.py --resume       # resume from DSE_STATE.json

Estimated time:
    Phase 1: ~8 configs x 2.5h = 20h (30-epoch train + 5-goal eval each)
    Phase 2: ~4 configs x 2.5h = 10h
    Phase 3: ~3 configs x 4h = 12h
    Total: ~42h (~2 overnight runs)
"""

import json
import os
import subprocess
import sys
import time
import csv

sys.stdout.reconfigure(line_buffering=True)

YOPO_DIR = "/home/jamine/yopo_ws/src/YOPO/YOPO"
PYTHON = "/home/jamine/miniconda3/envs/yopo/bin/python"
DSE_DIR = os.path.join(YOPO_DIR, "dse_results")
LOG_CSV = os.path.join(DSE_DIR, "dse_log.csv")
STATE_FILE = os.path.join(DSE_DIR, "DSE_STATE.json")
REPORT_FILE = os.path.join(DSE_DIR, "DSE_REPORT.md")


def run_config(run_id, wd, wa, wc=1.5, ws=10.0, score_dyn_boost=5.0,
               gradient_decay=1, convergence=1, epochs=30):
    """Run a single DSE configuration."""
    cmd = [
        PYTHON, "-u", os.path.join(YOPO_DIR, "dse_train_eval.py"),
        "--wd", str(wd),
        "--wa", str(wa),
        "--wc", str(wc),
        "--ws", str(ws),
        "--score_dyn_boost", str(score_dyn_boost),
        "--gradient_decay", str(gradient_decay),
        "--convergence", str(convergence),
        "--epochs", str(epochs),
        "--run_id", run_id,
    ]
    print(f"\n{'='*70}")
    print(f"[DSE-LOOP] Starting {run_id}: wd={wd} wa={wa} decay={gradient_decay} conv={convergence}")
    print(f"{'='*70}")

    proc = subprocess.run(cmd, timeout=7200 * 4)
    return proc.returncode == 0


def read_results():
    """Read all results from dse_log.csv."""
    if not os.path.exists(LOG_CSV):
        return []
    with open(LOG_CSV, 'r') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def get_best(results, n=3):
    """Return top-n results by objective (lower is better)."""
    sorted_results = sorted(results, key=lambda r: float(r["objective"]))
    return sorted_results[:n]


def already_done(run_id):
    """Check if a run_id already exists in the log."""
    results = read_results()
    return any(r["run_id"] == run_id for r in results)


def save_loop_state(phase, completed, total):
    """Save loop progress for crash recovery."""
    state = {
        "phase": phase,
        "completed": completed,
        "total": total,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(DSE_DIR, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def phase1_weight_sweep():
    """Phase 1: Sweep wd and wa — the two most impactful knobs.

    Based on diagnosis:
    - wd=16 (current) -> dynamics is 6.6% of loss -> too weak
    - wa=0.3 (current) -> accel is 2.1% of loss -> too weak
    - Need to increase both to make swing matter
    """
    configs = [
        # (run_id, wd, wa) — systematic sweep
        ("P1_wd00_wa03", 0.0, 0.3),    # No dynamics loss (baseline)
        ("P1_wd16_wa03", 16.0, 0.3),   # Current config
        ("P1_wd16_wa20", 16.0, 2.0),   # Current wd + stronger accel
        ("P1_wd48_wa03", 48.0, 0.3),   # Strong dynamics + current accel
        ("P1_wd48_wa20", 48.0, 2.0),   # Strong dynamics + strong accel
        ("P1_wd75_wa03", 75.0, 0.3),   # Very strong dynamics
        ("P1_wd75_wa20", 75.0, 2.0),   # Very strong dynamics + strong accel
        ("P1_wd32_wa10", 32.0, 1.0),   # Moderate both
    ]

    completed = 0
    for run_id, wd, wa in configs:
        if already_done(run_id):
            print(f"[DSE-LOOP] Skipping {run_id} (already done)")
            completed += 1
            continue
        save_loop_state(1, completed, len(configs))
        run_config(run_id, wd=wd, wa=wa, epochs=30)
        completed += 1

    return completed


def phase2_ablations(best_wd, best_wa):
    """Phase 2: Ablate gradient decay and convergence incentive at best wd/wa."""
    configs = [
        (f"P2_best_nodecay", best_wd, best_wa, 0, 1),    # No gradient decay
        (f"P2_best_noconv", best_wd, best_wa, 1, 0),      # No convergence
        (f"P2_best_neither", best_wd, best_wa, 0, 0),     # Neither
        (f"P2_best_both", best_wd, best_wa, 1, 1),        # Both (should be same as P1 best)
    ]

    completed = 0
    for run_id, wd, wa, decay, conv in configs:
        if already_done(run_id):
            print(f"[DSE-LOOP] Skipping {run_id} (already done)")
            completed += 1
            continue
        save_loop_state(2, completed, len(configs))
        run_config(run_id, wd=wd, wa=wa, gradient_decay=decay, convergence=conv, epochs=30)
        completed += 1

    return completed


def phase3_full_training(top_configs):
    """Phase 3: Full 50-epoch training of top configs."""
    completed = 0
    for i, cfg in enumerate(top_configs):
        run_id = f"P3_top{i+1}_50ep"
        if already_done(run_id):
            print(f"[DSE-LOOP] Skipping {run_id} (already done)")
            completed += 1
            continue
        save_loop_state(3, completed, len(top_configs))
        run_config(
            run_id,
            wd=float(cfg["wd"]),
            wa=float(cfg["wa"]),
            gradient_decay=int(cfg["gradient_decay"]),
            convergence=int(cfg["convergence"]),
            epochs=50,
        )
        completed += 1

    return completed


def generate_report():
    """Generate DSE_REPORT.md with results summary."""
    results = read_results()
    if not results:
        return

    best = get_best(results, n=5)

    report = f"""# DSE Report: YOPO Slung-Load Weight Tuning

**Date**: {time.strftime("%Y-%m-%d %H:%M")}
**Total runs**: {len(results)}

## Best Configurations (top 5)

| Rank | Run ID | wd | wa | decay | conv | Peak Swing | Mean Swing | Success | Objective |
|------|--------|----|----|-------|------|-----------|-----------|---------|-----------|
"""
    for i, r in enumerate(best):
        report += (f"| {i+1} | {r['run_id']} | {r['wd']} | {r['wa']} | "
                   f"{r['gradient_decay']} | {r['convergence']} | "
                   f"{r['avg_peak_swing']} | {r['avg_mean_swing']} | "
                   f"{r['success_rate']} | {r['objective']} |\n")

    report += f"\n## All Results\n\n"
    report += "| Run ID | wd | wa | decay | conv | Peak | Mean | Success | Obj | Time (min) |\n"
    report += "|--------|----|----|-------|------|------|------|---------|-----|------------|\n"
    for r in sorted(results, key=lambda x: float(x["objective"])):
        report += (f"| {r['run_id']} | {r['wd']} | {r['wa']} | "
                   f"{r['gradient_decay']} | {r['convergence']} | "
                   f"{r['avg_peak_swing']} | {r['avg_mean_swing']} | "
                   f"{r['success_rate']} | {r['objective']} | {r['elapsed_min']} |\n")

    report += f"\n## Conclusions\n\n"
    if best:
        b = best[0]
        report += f"- Best config: wd={b['wd']}, wa={b['wa']}, decay={b['gradient_decay']}, conv={b['convergence']}\n"
        report += f"- Best objective: {b['objective']} (peak={b['avg_peak_swing']}, mean={b['avg_mean_swing']})\n"

    with open(REPORT_FILE, 'w') as f:
        f.write(report)
    print(f"\n[DSE-LOOP] Report saved to {REPORT_FILE}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0, help="Run specific phase (0=all)")
    parser.add_argument("--resume", action="store_true", help="Resume from state")
    args = parser.parse_args()

    os.makedirs(DSE_DIR, exist_ok=True)
    start = time.time()

    if args.phase == 0 or args.phase == 1:
        print("\n[DSE-LOOP] ===== PHASE 1: Weight Sweep =====")
        phase1_weight_sweep()
        generate_report()

    if args.phase == 0 or args.phase == 2:
        print("\n[DSE-LOOP] ===== PHASE 2: Ablations =====")
        results = read_results()
        best = get_best(results, n=1)
        if best:
            best_wd = float(best[0]["wd"])
            best_wa = float(best[0]["wa"])
            print(f"[DSE-LOOP] Best from Phase 1: wd={best_wd} wa={best_wa}")
            phase2_ablations(best_wd, best_wa)
        generate_report()

    if args.phase == 0 or args.phase == 3:
        print("\n[DSE-LOOP] ===== PHASE 3: Full Training =====")
        results = read_results()
        top3 = get_best(results, n=3)
        if top3:
            phase3_full_training(top3)
        generate_report()

    elapsed = (time.time() - start) / 3600
    print(f"\n[DSE-LOOP] Total elapsed: {elapsed:.1f} hours")
    print("[DSE-LOOP] DONE. See dse_results/DSE_REPORT.md")


if __name__ == "__main__":
    main()
