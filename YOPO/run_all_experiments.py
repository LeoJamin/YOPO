"""
Master experiment runner for YOPO-Payload.

Orchestrates all pending experiments from EXPERIMENT_PLAN.md:
  M0: Sanity check (2 epochs)
  M1: Baseline training B0-B3
  M2: Ablation studies A1-A3
  M3: Critical ablations D1 (detach), M1 (no mass)
  M4: Closed-loop evaluation with pendulum ODE

Usage:
    # Full suite (sequential)
    python run_all_experiments.py

    # Sanity only
    python run_all_experiments.py --milestone 0

    # Specific milestone
    python run_all_experiments.py --milestone 1

    # Skip sanity, start from baselines
    python run_all_experiments.py --skip_sanity

    # Only closed-loop evaluation (after training is done)
    python run_all_experiments.py --milestone 4
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "saved")
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "..", "..", "refine-logs")

# Use the yopo conda environment Python
PYTHON = "/home/jamine/miniconda3/envs/yopo/bin/python"


def run_command(cmd, desc, timeout=None):
    """Run a subprocess command, print output in real time."""
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"  CMD: {cmd}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    start = time.time()
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=SCRIPT_DIR,
            timeout=timeout,
        )
        elapsed = time.time() - start
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"\n  [{status}] {desc} — {elapsed/60:.1f} min")
        return proc.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"\n  [TIMEOUT after {elapsed/60:.1f} min] {desc}")
        return -1


def milestone_0_sanity(args):
    """Sanity check: train A4_full_model for 2 epochs."""
    print("\n" + "="*70)
    print("  MILESTONE 0: SANITY CHECK")
    print("="*70)

    rc = run_command(
        f"{PYTHON} train_ablation.py --epochs 2 --batch_size {args.batch_size} "
        f"--save_dir {SAVE_DIR} --ablations A4_full_model --seed 0",
        "Sanity: A4_full_model (2 epochs)",
        timeout=600,  # 10 min max
    )
    if rc != 0:
        print("SANITY FAILED! Fix before proceeding.")
        return False
    print("Sanity passed.")
    return True


def milestone_1_baselines(args):
    """Train B0-B3 baselines."""
    print("\n" + "="*70)
    print("  MILESTONE 1: BASELINE TRAINING (B0-B3)")
    print("="*70)

    # B0 = A0_baseline (9D, no dynamics)
    # B1 and B2 are different training configs of 9D models
    # B3 = A4_full_model (15D, full dynamics)

    runs = [
        ("A0_baseline", "B0: Original YOPO (9D, no payload)"),
        ("B1_inflated", "B1: YOPO + Inflated Margin (r += 1.5m)"),
        ("B2_acc_penalty", "B2: YOPO + Acc Penalty (wa=5.0)"),
        ("A4_full_model", "B3: YOPO-Payload (15D, full model)"),
    ]

    for ablation_name, desc in runs:
        rc = run_command(
            f"{PYTHON} train_ablation.py --epochs {args.epochs} --batch_size {args.batch_size} "
            f"--learning_rate {args.lr} --save_dir {SAVE_DIR} "
            f"--ablations {ablation_name} --seed {args.seed}",
            desc,
            timeout=args.timeout_per_run,
        )
        if rc != 0:
            print(f"WARNING: {desc} failed, continuing...")

    return True


def milestone_2_ablations(args):
    """Ablation studies A1-A3."""
    print("\n" + "="*70)
    print("  MILESTONE 2: ABLATION STUDIES (A1-A3)")
    print("="*70)

    ablations = ["A1_cable_L", "A2_swing_ang", "A3_swing_full"]

    for name in ablations:
        rc = run_command(
            f"{PYTHON} train_ablation.py --epochs {args.epochs} --batch_size {args.batch_size} "
            f"--learning_rate {args.lr} --save_dir {SAVE_DIR} "
            f"--ablations {name} --seed {args.seed}",
            f"Ablation: {name}",
            timeout=args.timeout_per_run,
        )
        if rc != 0:
            print(f"WARNING: {name} failed, continuing...")

    return True


def milestone_3_critical(args):
    """Critical ablations: detach_qvec and mass."""
    print("\n" + "="*70)
    print("  MILESTONE 3: CRITICAL ABLATIONS (detach, mass)")
    print("="*70)

    critical = ["D1_no_detach", "A4_no_mass"]

    for name in critical:
        rc = run_command(
            f"{PYTHON} train_ablation.py --epochs {args.epochs} --batch_size {args.batch_size} "
            f"--learning_rate {args.lr} --save_dir {SAVE_DIR} "
            f"--ablations {name} --seed {args.seed}",
            f"Critical ablation: {name}",
            timeout=args.timeout_per_run,
        )
        if rc != 0:
            print(f"WARNING: {name} failed, continuing...")

    return True


def milestone_4_eval(args):
    """Closed-loop evaluation of all trained models."""
    print("\n" + "="*70)
    print("  MILESTONE 4: CLOSED-LOOP EVALUATION")
    print("="*70)

    methods = [
        "B0_YOPO_original", "B1_YOPO_inflated", "B2_YOPO_acc_penalty",
        "B3_YOPO_Payload",
        "A1_cable_L", "A2_swing_ang", "A3_swing_full",
        "A4_no_mass", "D1_no_detach",
        # B_MPC is the legacy 7x7 grid search; B_MPC_RH is the
        # receding-horizon swing-aware MPC that should be the principal
        # MPC baseline in the paper.
        "B_MPC", "B_MPC_RH",
    ]

    methods_str = " ".join(methods)
    rc = run_command(
        f"{PYTHON} evaluate_closed_loop.py "
        f"--checkpoint_dir {SAVE_DIR} --epoch {args.epochs} "
        f"--n_samples {args.eval_samples} --seed {args.seed} "
        f"--methods {methods_str} "
        f"--output {SAVE_DIR}/closed_loop_results.csv "
        f"--batch_size {args.batch_size}",
        "Closed-loop evaluation (all methods + MPC)",
        timeout=3600,
    )

    # Also run baseline comparison (validation loss metrics)
    run_command(
        f"{PYTHON} evaluate_baselines.py --checkpoint_dir {SAVE_DIR} "
        f"--epoch {args.epochs} --batch_size {args.batch_size} --n_batches 200",
        "Baseline validation loss comparison",
        timeout=1800,
    )

    return rc == 0


def update_tracker(milestone, status):
    """Update EXPERIMENT_TRACKER.md with results."""
    tracker_path = os.path.join(RESULTS_DIR, "EXPERIMENT_TRACKER.md")
    if not os.path.exists(tracker_path):
        return

    with open(tracker_path, "r") as f:
        content = f.read()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = content.replace(
        f"**Last Updated**: 2026-03-24",
        f"**Last Updated**: {timestamp}"
    )

    with open(tracker_path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="YOPO-Payload Experiment Runner")
    parser.add_argument("--milestone", type=int, default=None,
                        help="Run specific milestone (0-4). Default: run all.")
    parser.add_argument("--skip_sanity", action="store_true",
                        help="Skip milestone 0 sanity check")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--timeout_per_run", type=int, default=18000,
                        help="Max seconds per training run (default: 5h)")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    milestones = {
        0: ("Sanity Check", milestone_0_sanity),
        1: ("Baseline Training", milestone_1_baselines),
        2: ("Ablation Studies", milestone_2_ablations),
        3: ("Critical Ablations", milestone_3_critical),
        4: ("Closed-Loop Eval", milestone_4_eval),
    }

    start_time = time.time()

    if args.milestone is not None:
        # Run single milestone
        name, func = milestones[args.milestone]
        print(f"\nRunning Milestone {args.milestone}: {name}")
        func(args)
    else:
        # Run all milestones
        for m_id, (name, func) in milestones.items():
            if m_id == 0 and args.skip_sanity:
                print("Skipping sanity check.")
                continue
            print(f"\n{'#'*70}")
            print(f"  Starting Milestone {m_id}: {name}")
            print(f"{'#'*70}")
            success = func(args)
            if m_id == 0 and not success:
                print("Aborting: sanity check failed.")
                sys.exit(1)

    total_time = (time.time() - start_time) / 3600
    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE — Total time: {total_time:.1f}h")
    print(f"{'='*70}")
    print(f"\nResults:")
    print(f"  Ablation CSV:      {SAVE_DIR}/ablation_results.csv")
    print(f"  Baseline CSV:      {SAVE_DIR}/baseline_comparison.csv")
    print(f"  Closed-loop CSV:   {SAVE_DIR}/closed_loop_results.csv")
    print(f"  Closed-loop JSON:  {SAVE_DIR}/closed_loop_results.json")
    print(f"\nNext step:")
    print(f"  → /auto-review-loop 'YOPO-Payload UAV slung-load transportation'")


if __name__ == "__main__":
    main()
