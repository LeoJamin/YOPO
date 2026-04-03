"""
YOPO Iterative Refinement Loop.

Staged strategy: get the base planner working first, then add payload physics.
Each stage trains, evaluates, analyzes, and adjusts parameters until success
criteria are met before advancing to the next stage.

Stages:
  Stage 1: Base planner (wd=0) -- navigation + collision avoidance
  Stage 2: Introduce dynamics loss (wd>0) -- swing-aware planning
  Stage 3: Convergence tuning -- post-turn oscillation damping

Usage:
    python refine_loop.py --stage 1
    python refine_loop.py --stage 2 --base_checkpoint saved/refine_S1/best.pth
    python refine_loop.py --stage 1 --max_iter 5 --epochs_per_iter 20
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ruamel.yaml import YAML


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES = {
    1: {
        "name": "Base Planner",
        "description": "Navigation + collision avoidance without payload dynamics",
        "param_space": {
            "ws": [5.0, 10.0, 15.0],       # smoothness weight
            "wc": [1.0, 1.5, 2.5],          # safety weight
            "wg": [0.1, 0.15, 0.25],        # goal weight
            "wa": [0.1, 0.3, 0.5],          # acceleration weight
        },
        "fixed": {
            "wd": 0.0,                       # no dynamics loss
        },
        "success_criteria": {
            "val_traj_loss_max": 5.0,        # trajectory loss converged
            "val_score_loss_max": 2.0,        # score prediction converged
            "goal_loss_decreasing": True,     # goal loss trending down
        },
        "epochs_per_iter": 30,
        "max_iterations": 8,
        "patience": 3,
    },
    2: {
        "name": "Dynamics Introduction",
        "description": "Add swing-aware planning with gradual dynamics weight ramp",
        "param_space": {
            "wd": [2.0, 4.0, 8.0, 16.0],    # dynamics weight (ramped up)
            "score_dyn_boost": [2.0, 5.0, 8.0],
        },
        "fixed": {},  # inherit ws, wc, wg, wa from Stage 1 best
        "success_criteria": {
            "mean_peak_swing_deg_max": 25.0,  # peak swing under 25 deg
            "mean_rms_swing_deg_max": 12.0,   # RMS swing under 12 deg
            "success_rate_min": 0.7,          # at least 70% reach goal
        },
        "epochs_per_iter": 30,
        "max_iterations": 10,
        "patience": 3,
    },
    3: {
        "name": "Convergence Tuning",
        "description": "Tune oscillation damping after sharp turns",
        "param_space": {
            "wd": [12.0, 16.0, 24.0, 32.0],
            "score_dyn_boost": [5.0, 8.0, 10.0],
            # These are in differentiable_pendulum.py, tuned via config override
            "pendulum_n_steps": [15, 20, 30],
        },
        "fixed": {},
        "success_criteria": {
            "mean_peak_swing_deg_max": 15.0,  # tighter swing constraint
            "mean_rms_swing_deg_max": 8.0,
            "success_rate_min": 0.85,
        },
        "epochs_per_iter": 30,
        "max_iterations": 10,
        "patience": 4,
    },
}


# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------

def load_config(yaml_path: str) -> dict:
    yaml = YAML()
    with open(yaml_path, 'r') as f:
        return dict(yaml.load(f))


def save_config(yaml_path: str, config: dict):
    yaml = YAML()
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)


def apply_params(base_config: dict, params: dict) -> dict:
    """Apply parameter overrides to config. Returns new dict (immutable)."""
    config = dict(base_config)
    for key, val in params.items():
        if key in config:
            config[key] = val
        elif key == "pendulum_n_steps":
            # Special: not in yaml, passed via env var
            pass
        else:
            config[key] = val
    return config


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------

def run_training(config_path: str, output_dir: str, epochs: int,
                 checkpoint_path: str = "", pendulum_n_steps: int = 20) -> dict:
    """Run training and return final loss metrics from TensorBoard logs."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "train_yopo.py",
        "--trial", "0",  # placeholder, we override save path
        "--epoch", str(epochs),
    ]
    if checkpoint_path and os.path.exists(checkpoint_path):
        cmd.extend(["--pretrained", "1"])

    env = os.environ.copy()
    env["YOPO_SAVE_DIR"] = output_dir
    env["YOPO_PENDULUM_N_STEPS"] = str(pendulum_n_steps)

    # Run training as subprocess
    print(f"  Training {epochs} epochs -> {output_dir}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  Training FAILED ({elapsed:.0f}s)")
        print(f"  stderr: {result.stderr[-500:]}")
        return {"status": "failed", "elapsed_s": elapsed}

    print(f"  Training done ({elapsed:.0f}s)")

    # Parse final losses from stdout
    metrics = parse_training_output(result.stdout)
    metrics["status"] = "ok"
    metrics["elapsed_s"] = elapsed
    return metrics


def parse_training_output(stdout: str) -> dict:
    """Extract final epoch metrics from training stdout."""
    metrics = {}
    lines = stdout.strip().split("\n")

    # Look for the last logged epoch metrics
    for line in reversed(lines):
        if "Traj Loss" in line and "Score Loss" in line:
            try:
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    if "Traj Loss" in part:
                        metrics["val_traj_loss"] = float(part.split(":")[-1].strip())
                    if "Score Loss" in part:
                        metrics["val_score_loss"] = float(part.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
            break

    return metrics


# ---------------------------------------------------------------------------
# Evaluation wrapper
# ---------------------------------------------------------------------------

def run_evaluation(checkpoint_path: str, obs_dim: int = 13,
                   n_samples: int = 200) -> dict:
    """Run closed-loop evaluation and return metrics."""
    if not os.path.exists(checkpoint_path):
        return {"status": "no_checkpoint"}

    cmd = [
        sys.executable, "evaluate_closed_loop.py",
        "--n_samples", str(n_samples),
        "--methods", "B3_YOPO_Payload",
        "--epoch", "0",  # we pass the full path
        "--checkpoint_dir", os.path.dirname(checkpoint_path),
    ]

    # For Stage 1 (wd=0), use original YOPO eval (9D obs)
    if obs_dim == 9:
        cmd[4] = "B0_YOPO_original"

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        # Fallback: extract what we can from loss values
        return {"status": "eval_failed"}

    return parse_eval_output(result.stdout)


def parse_eval_output(stdout: str) -> dict:
    """Parse evaluation summary table from stdout."""
    metrics = {"status": "ok"}
    for line in stdout.split("\n"):
        if "peak=" in line:
            try:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.startswith("peak="):
                        metrics["mean_peak_swing_deg"] = float(p.split("=")[1].rstrip("°"))
                    if p.startswith("rms="):
                        metrics["mean_rms_swing_deg"] = float(p.split("=")[1].rstrip("°"))
                    if p.startswith("success="):
                        metrics["success_rate"] = float(p.split("=")[1].rstrip("%")) / 100
            except (ValueError, IndexError):
                pass
    return metrics


# ---------------------------------------------------------------------------
# Analysis and parameter selection
# ---------------------------------------------------------------------------

def check_success(metrics: dict, criteria: dict) -> tuple:
    """Check if metrics meet success criteria. Returns (passed, reasons)."""
    passed = True
    reasons = []

    for key, threshold in criteria.items():
        if key == "goal_loss_decreasing":
            continue  # checked separately via trend

        metric_key = key.replace("_max", "").replace("_min", "")
        if metric_key not in metrics:
            reasons.append(f"{metric_key}: not available")
            continue

        val = metrics[metric_key]
        if key.endswith("_max") and val > threshold:
            passed = False
            reasons.append(f"{metric_key}={val:.3f} > {threshold} (FAIL)")
        elif key.endswith("_min") and val < threshold:
            passed = False
            reasons.append(f"{metric_key}={val:.3f} < {threshold} (FAIL)")
        else:
            reasons.append(f"{metric_key}={val:.3f} (OK)")

    return passed, reasons


def select_next_params(history: list, param_space: dict, rng) -> dict:
    """Select next parameter combination based on history.

    Strategy:
    - First iteration: use middle values
    - If loss increased: revert to best params and perturb one dimension
    - If loss decreased: continue in same direction
    """
    if not history:
        # First iteration: pick middle of each range
        return {k: sorted(v)[len(v) // 2] for k, v in param_space.items()}

    # Find best iteration so far
    best_idx = 0
    best_score = float("inf")
    for i, h in enumerate(history):
        score = h.get("composite_score", float("inf"))
        if score < best_score:
            best_score = score
            best_idx = i

    best_params = history[best_idx]["params"]

    # Randomly perturb one parameter from best
    keys = list(param_space.keys())
    perturb_key = rng.choice(keys)
    candidates = param_space[perturb_key]

    # Pick a neighbor of the current value
    current_val = best_params.get(perturb_key, candidates[len(candidates) // 2])
    sorted_candidates = sorted(candidates)
    if current_val in sorted_candidates:
        idx = sorted_candidates.index(current_val)
    else:
        idx = len(sorted_candidates) // 2

    # Move to a neighbor (or random if at boundary)
    direction = rng.choice([-1, 1])
    new_idx = max(0, min(len(sorted_candidates) - 1, idx + direction))
    if new_idx == idx:
        new_idx = rng.integers(0, len(sorted_candidates))

    new_params = dict(best_params)
    new_params[perturb_key] = sorted_candidates[new_idx]

    return new_params


def compute_composite_score(metrics: dict, stage: int) -> float:
    """Compute a single scalar score for comparison. Lower is better."""
    if stage == 1:
        traj = metrics.get("val_traj_loss", 100.0)
        score = metrics.get("val_score_loss", 100.0)
        return traj + 0.5 * score
    else:
        peak = metrics.get("mean_peak_swing_deg", 90.0)
        rms = metrics.get("mean_rms_swing_deg", 45.0)
        sr = metrics.get("success_rate", 0.0)
        return peak + rms - 50.0 * sr  # reward success, penalize swing


# ---------------------------------------------------------------------------
# Main refinement loop
# ---------------------------------------------------------------------------

def run_refinement(stage_num: int, base_checkpoint: str = "",
                   max_iter: int = None, epochs_per_iter: int = None):
    stage = STAGES[stage_num]
    print(f"\n{'='*70}")
    print(f"Stage {stage_num}: {stage['name']}")
    print(f"  {stage['description']}")
    print(f"{'='*70}\n")

    max_iterations = max_iter or stage["max_iterations"]
    epochs = epochs_per_iter or stage["epochs_per_iter"]
    patience = stage["patience"]

    # Setup directories
    refine_dir = Path(f"saved/refine_S{stage_num}")
    refine_dir.mkdir(parents=True, exist_ok=True)
    log_path = refine_dir / "refine_log.json"

    # Load existing history if resuming
    history = []
    if log_path.exists():
        with open(log_path) as f:
            history = json.load(f)
        print(f"  Resuming from iteration {len(history)}")

    # Load base config
    yaml_path = os.path.join(os.path.dirname(__file__), "config", "traj_opt.yaml")
    base_config = load_config(yaml_path)

    # Apply fixed params for this stage
    for k, v in stage["fixed"].items():
        base_config[k] = v

    rng = np.random.default_rng(42 + len(history))
    best_score = min((h["composite_score"] for h in history), default=float("inf"))
    no_improve_count = sum(1 for _ in [])  # reset

    # Count consecutive no-improvement
    if history:
        no_improve_count = 0
        for h in reversed(history):
            if h["composite_score"] >= best_score:
                no_improve_count += 1
            else:
                break

    for iteration in range(len(history), max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        # 1. Select parameters
        params = select_next_params(history, stage["param_space"], rng)
        print(f"  Params: {params}")

        # 2. Apply to config
        config = apply_params(base_config, params)
        save_config(yaml_path, config)

        # 3. Train
        iter_dir = str(refine_dir / f"iter_{iteration:02d}")
        checkpoint = base_checkpoint
        if history:
            # Use best checkpoint from previous iterations
            best_iter = min(history, key=lambda h: h["composite_score"])
            prev_best_ckpt = best_iter.get("checkpoint_path", "")
            if prev_best_ckpt and os.path.exists(prev_best_ckpt):
                checkpoint = prev_best_ckpt

        n_steps = params.get("pendulum_n_steps", 20)
        train_metrics = run_training(
            yaml_path, iter_dir, epochs,
            checkpoint_path=checkpoint,
            pendulum_n_steps=n_steps,
        )

        if train_metrics["status"] != "ok":
            print(f"  Training failed, skipping evaluation")
            entry = {
                "iteration": iteration,
                "params": params,
                "train_metrics": train_metrics,
                "eval_metrics": {},
                "composite_score": float("inf"),
                "timestamp": datetime.now().isoformat(),
            }
            history.append(entry)
            with open(log_path, "w") as f:
                json.dump(history, f, indent=2)
            continue

        # 4. Evaluate
        ckpt_path = os.path.join(iter_dir, f"epoch{epochs}.pth")
        if not os.path.exists(ckpt_path):
            # Try finding any checkpoint
            for ep in [epochs, epochs - 10, epochs - 20, 10]:
                candidate = os.path.join(iter_dir, f"epoch{ep}.pth")
                if os.path.exists(candidate):
                    ckpt_path = candidate
                    break

        obs_dim = 9 if config.get("wd", 0) == 0 else 13
        eval_metrics = {}
        if stage_num >= 2:
            eval_metrics = run_evaluation(ckpt_path, obs_dim=obs_dim, n_samples=200)
            print(f"  Eval: {eval_metrics}")

        # 5. Compute composite score
        all_metrics = {**train_metrics, **eval_metrics}
        score = compute_composite_score(all_metrics, stage_num)
        print(f"  Composite score: {score:.3f} (best: {best_score:.3f})")

        # 6. Check success criteria
        passed, reasons = check_success(all_metrics, stage["success_criteria"])
        for r in reasons:
            print(f"    {r}")

        # 7. Record
        entry = {
            "iteration": iteration,
            "params": params,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "composite_score": score,
            "success": passed,
            "checkpoint_path": ckpt_path,
            "timestamp": datetime.now().isoformat(),
        }
        history.append(entry)

        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        # 8. Check termination
        if passed:
            print(f"\n  SUCCESS: Stage {stage_num} criteria met at iteration {iteration + 1}!")
            # Save best checkpoint
            best_path = str(refine_dir / "best.pth")
            if os.path.exists(ckpt_path):
                shutil.copy2(ckpt_path, best_path)
                print(f"  Best checkpoint saved to {best_path}")
            break

        if score < best_score:
            best_score = score
            no_improve_count = 0
            # Save current best
            best_path = str(refine_dir / "best.pth")
            if os.path.exists(ckpt_path):
                shutil.copy2(ckpt_path, best_path)
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"\n  PATIENCE EXHAUSTED: {patience} iterations without improvement.")
            print(f"  Best score: {best_score:.3f}")
            break

    # Restore original config
    save_config(yaml_path, base_config)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Stage {stage_num} Summary ({len(history)} iterations)")
    print(f"{'='*70}")
    print(f"{'Iter':>4} | {'Score':>8} | {'Status':>7} | Params")
    print("-" * 70)
    for h in history:
        status = "PASS" if h.get("success") else "fail"
        params_str = ", ".join(f"{k}={v}" for k, v in h["params"].items())
        print(f"{h['iteration']+1:>4} | {h['composite_score']:>8.3f} | {status:>7} | {params_str}")

    best_entry = min(history, key=lambda h: h["composite_score"])
    print(f"\nBest: iteration {best_entry['iteration']+1}, score={best_entry['composite_score']:.3f}")
    print(f"  Params: {best_entry['params']}")
    print(f"  Checkpoint: {best_entry.get('checkpoint_path', 'N/A')}")

    return best_entry


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOPO Iterative Refinement Loop")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="Which stage to run")
    parser.add_argument("--base_checkpoint", type=str, default="",
                        help="Checkpoint to fine-tune from (e.g., Stage 1 best)")
    parser.add_argument("--max_iter", type=int, default=None,
                        help="Override max iterations for this stage")
    parser.add_argument("--epochs_per_iter", type=int, default=None,
                        help="Override epochs per iteration")
    args = parser.parse_args()

    best = run_refinement(
        stage_num=args.stage,
        base_checkpoint=args.base_checkpoint,
        max_iter=args.max_iter,
        epochs_per_iter=args.epochs_per_iter,
    )

    print(f"\nDone. Next step:")
    if args.stage < 3:
        next_stage = args.stage + 1
        print(f"  python refine_loop.py --stage {next_stage} "
              f"--base_checkpoint saved/refine_S{args.stage}/best.pth")
    else:
        print(f"  All stages complete. Best model at saved/refine_S3/best.pth")
        print(f"  Ready for ablation and baseline comparison.")
