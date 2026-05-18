"""
train.py — Main entry point for running all experiments.

Usage
-----
    python train.py                    # run both experiments with default settings
    python train.py --exp noisy        # only Experiment 1 (noisy labels)
    python train.py --exp imbalance    # only Experiment 2 (class imbalance)
    python train.py --epochs 30        # override training epochs for reweighting model
    python train.py --seed 123         # custom random seed

Outputs
-------
    plots/    — all figures (PNG)
    results/  — metrics tables (CSV + JSON)
"""

import argparse
import sys
import os
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def set_global_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learning to Reweight Examples — experiment runner"
    )
    parser.add_argument(
        "--exp",
        choices=["noisy", "imbalance", "both"],
        default="both",
        help="Which experiment to run (default: both)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for the reweighting model (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size for the reweighting model (default: 64)",
    )
    parser.add_argument(
        "--val_batch",
        type=int,
        default=32,
        help="Validation mini-batch size per step (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seeds(args.seed)

    print("\n" + "#" * 62)
    print("#  Learning to Reweight Examples for Robust Deep Learning  #")
    print("#  Simplified Implementation — FML Course Project          #")
    print("#" * 62)
    print(f"\n  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Val batch   : {args.val_batch}")
    print(f"  Seed        : {args.seed}")
    print(f"  Experiment  : {args.exp}")

    common_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch,
        random_state=args.seed,
    )

    if args.exp in ("noisy", "both"):
        from experiments.noisy_labels import run_experiment as run_noisy
        run_noisy(**common_kwargs)

    if args.exp in ("imbalance", "both"):
        from experiments.imbalance import run_experiment as run_imbalance
        run_imbalance(**common_kwargs)

    print("\n" + "=" * 62)
    print("  All experiments finished.")
    print(f"  Plots   → {os.path.join(PROJECT_ROOT, 'plots/')}")
    print(f"  Results → {os.path.join(PROJECT_ROOT, 'results/')}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
