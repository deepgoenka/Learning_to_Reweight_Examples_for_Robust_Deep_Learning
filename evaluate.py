"""
evaluate.py — Load saved experiment results and print a final summary report.

Run this AFTER train.py has completed.

Usage
-----
    python evaluate.py
    python evaluate.py --exp noisy       # only the noisy-labels results
    python evaluate.py --exp imbalance   # only the imbalance results
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

MODEL_NAMES = ["Logistic Regression", "Random Forest", "SVM", "Reweighting (Ours)"]
COLORS = {"Logistic Regression": "#e74c3c",
          "Random Forest": "#2ecc71",
          "SVM": "#3498db",
          "Reweighting (Ours)": "#9b59b6"}


def load_json(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"  [WARN] Results file not found: {path}")
        print("  Run 'python train.py' first to generate results.")
        return None
    with open(path) as f:
        return json.load(f)


def print_summary_table(results, condition_name, metric_keys=("accuracy", "f1")):
    """Print a formatted table: rows = conditions, columns = model × metric."""
    print(f"\n{'='*70}")
    print(f"  Summary — {condition_name}")
    print(f"{'='*70}")

    conditions = list(results.keys())
    header = f"{'Condition':<12}" + "".join(
        f"{m[:4]:>9}" for m in MODEL_NAMES for _ in metric_keys
    )
    sub_header = f"{'':12}" + "".join(
        f"{'  acc':>5}{'  f1':>4}" for _ in MODEL_NAMES
    )
    print(header)
    print(sub_header)
    print("-" * 70)

    for cond in conditions:
        row = f"{str(cond):<12}"
        for model_name in MODEL_NAMES:
            m = results[cond].get(model_name, {})
            acc = m.get("accuracy", float("nan"))
            f1  = m.get("f1",       float("nan"))
            row += f"{acc:>6.3f}{f1:>5.3f}"
        print(row)


def plot_bar_comparison(results, condition_name, metric_key, ylabel, title, fname):
    """Grouped bar chart comparing all models across conditions."""
    conditions = list(results.keys())
    x = np.arange(len(conditions))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model_name in enumerate(MODEL_NAMES):
        vals = [results[c].get(model_name, {}).get(metric_key, 0) for c in conditions]
        ax.bar(x + i * width, vals, width, label=model_name,
               color=COLORS[model_name], alpha=0.85)

    ax.set_xlabel(condition_name, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def evaluate_noisy():
    print("\n" + "#" * 50)
    print("#  EXPERIMENT 1 — Noisy Labels Results")
    print("#" * 50)

    data = load_json("noisy_labels_results.json")
    if data is None:
        return

    # Keys were stored as strings (JSON requirement)
    results = {float(k): v for k, v in data.items()}

    print_summary_table(
        {f"{int(k*100)}%": v for k, v in sorted(results.items())},
        condition_name="Noise Ratio",
    )

    # Bar charts
    labelled = {f"{int(k*100)}%": v for k, v in sorted(results.items())}
    plot_bar_comparison(
        labelled, "Noise Ratio (%)", "accuracy", "Accuracy",
        "Model Accuracy under Noisy Labels",
        "eval_noisy_accuracy_bar.png",
    )
    plot_bar_comparison(
        labelled, "Noise Ratio (%)", "f1", "F1-Score (weighted)",
        "Model F1-Score under Noisy Labels",
        "eval_noisy_f1_bar.png",
    )

    # Find best model per noise level
    print("\n  Best model at each noise level:")
    for k, models in sorted(results.items()):
        best = max(models, key=lambda m: models[m].get("f1", 0))
        print(f"    {int(k*100)}% noise → {best}  (F1={models[best]['f1']:.3f})")


def evaluate_imbalance():
    print("\n" + "#" * 50)
    print("#  EXPERIMENT 2 — Class Imbalance Results")
    print("#" * 50)

    data = load_json("imbalance_results.json")
    if data is None:
        return

    print_summary_table(data, condition_name="Imbalance Ratio")

    plot_bar_comparison(
        data, "Imbalance Ratio", "accuracy", "Accuracy",
        "Model Accuracy under Class Imbalance",
        "eval_imbalance_accuracy_bar.png",
    )
    plot_bar_comparison(
        data, "Imbalance Ratio", "f1", "F1-Score (weighted)",
        "Model F1-Score under Class Imbalance",
        "eval_imbalance_f1_bar.png",
    )

    print("\n  Best model at each imbalance ratio:")
    for ratio, models in data.items():
        best = max(models, key=lambda m: models[m].get("f1", 0))
        print(f"    {ratio} → {best}  (F1={models[best]['f1']:.3f})")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and summarise experiment results")
    p.add_argument("--exp", choices=["noisy", "imbalance", "both"], default="both")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if args.exp in ("noisy", "both"):
        evaluate_noisy()

    if args.exp in ("imbalance", "both"):
        evaluate_imbalance()

    print("\nDone. Evaluation plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
