"""
Experiment 2 — Class Imbalance.

Dataset: Pima Indians Diabetes (UCI ID 34)
  768 samples, 8 features, binary classification.
  Natural imbalance: 65.1% no-diabetes vs 34.9% diabetes.
  Source: https://archive.ics.uci.edu/dataset/34/diabetes

Goal: Verify that the reweighting method preserves minority-class performance
      better than baseline classifiers under severe class imbalance.

Procedure:
  1. Load Pima Indians Diabetes dataset (different from Exp 1).
  2. Split 70/15/15 (train/val/test). Validation and test stay balanced.
  3. For each imbalance ratio in [1:5, 1:10, 1:20]:
       a. Reduce minority-class (diabetes=1) samples in the TRAINING set only.
       b. Train all four models.
       c. Evaluate on the BALANCED clean test set.
  4. Plot Accuracy vs Imbalance Ratio, F1 vs Imbalance Ratio,
     and Confusion Matrices (at worst imbalance).
  5. Save results to results/ and plots to plots/.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import load_pima_diabetes_data, preprocess_and_split, get_dataset_info
from utils.imbalance import create_imbalanced_dataset, imbalance_stats
from utils.metrics import (
    compute_metrics, print_metrics, get_classification_report,
    save_metrics_json, results_to_dataframe,
)
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.svm import SVMModel
from models.reweighting_model import ReweightingModel

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAMES = ["Logistic Regression", "Random Forest", "SVM", "Reweighting (Ours)"]
COLORS = {"Logistic Regression": "#e74c3c",
          "Random Forest": "#2ecc71",
          "SVM": "#3498db",
          "Reweighting (Ours)": "#9b59b6"}

# Imbalance ratios expressed as minority:majority (e.g. 1/5 = 1:5)
DEFAULT_RATIOS = {
    "1:5":  1 / 5,
    "1:10": 1 / 10,
    "1:20": 1 / 20,
}


def _make_models(input_dim, num_classes, seed):
    return {
        "Logistic Regression": LogisticRegressionModel(random_state=seed),
        "Random Forest":       RandomForestModel(random_state=seed),
        "SVM":                 SVMModel(random_state=seed),
        "Reweighting (Ours)":  ReweightingModel(input_dim=input_dim,
                                                 num_classes=num_classes,
                                                 lr=1e-3, dropout=0.3),
    }


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def _plot_metric_vs_ratio(results, metric_key, ylabel, title, fname):
    """Line plot: metric vs imbalance ratio label (e.g. '1:5') for all models."""
    ratio_labels = list(results.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in MODEL_NAMES:
        values = [results[rl][model_name][metric_key] for rl in ratio_labels]
        ax.plot(ratio_labels, values,
                marker="o", label=model_name,
                color=COLORS[model_name], linewidth=2)

    ax.set_xlabel("Imbalance Ratio (minority : majority)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_training_loss(loss_history, ratio_label, fname):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_history, color=COLORS["Reweighting (Ours)"], linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Weighted Training Loss", fontsize=12)
    ax.set_title(f"Training Loss — Reweighting Model (imbalance {ratio_label})",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_confusion_matrices(cms, ratio_label, fname):
    """2×2 grid of confusion matrices for all models at the worst imbalance."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    class_labels = ["No Diabetes", "Diabetes"]

    for ax, (model_name, cm) in zip(axes, cms.items()):
        cm_arr = np.array(cm)
        sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Purples",
                    xticklabels=class_labels, yticklabels=class_labels,
                    ax=ax, cbar=False)
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    fig.suptitle(f"Confusion Matrices — Imbalance Ratio {ratio_label}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# -------------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------------

def run_experiment(
    imbalance_ratios: dict = None,
    epochs: int = 50,
    batch_size: int = 64,
    val_batch_size: int = 32,
    random_state: int = 42,
):
    """
    Run the full class-imbalance experiment.

    Args:
        imbalance_ratios : dict mapping label (e.g. '1:5') → float ratio (e.g. 1/5).
                           Defaults to {1:5, 1:10, 1:20}.
        epochs           : Training epochs for the reweighting model.
        batch_size       : Mini-batch size.
        val_batch_size   : Validation subset size per step.
        random_state     : Global RNG seed.
    """
    if imbalance_ratios is None:
        imbalance_ratios = DEFAULT_RATIOS

    np.random.seed(random_state)

    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Class Imbalance")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load and split data (balanced split)
    # ------------------------------------------------------------------
    print("\n[1/4] Loading Pima Indians Diabetes dataset (UCI ID 34)...")
    print("      (Different dataset from Experiment 1 — Breast Cancer Wisconsin)")
    X, y, feat_names, class_names = load_pima_diabetes_data()
    X_train_full, X_val, X_test, y_train_full, y_val, y_test, _ = \
        preprocess_and_split(X, y, random_state=random_state)

    input_dim = X_train_full.shape[1]
    num_classes = len(np.unique(y))

    print(f"  Input dim: {input_dim}  |  Classes: {num_classes}  |  "
          f"Labels: {class_names}")
    get_dataset_info(y_train_full, y_val, y_test)

    # Identify minority class (class with fewer samples = diabetes=1)
    unique, counts = np.unique(y_train_full, return_counts=True)
    minority_class = int(unique[np.argmin(counts)])
    print(f"  Minority class: {minority_class} ({class_names[minority_class]}) "
          f"with {counts.min()} samples in train split")

    # ------------------------------------------------------------------
    # 2. Run for each imbalance ratio
    # ------------------------------------------------------------------
    all_results = {}
    loss_histories = {}

    import torch
    torch.manual_seed(random_state)

    for ratio_label, ratio_val in imbalance_ratios.items():
        print(f"\n[2/4] Imbalance ratio = {ratio_label}")
        print("-" * 40)

        # Reduce minority class in TRAINING set only
        X_train_imb, y_train_imb = create_imbalanced_dataset(
            X_train_full, y_train_full,
            minority_class=minority_class,
            imbalance_ratio=ratio_val,
            random_state=random_state,
        )
        stats = imbalance_stats(y_train_imb)
        print(f"  Imbalanced training set: {dict(stats)}")

        all_results[ratio_label] = {}
        models = _make_models(input_dim, num_classes, seed=random_state)

        for model_name, model in models.items():
            print(f"\n  Training: {model_name}")
            t0 = time.time()

            if model_name == "Reweighting (Ours)":
                model.fit(
                    X_train_imb, y_train_imb,
                    X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    val_batch_size=val_batch_size,
                    verbose=True,
                )
                loss_histories[ratio_label] = model.get_train_losses()
            else:
                model.fit(X_train_imb, y_train_imb)

            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s")

            # Evaluate on balanced clean test set
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            all_results[ratio_label][model_name] = metrics
            print_metrics(metrics, model_name=f"{model_name} @ {ratio_label}")

            report = get_classification_report(y_test, y_pred,
                                               target_names=class_names)
            print(report)

    # ------------------------------------------------------------------
    # 3. Save results
    # ------------------------------------------------------------------
    print("\n[3/4] Saving results...")
    json_path = os.path.join(RESULTS_DIR, "imbalance_results.json")
    save_metrics_json(all_results, json_path)
    print(f"  Saved: {json_path}")

    df = results_to_dataframe(all_results)
    df.rename(columns={"condition": "imbalance_ratio"}, inplace=True)
    csv_path = os.path.join(RESULTS_DIR, "imbalance_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print("\n" + df.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Generate plots
    # ------------------------------------------------------------------
    print("\n[4/4] Generating plots...")

    _plot_metric_vs_ratio(
        all_results, "accuracy", "Accuracy",
        "Accuracy vs Imbalance Ratio (Breast Cancer)",
        "imbalance_accuracy_vs_ratio.png",
    )
    _plot_metric_vs_ratio(
        all_results, "f1", "F1-Score (weighted)",
        "F1-Score vs Imbalance Ratio (Breast Cancer)",
        "imbalance_f1_vs_ratio.png",
    )

    for ratio_label, losses in loss_histories.items():
        safe_label = ratio_label.replace(":", "_")
        _plot_training_loss(
            losses, ratio_label,
            f"imbalance_training_loss_{safe_label}.png",
        )

    # Confusion matrices at the worst imbalance (smallest ratio value)
    worst_label = list(imbalance_ratios.keys())[-1]
    cms = {m: all_results[worst_label][m]["confusion_matrix"] for m in MODEL_NAMES}
    _plot_confusion_matrices(
        cms, worst_label,
        f"imbalance_confusion_matrices_{worst_label.replace(':', '_')}.png",
    )

    print("\n✓  Experiment 2 complete.")
    return all_results
