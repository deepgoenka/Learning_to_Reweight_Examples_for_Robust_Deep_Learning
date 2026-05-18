"""
Experiment 1 — Noisy Labels.

Goal: Verify that the reweighting method degrades less than baseline classifiers
      when training labels are randomly corrupted.

Procedure:
  1. Load Breast Cancer Wisconsin dataset (binary classification).
  2. Split 70/15/15 (train/val/test). Validation and test labels stay clean.
  3. For each noise ratio in [10%, 20%, 30%, 40%]:
       a. Randomly flip that fraction of TRAINING labels.
       b. Train all four models on the noisy training set.
       c. Evaluate every model on the CLEAN test set.
  4. Plot Accuracy vs Noise Ratio, F1 vs Noise Ratio, Training Loss Curves,
     and Confusion Matrices (at highest noise level).
  5. Save all results to results/ and plots to plots/.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

# Make imports work whether running as a script or from the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import load_breast_cancer_data, preprocess_and_split, get_dataset_info
from utils.noise import add_label_noise, noise_stats
from utils.metrics import (
    compute_metrics, print_metrics, get_classification_report,
    save_metrics_json, results_to_dataframe,
)
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.svm import SVMModel
from models.reweighting_model import ReweightingModel

# -------------------------------------------------------------------------
# Helpers
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


def _make_models(input_dim, num_classes, seed):
    """Instantiate one of each model type with a fixed seed."""
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

def _plot_metric_vs_noise(results, metric_key, ylabel, title, fname):
    """Line plot: metric vs noise ratio for all models."""
    noise_ratios = sorted(results.keys())
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in MODEL_NAMES:
        values = [results[nr][model_name][metric_key] for nr in noise_ratios]
        ax.plot([int(nr * 100) for nr in noise_ratios], values,
                marker="o", label=model_name, color=COLORS[model_name], linewidth=2)

    ax.set_xlabel("Noise Ratio (%)", fontsize=12)
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


def _plot_training_loss(loss_history, noise_ratio, fname):
    """Training loss curve for the reweighting model at a given noise level."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_history, color=COLORS["Reweighting (Ours)"], linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Weighted Training Loss", fontsize=12)
    ax.set_title(f"Training Loss — Reweighting Model (noise={int(noise_ratio*100)}%)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_confusion_matrices(cms, noise_ratio, fname):
    """2×2 grid of confusion matrices for all models at the highest noise level."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    class_labels = ["Malignant", "Benign"]

    for ax, (model_name, cm) in zip(axes, cms.items()):
        cm_arr = np.array(cm)
        sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_labels, yticklabels=class_labels,
                    ax=ax, cbar=False)
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    fig.suptitle(f"Confusion Matrices — Noise Ratio = {int(noise_ratio*100)}%",
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
    noise_ratios=(0.1, 0.2, 0.3, 0.4),
    epochs: int = 50,
    batch_size: int = 64,
    val_batch_size: int = 32,
    random_state: int = 42,
):
    """
    Run the full noisy-labels experiment.

    Args:
        noise_ratios  : Sequence of noise levels to test.
        epochs        : Training epochs for the reweighting model.
        batch_size    : Mini-batch size for the reweighting model.
        val_batch_size: Validation subset size per step.
        random_state  : Global RNG seed.
    """
    np.random.seed(random_state)
    torch_seed = random_state

    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Noisy Labels")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load and split data (clean split — noise added later)
    # ------------------------------------------------------------------
    print("\n[1/4] Loading and splitting Breast Cancer dataset...")
    X, y, feat_names, class_names = load_breast_cancer_data()
    X_train, X_val, X_test, y_train_clean, y_val, y_test, _ = \
        preprocess_and_split(X, y, random_state=random_state)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))

    print(f"  Input dim: {input_dim}  |  Classes: {num_classes}  |  "
          f"Labels: {class_names}")
    get_dataset_info(y_train_clean, y_val, y_test)

    # ------------------------------------------------------------------
    # 2. Run for each noise ratio
    # ------------------------------------------------------------------
    all_results = {}     # {noise_ratio: {model_name: metrics_dict}}
    loss_histories = {}  # {noise_ratio: [epoch losses]}

    import torch
    torch.manual_seed(torch_seed)

    for nr in noise_ratios:
        print(f"\n[2/4] Noise ratio = {int(nr * 100)}%")
        print("-" * 40)

        # Corrupt TRAINING labels only — val and test stay clean
        y_train_noisy = add_label_noise(
            y_train_clean, noise_ratio=nr,
            num_classes=num_classes, random_state=random_state
        )
        stats = noise_stats(y_train_clean, y_train_noisy)
        print(f"  Flipped {stats['n_flipped']} / {len(y_train_clean)} "
              f"labels (actual ratio: {stats['actual_ratio']:.2%})")

        all_results[nr] = {}
        models = _make_models(input_dim, num_classes, seed=random_state)

        for model_name, model in models.items():
            print(f"\n  Training: {model_name}")
            t0 = time.time()

            if model_name == "Reweighting (Ours)":
                # Reweighting model uses clean validation for guidance
                model.fit(
                    X_train, y_train_noisy,
                    X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    val_batch_size=val_batch_size,
                    verbose=True,
                )
                loss_histories[nr] = model.get_train_losses()
            else:
                model.fit(X_train, y_train_noisy)

            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s")

            # Evaluate on CLEAN test set
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            all_results[nr][model_name] = metrics
            print_metrics(metrics, model_name=f"{model_name} @ noise={int(nr*100)}%")

            # Print classification report
            report = get_classification_report(y_test, y_pred,
                                               target_names=class_names)
            print(report)

    # ------------------------------------------------------------------
    # 3. Save results
    # ------------------------------------------------------------------
    print("\n[3/4] Saving results...")

    # JSON
    json_path = os.path.join(RESULTS_DIR, "noisy_labels_results.json")
    # Convert keys to strings for JSON serialisation
    serialisable = {str(k): v for k, v in all_results.items()}
    save_metrics_json(serialisable, json_path)
    print(f"  Saved: {json_path}")

    # CSV
    df = results_to_dataframe(all_results)
    df.rename(columns={"condition": "noise_ratio"}, inplace=True)
    csv_path = os.path.join(RESULTS_DIR, "noisy_labels_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print("\n" + df.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Generate plots
    # ------------------------------------------------------------------
    print("\n[4/4] Generating plots...")

    _plot_metric_vs_noise(
        all_results, "accuracy", "Accuracy",
        "Accuracy vs Noise Ratio (Breast Cancer)",
        "noisy_accuracy_vs_noise.png",
    )
    _plot_metric_vs_noise(
        all_results, "f1", "F1-Score (weighted)",
        "F1-Score vs Noise Ratio (Breast Cancer)",
        "noisy_f1_vs_noise.png",
    )

    # Training loss curves for each noise level
    for nr, losses in loss_histories.items():
        _plot_training_loss(
            losses, nr,
            f"noisy_training_loss_{int(nr * 100)}pct.png",
        )

    # Confusion matrices at the highest noise level
    highest_nr = max(noise_ratios)
    cms = {m: all_results[highest_nr][m]["confusion_matrix"] for m in MODEL_NAMES}
    _plot_confusion_matrices(
        cms, highest_nr,
        f"noisy_confusion_matrices_{int(highest_nr * 100)}pct.png",
    )

    print("\n✓  Experiment 1 complete.")
    return all_results
