"""
Evaluation metrics utilities.
Computes accuracy, precision, recall, F1, confusion matrix, and classification report.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


def compute_metrics(y_true, y_pred, average="weighted"):
    """
    Compute all required evaluation metrics.

    Args:
        y_true:   Ground-truth labels.
        y_pred:   Predicted labels.
        average:  Averaging strategy for multi-class metrics.

    Returns:
        dict with keys: accuracy, precision, recall, f1, confusion_matrix
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_metrics(metrics, model_name="Model"):
    """Pretty-print a metrics dict."""
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1']:.4f}")
    print(f"  Confusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    print(f"    {cm}")


def get_classification_report(y_true, y_pred, target_names=None):
    """Return sklearn classification report as a string."""
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def save_metrics_json(metrics_dict, filepath):
    """Save a dict of metrics to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics_dict, f, indent=2)


def results_to_dataframe(all_results):
    """
    Convert a nested results dict to a tidy DataFrame.

    Expected structure:
        all_results[condition_value][model_name] = metrics_dict
    """
    rows = []
    for condition, model_results in all_results.items():
        for model_name, metrics in model_results.items():
            row = {"condition": condition, "model": model_name}
            row.update({k: v for k, v in metrics.items() if k != "confusion_matrix"})
            rows.append(row)
    return pd.DataFrame(rows)
