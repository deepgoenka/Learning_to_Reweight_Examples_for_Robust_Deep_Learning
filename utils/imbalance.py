"""
Class imbalance utilities.
Used for Experiment 2: artificially reduces the minority class in the TRAINING set only.
The validation and test sets are kept balanced throughout.
"""

import numpy as np


def create_imbalanced_dataset(X_train, y_train, minority_class=0,
                               imbalance_ratio=0.1, random_state=42):
    """
    Reduce the minority class in the training set to achieve a target ratio.

    Args:
        X_train:          Training features.
        y_train:          Training labels.
        minority_class:   Class label to reduce (default 0 = malignant in breast cancer).
        imbalance_ratio:  Desired minority:majority ratio (e.g. 1/5, 1/10, 1/20).
        random_state:     RNG seed.

    Returns:
        X_imb, y_imb: Imbalanced training arrays (shuffled).
    """
    rng = np.random.RandomState(random_state)

    minority_idx = np.where(y_train == minority_class)[0]
    majority_idx = np.where(y_train != minority_class)[0]

    n_majority = len(majority_idx)
    n_minority_target = max(2, int(n_majority * imbalance_ratio))

    if n_minority_target >= len(minority_idx):
        # No reduction needed; dataset already satisfies the ratio
        return X_train.copy(), y_train.copy()

    kept_minority = rng.choice(minority_idx, size=n_minority_target, replace=False)
    all_idx = np.concatenate([kept_minority, majority_idx])
    rng.shuffle(all_idx)

    return X_train[all_idx], y_train[all_idx]


def imbalance_stats(y):
    """Print class distribution of a label array."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    info = {int(u): {"count": int(c), "pct": round(100 * c / total, 1)}
            for u, c in zip(unique, counts)}
    return info
