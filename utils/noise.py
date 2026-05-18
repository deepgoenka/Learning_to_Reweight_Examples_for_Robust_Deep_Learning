"""
Label noise injection utilities.
Used for Experiment 1: randomly flips training labels to simulate noisy annotation.
"""

import numpy as np


def add_label_noise(y, noise_ratio, num_classes=2, random_state=42):
    """
    Randomly flip a fraction of labels to a different class.

    Args:
        y:           Original label array (int).
        noise_ratio: Fraction of samples to corrupt, e.g. 0.2 for 20%.
        num_classes: Total number of classes.
        random_state: RNG seed for reproducibility.

    Returns:
        y_noisy: Corrupted label array (copy — original is unchanged).
    """
    rng = np.random.RandomState(random_state)
    y_noisy = y.copy()
    n = len(y_noisy)
    n_noisy = int(noise_ratio * n)

    noisy_indices = rng.choice(n, size=n_noisy, replace=False)
    for idx in noisy_indices:
        current = y_noisy[idx]
        other_classes = [c for c in range(num_classes) if c != current]
        y_noisy[idx] = rng.choice(other_classes)

    return y_noisy


def noise_stats(y_original, y_noisy):
    """Return a dict summarising how many / which labels were flipped."""
    flipped = y_original != y_noisy
    return {
        "n_flipped": int(flipped.sum()),
        "actual_ratio": float(flipped.mean()),
        "flipped_indices": np.where(flipped)[0].tolist(),
    }
