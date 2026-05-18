"""
Data loading and preprocessing utilities.

Two UCI datasets are used:
  1. Breast Cancer Wisconsin  (sklearn built-in, UCI source)
     → Experiment 1: Noisy Labels
     → Binary: 0=malignant, 1=benign  |  569 samples, 30 features

  2. Pima Indians Diabetes  (fetched from OpenML / UCI)
     → Experiment 2: Class Imbalance
     → Binary: 0=no diabetes, 1=diabetes  |  768 samples, 8 features
     → Natural imbalance: 65.1% negative vs 34.9% positive
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

RANDOM_SEED = 42


# -------------------------------------------------------------------------
# Dataset 1 — Breast Cancer Wisconsin  (Noisy Labels experiment)
# UCI: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
# -------------------------------------------------------------------------

def load_breast_cancer_data():
    """
    Load the Breast Cancer Wisconsin (Diagnostic) dataset.
    Bundled with scikit-learn — no download required.
    Source: UCI ML Repository, dataset ID 17.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    # y=0: malignant (212 samples), y=1: benign (357 samples)
    return X, y, list(data.feature_names), list(data.target_names)


# -------------------------------------------------------------------------
# Dataset 2 — Pima Indians Diabetes  (Class Imbalance experiment)
# UCI: https://archive.ics.uci.edu/dataset/34/diabetes
# -------------------------------------------------------------------------

def load_pima_diabetes_data():
    """
    Load the Pima Indians Diabetes dataset from OpenML (mirrors UCI ID 34).
    Downloaded and cached the first time; no internet needed after that.

    Features (8): pregnancies, glucose, blood pressure, skin thickness,
                  insulin, BMI, diabetes pedigree function, age.
    Labels: 0 = no diabetes (500 samples, 65.1%)
            1 = diabetes    (268 samples, 34.9%)
    Total: 768 samples
    """
    print("  Loading Pima Indians Diabetes dataset from OpenML (UCI ID 34)...")
    dataset = fetch_openml(
        name="diabetes",
        version=1,
        as_frame=False,
        parser="auto",
    )
    X = dataset.data.astype(np.float64)
    # OpenML labels are strings 'tested_positive' / 'tested_negative'
    raw_y = dataset.target
    y = (raw_y == "tested_positive").astype(int)

    feature_names = list(dataset.feature_names) if hasattr(dataset, "feature_names") \
        else [f"feature_{i}" for i in range(X.shape[1])]
    class_names = ["no_diabetes", "diabetes"]

    return X, y, feature_names, class_names


# -------------------------------------------------------------------------
# Shared preprocessing (works for both datasets)
# -------------------------------------------------------------------------

def preprocess_and_split(X, y, val_size=0.15, test_size=0.15, random_state=RANDOM_SEED):
    """
    Handle missing values, standardize features, and split 70 / 15 / 15.
    The validation and test sets are NEVER corrupted (noise / imbalance is
    applied to the training portion only, after this function returns).

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    # --- Missing-value imputation (median strategy, safe for both datasets) ---
    if np.isnan(X).any():
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)

    # --- Train / test split first ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # --- Then split the remainder into train / val ---
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp,
    )

    # --- Standardise: fit on train only to avoid data leakage ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def get_dataset_info(y_train, y_val, y_test):
    """Print a quick class-distribution summary for each split."""
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"  {name}: {len(y)} samples | class dist: {dist}")
