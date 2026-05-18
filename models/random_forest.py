"""
Random Forest baseline model.
Wraps sklearn.ensemble.RandomForestClassifier with a uniform interface.
"""

from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    """Baseline random forest classifier."""

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.train_losses = []

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train on X_train / y_train (val args accepted but unused)."""
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_train_losses(self):
        return self.train_losses
