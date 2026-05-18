"""
Logistic Regression baseline model.
Wraps sklearn.linear_model.LogisticRegression with a uniform interface.
"""

from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    """Baseline logistic regression classifier."""

    def __init__(self, C=1.0, max_iter=1000, random_state=42):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
            multi_class="auto",
        )
        self.train_losses = []  # placeholder for interface compatibility

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
