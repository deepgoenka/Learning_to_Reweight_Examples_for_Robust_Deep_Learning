"""
Support Vector Machine baseline model.
Wraps sklearn.svm.SVC with a uniform interface.
"""

from sklearn.svm import SVC


class SVMModel:
    """Baseline support vector machine classifier."""

    def __init__(self, C=1.0, kernel="rbf", random_state=42):
        self.model = SVC(
            C=C,
            kernel=kernel,
            random_state=random_state,
            probability=True,   # needed for predict_proba
            cache_size=500,
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
