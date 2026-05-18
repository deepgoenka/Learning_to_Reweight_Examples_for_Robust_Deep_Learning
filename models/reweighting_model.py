"""
Validation-Guided Example Reweighting Model.

Simplified implementation of "Learning to Reweight Examples for Robust Deep Learning"
(Ren et al., ICML 2018).

Core idea
---------
For each training mini-batch, assign a higher weight to samples whose gradient
direction *aligns* with the gradient direction that would reduce the CLEAN
validation loss.  Mathematically, if

    grad_val  = gradient of validation loss w.r.t. model parameters
    grad_i    = gradient of sample i's loss w.r.t. model parameters

then

    alignment_i  = dot(grad_val, grad_i)          (scalar)
    weight_i     = max(alignment_i, 0)             (positive alignment → helpful)
    weight_i     = weight_i / sum(weight_j)        (normalize to sum 1)

The final training loss is the weighted sum of per-sample losses.

This is the simplified version described in the implementation guide (no second-order
MAML-style optimization), while capturing the same gradient-alignment intuition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# MLP architecture
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Simple multi-layer perceptron.

    Architecture:
        Input  →  Linear(128) → ReLU → Dropout
               →  Linear(64)  → ReLU
               →  Linear(num_classes)
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def functional_forward(self, x: torch.Tensor, params: list) -> torch.Tensor:
        """
        Forward pass using an externally supplied parameter list.
        Used during weight computation to avoid modifying the model in-place.

        params order: [fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias]
        Dropout is intentionally skipped here (evaluation-style pass for gradient alignment).
        """
        w1, b1, w2, b2, w3, b3 = params
        x = F.relu(F.linear(x, w1, b1))
        x = F.relu(F.linear(x, w2, b2))
        return F.linear(x, w3, b3)


# ---------------------------------------------------------------------------
# Reweighting model (training logic)
# ---------------------------------------------------------------------------

class ReweightingModel:
    """
    MLP trained with validation-guided example reweighting.

    Usage
    -----
    model = ReweightingModel(input_dim=30, num_classes=2)
    model.fit(X_train, y_train, X_val, y_val, epochs=50)
    preds = model.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        dropout: float = 0.3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.model = MLP(input_dim, num_classes, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_losses: list = []

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _compute_weights(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample weights via gradient alignment with the validation gradient.

        Steps
        -----
        1. Compute the gradient of the validation loss (w.r.t. model parameters).
        2. Compute the gradient of each training sample's loss.
        3. weight_i = max( dot(grad_val, grad_i), 0 )
        4. Normalize so weights sum to 1.
        """
        n = len(X_batch)

        # --- Step 1: validation gradient (eval mode for no-dropout consistency) ---
        self.model.eval()
        val_logits = self.model(X_val)
        val_loss = F.cross_entropy(val_logits, y_val)
        val_grads = torch.autograd.grad(val_loss, self.model.parameters())
        # Flatten into a single vector for dot-product comparison
        val_grad_flat = torch.cat([g.flatten() for g in val_grads]).detach()

        # --- Step 2 & 3: per-sample gradient alignment ---
        # Single forward pass for the whole batch (shares graph across samples)
        logits = self.model(X_batch)
        per_sample_loss = F.cross_entropy(logits, y_batch, reduction="none")

        alignments = []
        for i in range(n):
            # retain_graph=True until the last sample to keep the shared graph alive
            retain = i < n - 1
            grads_i = torch.autograd.grad(
                per_sample_loss[i],
                self.model.parameters(),
                retain_graph=retain,
            )
            grad_i_flat = torch.cat([g.flatten() for g in grads_i]).detach()
            alignments.append(torch.dot(val_grad_flat, grad_i_flat).item())

        self.model.train()  # restore training mode for actual update

        # --- Step 4: clamp negative alignments and normalize ---
        alignments_t = torch.tensor(alignments, dtype=torch.float32, device=self.device)
        raw_weights = F.relu(alignments_t)

        weight_sum = raw_weights.sum()
        if weight_sum > 1e-8:
            return raw_weights / weight_sum
        # Fallback: uniform weights when all alignments are ≤ 0
        return torch.ones(n, dtype=torch.float32, device=self.device) / n

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_step(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> float:
        """One mini-batch update with validation-guided reweighting."""

        # 1. Compute sample weights from gradient alignment
        weights = self._compute_weights(X_batch, y_batch, X_val, y_val)

        # 2. Actual forward pass (now in train mode → dropout active)
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(X_batch)
        per_sample_loss = F.cross_entropy(logits, y_batch, reduction="none")

        # 3. Weighted loss (L = Σ w_i * f_i)
        loss = (weights * per_sample_loss).sum()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        val_batch_size: int = 32,
        verbose: bool = True,
    ) -> "ReweightingModel":
        """
        Train the model.

        Args:
            X_train / y_train : Noisy training data.
            X_val   / y_val   : Clean validation data (used ONLY for guidance).
            epochs            : Number of full passes over the training set.
            batch_size        : Training mini-batch size.
            val_batch_size    : Validation mini-batch size used per step (kept small
                                for speed; random subset sampled each step).
            verbose           : Print loss every 10 epochs.
        """
        X_tr = torch.FloatTensor(X_train).to(self.device)
        y_tr = torch.LongTensor(y_train).to(self.device)
        X_vl = torch.FloatTensor(X_val).to(self.device)
        y_vl = torch.LongTensor(y_val).to(self.device)

        n = len(X_tr)
        self.train_losses = []

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = perm[start: start + batch_size]
                X_batch = X_tr[batch_idx]
                y_batch = y_tr[batch_idx]

                # Use a random subset of the validation set each step (efficiency)
                val_idx = torch.randperm(len(X_vl), device=self.device)[:val_batch_size]
                X_val_mini = X_vl[val_idx]
                y_val_mini = y_vl[val_idx]

                loss = self._train_step(X_batch, y_batch, X_val_mini, y_val_mini)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            self.train_losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch [{epoch + 1:3d}/{epochs}]  Loss: {avg_loss:.4f}")

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X).to(self.device))
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X).to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def get_train_losses(self) -> list:
        return self.train_losses
