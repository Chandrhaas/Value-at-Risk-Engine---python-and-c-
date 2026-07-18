"""
Trains the quantile regression neural network for VaR estimation: a small
feedforward net with two output heads (q=0.05 and q=0.01), fit jointly with
pinball loss, on the pooled single-asset dataset from prepare_data.py.

note: 
q=0.01 is a more extreme (lower) tail than q=0.05, so by definition the
true q01 return level must be <= the true q05 return level. Two
independent output heads trained only with a joint pinball loss have no
guarantee of preserving that ordering. To account for this the model outputs q05 directly and a non-negative gap
that is SUBTRACTED to get q01: q01 = q05 - softplus(gap).

Outputs
-------
models/var_model.pth     -- torch state_dict + architecture metadata
models/feature_scaler.pkl -- the StandardScaler fit on TRAIN_TICKERS only
"""

import os
import random
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from src_python.prepare_data import (DEFAULT_YEARS, FEATURE_COLUMNS, build_feature_table, prepare_var_dataset, transform_with_scaler)

QUANTILES: List[float] = [0.05, 0.01]  # order fixed: index 0 -> q05, index 1 -> q01

# Training universe: liquid, sector-diverse names.
TRAIN_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "NVDA",   # tech
    "JPM", "BAC", "GS",                # financials
    "JNJ", "UNH", "PFE",               # healthcare
    "PG", "KO", "MCD",                 # consumer
    "XOM", "CVX",                      # energy
    "HON",                             # industrials
]

HOLDOUT_TICKERS: List[str] = [
    "AMD",   # tech
    "WFC",   # financials
    "ABBV",  # healthcare
    "NKE",   # consumer
    "COP",   # energy
    "CAT",   # industrials
]
assert not set(TRAIN_TICKERS) & set(HOLDOUT_TICKERS), "TRAIN/HOLDOUT ticker overlap!"

HIDDEN_DIMS: Tuple[int, ...] = (64, 32)
BATCH_SIZE: int = 256
MAX_EPOCHS: int = 200
LEARNING_RATE: float = 1e-3
EARLY_STOPPING_PATIENCE: int = 15
SEED: int = 42
MODEL_DIR: str = os.path.join(root, "models")


def set_seed(seed: int = SEED) -> None:
    """Seed python/numpy/torch for a reproducible training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QuantileVaRNet(nn.Module):
    """
    Small feedforward net predicting two return quantiles (q05, q01) from
    an 8-feature, single-asset input vector (see prepare_data.py).

    Outputs q05 directly from a linear head, and q01 as q05 minus a
    softplus-transformed (always >= 0) "gap" from a second head.
    """

    def __init__(self, n_features: int, hidden_dims: Tuple[int, ...] = HIDDEN_DIMS):
        super().__init__()
        layers = []
        prev_dim = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        self.trunk = nn.Sequential(*layers)
        self.q05_head = nn.Linear(prev_dim, 1)
        self.gap_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_features)

        Returns
        -------
        torch.Tensor, shape (batch, 2)
            Column 0 = predicted q05 return, column 1 = predicted q01 return.
            Column 1 <= column 0 for every row, always.
        """
        hidden = self.trunk(x)
        q05 = self.q05_head(hidden)
        gap = F.softplus(self.gap_head(hidden))  # >= 0 for any input
        q01 = q05 - gap
        return torch.cat([q05, q01], dim=1)


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """
    Standard Koenker & Bassett formulation: for error e = y_true - y_pred,
        loss = max(q * e, (q - 1) * e)
    which is equivalent to the textbook piecewise definition
        loss = q * e         if e >= 0   (under-prediction, y_true above y_pred)
        loss = (q - 1) * e   if e < 0    (over-prediction, y_true below y_pred)

    Parameters
    ----------
    y_true : torch.Tensor, shape (batch, 1)
    y_pred : torch.Tensor, shape (batch, n_quantiles)
    quantiles : torch.Tensor, shape (1, n_quantiles)
        Precomputed once outside the training loop, not rebuilt every call.

    Returns
    -------
    torch.Tensor, scalar
    """
    errors = y_true - y_pred  # broadcasts (batch,1) against (batch,n_quantiles)
    loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
    return loss.mean()


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    quantiles: List[float] = QUANTILES,
    epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """
    Train with Adam + early stopping on validation pinball loss.

    IMPORTANT: X_val/y_val here should be the TEMPORAL test set (same
    tickers as training, later dates) used for early stopping decisions.
    Do NOT pass the held-out-ticker set here; early stopping on it would
    make it no longer a clean, uncontaminated generalization check.

    DataLoader(shuffle=True) shuffles MINIBATCH order
    within the already-fixed training set every epoch and NOT the same thing as shuffling the train/test split
    itself (which prepare_data.py correctly never does, since that would leak future dates into training).

    Returns
    -------
    (model, history) : trained model (best validation checkpoint restored) and per-epoch train/val loss history.
    """
    model = model.to(device)
    quantiles_t = torch.tensor(quantiles, dtype=torch.float32, device=device).view(1, -1)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = pinball_loss(yb, preds, quantiles_t)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = pinball_loss(y_val_t, val_preds, quantiles_t).item()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[train_model] epoch {epoch:3d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[train_model] Early stopping at epoch {epoch} "
                      f"(best val_loss={best_val_loss:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

def evaluate(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    quantiles: List[float] = QUANTILES,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Compute pinball loss and empirical coverage for each quantile head.

    Empirical coverage for quantile q: 
    fraction of true returns <= the predicted q-quantile.
    A well-calibrated q=0.05 head should have coverage close to 0.05; likewise q=0.01 close to 0.01
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    quantiles_t = torch.tensor(quantiles, dtype=torch.float32, device=device).view(1, -1)

    with torch.no_grad():
        preds = model(X_t)
        loss = pinball_loss(y_t, preds, quantiles_t).item()

    preds_np = preds.cpu().numpy()
    y_flat = y.flatten()

    metrics = {"pinball_loss": loss}
    for i, q in enumerate(quantiles):
        coverage = float(np.mean(y_flat <= preds_np[:, i]))
        metrics[f"coverage_q{q}"] = coverage
    return metrics


if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Using device: {device}")

    print("\n=== Building training dataset ===")
    dataset = prepare_var_dataset(TRAIN_TICKERS, years=DEFAULT_YEARS)

    print("\n=== Building held-out ticker dataset (never used in training) ===")
    holdout_table = build_feature_table(HOLDOUT_TICKERS, years=DEFAULT_YEARS)
    X_holdout, y_holdout = transform_with_scaler(holdout_table, dataset["scaler"])

    print("\n=== Training ===")
    model = QuantileVaRNet(n_features=len(FEATURE_COLUMNS), hidden_dims=HIDDEN_DIMS)
    model, history = train_model(
        model,
        dataset["X_train"], dataset["y_train"],
        dataset["X_test"], dataset["y_test"],  # temporal val set, for early stopping
        quantiles=QUANTILES,
        device=device,
    )

    print("\n=== Final evaluation ===")
    temporal_metrics = evaluate(model, dataset["X_test"], dataset["y_test"], device=device)
    holdout_metrics = evaluate(model, X_holdout, y_holdout, device=device)

    print(f"Temporal test set (same tickers, later dates): {temporal_metrics}")
    print(f"Held-out ticker set (never seen at all): {holdout_metrics}")
    print("(Target coverage: ~0.05 for the q05 head, ~0.01 for the q01 head. "
          "A holdout-ticker coverage close to the temporal test's is the actual "
          "evidence the single-asset architecture generalizes across tickers.)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "var_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dims": HIDDEN_DIMS,
            "n_features": len(FEATURE_COLUMNS),
            "feature_columns": FEATURE_COLUMNS,
            "quantiles": QUANTILES,
        },
        model_path,
    )
    scaler_path = os.path.join(MODEL_DIR, "feature_scaler.pkl")
    joblib.dump(dataset["scaler"], scaler_path)

    print(f"\n[main] Saved model  -> {model_path}")
    print(f"[main] Saved scaler -> {scaler_path}")