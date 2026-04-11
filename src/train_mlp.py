"""Small PyTorch MLP: dropout, weighted cross-entropy, early stopping."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

MODELS_DIR = "models"
MLP_WEIGHTS = os.path.join(MODELS_DIR, "mlp.pt")
MLP_META = os.path.join(MODELS_DIR, "mlp_meta.joblib")


@dataclass
class MLPMeta:
    input_dim: int
    hidden_dims: tuple
    num_classes: int
    dropout: float
    class_names: list


class SmallMLP(nn.Module):
    def __init__(
        self, input_dim: int, num_classes: int, hidden_dims=(256, 128, 64), dropout=0.3
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d_in, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d_in = h
        layers.append(nn.Linear(d_in, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifierWrapper:
    """Sklearn-like API for test.py and inference."""

    def __init__(self, model: SmallMLP, meta: MLPMeta, device: torch.device):
        self.model = model
        self.meta = meta
        self.device = device
        self.classes_ = np.array(meta.class_names)

    def _as_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(X, dtype=torch.float32, device=self.device)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._as_tensor(X))
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._as_tensor(X))
            return torch.softmax(logits, dim=1).cpu().numpy()


def _make_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_save_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: list,
    batch_size: int = 4096,
    max_epochs: int = 100,
    patience: int = 7,
    lr: float = 1e-3,
    dropout: float = 0.3,
    hidden_dims: tuple = (256, 128, 64),
    seed: int = 42,
) -> MLPClassifierWrapper:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _make_device()
    n_classes = len(class_names)
    input_dim = X_train.shape[1]

    weights = compute_class_weight(
        "balanced", classes=np.arange(n_classes), y=y_train
    )
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_t = torch.as_tensor(y_train, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0
    )

    model = SmallMLP(input_dim, n_classes, hidden_dims=hidden_dims, dropout=dropout)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)

    best_val = float("inf")
    best_state = None
    stale = 0
    t0 = time.perf_counter()

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(X_val_t)
            v_loss = crit(v_logits, y_val_t).item()

        if v_loss < best_val - 1e-6:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  MLP epoch {epoch + 1}/{max_epochs}  val_loss={v_loss:.5f}")
        if stale >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    elapsed = time.perf_counter() - t0
    print(f"  MLP training time: {elapsed:.1f}s")

    meta = MLPMeta(
        input_dim=input_dim,
        hidden_dims=tuple(hidden_dims),
        num_classes=n_classes,
        dropout=dropout,
        class_names=list(class_names),
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), MLP_WEIGHTS)
    joblib.dump(meta, MLP_META)

    wrapper = MLPClassifierWrapper(model, meta, device)
    return wrapper


def load_mlp_wrapper() -> MLPClassifierWrapper:
    meta: MLPMeta = joblib.load(MLP_META)
    device = _make_device()
    model = SmallMLP(
        meta.input_dim,
        meta.num_classes,
        hidden_dims=tuple(meta.hidden_dims),
        dropout=meta.dropout,
    )
    state = torch.load(MLP_WEIGHTS, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return MLPClassifierWrapper(model, meta, device)
