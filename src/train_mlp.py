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

from evaluation import security_metrics_dict

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
    class_weight_mode: str = "balanced"
    lr: float = 1e-3
    selection_metric: str = "val_loss"


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


def _le_proxy(class_names: list):
    return type("LE", (), {"classes_": np.array(class_names)})()


def _build_class_weights(
    y_train: np.ndarray,
    n_classes: int,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    if mode == "none":
        return torch.ones(n_classes, dtype=torch.float32, device=device)
    if mode not in ("balanced", "balanced_sqrt"):
        raise ValueError(f"Unknown class_weight mode: {mode!r}")
    w = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    if mode == "balanced_sqrt":
        w = np.sqrt(w)
    return torch.tensor(w, dtype=torch.float32, device=device)


def artifact_paths(artifact_basename: str) -> tuple[str, str]:
    """e.g. mlp -> models/mlp.pt, models/mlp_meta.joblib"""
    base = artifact_basename.strip()
    if not base:
        raise ValueError("artifact_basename must be non-empty")
    pt = os.path.join(MODELS_DIR, f"{base}.pt")
    meta = os.path.join(MODELS_DIR, f"{base}_meta.joblib")
    return pt, meta


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
    *,
    artifact_basename: str = "mlp",
    class_weight_mode: str = "balanced",
    selection_metric: str = "val_loss",
    min_benign_recall: float | None = None,
    max_benign_fpr: float | None = None,
) -> MLPClassifierWrapper:
    """
    Train SmallMLP and save weights under models/{artifact_basename}.pt
    (does not touch mlp_baseline.pt unless artifact_basename is that name).

    class_weight_mode:
      - "balanced" — sklearn balanced weights (unchanged default behavior)
      - "balanced_sqrt" — sqrt(balanced) to soften minority emphasis
      - "none" — uniform class weights (1.0 each)

    selection_metric: "val_loss" (minimize) or "macro_f1" (maximize).
    For macro_f1, optional min_benign_recall / max_benign_fpr gate which epochs
    can become the best checkpoint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _make_device()
    n_classes = len(class_names)
    input_dim = X_train.shape[1]
    le_proxy = _le_proxy(class_names)

    class_weights = _build_class_weights(y_train, n_classes, class_weight_mode, device)

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
    y_val_np = np.asarray(y_val)

    best_val_loss = float("inf")
    best_macro_f1 = float("-inf")
    best_state = None
    stale = 0
    t0 = time.perf_counter()

    sel = selection_metric.lower().strip()
    if sel not in ("val_loss", "macro_f1"):
        raise ValueError(f"selection_metric must be 'val_loss' or 'macro_f1', got {sel!r}")

    def epoch_eligible(m: dict) -> bool:
        if min_benign_recall is not None:
            br = m["benign_recall"]
            if not np.isnan(br) and br < min_benign_recall:
                return False
        if max_benign_fpr is not None:
            bf = m["benign_fpr"]
            if not np.isnan(bf) and bf > max_benign_fpr:
                return False
        return True

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
            pred_idx = v_logits.argmax(dim=1).cpu().numpy()
            vm = security_metrics_dict(y_val_np, pred_idx, le_proxy)
            val_acc = float(np.mean(pred_idx == y_val_np))

        improved = False
        if sel == "val_loss":
            if v_loss < best_val_loss - 1e-6:
                best_val_loss = v_loss
                improved = True
        else:
            score = vm["macro_f1"]
            if epoch_eligible(vm) and score > best_macro_f1 + 1e-6:
                best_macro_f1 = score
                improved = True

        if improved:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        br = vm["benign_recall"]
        print(
            f"  MLP epoch {epoch + 1}/{max_epochs}  "
            f"val_loss={v_loss:.5f}  val_acc={val_acc:.4f}  "
            f"macro_f1={vm['macro_f1']:.4f}  w_f1={vm['weighted_f1']:.4f}  "
            f"benign_rec={br:.4f}  benign_fpr={vm['benign_fpr']:.4f}"
        )
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
        class_weight_mode=class_weight_mode,
        lr=lr,
        selection_metric=sel,
    )

    weights_path, meta_path = artifact_paths(artifact_basename)
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    joblib.dump(meta, meta_path)
    print(f"  Saved: {weights_path}")
    print(f"  Saved: {meta_path}")

    wrapper = MLPClassifierWrapper(model, meta, device)
    return wrapper


def load_mlp_wrapper(
    weights_path: str | None = None,
    meta_path: str | None = None,
) -> MLPClassifierWrapper:
    wp = weights_path or MLP_WEIGHTS
    mp = meta_path or MLP_META
    meta: MLPMeta = joblib.load(mp)
    for attr, default in (
        ("class_weight_mode", "balanced"),
        ("lr", 1e-3),
        ("selection_metric", "val_loss"),
    ):
        if not hasattr(meta, attr):
            setattr(meta, attr, default)
    device = _make_device()
    model = SmallMLP(
        meta.input_dim,
        meta.num_classes,
        hidden_dims=tuple(meta.hidden_dims),
        dropout=meta.dropout,
    )
    state = torch.load(wp, map_location=device)
    if isinstance(state, dict) and state and "net.0.weight" not in state:
        raise RuntimeError(
            f"Checkpoint {wp!r} is not a SmallMLP (expected keys like net.0.weight). "
            "Replace it or retrain: python src/train.py --model mlp"
        )
    model.load_state_dict(state)
    model.to(device)
    return MLPClassifierWrapper(model, meta, device)
