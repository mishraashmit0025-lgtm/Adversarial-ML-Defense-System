"""Classifiers: a PyTorch MLP with an sklearn-style API, and tree models."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from torch import nn


class FlowScaler:
    """log1p on non-negative heavy-tailed features, then standardization. Differentiable in torch."""

    def fit(self, X: np.ndarray) -> "FlowScaler":
        self.shift = np.minimum(X.min(axis=0), 0.0)  # e.g. Init_Win_bytes can be -1
        Z = np.log1p(X - self.shift)
        self.mu, self.sd = Z.mean(axis=0), Z.std(axis=0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.log1p(np.maximum(X - self.shift, 0)) - self.mu) / self.sd

    def torch_transform(self, X: torch.Tensor) -> torch.Tensor:
        shift, mu, sd = (torch.as_tensor(a, dtype=X.dtype) for a in (self.shift, self.mu, self.sd))
        return (torch.log1p(torch.clamp(X - shift, min=0)) - mu) / sd


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden=(128, 64), dropout=0.1):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchFlowClassifier:
    """MLP on raw flow features (scaling happens inside, so gradients are w.r.t. raw features).

    ``adversary`` (optional) is called each epoch as ``adversary(model, Xb_raw, yb)`` and
    returns perturbed raw inputs used for adversarial training.
    """

    def __init__(self, epochs: int = 25, lr: float = 2e-3, batch: int = 256, seed: int = 0,
                 adversary=None, adv_weight: float = 0.5):
        self.epochs, self.lr, self.batch, self.seed = epochs, lr, batch, seed
        self.adversary, self.adv_weight = adversary, adv_weight

    def logits_raw(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(self.scaler.torch_transform(X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchFlowClassifier":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.scaler = FlowScaler().fit(X)
        self.model = MLP(X.shape[1])
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        pos_weight = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        for _ in range(self.epochs):
            self.model.train()
            for idx in np.array_split(rng.permutation(len(X)), max(1, len(X) // self.batch)):
                xb, yb = Xt[idx], yt[idx]
                loss = lossf(self.logits_raw(xb), yb)
                if self.adversary is not None:
                    self.model.eval()
                    xa = torch.tensor(self.adversary(self, xb.numpy().astype(np.float64), yb.numpy().astype(int)),
                                      dtype=torch.float32)
                    self.model.train()
                    loss = (1 - self.adv_weight) * loss + self.adv_weight * lossf(self.logits_raw(xa), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.model.eval()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            p = torch.sigmoid(self.logits_raw(torch.tensor(X, dtype=torch.float32))).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def random_forest(seed: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=seed)


def gradient_boosting(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=seed)
