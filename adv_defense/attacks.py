"""Constrained evasion attacks on flow features.

The attacker starts from a malicious flow and tries to make the detector call it
benign. Perturbations are limited to features the attacker controls
(``MUTABLE_FEATURES``), may only increase them (padding, delay, extra packets),
by at most a factor ``1 + eps``. After every step, integer features are rounded
and derived features recomputed, so every adversarial example is a valid flow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data import FEATURES, INTEGER_FEATURES, MUTABLE_FEATURES, recompute_derived

MUT_IDX = np.array([FEATURES.index(f) for f in MUTABLE_FEATURES])
INT_IDX = np.array([FEATURES.index(f) for f in INTEGER_FEATURES])


def project(X_adv: np.ndarray, X0: np.ndarray, eps: float) -> np.ndarray:
    """Map an arbitrary candidate back onto the feasible set around X0."""
    out = X0.copy()
    lo = X0[:, MUT_IDX]
    hi = lo + eps * np.maximum(np.abs(lo), 1.0)  # zero-valued features may grow to eps (e.g. 1 PSH flag)
    out[:, MUT_IDX] = np.clip(X_adv[:, MUT_IDX], lo, hi)
    out[:, INT_IDX] = np.floor(out[:, INT_IDX] + 1e-9)
    out[:, INT_IDX] = np.maximum(out[:, INT_IDX], X0[:, INT_IDX])
    df = recompute_derived(pd.DataFrame(out, columns=FEATURES))
    return df[FEATURES].to_numpy(np.float64)


def pgd_attack(clf, X0: np.ndarray, eps: float = 0.5, steps: int = 20, step_frac: float = 0.15,
               random_start: bool = True, seed: int = 0) -> np.ndarray:
    """Projected gradient descent on the benign-vs-attack logit of a TorchFlowClassifier.

    Steps are taken in log space for each mutable feature (scale-free), which suits
    features spanning many orders of magnitude.
    """
    rng = np.random.default_rng(seed)
    X = X0.copy()
    if random_start:
        noise = np.zeros_like(X)
        noise[:, MUT_IDX] = rng.uniform(0, eps, (len(X), len(MUT_IDX))) * np.maximum(np.abs(X0[:, MUT_IDX]), 1.0)
        X = project(X + noise, X0, eps)
    for _ in range(steps):
        xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        logit = clf.logits_raw(xt).sum()  # we want to *decrease* the attack logit
        (g,) = torch.autograd.grad(logit, xt)
        g = g.numpy().astype(np.float64)
        cand = X.copy()
        scale = np.maximum(np.abs(X0[:, MUT_IDX]), 1.0)
        cand[:, MUT_IDX] = X[:, MUT_IDX] - step_frac * eps * scale * np.sign(g[:, MUT_IDX])
        X = project(cand, X0, eps)
    return X


def fgsm_attack(clf, X0: np.ndarray, eps: float = 0.5) -> np.ndarray:
    return pgd_attack(clf, X0, eps=eps, steps=1, step_frac=1.0, random_start=False)


def greedy_blackbox_attack(predict_proba, X0: np.ndarray, eps: float = 0.5, grid: int = 4) -> np.ndarray:
    """Model-agnostic coordinate search (works on trees / ensembles): for each mutable
    feature try a few increases and keep the one that lowers P(attack) the most."""
    X = X0.copy()
    best = predict_proba(X)[:, 1]
    for j in MUT_IDX:
        for frac in np.linspace(eps / grid, eps, grid):
            cand = X.copy()
            cand[:, j] = X0[:, j] + frac * np.maximum(np.abs(X0[:, j]), 1.0)
            cand = project(cand, X0, eps)
            p = predict_proba(cand)[:, 1]
            better = p < best
            X[better], best[better] = cand[better], p[better]
    return X


class pgd_adversary:
    """Adversary hook for adversarial training: perturb only the malicious rows (picklable)."""

    def __init__(self, eps: float = 0.5, steps: int = 5):
        self.eps, self.steps, self.calls = eps, steps, 0

    def __call__(self, clf, Xb: np.ndarray, yb: np.ndarray) -> np.ndarray:
        self.calls += 1
        out = Xb.copy()
        m = yb == 1
        if m.any():
            out[m] = pgd_attack(clf, Xb[m], eps=self.eps, steps=self.steps, seed=self.calls)
        return out
