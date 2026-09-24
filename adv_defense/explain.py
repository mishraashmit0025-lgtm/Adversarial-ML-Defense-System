"""SHAP explanations for the tree members of the ensemble."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .data import FEATURES


def _attack_class(values) -> np.ndarray:
    v = np.asarray(values)
    if v.ndim == 3:  # (n, features, classes)
        return v[:, :, 1]
    return v


def global_importance(model, X: np.ndarray, n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Mean |SHAP| per feature for P(attack) on a sample of X."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    sv = _attack_class(shap.TreeExplainer(model).shap_values(X[idx]))
    imp = np.abs(sv).mean(axis=0)
    return pd.DataFrame({"feature": FEATURES, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False,
                                                                                  ignore_index=True)


def explain_one(model, x: np.ndarray, top: int = 5) -> list[dict]:
    """Top features pushing a single flow toward (+) or away from (-) the attack class."""
    sv = _attack_class(shap.TreeExplainer(model).shap_values(x.reshape(1, -1)))[0]
    order = np.argsort(-np.abs(sv))[:top]
    return [{"feature": FEATURES[i], "value": float(x[i]), "shap": float(sv[i])} for i in order]


def attribution_shift(model, X_clean: np.ndarray, X_adv: np.ndarray) -> pd.DataFrame:
    """How the attack moved attributions: which features the adversary exploited."""
    a = _attack_class(shap.TreeExplainer(model).shap_values(X_clean))
    b = _attack_class(shap.TreeExplainer(model).shap_values(X_adv))
    d = (b - a).mean(axis=0)
    return pd.DataFrame({"feature": FEATURES, "mean_shap_change": d}).sort_values("mean_shap_change", ignore_index=True)
