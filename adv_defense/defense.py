"""Ensemble defense.

Three layers:
1. Diverse members: RandomForest, HistGradientBoosting and a PGD-adversarially-trained MLP,
   combined by soft voting. Gradient attacks on the MLP transfer poorly to trees and vice versa.
2. Disagreement check: a flow is also flagged when any member is highly confident it is
   an attack. Evasion usually has to fool every member at once.
3. Novelty check: an IsolationForest fit on *benign* training flows flags inputs that do not
   look like normal traffic. Evasion pushes malicious flows into odd regions, not onto real
   benign traffic.

The thresholds of layers 2 and 3 are calibrated on held-out benign flows so the *combined*
false-alarm rate stays within ``max_false_alarm``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from .attacks import pgd_adversary
from .models import FlowScaler, TorchFlowClassifier, gradient_boosting, random_forest


@dataclass
class Decision:
    attack: np.ndarray  # final 0/1
    vote: np.ndarray  # soft-vote P(attack)
    reason: np.ndarray  # "vote" | "member" | "novelty" | ""


def threshold_for(scores: np.ndarray, max_flags: float) -> float:
    """Smallest threshold t such that at most floor(max_flags) of ``scores`` satisfy score >= t."""
    k = int(np.floor(max_flags + 1e-9))
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    if k >= len(s):
        return -np.inf
    return float(np.nextafter(s[k], np.inf))


class Calibrated:
    """Wrap any probabilistic classifier with a threshold set for a target false-alarm rate on benign data."""

    def __init__(self, model, X_cal_benign: np.ndarray, max_false_alarm: float):
        self.model = model
        self.threshold = threshold_for(model.predict_proba(X_cal_benign)[:, 1], max_false_alarm * len(X_cal_benign))

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.model.predict_proba(X)[:, 1] >= self.threshold).astype(int)


class EnsembleDefense:
    def __init__(self, max_false_alarm: float = 0.003, adv_eps: float = 0.5, mlp_epochs: int = 20, seed: int = 0):
        self.max_false_alarm, self.adv_eps, self.mlp_epochs, self.seed = max_false_alarm, adv_eps, mlp_epochs, seed

    def fit(self, X: np.ndarray, y: np.ndarray, X_cal_benign: np.ndarray) -> "EnsembleDefense":
        self.members = {
            "random_forest": random_forest(self.seed).fit(X, y),
            "grad_boosting": gradient_boosting(self.seed).fit(X, y),
            "adv_mlp": TorchFlowClassifier(epochs=self.mlp_epochs, seed=self.seed,
                                           adversary=pgd_adversary(self.adv_eps, steps=5)).fit(X, y),
        }
        self.scaler = FlowScaler().fit(X)
        self.iso = IsolationForest(n_estimators=300, random_state=self.seed).fit(self.scaler.transform(X[y == 0]))
        self._calibrate(X_cal_benign)
        return self

    def member_probas(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.members.values()])

    def novelty(self, X: np.ndarray) -> np.ndarray:
        return -self.iso.score_samples(self.scaler.transform(X))  # higher = more unusual

    def _calibrate(self, Xb: np.ndarray) -> None:
        """Spend the false-alarm budget: half on the vote, a quarter each on the member and novelty layers.

        Thresholds are set by counting flagged calibration flows, so the combined false-alarm
        rate on the calibration set never exceeds ``max_false_alarm``.
        """
        n, allowed = len(Xb), self.max_false_alarm * len(Xb)
        P = self.member_probas(Xb)
        vote = P.mean(axis=1)
        self.vote_threshold = threshold_for(vote, allowed / 2)
        flagged = vote >= self.vote_threshold
        mx = P.max(axis=1)
        self.member_threshold = threshold_for(mx[~flagged], allowed * 3 / 4 - flagged.sum())
        flagged |= mx >= self.member_threshold
        nov = self.novelty(Xb)
        self.novelty_threshold = threshold_for(nov[~flagged], allowed - flagged.sum())
        assert (flagged | (nov >= self.novelty_threshold)).sum() <= allowed + 1e-9 or n == 0

    def decide(self, X: np.ndarray) -> Decision:
        P = self.member_probas(X)
        vote = P.mean(axis=1)
        by_vote = vote >= self.vote_threshold
        by_member = ~by_vote & (P.max(axis=1) >= self.member_threshold)
        by_novelty = ~by_vote & ~by_member & (self.novelty(X) >= self.novelty_threshold)
        reason = np.where(by_vote, "vote", np.where(by_member, "member", np.where(by_novelty, "novelty", "")))
        return Decision((by_vote | by_member | by_novelty).astype(int), vote, reason)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decide(X).attack

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft score in [0, 1] consistent with ``decide`` (used by black-box attacks)."""
        d = self.decide(X)
        # map the vote onto [0, 1] so that 0.5 is the calibrated decision boundary
        t = self.vote_threshold
        v = np.where(d.vote < t, 0.5 * d.vote / max(t, 1e-9), 0.5 + 0.5 * (d.vote - t) / max(1 - t, 1e-9))
        s = np.where(d.attack == 1, np.maximum(v, 0.5 + 1e-6), np.minimum(v, 0.5 - 1e-6))
        return np.column_stack([1 - s, s])
