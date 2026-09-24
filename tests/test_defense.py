import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from adv_defense.attacks import MUT_IDX, greedy_blackbox_attack, pgd_attack, project
from adv_defense.data import (
    BASE_FEATURES, FEATURES, INTEGER_FEATURES, load_cicids2017, recompute_derived, synthetic_flows, to_xy,
)
from adv_defense.defense import Calibrated, EnsembleDefense
from adv_defense.explain import global_importance
from adv_defense.models import TorchFlowClassifier, random_forest

EPS = 1.0


@pytest.fixture(scope="module")
def split():
    X, y = to_xy(synthetic_flows(8000, seed=1))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_tr, y_tr, test_size=0.3, stratify=y_tr, random_state=0)
    return X_tr, y_tr, X_cal[y_cal == 0], X_te, y_te


@pytest.fixture(scope="module")
def mlp(split):
    X_tr, y_tr, *_ = split
    return TorchFlowClassifier(epochs=6).fit(X_tr, y_tr)


def test_synthetic_schema_and_consistency():
    df = synthetic_flows(500, seed=2)
    assert list(df.columns) == FEATURES + ["Label"]
    again = recompute_derived(df)
    np.testing.assert_allclose(again[FEATURES].to_numpy(), df[FEATURES].to_numpy())
    assert (df["Label"] == "BENIGN").mean() > 0.5


def test_projection_produces_valid_flows(split):
    *_, X_te, y_te = split
    X0 = X_te[y_te == 1][:200]
    rng = np.random.default_rng(0)
    wild = X0 + rng.normal(0, 1, X0.shape) * np.abs(X0) * 3  # arbitrary junk, incl. decreases
    P = project(wild, X0, EPS)
    immut = np.setdiff1d(np.arange(len(FEATURES)), MUT_IDX)
    base_immut = [i for i in immut if FEATURES[i] in BASE_FEATURES]
    np.testing.assert_array_equal(P[:, base_immut], X0[:, base_immut])  # untouchable features unchanged
    assert np.all(P[:, MUT_IDX] >= X0[:, MUT_IDX] - 1e-9)  # only increases
    assert np.all(P[:, MUT_IDX] <= X0[:, MUT_IDX] + EPS * np.maximum(np.abs(X0[:, MUT_IDX]), 1) + 1e-6)
    ints = [FEATURES.index(f) for f in INTEGER_FEATURES]
    np.testing.assert_array_equal(P[:, ints], np.round(P[:, ints]))
    df = pd.DataFrame(P, columns=FEATURES)
    np.testing.assert_allclose(recompute_derived(df)[FEATURES].to_numpy(), P)  # derived features consistent


def test_pgd_reduces_attack_probability(split, mlp):
    *_, X_te, y_te = split
    A = X_te[y_te == 1][:300]
    adv = pgd_attack(mlp, A, eps=EPS, steps=15)
    assert mlp.predict_proba(adv)[:, 1].mean() < mlp.predict_proba(A)[:, 1].mean() - 0.02


def test_blackbox_never_increases_score(split):
    X_tr, y_tr, _, X_te, y_te = split
    rf = random_forest().fit(X_tr, y_tr)
    A = X_te[y_te == 1][:200]
    adv = greedy_blackbox_attack(rf.predict_proba, A, eps=EPS)
    assert np.all(rf.predict_proba(adv)[:, 1] <= rf.predict_proba(A)[:, 1] + 1e-12)


def test_calibrated_false_alarm(split, mlp):
    _, _, Xb_cal, X_te, y_te = split
    c = Calibrated(mlp, Xb_cal, 0.01)
    assert c.predict(Xb_cal).mean() <= 0.011
    assert c.predict(X_te[y_te == 0]).mean() < 0.03


def test_defense_is_calibrated_and_more_robust(split, mlp):
    X_tr, y_tr, Xb_cal, X_te, y_te = split
    budget = 0.01
    d = EnsembleDefense(budget, adv_eps=EPS, mlp_epochs=4).fit(X_tr, y_tr, Xb_cal)
    assert d.predict(Xb_cal).mean() <= budget + 1e-9
    assert d.predict(X_te[y_te == 0]).mean() < 3 * budget
    A = X_te[y_te == 1][:300]
    adv = pgd_attack(mlp, A, eps=EPS, steps=15)
    base = Calibrated(mlp, Xb_cal, budget)
    assert d.predict(adv).mean() >= base.predict(adv).mean()
    reasons = set(d.decide(X_te).reason)
    assert "vote" in reasons


def test_shap_importance(split):
    X_tr, y_tr, _, X_te, _ = split
    rf = random_forest().fit(X_tr, y_tr)
    imp = global_importance(rf, X_te, n=100)
    assert list(imp.columns) == ["feature", "mean_abs_shap"]
    assert len(imp) == len(FEATURES) and imp["mean_abs_shap"].iloc[0] > 0


def test_cicids_loader(tmp_path):
    df = synthetic_flows(300, seed=3)
    raw = df[BASE_FEATURES + ["Label"]].copy()
    raw["Label"] = raw["Label"].replace({"DoS": "DoS Hulk", "BruteForce": "SSH-Patator"})
    raw.loc[raw.index[0], "Flow Duration"] = np.inf
    raw.columns = [" " + c for c in raw.columns]  # CICIDS headers have leading spaces
    raw.to_csv(tmp_path / "Monday-WorkingHours.pcap_ISCX.csv", index=False)
    out = load_cicids2017(tmp_path)
    assert len(out) == 299
    assert set(out["Label"]) <= {"BENIGN", "DoS", "DDoS", "PortScan", "Bot", "BruteForce"}
    assert list(out.columns) == FEATURES + ["Label"]
