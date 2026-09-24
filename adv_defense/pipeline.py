"""End-to-end experiment: train baselines + defense, attack them, report."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from .attacks import greedy_blackbox_attack, pgd_attack
from .data import load_cicids2017, synthetic_flows, to_xy
from .defense import Calibrated, EnsembleDefense
from .explain import attribution_shift, global_importance
from .models import TorchFlowClassifier, random_forest


def rates(pred: np.ndarray, y: np.ndarray) -> dict:
    return {
        "detection_rate": float(pred[y == 1].mean()) if (y == 1).any() else None,
        "false_alarm_rate": float(pred[y == 0].mean()) if (y == 0).any() else None,
    }


def run(data: str | None = None, n: int = 40000, eps: float = 1.0, n_attack: int = 1000, seed: int = 0,
        out_dir: str = "results", max_rows_per_file: int | None = 50000, mlp_epochs: int = 20,
        max_false_alarm: float = 0.003) -> dict:
    t0 = time.time()
    df = load_cicids2017(data, max_rows_per_file, seed) if data else synthetic_flows(n, seed)
    X, y = to_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_tr, y_tr, test_size=0.3, stratify=y_tr, random_state=seed)

    mlp = TorchFlowClassifier(epochs=mlp_epochs, seed=seed).fit(X_tr, y_tr)
    rf = random_forest(seed).fit(X_tr, y_tr)
    Xb_cal = X_cal[y_cal == 0]
    defense = EnsembleDefense(max_false_alarm, adv_eps=eps, mlp_epochs=mlp_epochs, seed=seed).fit(X_tr, y_tr, Xb_cal)
    # every system operates at the same false-alarm budget, calibrated on held-out benign flows
    mlp_c, rf_c = Calibrated(mlp, Xb_cal, max_false_alarm), Calibrated(rf, Xb_cal, max_false_alarm)

    rng = np.random.default_rng(seed)
    mal = np.where(y_te == 1)[0]
    A = X_te[rng.choice(mal, size=min(n_attack, len(mal)), replace=False)]

    adv_mlp = pgd_attack(mlp, A, eps=eps, steps=30, seed=seed)  # white-box vs undefended MLP
    adv_member = pgd_attack(defense.members["adv_mlp"], A, eps=eps, steps=30, seed=seed)  # white-box vs defense's MLP
    adv_rf_bb = greedy_blackbox_attack(rf.predict_proba, A, eps=eps)  # query attack vs RF
    adv_def_bb = greedy_blackbox_attack(defense.predict_proba, A, eps=eps)  # query attack vs full defense

    systems = {"mlp (undefended)": mlp_c.predict, "random_forest": rf_c.predict, "ensemble_defense": defense.predict}
    attacks = {
        "clean": A,
        "pgd_on_mlp": adv_mlp,
        "pgd_on_defense_mlp": adv_member,
        "blackbox_on_rf": adv_rf_bb,
        "blackbox_on_defense": adv_def_bb,
    }
    report: dict = {"config": dict(data=data or f"synthetic(n={n})", eps=eps, n_attack=len(A), seed=seed,
                                   max_false_alarm=max_false_alarm,
                                   n_train=len(X_tr), n_cal=len(X_cal), n_test=len(X_te)),
                    "clean_test": {}, "adversarial_detection_rate": {}}
    for name, pred in systems.items():
        report["clean_test"][name] = rates(pred(X_te), y_te)
        report["adversarial_detection_rate"][name] = {k: float(pred(v).mean()) for k, v in attacks.items()}
    d = defense.decide(X_te)
    report["defense_flag_reasons"] = {r: int((d.reason == r).sum()) for r in ("vote", "member", "novelty")}
    report["calibration"] = {"vote_threshold": defense.vote_threshold, "member_threshold": defense.member_threshold,
                             "novelty_threshold": defense.novelty_threshold, "mlp_threshold": mlp_c.threshold,
                             "rf_threshold": rf_c.threshold}
    report["shap_top_features"] = global_importance(rf, X_te, n=400, seed=seed).head(8).to_dict(orient="records")
    report["shap_features_exploited_by_attack"] = attribution_shift(rf, A[:200], adv_rf_bb[:200]).head(5).to_dict(orient="records")
    report["seconds"] = round(time.time() - t0, 1)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(to_markdown(report))
    joblib.dump(defense, out / "defense.joblib")
    return report


def to_markdown(r: dict) -> str:
    pct = lambda v: "-" if v is None else f"{100 * v:.2f}%"
    lines = [f"# Results ({r['config']['data']}, eps={r['config']['eps']}, "
             f"false-alarm budget {100 * r['config']['max_false_alarm']:.1f}%)", "",
             "## Clean test set", "", "| system | detection rate | false alarm rate |", "|---|---|---|"]
    for k, v in r["clean_test"].items():
        lines.append(f"| {k} | {pct(v['detection_rate'])} | {pct(v['false_alarm_rate'])} |")
    atk = list(next(iter(r["adversarial_detection_rate"].values())))
    lines += ["", f"## Detection rate on {r['config']['n_attack']} malicious flows under attack", "",
              "| system | " + " | ".join(atk) + " |", "|---|" + "---|" * len(atk)]
    for k, v in r["adversarial_detection_rate"].items():
        lines.append(f"| {k} | " + " | ".join(pct(v[a]) for a in atk) + " |")
    lines += ["", "## Top SHAP features (random forest)", ""]
    lines += [f"- {f['feature']}: {f['mean_abs_shap']:.4f}" for f in r["shap_top_features"]]
    return "\n".join(lines) + "\n"
