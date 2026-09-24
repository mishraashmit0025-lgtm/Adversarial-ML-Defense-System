# Adversarial ML Defense System

A network-intrusion detector for IoT/enterprise flow data that is built to hold up against **adversarial evasion**: attackers who tweak their traffic so a machine-learning detector labels it benign.

- **Data**: CICFlowMeter flow features in the **CICIDS 2017** schema. It includes a loader for the official `MachineLearningCVE` CSVs and a CICIDS-style synthetic generator, so the whole pipeline runs without the multi-GB download.
- **Realistic threat model**: the attacker controls only what the malicious endpoint can change (payload padding, delays, extra packets). Those features can only *increase*, by at most a factor `1 + eps`. After every step, integer counts are rounded and derived features (rates, averages) are recomputed, so **every adversarial example is a physically valid flow**.
- **Attacks**: white-box PGD/FGSM on a PyTorch MLP (with gradients through the feature scaler), and a black-box query attack that works against any model, including tree ensembles and the full defense.
- **Defense** (`adv_defense/defense.py`), in three layers:
  1. **Diverse ensemble**: soft vote of a RandomForest, a HistGradientBoosting model and a **PGD-adversarially-trained** PyTorch MLP.
  2. **Member check**: flags a flow when any single member is near-certain it is an attack.
  3. **Novelty check**: an IsolationForest trained on benign traffic only flags evasive flows pushed into unusual regions.

  Thresholds are calibrated on held-out benign flows so the combined **false-alarm rate stays within a budget** (default 0.3%).
- **Explainability**: SHAP TreeExplainer shows global feature importance, per-flow explanations in the API, and *attribution shift*, i.e. which features an attack exploited.
- **Docker**: the image trains a model at build time and serves it through FastAPI.

## Results (synthetic CICIDS-style data, seed 0)

`python -m adv_defense` trains everything, runs the attacks and writes `results/report.md`. All systems are calibrated to the **same 0.3% false-alarm budget** so the comparison is fair. eps = 1.0 means each attacker-controlled feature can at most double.

| system | clean detection | false alarms | PGD (white-box on MLP) | PGD on defense's own MLP | black-box vs RF | black-box vs **full defense** |
|---|---|---|---|---|---|---|
| MLP (undefended) | 85.2% | 0.25% | 65.4% | 65.1% | 57.9% | 60.4% |
| Random forest | 93.2% | 0.35% | 69.1% | 69.3% | 61.8% | 66.2% |
| **Ensemble defense** | 92.4% | 0.35% | **76.9%** | **75.8%** | **71.1%** | **69.9%** |

Under every attack the defense keeps **7 to 11 points more detection** than the best single model, at the same false-alarm rate. The black-box attack against the full defense is *adaptive*: it queries the defense itself.

> These numbers come from **synthetic** flows whose class overlap was tuned so attacks are non-trivial. They are not CICIDS 2017 results. To measure on the real dataset:
>
> ```bash
> # download MachineLearningCVE.zip from https://www.unb.ca/cic/datasets/ids-2017.html and unzip to data/
> python -m adv_defense --data data/MachineLearningCVE --max-rows-per-file 100000
> ```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m adv_defense                  # full experiment on synthetic data (~2 min on CPU)
python -m pytest -q                    # tests

uvicorn adv_defense.service:app        # serve results/defense.joblib
```

Docker:

```bash
docker build -t adv-defense .
docker run -p 8000:8000 adv-defense
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"explain": true, "flows": [{"Destination Port": 80, "Flow Duration": 1200000, "Total Fwd Packets": 6,
       "Total Backward Packets": 4, "Total Length of Fwd Packets": 360, "Total Length of Bwd Packets": 1400,
       "Flow IAT Std": 90000, "SYN Flag Count": 1, "ACK Flag Count": 1, "PSH Flag Count": 1,
       "Init_Win_bytes_forward": 29200, "Init_Win_bytes_backward": 28960, "Idle Mean": 0}]}'
```

## Layout

```
adv_defense/
  data.py       CICIDS 2017 loader, synthetic generator, feature schema, derived-feature recomputation
  models.py     PyTorch MLP (sklearn-style API, differentiable scaler) + tree models
  attacks.py    feasibility projection, PGD/FGSM, black-box query attack, adversarial-training hook
  defense.py    ensemble + member/novelty layers + false-alarm-budget calibration
  explain.py    SHAP global importance, per-flow explanations, attribution shift
  pipeline.py   end-to-end experiment and report
  service.py    FastAPI inference service
tests/          feasibility of adversarial examples, attack/defense behavior, calibration, SHAP, loader, API
```
