"""FastAPI inference service for a trained defense.

    python -m adv_defense            # trains and writes results/defense.joblib
    uvicorn adv_defense.service:app  # serves it (MODEL_PATH overrides the path)
"""
from __future__ import annotations

import os
from functools import lru_cache

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import BASE_FEATURES, FEATURES, recompute_derived
from .explain import explain_one

app = FastAPI(title="Adversarial ML Defense System", version="0.1.0")


@lru_cache(maxsize=1)
def get_defense():
    path = os.environ.get("MODEL_PATH", "results/defense.joblib")
    if not os.path.exists(path):
        raise HTTPException(503, f"model not found at {path}; run `python -m adv_defense` first")
    return joblib.load(path)


class Flows(BaseModel):
    flows: list[dict[str, float]] = Field(min_length=1, max_length=10000,
                                          description="base CICFlowMeter features per flow; derived ones are recomputed")
    explain: bool = False


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/features")
def features():
    return {"required": BASE_FEATURES, "derived": [f for f in FEATURES if f not in BASE_FEATURES]}


@app.post("/predict")
def predict(req: Flows):
    import pandas as pd

    missing = sorted({f for row in req.flows for f in BASE_FEATURES if f not in row})
    if missing:
        raise HTTPException(422, f"missing features: {missing}")
    X = recompute_derived(pd.DataFrame(req.flows)[BASE_FEATURES])[FEATURES].to_numpy(np.float64)
    d = get_defense()
    dec = d.decide(X)
    out = []
    for i in range(len(X)):
        item = {"attack": bool(dec.attack[i]), "vote": float(dec.vote[i]), "reason": dec.reason[i] or None}
        if req.explain:
            item["explanation"] = explain_one(d.members["random_forest"], X[i])
        out.append(item)
    return {"results": out}
