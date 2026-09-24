import joblib
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from adv_defense import service
from adv_defense.data import BASE_FEATURES, synthetic_flows, to_xy
from adv_defense.defense import EnsembleDefense


def test_predict_endpoint(tmp_path, monkeypatch):
    df = synthetic_flows(3000, seed=4)
    X, y = to_xy(df)
    X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    d = EnsembleDefense(0.01, mlp_epochs=2).fit(X_tr, y_tr, X_cal[y_cal == 0])
    path = tmp_path / "defense.joblib"
    joblib.dump(d, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    service.get_defense.cache_clear()

    client = TestClient(service.app)
    flows = df[BASE_FEATURES].head(5).to_dict(orient="records")
    r = client.post("/predict", json={"flows": flows, "explain": True})
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert len(res) == 5
    assert all(0 <= x["vote"] <= 1 and len(x["explanation"]) == 5 for x in res)

    bad = client.post("/predict", json={"flows": [{"Flow Duration": 1.0}]})
    assert bad.status_code == 422
