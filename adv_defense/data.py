"""Flow-feature data: CICIDS 2017 loader and a CICIDS-style synthetic generator.

Feature names follow the CICFlowMeter columns used in CICIDS 2017
("MachineLearningCVE" CSVs), so a model trained on either source uses the same
schema. Derived features (rates and averages) are always recomputed from the
base features with ``recompute_derived`` so that data, and any adversarial
perturbation of it, stays physically consistent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "Destination Port",
    "Flow Duration",  # microseconds
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow IAT Std",
    "SYN Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Idle Mean",
]
DERIVED_FEATURES = [
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Average Packet Size",
    "Down/Up Ratio",
]
FEATURES = BASE_FEATURES + DERIVED_FEATURES

# What an attacker controlling the malicious endpoint can change without breaking the attack:
# pad payloads, add delay, send extra (e.g. keep-alive) packets. All changes can only *increase* values.
MUTABLE_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Flow IAT Std",
    "PSH Flag Count",
    "Idle Mean",
]
INTEGER_FEATURES = [
    "Destination Port", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "SYN Flag Count", "ACK Flag Count", "PSH Flag Count",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
]


def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    fwd, bwd = d["Total Fwd Packets"].clip(lower=1), d["Total Backward Packets"]
    dur_s = d["Flow Duration"].clip(lower=1) / 1e6
    pkts = fwd + bwd
    byts = d["Total Length of Fwd Packets"] + d["Total Length of Bwd Packets"]
    d["Fwd Packet Length Mean"] = d["Total Length of Fwd Packets"] / fwd
    d["Bwd Packet Length Mean"] = np.where(bwd > 0, d["Total Length of Bwd Packets"] / bwd.clip(lower=1), 0.0)
    d["Flow Bytes/s"] = byts / dur_s
    d["Flow Packets/s"] = pkts / dur_s
    d["Flow IAT Mean"] = d["Flow Duration"] / (pkts - 1).clip(lower=1)
    d["Average Packet Size"] = byts / pkts.clip(lower=1)
    d["Down/Up Ratio"] = bwd / fwd
    return d


CLASS_PROFILES = {
    # name: (share, port choices, duration lognormal (mu, sigma), fwd pkts (lam), bwd pkts (lam),
    #        fwd bytes/pkt (mu, sd), bwd bytes/pkt (mu, sd), syn p, ack p, psh p, win fwd, win bwd)
    "BENIGN": (0.70, [80, 443, 53, 22, 8080, 123], (12.0, 1.8), 10, 9, (320, 150), (500, 400), 0.9, 0.85, 0.6, 29200, 28960),
    "DoS": (0.08, [80, 443], (14.0, 0.9), 5, 4, (60, 30), (350, 250), 1.0, 0.8, 0.5, 29200, 28960),
    "DDoS": (0.07, [80], (10.5, 1.2), 3, 4, (25, 15), (700, 500), 1.0, 0.6, 0.4, 8192, 28960),
    "PortScan": (0.08, list(range(1, 1025)), (5.0, 1.2), 1.2, 1.0, (4, 4), (8, 6), 1.0, 0.2, 0.05, 1024, 0),
    "Bot": (0.03, [8080, 443], (11.5, 1.5), 5, 5, (120, 60), (300, 200), 0.9, 0.9, 0.6, 8192, 8192),
    "BruteForce": (0.04, [21, 22], (13.0, 0.9), 16, 15, (45, 25), (200, 120), 1.0, 1.0, 0.7, 29200, 28960),
}


def synthetic_flows(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """CICIDS-2017-style labelled flows with overlapping class distributions."""
    rng = np.random.default_rng(seed)
    names = list(CLASS_PROFILES)
    shares = np.array([CLASS_PROFILES[k][0] for k in names])
    labels = rng.choice(names, size=n, p=shares / shares.sum())
    rows = []
    for lab in names:
        m = int((labels == lab).sum())
        if m == 0:
            continue
        _, ports, (dmu, dsd), lf, lb, (fm, fs), (bm, bs), psyn, pack, ppsh, wf, wb = CLASS_PROFILES[lab]
        fwd = 1 + rng.poisson(lf, m)
        bwd = rng.poisson(lb, m)
        dur = np.exp(rng.normal(dmu, dsd, m))
        blend = rng.random(m) < 0.06  # a slice of every class looks like generic web traffic
        dur[blend] = np.exp(rng.normal(11.0, 2.2, blend.sum()))
        df = pd.DataFrame({
            "Destination Port": rng.choice(ports, m),
            "Flow Duration": dur.round(),
            "Total Fwd Packets": fwd,
            "Total Backward Packets": bwd,
            "Total Length of Fwd Packets": (fwd * np.abs(rng.normal(fm, fs, m))).round(),
            "Total Length of Bwd Packets": (bwd * np.abs(rng.normal(bm, bs, m))).round(),
            "Flow IAT Std": np.abs(dur / (fwd + bwd) * rng.gamma(1.5, 0.6, m)),
            "SYN Flag Count": rng.binomial(1, psyn, m),
            "ACK Flag Count": rng.binomial(1, pack, m),
            "PSH Flag Count": rng.binomial(1, ppsh, m),
            "Init_Win_bytes_forward": np.where(rng.random(m) < 0.65, wf, rng.choice([-1, 256, 1024, 8192, 29200, 65535], m)),
            "Init_Win_bytes_backward": np.where(rng.random(m) < 0.65, wb, rng.choice([-1, 0, 229, 8192, 28960, 65535], m)),
            "Idle Mean": np.where(rng.random(m) < 0.3, np.exp(rng.normal(14, 1.5, m)), 0.0),
            "Label": lab,
        })
        rows.append(df)
    out = recompute_derived(pd.concat(rows, ignore_index=True))
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)[FEATURES + ["Label"]]


def load_cicids2017(path: str | Path, max_rows_per_file: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Load the CICIDS 2017 MachineLearningCVE CSVs from a directory (or one CSV).

    Download from https://www.unb.ca/cic/datasets/ids-2017.html. Column names are
    stripped, inf/NaN rows dropped, label families merged (e.g. all "DoS *" -> DoS),
    and derived features recomputed so the schema matches ``FEATURES``.
    """
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no CSV files under {path}")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        if max_rows_per_file and len(df) > max_rows_per_file:
            df = df.sample(max_rows_per_file, random_state=seed)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in BASE_FEATURES + ["Label"] if c not in df.columns]
    if missing:
        raise ValueError(f"not a CICIDS 2017 flow file, missing columns: {missing}")
    df = df[BASE_FEATURES + ["Label"]].replace([np.inf, -np.inf], np.nan).dropna()
    lab = df["Label"].astype(str).str.strip()
    fam = np.select(
        [lab.eq("BENIGN"), lab.str.startswith("DoS") & ~lab.str.contains("DDoS"), lab.eq("DDoS"), lab.eq("PortScan"),
         lab.eq("Bot"), lab.isin(["FTP-Patator", "SSH-Patator"]) | lab.str.contains("Brute Force")],
        ["BENIGN", "DoS", "DDoS", "PortScan", "Bot", "BruteForce"], default="Other",
    )
    df["Label"] = fam
    return recompute_derived(df).reset_index(drop=True)[FEATURES + ["Label"]]


def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Features as float32 and binary target (1 = attack)."""
    return df[FEATURES].to_numpy(np.float64), (df["Label"] != "BENIGN").to_numpy(int)
