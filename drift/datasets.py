"""Real IIoT / network-intrusion streams (Papers 2 and 3) with optional injected concept drift.

Data: the sampled IoTID20 and CICIDS2017 files published with Paper 2 (Yang & Shami, MSANA, MIT licence):
https://github.com/Western-OC2-Lab/MSANA-Online-Data-Stream-Analytics-And-Concept-Drift-Adaptation/tree/main/Data
In file order the streams drift naturally (CICIDS2017 strongly), but there the drift positions are unknown.

Real datasets carry no drift labels, so for F1/delay we inject drift at known positions: at each onset a few of the
most predictive features swap columns with other features (like a sensor/firmware remapping). P(y|x) as seen by the
model changes while the data stays real. `drift.adaptive` shuffles the rows before injecting, so the
injected drifts are the only drifts. Drift 1 and 2 are abrupt, drift 3 is gradual (rows switch over `GRADUAL`)."""
import os
import numpy as np, pandas as pd
import lightgbm as lgb

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FILES = {"iotid20": "IoT_2020_b_0.01.csv", "cicids2017": "cic_0.005km_78.csv"}
ID_COLS = ["Flow_ID", "Src_IP", "Dst_IP", "Timestamp"]   # identifiers / time, not behaviour
GRADUAL, N_SWAP = 500, 5


def load(name):
    df = pd.read_csv(os.path.join(DATA, FILES[name]))
    y = df.pop("Label").astype(int).values
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    X = df.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float).values
    return X, y, list(df.columns)


def inject_drift(X, y, init, seed=0):
    """Swap the N_SWAP most important features (by a LightGBM fit on the first `init` rows) with random other
    columns at three onsets spread over the post-training part of the stream. Swaps accumulate."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    onsets = [int(init + (n - init) * f) for f in (0.25, 0.5, 0.75)]
    widths = [1, 1, GRADUAL]
    imp = lgb.LGBMClassifier(n_estimators=80, num_leaves=15, verbose=-1, n_jobs=1).fit(X[:init], y[:init]).feature_importances_
    top = np.argsort(-imp)[:N_SWAP]
    perm, perms = np.arange(d), []
    for _ in onsets:   # a new remapping per drift
        p = perm.copy()
        for f in top:
            g = rng.choice([c for c in range(d) if c not in top])
            p[f], p[g] = p[g], p[f]
        perms.append(p); perm = p
    Xd = X.copy()
    for k, (t, w) in enumerate(zip(onsets, widths)):
        end = onsets[k + 1] if k + 1 < len(onsets) else n
        rows = np.arange(t, end)
        if w > 1:   # gradual: row belongs to the new concept with linearly rising probability
            rows = rows[rng.random(len(rows)) < np.clip((rows - t) / w, 0, 1)]
        Xd[rows] = X[rows][:, perms[k]]
        if w > 1 and k > 0:   # rows not yet switched stay on the previous concept
            old = np.setdiff1d(np.arange(t, end), rows)
            Xd[old] = X[old][:, perms[k - 1]]
    return Xd, y, dict(onsets=onsets, width=widths)
