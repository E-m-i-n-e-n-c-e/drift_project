"""Downstream evaluation: does drift detection sustain classifier accuracy? (Paper 3 style)
Prequential test-then-train in micro-batches. Strategies: static LightGBM | drift-triggered LightGBM retrain
(per detector) | River ARF / HAT (online).
Streams: simulated IIoT (true drift), IoTID20 and CICIDS2017 with injected drift, and both real streams as-is."""
import argparse, time, os
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from river import forest, tree
from .detectors import REGISTRY
from .generator import make_industrial_stream
from .datasets import load, inject_drift
from .metrics import score

BATCH, INIT, WINDOW, WAIT = 25, 1000, 500, 300   # micro-batch, initial train size, max retrain window, wait after alarm
DETECTORS = ["DDM", "HDDM-A", "ADWIN", "EDDM", "Hybrid-OR(HDDM-A,ADWIN)"]


def fit(X, y):
    return lgb.LGBMClassifier(n_estimators=80, learning_rate=0.1, num_leaves=15, verbose=-1, n_jobs=1).fit(X, y)


def run_lgbm(X, y, det_name=None):
    """Returns per-sample predictions (from INIT on), alarm indices and retrain indices.
    After an alarm at a, wait WAIT samples, then retrain on post-alarm data only (at most WINDOW samples), so the new
    model does not learn the old concept; the detector is reset with the new model."""
    n = len(y)
    model = fit(X[:INIT], y[:INIT])
    det = REGISTRY[det_name]() if det_name else None
    pred, alarms, retrains, pending = np.zeros(n, int), [], [], None
    t0 = time.perf_counter()
    for s in range(INIT, n, BATCH):
        e = min(s + BATCH, n)
        pred[s:e] = model.predict(X[s:e])
        if det:
            for i in range(s, e):
                if det.update(int(pred[i] != y[i])):
                    alarms.append(i)
                    if pending is None:
                        pending = i
            if pending is not None and e >= pending + WAIT:
                lo = max(pending, e - WINDOW)
                if len(np.unique(y[lo:e])) < 2:   # one class only: widen to the last WINDOW samples
                    lo = e - WINDOW
                if len(np.unique(y[lo:e])) == 2:
                    model = fit(X[lo:e], y[lo:e]); retrains.append(e); det = REGISTRY[det_name]()
                pending = None
    return pred[INIT:], alarms, retrains, (time.perf_counter() - t0) / (n - INIT) * 1e3


def run_river(X, y, model):
    n, pred = len(y), np.zeros(len(y), int)
    t0 = time.perf_counter()
    for i in range(n):
        x = dict(enumerate(X[i]))
        pred[i] = model.predict_one(x) or 0
        model.learn_one(x, int(y[i]))
    return pred[INIT:], [], [], (time.perf_counter() - t0) / n * 1e3


def strategies(X, y, seed):
    runs = {"Static LightGBM": lambda: run_lgbm(X, y, None)}
    for d in DETECTORS:
        runs[f"LightGBM + {d}"] = (lambda d=d: run_lgbm(X, y, d))
    runs["River ARF (online)"] = lambda: run_river(X, y, forest.ARFClassifier(n_models=5, seed=seed))
    runs["River HAT (online)"] = lambda: run_river(X, y, tree.HoeffdingAdaptiveTreeClassifier(seed=seed))
    return runs


def stream(dataset, seed):
    """(X, y, meta); meta is None when the drift positions are unknown (real stream as-is)."""
    if dataset == "simulated":
        X, y, meta = make_industrial_stream(seed=seed)
        return X, y, dict(onsets=meta["onsets"], width=[1, 1, meta["width"]])
    name, injected = dataset.split("-")[0], not dataset.endswith("natural")
    X, y, _ = load(name)
    if not injected:
        return X, y, None
    order = np.random.default_rng(seed).permutation(len(y))   # shuffle away the natural drift: injected drift is the only drift
    return inject_drift(X[order], y[order], INIT, seed)


def evaluate(dataset, seed, curves=False):
    X, y, meta = stream(dataset, seed)
    yt = y[INIT:]
    rows, curve_rows = [], []
    for name, f in strategies(X, y, seed).items():
        pred, alarms, retrains, ms = f()
        ok = (pred == yt).astype(int)
        row = dict(dataset=dataset, seed=seed, strategy=name, accuracy=ok.mean(), f1=f1_score(yt, pred, zero_division=0),
                   alarms=len(alarms), retrains=len(retrains), ms_per_sample=ms)
        if meta:
            on = [o - INIT for o in meta["onsets"]]
            row["acc_after_1st_drift"] = ok[on[0]:].mean()
            if name.startswith("LightGBM +"):
                sc = score([a - INIT for a in alarms], on, meta["width"], len(yt))
                row.update(drift_F1=sc["f1"], drift_delay=sc["delay"])
        rows.append(row)
        if curves:
            c = pd.DataFrame(dict(t=np.arange(INIT, len(y)), strategy=name, correct=ok, alarm=0, retrain=0, true_drift=0))
            c.loc[np.array(alarms, int) - INIT, "alarm"] = 1
            c.loc[np.array(retrains, int) - INIT - 1, "retrain"] = 1
            if meta:
                c.loc[np.array(meta["onsets"]) - INIT, "true_drift"] = 1
            curve_rows.append(c)
    return rows, (pd.concat(curve_rows) if curves else None)


DATASETS = ["simulated", "iotid20", "cicids2017", "iotid20-natural", "cicids2017-natural"]
FILE = {"simulated": "", "iotid20": "_iotid20", "cicids2017": "_cicids2017",
        "iotid20-natural": "_iotid20_natural", "cicids2017-natural": "_cicids2017_natural"}


def main(out="results", seeds=10):
    os.makedirs(out, exist_ok=True)
    allrows = []
    for ds in DATASETS:
        n_seeds = 1 if ds.endswith("natural") else seeds   # natural streams: only River's seed would change
        for seed in range(n_seeds):
            rows, curves = evaluate(ds, seed, curves=(seed == 0))
            allrows += rows
            if curves is not None:
                curves.to_csv(f"{out}/downstream_curves{FILE[ds]}.csv", index=False)
            print(ds, seed, "done", flush=True)
    raw = pd.DataFrame(allrows); raw.to_csv(f"{out}/downstream_runs.csv", index=False)
    order = list(dict.fromkeys(raw.strategy))
    for ds in DATASETS:
        d = raw[raw.dataset == ds].drop(columns=["dataset", "seed"]).dropna(axis=1, how="all")
        m = d.groupby("strategy", sort=False).mean().loc[order]
        sd = d.groupby("strategy", sort=False)[["accuracy", "f1"]].std().add_suffix("_std").loc[order]
        summ = m.join(sd).round(4)
        summ.to_csv(f"{out}/downstream{FILE[ds]}.csv")
        print(f"\n== {ds}\n", summ.to_string())
    plot(out)


def plot(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    show = ["Static LightGBM", "LightGBM + ADWIN", "LightGBM + HDDM-A", "LightGBM + Hybrid-OR(HDDM-A,ADWIN)", "River HAT (online)"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for ax, ds in zip(axes, ["simulated", "iotid20", "cicids2017"]):
        c = pd.read_csv(f"{out}/downstream_curves{FILE[ds]}.csv")
        for k in show:
            v = c[c.strategy == k]
            ax.plot(v.t, v.correct.rolling(300).mean(), label=k, lw=1.1)
        for t in c[(c.strategy == show[0]) & (c.true_drift == 1)].t:
            ax.axvline(t, color="k", ls="--", lw=0.8)
        ax.set_title(f"{ds}: prequential accuracy (rolling 300), dashed = drift onsets (seed 0)"); ax.set_ylabel("accuracy")
    axes[0].legend(fontsize=7, ncol=3); axes[-1].set_xlabel("sample")
    plt.tight_layout(); plt.savefig(f"{out}/downstream.png", dpi=130); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); main(a.out, a.seeds)
