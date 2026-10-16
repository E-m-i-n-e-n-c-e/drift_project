"""Downstream evaluation: does drift detection sustain classifier accuracy? (Paper 3 style)
Prequential test-then-train in micro-batches on a simulated IIoT stream with real concept drift.
Strategies: static LightGBM | drift-triggered LightGBM retrain (per detector) | River ARF / HAT (online)."""
import time, os
import numpy as np, pandas as pd
import lightgbm as lgb
from river import forest, tree
from .detectors import REGISTRY
from .generator import make_industrial_stream
from .metrics import score

BATCH, INIT, WINDOW, WAIT = 25, 1000, 500, 250   # micro-batch, initial train size, retrain window, samples to wait after alarm


def fit(X, y):
    return lgb.LGBMClassifier(n_estimators=80, learning_rate=0.1, num_leaves=15, verbose=-1, n_jobs=1).fit(X, y)


def run_lgbm(X, y, det_name=None):
    n = len(y)
    model = fit(X[:INIT], y[:INIT])
    det = REGISTRY[det_name]() if det_name else None
    err, alarms, retrains, pending = np.zeros(n, int), [], [], None
    t0 = time.perf_counter()
    for s in range(INIT, n, BATCH):
        e = min(s + BATCH, n)
        pred = model.predict(X[s:e])
        err[s:e] = pred != y[s:e]
        if det:
            for i in range(s, e):
                if det.update(int(err[i])):
                    alarms.append(i)
                    if pending is None:
                        pending = i + WAIT
            if pending is not None and e >= pending:   # enough post-alarm data collected
                model = fit(X[e - WINDOW:e], y[e - WINDOW:e]); retrains.append(e); pending = None
    return err[INIT:], alarms, len(retrains), (time.perf_counter() - t0) / (n - INIT) * 1e3


def run_river(X, y, model):
    n, err = len(y), np.zeros(len(y), int)
    t0 = time.perf_counter()
    for i in range(n):
        x = dict(enumerate(X[i]))
        err[i] = model.predict_one(x) != y[i]
        model.learn_one(x, int(y[i]))
    return err[INIT:], [], 0, (time.perf_counter() - t0) / n * 1e3


def main(out="results", seed=0):
    os.makedirs(out, exist_ok=True)
    X, y, meta = make_industrial_stream(seed=seed)
    runs = {"Static LightGBM": lambda: run_lgbm(X, y, None)}
    for d in ["DDM", "HDDM-A", "ADWIN", "EDDM", "Hybrid-OR(HDDM-A,ADWIN)"]:
        runs[f"LightGBM + {d}"] = (lambda d=d: run_lgbm(X, y, d))
    runs["River ARF (online)"] = lambda: run_river(X, y, forest.ARFClassifier(n_models=5, seed=1))
    runs["River HAT (online)"] = lambda: run_river(X, y, tree.HoeffdingAdaptiveTreeClassifier(seed=1))
    rows, curves = [], {}
    for name, f in runs.items():
        err, alarms, nre, ms = f()
        sc = score([a - INIT for a in alarms], [o - INIT for o in meta["onsets"]], meta["width"], len(err)) if alarms or name.startswith("LightGBM") else {}
        rows.append(dict(strategy=name, accuracy=1 - err.mean(), acc_after_1st_drift=1 - err[meta["onsets"][0] - INIT:].mean(),
                         alarms=len(alarms), retrains=nre, drift_F1=sc.get("f1", np.nan), drift_delay=sc.get("delay", np.nan),
                         ms_per_sample=ms))
        curves[name] = pd.Series(1 - err).rolling(500).mean().values
        print(rows[-1])
    df = pd.DataFrame(rows).round(4); df.to_csv(f"{out}/downstream.csv", index=False)
    print("\n", df.to_string(index=False))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4.5))
    for k, v in curves.items(): plt.plot(np.arange(INIT, INIT + len(v)), v, label=k, lw=1.2)
    for o in meta["onsets"]: plt.axvline(o, color="k", ls="--", lw=0.8)
    plt.ylim(0.3, 1); plt.xlabel("sample"); plt.ylabel("rolling accuracy (500)"); plt.legend(fontsize=7, ncol=2)
    plt.title("Prequential accuracy under concept drift (dashed = true drift onsets)")
    plt.tight_layout(); plt.savefig(f"{out}/downstream.png", dpi=130)


if __name__ == "__main__":
    main()
