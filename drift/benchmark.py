"""Benchmark all detectors on synthetic drift streams (Paper 1 protocol) + Friedman/Nemenyi test."""
import argparse, time, tracemalloc, os
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, rankdata
from .detectors import REGISTRY
from .generator import make_stream
from .metrics import score

Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def run_detector(name, s):
    det = REGISTRY[name]()
    alarms = []
    t0 = time.perf_counter()
    for i, e in enumerate(s.errors):
        if det.update(int(e)):
            alarms.append(i)
    dt = time.perf_counter() - t0
    return alarms, dt / len(s.errors) * 1e6   # microseconds / sample


def peak_memory_kb(name, s):
    tracemalloc.start()
    run_detector(name, s)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 1024


def main(seeds=30, n=10000, out="results", start=0):
    os.makedirs(out, exist_ok=True)
    rows = []
    for kind in ["abrupt", "gradual", "incremental"]:
        for seed in range(start, start + seeds):
            s = make_stream(kind, n=n, seed=seed)
            for name in REGISTRY:
                alarms, us = run_detector(name, s)
                rows.append(dict(kind=kind, seed=seed, detector=name, us_per_sample=us,
                                 n_alarms=len(alarms), **score(alarms, s.onsets, s.width, n)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/benchmark_raw.csv", index=False)

    summ = df.groupby(["kind", "detector"]).agg(
        F1=("f1", "mean"), precision=("precision", "mean"), recall=("recall", "mean"),
        delay=("delay", "mean"), FA_per_1k=("far", "mean"), us_per_sample=("us_per_sample", "mean")).round(3)
    overall = df.groupby("detector").agg(
        F1=("f1", "mean"), delay=("delay", "mean"), FA_per_1k=("far", "mean"),
        us_per_sample=("us_per_sample", "mean")).round(3)
    s0 = make_stream("abrupt", n=n, seed=start)
    overall["peak_mem_kB"] = [round(peak_memory_kb(d, s0), 1) for d in overall.index]
    summ.to_csv(f"{out}/benchmark_by_kind.csv"); overall.to_csv(f"{out}/benchmark_overall.csv")

    # Friedman + Nemenyi on F1 (blocks = stream)
    piv = df.pivot_table(index=["kind", "seed"], columns="detector", values="f1")
    stat, p = friedmanchisquare(*[piv[c] for c in piv.columns])
    ranks = pd.Series(rankdata(-piv.values, axis=1).mean(0), index=piv.columns).sort_values()
    k, N = piv.shape[1][0] if False else len(piv.columns), len(piv)
    cd = Q05.get(k, 3.164) * np.sqrt(k * (k + 1) / (6 * N))
    ranks.rename("mean_rank").to_csv(f"{out}/friedman_ranks.csv")
    with open(f"{out}/friedman.txt", "w") as f:
        f.write(f"Friedman chi2={stat:.2f}, p={p:.3g}, k={k}, N={N}\nNemenyi CD(0.05)={cd:.3f}\n{ranks.round(2).to_string()}\n")
    print(summ.to_string()); print("\nOVERALL\n", overall.to_string())
    print(f"\nFriedman chi2={stat:.1f} p={p:.2g}  Nemenyi CD={cd:.2f}\n{ranks.round(2).to_string()}")
    plot(summ, out)


def plot(summ, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    summ["F1"].unstack(0).plot.bar(ax=ax[0]); ax[0].set_title("Detection F1 by drift type"); ax[0].set_ylabel("F1")
    summ["delay"].unstack(0).plot.bar(ax=ax[1]); ax[1].set_title("Mean detection delay (samples)")
    for a in ax: a.tick_params(axis="x", labelsize=7); plt.setp(a.get_xticklabels(), rotation=35, ha="right")
    plt.tight_layout(); plt.savefig(f"{out}/benchmark.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--out", default="results")
    a = ap.parse_args(); main(a.seeds, out=a.out, start=a.start)
