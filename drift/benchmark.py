"""Benchmark all detectors on synthetic drift streams (Paper 1 protocol) + Friedman/Nemenyi and Wilcoxon tests.
Default: the fixed setting (10k samples, 3 drifts, amplitude 0.12). --varied: a random setting per stream."""
import argparse, time, tracemalloc, os
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from .detectors import REGISTRY
from .generator import make_stream, make_varied_stream
from .metrics import score

# Nemenyi critical values q_0.05 (Demsar 2006)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}
PROPOSED = "Hybrid-OR(HDDM-A,ADWIN)"
KINDS = ["abrupt", "gradual", "incremental"]


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


def friedman(df, col, higher_better):
    """Friedman test over streams (blocks) + Nemenyi CD + Holm-corrected Wilcoxon of the proposed detector vs each other."""
    piv = df.pivot_table(index=["kind", "seed"], columns="detector", values=col)
    stat, p = friedmanchisquare(*[piv[c] for c in piv.columns])
    ranks = pd.Series(rankdata(-piv.values if higher_better else piv.values, axis=1).mean(0),
                      index=piv.columns).sort_values()
    k, N = len(piv.columns), len(piv)
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
    others = [c for c in piv.columns if c != PROPOSED]
    raw = []
    for c in others:
        d = piv[PROPOSED] - piv[c]
        raw.append(wilcoxon(d, zero_method="zsplit").pvalue if (d != 0).any() else 1.0)
    order, holm, run = np.argsort(raw), np.empty(len(raw)), 0.0
    for j, i in enumerate(order):   # Holm step-down
        run = max(run, min(1.0, (len(raw) - j) * raw[i])); holm[i] = run
    wil = pd.DataFrame(dict(other=others, mean_diff=[(piv[PROPOSED] - piv[c]).mean() for c in others],
                            rank_gap=[ranks[c] - ranks[PROPOSED] for c in others], p_holm=holm))
    return stat, p, k, N, cd, ranks, wil


def main(seeds=30, n=10000, out="results", start=0, varied=False):
    os.makedirs(out, exist_ok=True)
    rows = []
    for kind in KINDS:
        for seed in range(start, start + seeds):
            s, cfg = make_varied_stream(kind, seed) if varied else (make_stream(kind, n=n, seed=seed), {})
            for name in REGISTRY:
                alarms, us = run_detector(name, s)
                rows.append(dict(kind=kind, seed=seed, detector=name, us_per_sample=us, n_alarms=len(alarms),
                                 **cfg, **score(alarms, s.onsets, s.width, len(s.errors))))
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/benchmark_raw.csv", index=False)

    agg = dict(F1=("f1", "mean"), precision=("precision", "mean"), recall=("recall", "mean"),
               delay=("delay", "mean"), pen_delay=("pen_delay", "mean"), FA_per_1k=("far", "mean"),
               us_per_sample=("us_per_sample", "mean"))
    summ = df.groupby(["kind", "detector"]).agg(**agg).round(3)
    overall = df.groupby("detector").agg(F1_std=("f1", "std"), **agg).round(3)
    s0 = make_stream("abrupt", n=n, seed=start)
    overall["peak_mem_kB"] = [round(peak_memory_kb(d, s0), 1) for d in overall.index]
    summ.to_csv(f"{out}/benchmark_by_kind.csv"); overall.to_csv(f"{out}/benchmark_overall.csv")
    if varied:
        df.groupby(["amp", "detector"]).f1.mean().unstack(0).round(3).to_csv(f"{out}/benchmark_by_amp.csv")

    lines, rank_cols = [], {}
    for col, hb, label in [("f1", True, "F1"), ("pen_delay", False, "penalised delay"), ("us_per_sample", False, "time")]:
        stat, p, k, N, cd, ranks, wil = friedman(df, col, hb)
        rank_cols[label] = ranks
        lines += [f"== {label}: Friedman chi2={stat:.2f}, p={p:.3g}, k={k}, N={N}, Nemenyi CD(0.05)={cd:.3f}",
                  ranks.round(2).to_string(), f"-- {PROPOSED} vs others (Wilcoxon signed-rank, Holm-corrected)",
                  wil.round(4).to_string(index=False), ""]
        if col == "f1":
            ranks.rename("mean_rank").to_csv(f"{out}/friedman_ranks.csv")
            wil.round(4).to_csv(f"{out}/wilcoxon_f1.csv", index=False)
    pd.DataFrame(rank_cols).round(2).to_csv(f"{out}/friedman_ranks_all.csv")
    with open(f"{out}/friedman.txt", "w") as f:
        f.write("\n".join(lines))
    print(summ.to_string()); print("\nOVERALL\n", overall.to_string()); print("\n" + "\n".join(lines))
    plot(summ, out)


def plot(summ, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    summ["F1"].unstack(0).plot.bar(ax=ax[0]); ax[0].set_title("Detection F1 by drift type"); ax[0].set_ylabel("F1")
    summ["delay"].unstack(0).plot.bar(ax=ax[1]); ax[1].set_title("Mean detection delay (samples)")
    for a in ax: a.tick_params(axis="x", labelsize=7); plt.setp(a.get_xticklabels(), rotation=35, ha="right")
    plt.tight_layout(); plt.savefig(f"{out}/benchmark.png", dpi=130); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--out", default="results")
    ap.add_argument("--varied", action="store_true", help="random stream setting per stream")
    a = ap.parse_args(); main(a.seeds, out=a.out, start=a.start, varied=a.varied)
