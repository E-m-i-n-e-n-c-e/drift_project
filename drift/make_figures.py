"""Figures for the report: architecture diagram, detector benchmark, downstream accuracy curves."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "report/figs"
plt.rcParams.update({"font.size": 8, "font.family": "serif"})


def arch():
    fig, ax = plt.subplots(figsize=(7.1, 2.5)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 36)
    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=c, ec="k", lw=0.8))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7)
    def arr(a, b, txt=""):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.9))
        if txt: ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 1.6, txt, ha="center", fontsize=6)
    box(1, 13, 15, 10, "Producer\n(simulated IIoT\nsensors + labels,\ndrift schedule)", "#e8f0fe")
    box(22, 13, 13, 10, "Apache Kafka\ntopic `sensors`", "#fde7c8")
    box(42, 22, 20, 12, "Spark Structured\nStreaming\n10 s / 5 s windows\n(console sink;\nparallel branch)", "#e3f4e1")
    box(42, 3, 24, 15, "Online scorer\nRiver Hoeffding\nAdaptive Tree\n+ drift detector\n(HDDM-A OR ADWIN)", "#fbe3e3")
    box(73, 13, 13, 10, "InfluxDB\n`drift_stream`", "#f3e5f5")
    box(89, 13, 10, 10, "Grafana\ndashboard", "#fff9c4")
    arr((16, 18), (22, 18)); arr((35, 20), (42, 27), "tap"); arr((35, 16), (42, 10), "tap")
    arr((66, 10), (73, 16), "  err, alarm"); arr((86, 18), (89, 18))
    arr((54, 3.3), (54, 0.5)); ax.text(54, -0.2, "alarm $\\rightarrow$ reset learner", ha="center", fontsize=6, va="top")
    plt.savefig(f"{OUT}/arch.pdf", bbox_inches="tight"); plt.close()


def bench():
    df = pd.read_csv("results/benchmark_by_kind.csv")
    keep = ["EDDM", "DDM", "HDDM-W", "ADWIN", "HDDM-A", "Hybrid-OR(HDDM-A,ADWIN)"]
    lab = {"Hybrid-OR(HDDM-A,ADWIN)": "Hybrid (ours)"}
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.4))
    for a, col, ttl in zip(ax, ["F1", "delay"], ["Detection F1 (higher is better)", "Mean detection delay, samples (lower is better)"]):
        w = 0.26
        for j, k in enumerate(["abrupt", "gradual", "incremental"]):
            d = df[df.kind == k].set_index("detector").loc[keep, col]
            a.bar(np.arange(len(keep)) + (j - 1) * w, d.values, w, label=k)
        a.set_xticks(range(len(keep))); a.set_xticklabels([lab.get(x, x) for x in keep], rotation=25, ha="right", fontsize=7)
        a.set_title(ttl, fontsize=8)
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/bench.pdf"); plt.close()


def downstream():
    from drift.adaptive import run_lgbm, run_river, INIT
    from drift.generator import make_industrial_stream
    from river import tree
    X, y, meta = make_industrial_stream(seed=0)
    curves = {}
    for name, f in {"Static LightGBM": lambda: run_lgbm(X, y, None),
                    "LightGBM + ADWIN": lambda: run_lgbm(X, y, "ADWIN"),
                    "LightGBM + HDDM-A": lambda: run_lgbm(X, y, "HDDM-A"),
                    "LightGBM + Hybrid (ours)": lambda: run_lgbm(X, y, "Hybrid-OR(HDDM-A,ADWIN)"),
                    "River HAT (online)": lambda: run_river(X, y, tree.HoeffdingAdaptiveTreeClassifier(seed=1))}.items():
        curves[name] = pd.Series(1 - f()[0]).rolling(500).mean().values
    fig, ax = plt.subplots(figsize=(7.1, 2.6))
    for k, v in curves.items():
        ax.plot(np.arange(INIT, INIT + len(v)), v, label=k, lw=1.1)
    for o in meta["onsets"]:
        ax.axvline(o, color="k", ls="--", lw=0.7)
    ax.set_ylim(0.3, 1.0); ax.set_xlabel("stream position (sample)"); ax.set_ylabel("rolling accuracy (500)")
    ax.legend(fontsize=6.5, ncol=3, loc="lower center"); plt.tight_layout(); plt.savefig(f"{OUT}/downstream.pdf"); plt.close()


if __name__ == "__main__":
    arch(); bench(); downstream()
