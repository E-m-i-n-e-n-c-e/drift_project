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
    box(42, 22, 20, 12, "Spark Structured\nStreaming\n10 s / 5 s windows\n(optional,\nnot run)", "#e3f4e1")
    box(42, 3, 24, 15, "Online scorer\nRiver Hoeffding\nAdaptive Tree\n+ drift detector\n(HDDM-A OR ADWIN)", "#fbe3e3")
    box(73, 13, 13, 10, "InfluxDB\n`drift_stream`", "#f3e5f5")
    box(89, 13, 10, 10, "Grafana\ndashboard", "#fff9c4")
    arr((16, 18), (22, 18)); arr((35, 20), (42, 27), "tap"); arr((35, 16), (42, 10), "tap")
    arr((66, 10), (73, 16)); ax.text(70.5, 9.5, "err, alarm,\nlatency", fontsize=6, ha="center", va="top"); arr((86, 18), (89, 18))
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
    """Rolling accuracy from the saved per-sample curves (python -m drift.adaptive writes them)."""
    show = {"Static LightGBM": "Static LightGBM", "LightGBM + ADWIN": "LightGBM + ADWIN",
            "LightGBM + Hybrid-OR(HDDM-A,ADWIN)": "LightGBM + Hybrid (ours)", "River ARF (online)": "River ARF (online)",
            "River HAT (online)": "River HAT (online)"}
    panels = [("", "Simulated IIoT stream (true drift)"), ("_iotid20", "IoTID20, injected drift"),
              ("_cicids2017_natural", "CICIDS2017, natural order (drift positions unknown)")]
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 6.2), sharey=False)
    for ax, (suf, ttl) in zip(axes, panels):
        c = pd.read_csv(f"results/downstream_curves{suf}.csv")
        for k, lab in show.items():
            v = c[c.strategy == k]
            ax.plot(v.t, v.correct.rolling(300).mean(), label=lab, lw=0.9)
        for t in c[(c.strategy == "Static LightGBM") & (c.true_drift == 1)].t:
            ax.axvline(t, color="k", ls="--", lw=0.7)
        ax.set_title(ttl, fontsize=8); ax.set_ylabel("rolling acc. (300)")
    axes[0].set_ylim(0.3, 1.0); axes[-1].set_xlabel("stream position (sample)")
    axes[0].legend(fontsize=6, ncol=3, loc="lower center"); plt.tight_layout(); plt.savefig(f"{OUT}/downstream.pdf"); plt.close()


def live():
    """Kafka -> River -> InfluxDB run (exported from InfluxDB to results/stream_kafka.csv)."""
    df = pd.read_csv("results/stream_kafka.csv")
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True)
    for ax, (det, lab) in zip(axes, [("Hybrid-OR(HDDM-A,ADWIN)", "Hybrid (two-sided ADWIN)"),
                                     ("Hybrid-OR(HDDM-A,ADWIN-up)", "Hybrid (one-sided ADWIN)")]):
        v = df[df.detector == det]
        ax.plot(v.t, v.rolling_err, lw=0.8, color="C0")
        for t in v[v.true_drift == 1].t: ax.axvline(t, color="k", ls="--", lw=0.7)
        for t in v[v.alarm == 1].t: ax.axvline(t, color="C3", lw=1.0)
        ax.set_title(lab, fontsize=7); ax.set_ylabel("rolling error")
    axes[-1].set_xlabel("record"); plt.tight_layout(); plt.savefig(f"{OUT}/live.pdf"); plt.close()


if __name__ == "__main__":
    arch(); bench(); downstream(); live()
