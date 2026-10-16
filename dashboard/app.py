"""Concept drift demo dashboard.

Run from the repo root:   streamlit run dashboard/app.py

Reads the final result files in results/, results_heldout/ and results_varied/ and never recomputes experiment
results. The only project code it calls is drift.generator.make_stream (Page 4).
Runs fully offline: no Docker, Kafka, InfluxDB or Grafana.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `drift.generator` imports when run with `streamlit run dashboard/app.py`

RESULTS = Path(os.environ.get("DRIFT_RESULTS", ROOT / "results"))
HELDOUT = ROOT / "results_heldout"
VARIED = ROOT / "results_varied"

# ------------------------------------------------------------------------------------------------
# Shared visual language: one colour per detector on every page (and in the slides); true drifts dashed
# black; alarms red. Palette validated for colour-blind separation on a light surface.
# ------------------------------------------------------------------------------------------------
HYBRID = "Hybrid-OR(HDDM-A,ADWIN)"
HYBRID_UP = "Hybrid-OR(HDDM-A,ADWIN-up)"
DET_COLOR = {
    HYBRID: "#2a78d6",    # blue: our detector
    "HDDM-A": "#eb6834",  # orange
    "ADWIN": "#1baf7a",   # aqua
    "HDDM-W": "#6250d6",  # violet
    "DDM": "#eda100",     # yellow
    "EDDM": "#e87ba4",    # magenta
}
VARIANT_COLOR = {         # design-isolation variants of the hybrid
    HYBRID_UP: "#184f95",       # dark blue: one-sided ADWIN
    "HDDM-A+R500": "#a8441b",   # dark orange: HDDM-A with the refractory period only
}
ABLATION_COLOR = {        # pairings that failed
    "Hybrid-OR(HDDM-W,EDDM)": "#8f8d87",
    "Hybrid-OR(ADWIN,EDDM)": "#6b6a65",
    "Hybrid-AND(HDDM-W,EDDM)": "#b3b1aa",
    "Hybrid-AND(HDDM-W,ADWIN)": "#4a4945",
}
STATIC_COLOR = "#52514e"
ONLINE_COLOR = {"River HAT (online)": "#5b2a86", "River ARF (online)": "#a0522d"}
ALARM = "#d62828"
DRIFT = "#111111"
INK, INK2, GRID, TINT = "#1a1a19", "#52514e", "#e6e5e0", "#dbe8f8"
BASELINES = ["DDM", "EDDM", "HDDM-W", "ADWIN", "HDDM-A", HYBRID]
SHORT = {
    HYBRID: "Hybrid (HDDM-A OR ADWIN)",
    HYBRID_UP: "Hybrid, one-sided ADWIN",
    "HDDM-A+R500": "HDDM-A + refractory only",
    "Hybrid-OR(HDDM-W,EDDM)": "OR(HDDM-W, EDDM): proposal",
    "Hybrid-OR(ADWIN,EDDM)": "OR(ADWIN, EDDM): Paper 2",
    "Hybrid-AND(HDDM-W,EDDM)": "AND(HDDM-W, EDDM)",
    "Hybrid-AND(HDDM-W,ADWIN)": "AND(HDDM-W, ADWIN)",
}
TOL = 300  # scoring margin from drift/metrics.py


def short(name: str) -> str:
    return SHORT.get(name, name)


def strat_label(name: str) -> str:
    if name.startswith("LightGBM + "):
        return "Retrain on " + short(name[len("LightGBM + "):])
    return name


def color_of(name: str) -> str:
    key = name.replace("LightGBM + ", "")
    for table in (DET_COLOR, VARIANT_COLOR, ABLATION_COLOR, ONLINE_COLOR):
        if key in table:
            return table[key]
    return STATIC_COLOR


def base_layout(fig: go.Figure, height=430, ytitle="", xtitle="sample", legend=True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        font=dict(size=16, color=INK, family="Inter, Segoe UI, Helvetica, Arial, sans-serif"),
        hoverlabel=dict(font_size=15, bgcolor="white"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=15)),
    )
    fig.update_xaxes(title=xtitle, gridcolor=GRID, zeroline=False, linecolor=INK2, tickfont=dict(color=INK2))
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False, linecolor=INK2, tickfont=dict(color=INK2))
    return fig


def drift_lines(fig: go.Figure, onsets, label=True):
    for i, o in enumerate(onsets):
        fig.add_vline(x=o, line=dict(color=DRIFT, dash="dash", width=1.6))
        if label:
            fig.add_annotation(x=o, y=1, yref="paper", text=f"drift {i + 1}", showarrow=False,
                               xanchor="left", yanchor="top", xshift=4, font=dict(size=13, color=INK2))


def legend_drift(fig: go.Figure):
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="True drift",
                             line=dict(color=DRIFT, dash="dash", width=1.6)))


def caption(text: str):
    st.markdown(f"<div class='cap'>{text}</div>", unsafe_allow_html=True)


def chips(names):
    html = " &nbsp;&nbsp; ".join(
        f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;background:{color_of(d)};"
        f"vertical-align:-2px'></span>&nbsp;{short(d)}" for d in names)
    st.markdown(f"<div style='font-size:1.0rem;margin:0.2rem 0 0.6rem 0;line-height:1.9'>{html}</div>",
                unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------
# Data access (cached; read-only)
# ------------------------------------------------------------------------------------------------
@st.cache_data
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def nemenyi_cd(path: str) -> float | None:
    """Critical difference for the F1 ranks, read from friedman.txt."""
    try:
        txt = Path(path).read_text()
        m = re.search(r"== F1:.*?CD\(0\.05\)=([\d.]+)", txt)
        return float(m.group(1)) if m else None
    except OSError:
        return None


def need(path: Path) -> bool:
    if not path.exists():
        st.error(f"`{path.relative_to(ROOT)}` not found. Regenerate it with the commands in README.md.")
        return False
    return True


# The live stream: true drifts at 5000, 10000 (abrupt) and 15000 (gradual over 1500).
STREAM_WIDTHS = {5000: 1, 10000: 1, 15000: 1500}


def label_alarms(alarms, widths):
    """Same rule as drift/metrics.py: the first alarm in [onset, onset + width + 300), capped at the next onset,
    detects that drift; every other alarm is a false alarm. Returns (alarm, onset, delay, hit, hover text)."""
    onsets = sorted(widths)
    ends = {}
    for i, o in enumerate(onsets):
        e = o + widths[o] + TOL
        if i + 1 < len(onsets):
            e = min(e, onsets[i + 1])
        ends[o] = e
    out, used = [], set()
    for a in sorted(alarms):
        prev = [o for o in onsets if o <= a]
        if not prev:
            out.append((a, None, None, False, "False alarm: no drift has happened yet"))
            continue
        o = max(prev)
        d = a - o
        if a < ends[o] and o not in used:
            used.add(o)
            out.append((a, o, d, True, f"Detected drift {onsets.index(o) + 1} at {o:,}<br>detection delay = <b>{d:,} records</b>"))
        else:
            out.append((a, o, d, False, f"False alarm<br>{d:,} records after drift {onsets.index(o) + 1} (outside its window)"))
    return out


# ------------------------------------------------------------------------------------------------
# Page 1: live stream (the Kafka run, replayed)
# ------------------------------------------------------------------------------------------------
LIVE_DET = {"Two-sided hybrid": HYBRID, "One-sided hybrid (ADWIN-up)": HYBRID_UP}


@st.cache_data
def live_stream(path: str, detector: str | None):
    df = pd.read_csv(path)
    if detector is not None and "detector" in df:
        df = df[df.detector == detector]
    df = df.sort_values("t").reset_index(drop=True)
    stats = {}
    if "time" in df and "latency_ms" in df:
        tt = pd.to_datetime(df["time"], format="mixed", utc=True)
        span = (tt.iloc[-1] - tt.iloc[0]).total_seconds()
        stats = dict(rate=len(df) / span if span > 0 else None, lat_med=df.latency_ms.median(),
                     lat_p95=df.latency_ms.quantile(0.95))
    return df, stats


def page_live():
    st.title("Live stream: watch drift happen")
    kafka = RESULTS / "stream_kafka.csv"
    use_kafka = kafka.exists()
    if not use_kafka and not need(RESULTS / "stream_local.csv"):
        return

    ss = st.session_state
    START = 1000
    ss.setdefault("pos", START)
    ss.setdefault("playing", False)

    c0, c1, c2, c3, c4 = st.columns([2.4, 0.9, 0.9, 0.9, 2])
    if use_kafka:
        det_label = c0.segmented_control("Detector", list(LIVE_DET), default=list(LIVE_DET)[0], key="live_det") \
            or list(LIVE_DET)[0]
        df, stats = live_stream(str(kafka), LIVE_DET[det_label])
    else:
        det_label = "Two-sided hybrid"
        df, stats = live_stream(str(RESULTS / "stream_local.csv"), None)
        c0.caption("stream_kafka.csv not found: replaying the in-process run")
    n = len(df)
    onsets = df.loc[df.true_drift == 1, "t"].tolist() or list(STREAM_WIDTHS)
    widths = {o: STREAM_WIDTHS.get(o, 1) for o in onsets}
    alarms_all = label_alarms(df.loc[df.alarm == 1, "t"].tolist(), widths)
    line_col = DET_COLOR[HYBRID] if LIVE_DET[det_label] == HYBRID else VARIANT_COLOR[HYBRID_UP]

    c1.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    c2.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    c3.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    if c1.button("Pause" if ss.playing else "Play", type="primary", width="stretch"):
        ss.playing = not ss.playing
        if ss.pos >= n:
            ss.pos = START
        st.rerun()
    if c2.button("Reset", width="stretch"):
        ss.pos, ss.playing = START, False
        st.rerun()
    if c3.button("Show all", width="stretch"):
        ss.pos, ss.playing = n, False
        st.rerun()
    speed = c4.select_slider("Speed (records per update)", options=[50, 100, 200, 400, 800], value=200, key="speed")

    @st.fragment(run_every=0.25 if ss.playing else None)
    def live():
        if ss.playing:
            ss.pos = min(ss.pos + speed, n)
            if ss.pos >= n:
                ss.playing = False
        pos = min(ss.pos, n)
        view = df.iloc[:pos]
        seen = [a for a in alarms_all if a[0] < pos]
        hits = sum(a[3] for a in seen)

        m = st.columns(5)
        m[0].metric("Records processed", f"{pos:,}")
        m[1].metric("True drifts so far", f"{sum(o < pos for o in onsets)}")
        m[2].metric("Alarms raised", f"{len(seen)}", help="red triangles")
        m[3].metric("False alarms", f"{len(seen) - hits}")
        m[4].metric("Error rate (last 500)", f"{view.rolling_err.iloc[-1]:.1%}")

        fig = go.Figure()
        v = view[view.t >= 300]  # skip the cold start, where the rolling window holds only a few records
        v = v.iloc[::5] if len(v) > 4000 else v
        fig.add_trace(go.Scatter(x=v.t, y=v.rolling_err, mode="lines", name="Error rate (last 500 records)",
                                 line=dict(color=line_col, width=2),
                                 hovertemplate="record %{x:,}<br>error rate %{y:.1%}<extra></extra>"))
        if pos > 15000:
            fig.add_vrect(x0=15000, x1=min(16500, pos), fillcolor="#dddcd6", opacity=0.35, line_width=0,
                          layer="below", annotation_text="gradual drift", annotation_position="bottom left",
                          annotation_font=dict(size=12, color=INK2))
        drift_lines(fig, [o for o in onsets if o < pos])
        for a, o, d, hit, _ in seen:  # delay bracket from the drift line to the alarm that detected it
            if hit:
                y = float(df.rolling_err.iloc[a]) + 0.025
                fig.add_shape(type="line", x0=o, x1=a, y0=y, y1=y, line=dict(color=ALARM, width=2))
                fig.add_annotation(x=a, y=y, text=f"delay {d:,}", showarrow=False, xanchor="left",
                                   yanchor="bottom", xshift=4, font=dict(size=13, color=ALARM),
                                   bgcolor="rgba(252,252,251,0.9)")
        if seen:
            ax = [a[0] for a in seen]
            fig.add_trace(go.Scatter(
                x=ax, y=df.rolling_err.iloc[ax], mode="markers", name="Detector alarm",
                marker=dict(color=ALARM, size=14, symbol=["triangle-down" if a[3] else "triangle-down-open" for a in seen],
                            line=dict(color=ALARM, width=2)),
                text=[a[4] for a in seen],
                hovertemplate="<b>Alarm at record %{x:,}</b><br>%{text}<extra></extra>"))
        legend_drift(fig)
        fig.update_xaxes(range=[0, n])
        fig.update_yaxes(range=[0, 0.3], tickformat=".0%")
        st.plotly_chart(base_layout(fig, 430, "error rate", "record"), width="stretch", key="live_chart",
                        config={"displayModeBar": False})

        rate = f"~{stats['rate']:.0f} records/s" if stats.get("rate") else ""
        lat = f"median {stats['lat_med']:.1f} ms end-to-end latency" if stats.get("lat_med") is not None else ""
        via = (f" This data went through Kafka into InfluxDB at {rate} with a {lat}." if use_kafka and rate else "")
        caption("The error rate jumps after each drift (dashed line); the detector should fire (red triangle) "
                "shortly after. Filled triangles detected a drift, hollow ones are false alarms; hover for the "
                f"delay.{via}")
        if pos >= n:
            fa = len(seen) - hits
            p = hits / len(seen) if seen else 0
            r = hits / len(onsets) if onsets else 0
            f1 = 2 * p * r / (p + r) if p + r else 0
            st.info(f"**{det_label}:** detected {hits} of {len(onsets)} drifts with {fa} false alarm"
                    f"{'s' if fa != 1 else ''}, detection F1 = {f1:.2f}.", icon=":material/flag:")

    live()

    with st.expander("What is being replayed?"):
        st.markdown(
            "- 20,000 records of a simulated IIoT stream (8 sensor channels, normal vs fault), sent by a producer "
            "into **Kafka**, scored by an online consumer and written to **InfluxDB**; this is the export of that run.\n"
            "- An online model (River Hoeffding Adaptive Tree) predicts each record, then learns from its label. "
            "Each prediction becomes 0 (right) or 1 (wrong) and the detector watches that 0/1 stream.\n"
            "- On an alarm the model is replaced, and the detector skips a 300-record warm-up.\n"
            "- **Two-sided** hybrid: ADWIN also fires when the error *falls*, e.g. when a fresh model improves. "
            "**One-sided** hybrid: ADWIN only reacts to rises, which removes those false alarms.\n"
            "- The alarm near record 2,500 comes from a real rise in the model's error with no change in the data.\n"
            "- True drifts: 5,000 (abrupt), 10,000 (abrupt), 15,000 (gradual over 1,500 records). A detection counts "
            "if the first alarm lands inside the drift plus 300 records."
        )


# ------------------------------------------------------------------------------------------------
# Page 2: does retraining help?
# ------------------------------------------------------------------------------------------------
DATASETS = {
    "CICIDS2017, natural order": ("downstream_curves_cicids2017_natural.csv", "downstream_cicids2017_natural.csv"),
    "Simulated IIoT stream": ("downstream_curves.csv", "downstream.csv"),
    "IoTID20, injected drift": ("downstream_curves_iotid20.csv", "downstream_iotid20.csv"),
    "CICIDS2017, injected drift": ("downstream_curves_cicids2017.csv", "downstream_cicids2017.csv"),
    "IoTID20, natural order": ("downstream_curves_iotid20_natural.csv", "downstream_iotid20_natural.csv"),
}
DATASET_NOTE = {
    "CICIDS2017, natural order": "Network traffic in its original order. Attack mixes change over time, so drift is real, "
                                 "but its positions are unknown (no dashed lines).",
    "Simulated IIoT stream": "Simulated sensor stream with three known drifts (abrupt, abrupt, gradual).",
    "IoTID20, injected drift": "IoTID20 shuffled (removes natural drift), then the most important features swap "
                               "columns at three known points, like a sensor remapping.",
    "CICIDS2017, injected drift": "CICIDS2017 shuffled, then the most important features swap columns at three known points.",
    "IoTID20, natural order": "IoTID20 in its original order. It is essentially stationary: good detectors should stay silent.",
}
PICK = {"DDM": "DDM", "ADWIN": "ADWIN", "EDDM": "EDDM", "HDDM-A": "HDDM-A", "Hybrid": HYBRID}
WINDOW = 300


def page_retrain():
    st.title("Does retraining help?")
    c1, c2, c3 = st.columns([1.3, 1.6, 1.4])
    ds = c1.selectbox("Dataset", list(DATASETS))
    det_label = c2.segmented_control("Retraining trigger", list(PICK), default="Hybrid", key="det_pick") or "Hybrid"
    online = c3.multiselect("Overlay online models", list(ONLINE_COLOR), default=[],
                            format_func=lambda s: s.replace(" (online)", ""))
    det = PICK[det_label]
    st.caption(DATASET_NOTE[ds])

    cpath, spath = RESULTS / DATASETS[ds][0], RESULTS / DATASETS[ds][1]
    if not need(cpath):
        return
    cur = read_csv(str(cpath))
    static_name = "Static LightGBM"
    det_name = f"LightGBM + {det}"
    if det_name not in set(cur.strategy):
        st.error(f"`{det_name}` is not in {cpath.name}.")
        return

    fig = go.Figure()
    onsets, lows = [], []
    order = [(static_name, "Static LightGBM (never retrained)", STATIC_COLOR, "dot", 2)]
    order += [(o, o.replace(" (online)", " (online model)"), ONLINE_COLOR[o], "dashdot", 2) for o in online]
    order += [(det_name, f"LightGBM retrained on {short(det)}", color_of(det_name), "solid", 2.6)]
    for name, label, col, dash, width in order:
        s = cur[cur.strategy == name].sort_values("t")
        acc = s.correct.rolling(WINDOW).mean()
        lows.append(acc.min())
        fig.add_trace(go.Scatter(x=s.t, y=acc, name=label, mode="lines", line=dict(color=col, width=width, dash=dash),
                                 hovertemplate="sample %{x:,}<br>accuracy %{y:.1%}<extra>" + label + "</extra>"))
        if name == det_name:
            onsets = s.loc[s.true_drift == 1, "t"].tolist()
            r = s[s.retrain == 1]
            fig.add_trace(go.Scatter(x=r.t, y=acc.loc[r.index], mode="markers", name="Retrain",
                                     marker=dict(symbol="diamond", size=12, color=col, line=dict(color="white", width=1.5)),
                                     hovertemplate="model retrained at sample %{x:,}<extra></extra>"))
    lo = max(0.0, np.floor(np.nanmin(lows) * 10) / 10 - 0.05)
    a = cur[(cur.strategy == det_name) & (cur.alarm == 1)]
    fig.add_trace(go.Scatter(x=a.t, y=np.full(len(a), lo + 0.02), mode="markers", name="Alarm",
                             marker=dict(symbol="line-ns", size=16, color=ALARM, line=dict(width=2.5, color=ALARM)),
                             hovertemplate="alarm at sample %{x:,}<extra></extra>"))
    drift_lines(fig, onsets)
    if onsets:
        legend_drift(fig)
    fig.update_yaxes(range=[lo, 1.01], tickformat=".0%")
    st.plotly_chart(base_layout(fig, 470, f"rolling accuracy ({WINDOW})"), width="stretch",
                    config={"displayModeBar": False})
    n_alarms = int(a.shape[0])
    if n_alarms == 0:
        why = ("This stream barely drifts, so silence is the right answer." if ds == "IoTID20, natural order" else
               "Here that is a miss: the injected drift raises the error by only a few percent, too little for it."
               if onsets else "")
        caption(f"{short(det)} never fired on this stream, so the model was never retrained and its line sits exactly "
                f"on top of the static model's. {why}")
    elif onsets:
        caption("After each drift (dashed) the static model falls and stays down; the retrained model dips, the "
                "detector fires (red ticks at the bottom), the model is retrained (diamonds), and accuracy comes back.")
    else:
        caption("When the traffic mix shifts, the static model collapses and only recovers if the old mix returns; "
                "the retrained model dips, the detector fires (red ticks), the model is retrained (diamonds), and "
                "accuracy comes back much sooner.")

    if not spath.exists():
        return
    summ = read_csv(str(spath))
    has_std = "accuracy_std" in summ and summ.accuracy_std.notna().any()
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.subheader("Summary" + (" (mean ± std over 10 seeds)" if has_std else " (one run)"))
        tab = summ.copy()
        fmt_pm = lambda m, s: f"{m:.1%} ± {s:.1%}" if pd.notna(s) else f"{m:.1%}"
        tab["accuracy "] = [fmt_pm(m, s) for m, s in zip(tab.accuracy, tab.get("accuracy_std", [np.nan] * len(tab)))]
        tab["F1 "] = [f"{m:.3f} ± {s:.3f}" if pd.notna(s) else f"{m:.3f}" for m, s in zip(tab.f1, tab.get("f1_std", [np.nan] * len(tab)))]
        cols = ["strategy", "accuracy ", "F1 ", "alarms", "retrains"]
        if "drift_F1" in tab and tab.drift_F1.notna().any():
            cols += ["drift_F1"]
        tab["strategy"] = tab.strategy.map(strat_label)
        show = tab[cols].rename(columns={"drift_F1": "detection F1"})
        sel = {strat_label(det_name), strat_label(static_name)}

        def hl(row):
            on = row["strategy"] in sel
            return [f"background-color:{TINT}; font-weight:600" if on else "" for _ in row]

        st.dataframe(show.style.apply(hl, axis=1).format({"alarms": "{:.1f}", "retrains": "{:.1f}",
                                                          "detection F1": "{:.2f}"}, na_rep="–"),
                     hide_index=True, width="stretch")
        st.caption("F1 here is the classifier's F1 on the attack / fault class. Detection F1 scores the alarms "
                   "against the known drift points (only where drift positions are known).")
    with right:
        metric = st.radio("Compare by", ["accuracy", "f1"], horizontal=True,
                          format_func=lambda m: "accuracy" if m == "accuracy" else "attack / fault F1",
                          index=1 if "cicids" in DATASETS[ds][1] else 0)
        s2 = summ.sort_values(metric)
        names = [strat_label(s) for s in s2.strategy]
        opac = [1.0 if s in (det_name, static_name) else 0.45 for s in s2.strategy]
        bar = go.Figure(go.Bar(x=s2[metric], y=names, orientation="h",
                               marker=dict(color=[color_of(s) for s in s2.strategy], opacity=opac),
                               text=[f"{v:.1%}" if metric == "accuracy" else f"{v:.3f}" for v in s2[metric]],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        bar.update_xaxes(range=[0, 1.2], tickformat=".0%" if metric == "accuracy" else ".1f")
        st.plotly_chart(base_layout(bar, 360, "", metric if metric == "accuracy" else "attack / fault F1", legend=False),
                        width="stretch", config={"displayModeBar": False})
        trig = summ[summ.strategy.str.startswith("LightGBM + ")]
        best = trig.loc[trig[metric].idxmax()]
        hy = summ.set_index("strategy").loc["LightGBM + " + HYBRID]
        st_ = summ.set_index("strategy").loc[static_name]
        best_all = summ.loc[summ[metric].idxmax(), "strategy"]
        gain = trig[metric].min() - st_[metric]
        parts = []
        if gain > 0.02:
            parts.append(f"Every retraining trigger beats the static model (+{gain:.2f} or more).")
        elif trig[metric].max() - st_[metric] < 0.01:
            parts.append("Retraining barely matters here: the data hardly drifts.")
        if best.strategy == "LightGBM + " + HYBRID:
            parts.append("The hybrid is the best trigger here.")
        else:
            parts.append(f"The hybrid is not the best trigger here ({hy[metric]:.3f}); "
                         f"{short(best.strategy.replace('LightGBM + ', ''))} is ({best[metric]:.3f}).")
        if best_all in ONLINE_COLOR:
            parts.append(f"{best_all.replace(' (online)', '')}, an online model, beats every LightGBM variant.")
        caption(" ".join(parts))


# ------------------------------------------------------------------------------------------------
# Page 3: detector comparison
# ------------------------------------------------------------------------------------------------
SETTINGS = {
    "Fixed (90 streams)": (RESULTS, "seeds 0–29: the setting the hybrid was designed on"),
    "Unseen seeds (90 streams)": (HELDOUT, "seeds 2000–2029, never used for design"),
    "Varied (300 streams)": (VARIED, "seeds 3000–3099; random length, number of drifts, width and size of drift"),
}


def verdict(row):
    if row.p_holm < 0.05:
        return "hybrid significantly better" if row.mean_diff > 0 else "hybrid significantly worse"
    return "not significant"


def page_compare():
    st.title("Detector comparison")
    avail = {k: v for k, v in SETTINGS.items() if (v[0] / "benchmark_overall.csv").exists()}
    c1, c2, c3 = st.columns([2.2, 1.3, 1.3])
    setting = c1.segmented_control("Setting", list(avail), default=list(avail)[0], key="setting") or list(avail)[0]
    show_var = c2.toggle("Show variants and failed pairings", value=False)
    delay_col = c3.radio("Delay", ["pen_delay", "delay"], horizontal=True,
                         format_func=lambda c: "penalised" if c == "pen_delay" else "detected only")
    base, desc = avail[setting]
    st.caption(desc)

    kind = read_csv(str(base / "benchmark_by_kind.csv"))
    overall = read_csv(str(base / "benchmark_overall.csv"))
    dets = BASELINES + ([d for d in [*VARIANT_COLOR, *ABLATION_COLOR] if d in set(overall.detector)] if show_var else [])
    chips(dets)

    kinds = ["abrupt", "gradual", "incremental"]
    kb = kind[kind.detector.isin(BASELINES)]
    leaders = kb.loc[kb.groupby("kind").F1.idxmax()]
    won = [k for k, d in zip(leaders.kind, leaders.detector) if d == HYBRID]
    worst = short(kb.groupby("detector").F1.mean().idxmin())
    lead_txt = ("The hybrid (blue) has the highest F1 on all three drift types" if len(won) == 3 else
                f"The hybrid (blue) has the highest F1 on {', '.join(won)} drift" if won else
                "No single detector wins every drift type")
    g1, g2 = st.columns(2)
    dname = "Penalised delay" if delay_col == "pen_delay" else "Delay over detected drifts"
    for col, metric, title, fmt, cap in [
        (g1, "F1", "F1 (higher is better)", ".2f",
         f"Did alarms land on real drifts? {lead_txt}; {worst} is lowest because it fires far too often."),
        (g2, delay_col, f"{dname} in samples (lower is better)", ".0f",
         "How long after the drift the alarm came. Penalised delay counts a missed drift as the whole window, "
         "so a detector cannot look fast by missing drifts. EDDM and HDDM-W are fastest, but pay with false alarms."),
    ]:
        fig = go.Figure()
        for d in dets:
            sub = kind[kind.detector == d].set_index("kind").reindex(kinds)
            fig.add_trace(go.Bar(x=kinds, y=sub[metric], name=short(d), marker_color=color_of(d),
                                 hovertemplate=f"{short(d)}<br>%{{x}}: %{{y:{fmt}}}<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
        if metric == "F1":
            fig.update_yaxes(range=[0, 1])
        with col:
            st.subheader(title)
            st.plotly_chart(base_layout(fig, 380, "F1" if metric == "F1" else "samples", "drift type", legend=False),
                            width="stretch", config={"displayModeBar": False})
            caption(cap)

    t1, t2 = st.columns([1.15, 1], gap="large")
    with t1:
        st.subheader(f"All {setting.split('(')[1].rstrip(')')}")
        tab = overall[overall.detector.isin(dets)].sort_values("F1", ascending=False).copy()
        tab["F1 "] = [f"{m:.3f} ± {s:.2f}" for m, s in zip(tab.F1, tab.F1_std)]
        tab["detector"] = tab.detector.map(short)
        tab = tab[["detector", "F1 ", "recall", "pen_delay", "FA_per_1k", "us_per_sample", "peak_mem_kB"]].rename(
            columns={"pen_delay": "pen. delay", "FA_per_1k": "false alarms / 1k", "us_per_sample": "µs / sample",
                     "peak_mem_kB": "kB"})

        def hl(row):
            on = row["detector"] == short(HYBRID)
            return [f"background-color:{TINT}; font-weight:600" if on else "" for _ in row]

        st.dataframe(tab.style.apply(hl, axis=1).format({"recall": "{:.2f}", "pen. delay": "{:.0f}",
                                                         "false alarms / 1k": "{:.2f}", "µs / sample": "{:.2f}",
                                                         "kB": "{:.1f}"}),
                     hide_index=True, width="stretch",
                     column_config={"detector": st.column_config.TextColumn(width="medium")})
        h = overall.set_index("detector").loc[HYBRID]
        caption(f"Highlighted row: our hybrid. Cost stays small: {h.us_per_sample:.1f} µs per sample and "
                f"{h.peak_mem_kB:.1f} kB, roughly the sum of its two parts.")

    with t2:
        st.subheader("Is the hybrid significantly better?")
        wpath = base / "wilcoxon_f1.csv"
        if wpath.exists():
            w = read_csv(str(wpath))
            w = w[w.other.isin(dets) & (w.other != HYBRID_UP)].copy()
            w["verdict"] = w.apply(verdict, axis=1)
            w = w.sort_values("mean_diff")
            vis = pd.DataFrame({"vs": w.other.map(short), "F1 gain": w.mean_diff, "p (Holm)": w.p_holm,
                                "verdict": w.verdict})

            def vc(v):
                return ("color:#0b6e4f;font-weight:600" if "better" in v else
                        "color:#d62828;font-weight:600" if "worse" in v else "color:#52514e")

            st.dataframe(vis.style.map(vc, subset=["verdict"]).format({"F1 gain": "{:+.3f}", "p (Holm)": "{:.4f}"}),
                         hide_index=True, width="stretch",
                         column_config={"vs": st.column_config.TextColumn(width="medium"),
                                        "verdict": st.column_config.TextColumn(width="medium")})
            ns = [short(o) for o, v in zip(w.other, w.verdict) if v == "not significant"]
            caption("Wilcoxon signed-rank test per stream, Holm-corrected. "
                    + (f"Not significantly different from: {', '.join(ns)}." if ns else
                       "The hybrid is significantly better than every detector shown.")
                    + (" The one-sided variant is identical on these streams (their errors only rise)."
                       if show_var else ""))

        rpath, cd = base / "friedman_ranks_all.csv", nemenyi_cd(str(base / "friedman.txt"))
        if rpath.exists() and cd:
            with st.expander(f"Mean F1 ranks (Nemenyi critical difference {cd:.2f} over all 12 detectors)"):
                r = read_csv(str(rpath))
                r = r[r.detector.isin(dets)].sort_values("F1", ascending=False)
                best = r.F1.min()
                fig = go.Figure()
                fig.add_vrect(x0=best, x1=best + cd, fillcolor=TINT, opacity=0.7, line_width=0, layer="below")
                fig.add_trace(go.Scatter(x=r.F1, y=[short(d) for d in r.detector], mode="markers+text",
                                         marker=dict(size=14, color=[color_of(d) for d in r.detector],
                                                     line=dict(color="white", width=1.5)),
                                         text=[f"{v:.2f}" for v in r.F1], textposition="middle right", cliponaxis=False,
                                         hovertemplate="%{y}: mean rank %{x:.2f}<extra></extra>"))
                fig.update_xaxes(range=[1, 12.5])
                st.plotly_chart(base_layout(fig, 340, "", "mean rank on F1 (lower = better)", legend=False),
                                width="stretch", config={"displayModeBar": False})
                caption("Shaded band: within one critical difference of the best rank. Nemenyi is the "
                        "conservative test; the Wilcoxon table above compares the hybrid pair by pair.")

    amp = VARIED / "benchmark_by_amp.csv"
    if amp.exists():
        st.subheader("F1 by size of drift (varied setting)")
        a = read_csv(str(amp)).set_index("detector")
        xs = [c for c in a.columns]
        fig = go.Figure()
        shown = [x for x in dets if x in a.index]
        for d in shown:
            fig.add_trace(go.Scatter(x=[f"+{float(c):.0%}" for c in xs], y=a.loc[d, xs], mode="lines+markers",
                                     name=short(d), line=dict(color=color_of(d), width=2.4 if d == HYBRID else 1.8),
                                     marker=dict(size=9), hovertemplate=f"{short(d)}<br>rise %{{x}}: F1 %{{y:.2f}}<extra></extra>"))
        fig.update_yaxes(range=[0, 1])
        fig.update_xaxes(type="category")
        st.plotly_chart(base_layout(fig, 380, "F1", "rise in error rate at each drift"), width="stretch",
                        config={"displayModeBar": False})
        small = a.loc[shown, xs[0]].max()
        caption(f"Small drifts are hard for every detector: at a {float(xs[0]):.0%} rise no detector exceeds F1 "
                f"{small:.2f}. As the drift grows, the mean-based detectors and the hybrid pull ahead; EDDM stays flat.")


# ------------------------------------------------------------------------------------------------
# Page 4: how a detector works
# ------------------------------------------------------------------------------------------------
def page_how():
    st.title("What does a detector actually see?")
    from drift.generator import make_stream  # the only project code the dashboard calls

    c1, c2, c3 = st.columns([2, 1, 1])
    kind = c1.segmented_control("Drift type", ["abrupt", "gradual", "incremental"], default="abrupt",
                                key="kind_pick") or "abrupt"
    amp = c2.select_slider("Error-rate jump per drift", options=[0.05, 0.10, 0.12, 0.15, 0.24], value=0.12)
    seed = c3.number_input("Random seed", 0, 999, 0)
    s = make_stream(kind=kind, amp=amp, seed=int(seed))
    n, p_low = len(s.errors), 0.10

    # expected error probability (same formula as the generator; gradual = mixture with the same expectation)
    idx = np.arange(n)
    on = np.array(s.onsets)
    levels = p_low + amp * np.arange(len(on) + 1)
    seg = np.searchsorted(on, idx, side="right")
    prev = np.maximum(seg - 1, 0)
    frac = np.where(seg == 0, 1.0, np.clip((idx - on[prev]) / max(s.width, 1), 0, 1))
    p = levels[prev] + (levels[seg] - levels[prev]) * (frac if kind != "abrupt" else 1.0)
    p[seg == 0] = levels[0]

    fig = go.Figure()
    for o in s.onsets:  # tolerance window used for scoring
        fig.add_vrect(x0=o, x1=o + s.width + TOL, fillcolor=TINT, opacity=0.6, line_width=0, layer="below")
    err_t = idx[s.errors == 1]
    fig.add_trace(go.Scatter(x=err_t, y=np.full(len(err_t), -0.04), mode="markers", name="Error (1)",
                             marker=dict(symbol="line-ns", size=10, color="#8f8d87", line=dict(width=1, color="#8f8d87")),
                             hoverinfo="skip"))
    roll = pd.Series(s.errors).rolling(300).mean()
    fig.add_trace(go.Scatter(x=idx, y=roll, mode="lines", name="Observed error rate (last 300)",
                             line=dict(color="#6250d6", width=1.6), hovertemplate="sample %{x:,}: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=idx, y=p, mode="lines", name="True error probability",
                             line=dict(color=INK, width=2.5), hovertemplate="sample %{x:,}: p = %{y:.2f}<extra></extra>"))
    drift_lines(fig, s.onsets)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Tolerance window",
                             marker=dict(symbol="square", size=14, color=TINT)))
    fig.update_yaxes(range=[-0.08, max(0.6, levels[-1] + 0.12)], tickformat=".0%")
    st.plotly_chart(base_layout(fig, 470, "error rate"), width="stretch", config={"displayModeBar": False})
    caption("A detector never sees the features, only this 0/1 stream (grey ticks = the model was wrong). "
            "Its job is to notice that the error probability (black) has stepped up despite the noise. "
            "An alarm inside the shaded window counts as a detection; anything else is a false alarm. "
            "Try a 5% jump: the steps almost vanish in the noise, which is why small drifts are hard.")
    st.markdown(
        "- **Abrupt:** the error rate switches instantly.\n"
        "- **Gradual:** old and new concepts are mixed, with the new one appearing more and more often over 500 samples.\n"
        "- **Incremental:** the error rate slides smoothly from the old level to the new one over 500 samples.\n"
        f"- This stream: {n:,} samples, {len(s.onsets)} drifts, error rate starts at {p_low:.0%} and rises by {amp:.0%} at each drift."
    )


# ------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Concept Drift Monitor", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
  .block-container {padding-top: 3.2rem; max-width: 1500px;}
  .cap {color:#52514e; font-size:1.02rem; margin:-0.4rem 0 1.0rem 0; line-height:1.4;}
  [data-testid="stMetricValue"] {font-size: 2.0rem;}
  h1 {font-size: 2.1rem !important;}
</style>
""", unsafe_allow_html=True)

nav = st.navigation([
    st.Page(page_live, title="1. Live stream", icon=":material/monitoring:", default=True),
    st.Page(page_retrain, title="2. Does retraining help?", icon=":material/autorenew:", url_path="retrain"),
    st.Page(page_compare, title="3. Detector comparison", icon=":material/leaderboard:", url_path="compare"),
    st.Page(page_how, title="4. How a detector works", icon=":material/school:", url_path="how"),
], position="top")
nav.run()
