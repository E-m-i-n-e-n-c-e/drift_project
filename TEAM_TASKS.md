# Team Tasks: Dashboard and Presentation

The technical work is done: IoTID20 and CICIDS2017 are in, all experiments have been re-run, and the live
Kafka → InfluxDB → Grafana pipeline has been run and measured. The final report is `report/report.pdf`, and the
status of every review item is in `REVIEW.md`. The rest of the team builds the **demo dashboard** and the
**presentation**. All the data you need is already in `results/` and `results_varied/`.

Suggested split: two people on the dashboard, one on the slides. Everyone should be able to explain the project
in their own words; Section 1 is the short version, and the report is the long one.

---

## 1. The project in one page

**Problem.** ML models running on industrial / IIoT data streams (sensor readings, network traffic) silently get
worse when the data changes over time: machines wear, firmware changes, new attack types appear. This is
**concept drift**.

**What we do.**
1. Turn the model's predictions into a stream of 0/1 values (1 = the model was wrong).
2. Feed that stream to a **drift detector**, which raises an **alarm** when the error rate jumps.
3. Compare detectors: **DDM, EDDM, HDDM-A, HDDM-W, ADWIN** (from the River library).
4. Our enhancement: a **hybrid detector = HDDM-A OR ADWIN**, ignoring repeat alarms for 500 samples after each
   alarm. A variant with a **one-sided ADWIN** (fires only when the error goes *up*) is better in the live pipeline.
5. When an alarm fires, **retrain the model** (LightGBM) and show that this rescues accuracy.

**Data.**
- **Synthetic error streams** (Paper 1 style), where we know exactly when each drift happens: 90 streams in one
  fixed setting, 90 more on unseen seeds, and 300 with varied length / number of drifts / width / size of drift.
- **A simulated IIoT sensor stream** with three known drifts (for the downstream study and the live demo).
- **IoTID20 and CICIDS2017**: the samples published by the authors of Papers 2 and 3. Used two ways:
  *natural order* (realistic, drift positions unknown) and *shuffled with injected drift* (known drift positions,
  so detection can be scored). Injected drift = at three points, the most important features swap columns,
  like a sensor or firmware remapping.

**How we measure detectors.**
- **F1:** did alarms land on real drifts? An alarm counts as correct if it comes within a tolerance window after a
  true drift (drift width + 300 samples); every other alarm is a false alarm.
- **Delay:** samples between the true drift and the alarm.
- **Cost:** microseconds per sample and memory (kB).

**Headline results.**
- Hybrid has the best detection F1 in every synthetic setting: **0.733** vs 0.672 for HDDM-A (fixed setting),
  **0.786** vs 0.723 (unseen seeds), **0.591** vs 0.550 for ADWIN (varied setting).
- Significance is mixed: the hybrid **beats HDDM-A** significantly in all settings, but is **tied with ADWIN**
  in the varied setting, and part of its gain over HDDM-A comes from the refractory period alone.
- The pairing in our proposal (DDM/HDDM-W + EDDM) **failed**, and so did Paper 2's ADWIN + EDDM: EDDM raises far
  too many false alarms. We did **not** reproduce Paper 1's finding that EDDM is the most accurate.
- **Retraining rescues accuracy:** static model **0.60** vs about **0.83** with retraining (simulated); on
  CICIDS2017 in natural order, attack-detection F1 goes from **0.02** (static) to **0.74–0.89**.
- But the hybrid is **not** the best *retraining trigger*: all detectors are close, and River's online models
  (HAT, ARF) are often as good or better.
- **Live Kafka pipeline:** ~820 records/s per consumer; median end-to-end latency 2–3 ms. One-sided hybrid F1
  0.86 (1 false alarm) vs 0.50 (3 false alarms) for the two-sided one.

**Base papers:** Sakurai et al. 2023 (detector benchmark, which we reproduce), Yang & Shami 2023 (MSANA: two
detectors trigger retraining; source of our IoTID20/CICIDS2017 samples), Yang & Shami 2021 (lightweight
LightGBM + adaptive windowing).

---

## 2. Dashboard (2 people)

### Goal
A replay of a data stream in which the audience **sees** drift happen, sees the detector react, and sees
retraining rescue accuracy. It is the demo moment of the presentation. It must run **offline on one laptop** and
must not break during the talk.

### Tool
Use **Streamlit** (Python; `pip install streamlit plotly pandas`). Put it in `dashboard/app.py` and run it with
`streamlit run dashboard/app.py`. Do **not** depend on Docker, Kafka or Grafana for the demo (see "Optional:
Grafana" below).

### Data

Everything is final and in the repo. Read the files; do not recompute anything inside the dashboard.

**Streams for the live replay (Page 1)**

| File | What it is | Columns |
|---|---|---|
| `results/stream_local.csv` | Online pipeline (River model + hybrid detector) on the simulated IIoT stream, 20,000 records | `t` record index, `err` 1 if model wrong, `rolling_err` error rate over last 500, `alarm` 1 if detector fired, `true_drift` 1 at a true drift, `pred` prediction, `latency_ms` (ignore: in-process, near zero) |
| `results/stream_kafka.csv` | The same stream sent **through Kafka into InfluxDB**, for two detectors | `detector` (`Hybrid-OR(HDDM-A,ADWIN)` or `Hybrid-OR(HDDM-A,ADWIN-up)`), `t`, `time`, `err`, `rolling_err`, `alarm`, `true_drift`, `latency_ms` (real end-to-end latency) |

True drifts in both are at records 5000, 10000 and 15000 (the last one gradual, over 1500 records).

**Retraining study (Page 2)**: one curves file per dataset, one row per sample per strategy (seed 0)

| File | Dataset |
|---|---|
| `results/downstream_curves.csv` | Simulated IIoT stream (true drift) |
| `results/downstream_curves_iotid20.csv` | IoTID20, injected drift |
| `results/downstream_curves_cicids2017.csv` | CICIDS2017, injected drift |
| `results/downstream_curves_iotid20_natural.csv` | IoTID20, natural order (no drift labels) |
| `results/downstream_curves_cicids2017_natural.csv` | CICIDS2017, natural order (no drift labels): **the most dramatic one for the demo** |

Columns: `t` sample index, `strategy`, `correct` 1 if right, `alarm` 1 if the detector fired, `retrain` 1 if the
model was retrained, `true_drift` 1 at a true drift onset (always 0 in the natural-order files).
Strategies: `Static LightGBM`, `LightGBM + DDM`, `LightGBM + HDDM-A`, `LightGBM + ADWIN`, `LightGBM + EDDM`,
`LightGBM + Hybrid-OR(HDDM-A,ADWIN)`, `River ARF (online)`, `River HAT (online)`.

Matching summary tables (mean over 10 seeds; natural-order files are one run):
`results/downstream.csv`, `downstream_iotid20.csv`, `downstream_cicids2017.csv`, `downstream_iotid20_natural.csv`,
`downstream_cicids2017_natural.csv`. Columns: `strategy`, `accuracy`, `f1`, `alarms`, `retrains`,
`ms_per_sample`, `acc_after_1st_drift`, `drift_F1`, `drift_delay`, `accuracy_std`, `f1_std` (the last few are
missing for natural-order files, which have no drift labels). `results/downstream_runs.csv` has every single run.

**Detector comparison (Page 3)**

| File | What it is |
|---|---|
| `results/benchmark_overall.csv` | Fixed setting, 90 streams: `detector`, `F1_std`, `F1`, `precision`, `recall`, `delay`, `pen_delay`, `FA_per_1k`, `us_per_sample`, `peak_mem_kB` |
| `results/benchmark_by_kind.csv` | Same, split by `kind` (abrupt / gradual / incremental) |
| `results_varied/benchmark_overall.csv`, `benchmark_by_kind.csv` | Same columns, 300 streams with varied settings |
| `results_varied/benchmark_by_amp.csv` | F1 by size of the drift (0.05 / 0.10 / 0.15 rise in error rate) |
| `results/friedman_ranks_all.csv` | Mean rank per detector for F1, penalised delay and time (lower = better) |
| `results/wilcoxon_f1.csv` | Hybrid vs every other detector: `mean_diff` in F1, `p_holm` (< 0.05 = significant) |
| `results_heldout/…` | Same files for the unseen seeds 2000–2029 |

`delay` is over detected drifts only; `pen_delay` counts a missed drift as the whole tolerance window (fairer).

### Pages / panels

**Page 1: Live stream (the demo).**
- Play / pause button and a speed slider; the chart grows as the stream plays (update every ~100 records).
- Line chart of `rolling_err`, dashed vertical line at each **true drift**, red marker at each **alarm**.
  The gap between them is the detection delay; show it on hover.
- Counters: records processed, alarms raised, current error rate.
- Toggle between the two detectors in `stream_kafka.csv` to show the one-sided version removes false alarms.
- Caption: "the error rate jumps after each drift; the detector should fire shortly after; this data went
  through Kafka and InfluxDB at ~800 records/s with ~3 ms latency".

**Page 2: Does retraining help?**
- **Dataset selector** (the five curves files).
- Rolling accuracy (window 300) of **Static LightGBM vs LightGBM + detector** on one chart, with drift lines and
  retrain markers.
- Detector selector (DDM, ADWIN, EDDM, HDDM-A, Hybrid) for the retrained curve; option to overlay River HAT / ARF.
- The selected dataset's summary table, showing `accuracy ± accuracy_std` and `f1`.

**Page 3: Detector comparison.**
- Setting selector: fixed / unseen seeds / varied.
- Bar charts of F1 and delay per detector, grouped by drift type.
- Overall table with the hybrid row highlighted; mean ranks; and the Wilcoxon table rendered as
  "significantly better / not significant" next to each detector.
- F1 vs drift size (`benchmark_by_amp.csv`): small drifts are hard for every detector.

**Page 4 (optional): How a detector works.**
- A slider for the drift type (abrupt / gradual / incremental) that shows a synthetic 0/1 error stream and its
  error probability. Use `drift.generator.make_stream` from the repo; that is the only place the dashboard may
  call project code.

### Optional: Grafana
The real pipeline works: `docker compose up -d`, then `python -m streaming.consumer --max 20000 &` and
`python -m streaming.producer --rate 1000`, then open http://localhost:3000 (dashboard "Concept Drift Monitor";
data appears within seconds). A **screenshot or short screen recording** of it makes a good slide showing the
system is real. Do not run it live during the talk.

### Rules
- Same colour for the same detector on every page; true drifts always dashed black, alarms always red.
- Every chart gets a one-line plain-English caption saying what to look at.
- Test it on the presentation laptop, offline, at the resolution of the projector.
- Record a backup screen video of the demo in case something fails on the day.

---

## 3. Presentation (1 person)

### Format
About 14 slides, 12–15 minutes plus questions. Diagrams and charts over text; at most ~4 short bullets per slide.
Final figures: `report/figs/arch.pdf` (architecture), `bench.pdf` (detector F1/delay), `downstream.pdf`
(retraining on three datasets), `live.pdf` (Kafka run). Numbers: `report/report.pdf` tables, or the CSVs above.

### Slide plan

| # | Slide | Content |
|---|---|---|
| 1 | Title | Title, team names and roll numbers |
| 2 | The problem | Picture: accuracy collapsing after drift (CICIDS2017 natural-order static curve is perfect). Why it matters (missed faults / attacks) |
| 3 | Types of drift | Three small diagrams: abrupt, gradual, incremental |
| 4 | Research question | As in the proposal: best trade-off between F1, delay and cost; can a hybrid improve it? |
| 5 | Base papers | One line each: what we took from it |
| 6 | System architecture | `arch.pdf`: Kafka → River model + detector → InfluxDB → Grafana |
| 7 | How drift detection works | Model → 0/1 error stream → detector → alarm → retrain. Tolerance window |
| 8 | Experimental setup | Synthetic (fixed, unseen seeds, varied), simulated IIoT, IoTID20 / CICIDS2017 (natural + injected), metrics, tests |
| 9 | Baseline results | `bench.pdf`. EDDM fastest but most false alarms; we did not reproduce Paper 1's ranking |
| 10 | Our hybrid | Idea, results across the three settings, the significance story (beats HDDM-A, tied with ADWIN), refractory-only check |
| 11 | What did not work | Proposed EDDM pairing, Paper 2's ADWIN+EDDM, AND-confirmation: why they fail |
| 12 | Retraining rescues accuracy | `downstream.pdf`. Static vs retrained; online River models are competitive; hybrid not the best trigger |
| 13 | Live system + demo | `live.pdf` + throughput/latency numbers, then switch to the dashboard |
| 14 | Limitations, future work, conclusion | Answer the research question in 2–3 sentences |

### Speaker split
Each of the four members presents a part: (1) problem and background, slides 1–5; (2) system and method,
slides 6–8; (3) results, slides 9–12; (4) live system, demo and conclusion, slides 13–14.

### Questions to prepare for (answers are in the report's Discussion section)
- Why did EDDM do so badly when Paper 1 says it is the most accurate?
- Is the hybrid's improvement significant? Against what, and in which setting?
- Why did you change the hybrid from what the proposal said?
- How can you measure detection F1 on IoTID20, which has no drift labels? (Injected drift; why shuffle first?)
- Why does the hybrid behave exactly like ADWIN on the real datasets?
- Why LightGBM, which is not an online learner? Why do River's online models do so well?
- What limits the live pipeline to ~820 records/s, and how would you scale it?

---

## 4. Workflow

- Work on your own git branch (`dashboard`, `slides`) and merge when done.
- Do not edit files in `drift/`, `streaming/`, `data/` or `results*/`. If you need data in another shape, ask
  Akhil rather than changing the code.
- Cut-off: dashboard and slides frozen by **[date]**, full rehearsal on **[date]**.
