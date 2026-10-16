# Team Tasks: Dashboard and Presentation

Akhil is finishing the technical work (IoTID20 data, reruns, report fixes; see `REVIEW.md`). The rest of the team
builds the **demo dashboard** and the **presentation**. Both can start right now using the results already in the repo.
Final numbers will arrive later as files in the same format, so nothing you build should need rework.

Suggested split: two people on the dashboard, one on the slides. Everyone should also be able to explain the
project in their own words; Section 1 is the short version.

---

## 1. The project in one page

**Problem.** ML models running on industrial / IIoT data streams (sensor readings, network traffic) silently get
worse when the data changes over time: machines wear, firmware changes, new attack types appear. This is
**concept drift**.

**What we do.**
1. Turn the model's predictions into a stream of 0/1 values (1 = the model was wrong).
2. Feed that stream to a **drift detector**, which raises an **alarm** when the error rate jumps.
3. Compare detectors: **DDM, EDDM, HDDM-A, HDDM-W, ADWIN** (from the River library).
4. Our enhancement: a **hybrid detector = HDDM-A OR ADWIN**, ignoring repeat alarms for 500 samples after
   each alarm. It catches more drifts than either detector alone without more false alarms.
5. When an alarm fires, **retrain the model** (LightGBM) and show that this rescues accuracy.

**How we measure detectors.**
- **F1:** did alarms land on real drifts? An alarm counts as correct if it comes within a tolerance window after
  a true drift; every other alarm is a false alarm.
- **Delay:** samples between the true drift and the alarm.
- **Cost:** microseconds per sample and memory (kB).

**Headline results so far** (may change slightly after the reruns):
- Hybrid has the best F1 (0.733 vs 0.672 for the best single detector, HDDM-A), but the difference is
  **not statistically significant**.
- The pairing in our proposal (HDDM-W/DDM + EDDM) **failed**: EDDM raises far too many false alarms.
- A static model falls to about **59%** accuracy after drift; drift-triggered retraining gives **81–86%**.
- Honest caveat: in the downstream test the hybrid was *not* the best retraining trigger (ADWIN was).

**Base papers:** Sakurai et al. 2023 (detector benchmark, which we reproduce), Yang & Shami 2023 (MSANA:
two detectors trigger retraining), Yang & Shami 2021 (lightweight LightGBM + adaptive windowing).

---

## 2. Dashboard (2 people)

### Goal
A live replay of a data stream in which the audience **sees** drift happen, sees the detector react, and sees
retraining rescue accuracy. It is the demo moment of the presentation. It must run **offline on one laptop** and
must not break during the talk.

### Tool
Use **Streamlit** (Python; `pip install streamlit plotly pandas`). Put it in `dashboard/app.py` and run it with
`streamlit run dashboard/app.py`. Do **not** depend on Docker, Kafka or Grafana for the demo.

### Data you can use now

All files are in `results/`. Read them; do not recompute anything inside the dashboard.

| File | What it is | Columns |
|---|---|---|
| `stream_local.csv` | One live run of the online pipeline (River model + hybrid detector), 20,000 samples | `t` sample index, `err` 1 if model wrong, `rolling_err` error rate over last 500, `alarm` 1 if detector fired, `true_drift` 1 at a true drift, `pred` prediction |
| `benchmark_by_kind.csv` | Detector results per drift type | `kind`, `detector`, `F1`, `precision`, `recall`, `delay`, `FA_per_1k`, `us_per_sample` |
| `benchmark_overall.csv` | Detector results over all 90 streams | `detector`, `F1`, `delay`, `FA_per_1k`, `us_per_sample`, `peak_mem_kB` |
| `friedman_ranks.csv` | Mean rank of each detector (lower = better) | `detector`, `mean_rank` |
| `downstream.csv` | Static vs retrained model, summary | `strategy`, `accuracy`, `acc_after_1st_drift`, `alarms`, `retrains`, `drift_F1`, `drift_delay`, `ms_per_sample` |

True drifts in `stream_local.csv` are at samples 5000, 10000 and 15000 (the last one gradual, over 1500 samples).

### Data that will arrive later

Akhil will add **`results/downstream_curves.csv`**, one row per sample per strategy:

| Column | Meaning |
|---|---|
| `t` | sample index |
| `strategy` | e.g. `Static LightGBM`, `LightGBM + ADWIN`, `LightGBM + Hybrid-OR(HDDM-A,ADWIN)` |
| `correct` | 1 if the model was right on this sample |
| `alarm` | 1 if the detector fired here |
| `retrain` | 1 if the model was retrained here |
| `true_drift` | 1 at a true drift onset |

Until it exists, build against a small fake file with exactly these columns, so the real one drops in.
An IoTID20 version will follow in the same format (`downstream_curves_iotid20.csv`); add a dataset selector for it.

### Pages / panels

**Page 1: Live stream (the demo).**
- Play / pause button and a speed slider; the chart grows sample by sample (update every ~100 samples, not every one).
- Line chart of `rolling_err` over time.
- Dashed vertical line at each **true drift** and a red marker at each **alarm**. The gap between them is the
  detection delay; label it on hover.
- Counters: samples processed, alarms raised, current error rate.
- Short caption in plain language: "the error rate jumps after each drift; the detector should fire shortly after".

**Page 2: Does retraining help?** (uses `downstream_curves.csv`)
- Rolling accuracy (window 500) of **Static LightGBM vs LightGBM + detector**, on the same chart, with drift lines.
- Detector selector (DDM, ADWIN, EDDM, HDDM-A, Hybrid) to switch which retrained curve is shown.
- Mark retrain points.
- Small table from `downstream.csv` for the selected strategy.

**Page 3: Detector comparison.**
- Bar charts of F1 and delay per detector, grouped by drift type (`benchmark_by_kind.csv`).
- Table of `benchmark_overall.csv`, with the hybrid row highlighted.
- Mean ranks from `friedman_ranks.csv`, with a note that a gap smaller than **1.27** (the Nemenyi critical
  difference) means "not significantly different".

**Page 4 (optional): How a detector works.**
- A slider for the drift type (abrupt / gradual / incremental) that shows a synthetic 0/1 error stream and its
  error probability, so the audience understands what the detectors receive. Use `drift.generator.make_stream`
  from the repo; that is the only place the dashboard may call project code.

### Rules
- Same colour for the same detector on every page; true drifts always dashed black, alarms always red.
- Every chart gets a one-line plain-English caption saying what to look at.
- Test it on the presentation laptop, offline, at the resolution of the projector.
- Record a backup screen video of the demo in case something fails on the day.

---

## 3. Presentation (1 person)

### Format
About 12 slides, 12–15 minutes plus questions. Diagrams and charts over text; at most ~4 short bullets per slide.
Charts already exist in `results/benchmark.png`, `results/downstream.png`, `report/figs/` (PDF figures and the
architecture diagram); final versions will replace them.

### Slide plan

| # | Slide | Content |
|---|---|---|
| 1 | Title | Title, team names and roll numbers |
| 2 | The problem | Picture: model accuracy falling over time after drift, with no alarm. One sentence on why it matters in industry (safety, missed faults/attacks) |
| 3 | Types of drift | Three small diagrams: abrupt, gradual, incremental |
| 4 | Research question | Exactly as in the proposal: best trade-off between F1, delay and cost; can a hybrid improve it? |
| 5 | Base papers | One line each: what we took from it |
| 6 | How drift detection works | Model → 0/1 error stream → detector → alarm → retrain. Explain the tolerance window |
| 7 | Experimental setup | Synthetic streams (90), detectors compared, metrics, Friedman/Nemenyi test, datasets |
| 8 | Baseline results | F1 and delay per drift type chart. Note: we did not reproduce Paper 1's ranking (EDDM came last), and why |
| 9 | Our hybrid | The idea (two detectors with different strengths + refractory period), results table, "best F1 but not significant" |
| 10 | What did not work | Proposed EDDM-based pairing and AND-confirmation both failed; explain why (too many false alarms / too slow) |
| 11 | Retraining rescues accuracy | Static vs retrained accuracy curve with drift lines. Mention the hybrid was not the best trigger here |
| 12 | Live demo | Switch to the dashboard (Pages 1 and 2) |
| 13 | Limitations & future work | Synthetic data, single setting, untuned baselines, streaming pipeline not measured |
| 14 | Conclusion | Answer the research question in 2–3 sentences: no single detector wins on everything; the hybrid gives the best F1 at low cost; drift-triggered retraining is essential |

Numbers for slides 8–11 are in the CSVs listed in Section 2 and in `report/report.tex`. Use placeholders for
now and update them once the final results arrive; Akhil will tell you when they are final.

### Speaker split
Each of the four members presents a part: (1) problem and background, slides 1–5; (2) method, slides 6–8;
(3) hybrid and results, slides 9–11; (4) demo and conclusion, slides 12–14.

### Questions to prepare for
- Why did EDDM do so badly when Paper 1 says it is the most accurate?
- Why is the hybrid's improvement not significant, and does it still matter?
- Why did you change the hybrid from what the proposal said?
- How do you know when a real dataset (IoTID20) has drift, given it has no drift labels?
- Why LightGBM, which is not an online learner?
- What is the cost of the hybrid compared with a single detector?

---

## 4. Workflow

- Work on your own git branch (`dashboard`, `slides`) and merge when done.
- Do not edit files in `drift/`, `streaming/` or `results/`; those belong to the technical side. If you need
  data in another shape, ask Akhil rather than changing the code.
- Cut-off: final numbers by **[date]**, dashboard and slides frozen by **[date]**, full rehearsal on **[date]**.
