# Project Review: Concept Drift Detection in Industrial Data Streams

Review of the current repository against the project proposal (`Concept_Drift_Detection_Proposal.pdf`).
Reviewed: 2026-10-16.

## 1. Overall verdict

Nearly the whole project has been built end to end, and the core of it is sound. The detector benchmark was
re-run during this review and gave exactly the numbers saved in `results/` and `results_heldout/`. The report
is honest about its own limits.

Three things the proposal promises are missing or weaker than they look:

- **No real industrial data.** IoTID20 / CICIDS2017 were never used. All results come from synthetic or simulated streams.
- **The hybrid detector is not the one the proposal describes.**
- **The hybrid does not help the downstream model.** It gives the lowest accuracy of all the retraining variants.

## 2. What is done, and whether it is correct

| Part | Status | Assessment |
|---|---|---|
| Synthetic drift generator: abrupt / gradual / incremental (`drift/generator.py`) | Done | Correct, but simplified (see change 4) |
| Scoring: F1, delay, false alarms per 1000 samples (`drift/metrics.py`) | Done | Correct. Strict but fair: every extra alarm counts as a false alarm |
| Paper 1 benchmark with Friedman / Nemenyi tests (`drift/benchmark.py`) | Done | Correct. The Nemenyi critical value for 9 detectors (q = 3.102) checks out |
| Hybrid detector and its variants (`drift/detectors.py`) | Done | Correct. Uses a different detector pair from the proposal |
| Confirmation run on unseen seeds 2000–2029 (`results_heldout/`) | Done | Good practice: the hybrid's settings were picked on the main seeds, so this checks for bias |
| Refractory-period sweep on seeds 1000–1019 (`drift/tune_hybrid.py`) | Done | Fine |
| Static vs drift-triggered LightGBM, River ARF / HAT (`drift/adaptive.py`) | Done | Only on one simulated stream with one seed |
| Kafka / Spark / InfluxDB / Grafana pipeline (`streaming/`, `docker-compose.yml`, `grafana/`) | Written | **Never run.** Not in the proposal either |
| In-process pipeline without Kafka (`streaming/run_local.py`) | Done | Ran; output in `results/stream_local.csv` |
| IEEE-style report (`report/report.tex`, `report.pdf`) | Drafted | Honest. Some numbers do not match the result files |

### 2.1 Independent check: is the hybrid's gain real?

The hybrid ignores repeat alarms for 500 samples after it fires (a "refractory period"); the single detectors do
not. Test: does that rule alone explain the hybrid's higher F1? Each single detector was given the same rule.
It mostly does not, so the gain from combining two detectors is real:

| Detector | F1, seeds 0–29 | F1, held-out seeds 2000–2029 |
|---|---|---|
| HDDM-A | 0.672 | 0.723 |
| HDDM-A + same 500-sample rule | 0.680 | 0.734 |
| ADWIN + same 500-sample rule | 0.611 | 0.641 |
| **Hybrid (HDDM-A OR ADWIN, R = 500)** | **0.733** | **0.786** |
| ADWIN OR EDDM, R = 500 (Paper 2's pair, not tested in the repo) | 0.347 | 0.360 |

The gain comes from higher recall (0.689 → 0.759 on seeds 0–29), while false alarms stay low.

However, the hybrid is **not significantly better** than HDDM-A or ADWIN: the rank gap to HDDM-A is 0.66, below
the Nemenyi critical difference of 1.27. The report already says this.

## 3. Changes needed

1. **State clearly that the hybrid changed from the proposal.**
   The proposal promised a fast detector (DDM or HDDM-W) plus EDDM, tuned for fewer false alarms. That pairing
   was tested and does badly (F1 0.22–0.42), and the final hybrid is HDDM-A + ADWIN. That is a valid finding, but the
   report should present it as: we tested what we proposed, it failed, here is why, and here is what works.
   Also add Paper 2's ADWIN + EDDM pair to the registry as a variant, since the proposal refers to it.

2. **The downstream result works against the hybrid, and the report skims over it.**
   In `results/downstream.csv`:

   | Strategy | Accuracy |
   |---|---|
   | River HAT (online) | 0.874 |
   | LightGBM + ADWIN | 0.863 |
   | LightGBM + DDM | 0.830 |
   | LightGBM + EDDM | 0.830 |
   | LightGBM + HDDM-A | 0.819 |
   | **LightGBM + Hybrid** | **0.811** |
   | Static LightGBM | 0.586 |

   The abstract's "restores accuracy to 0.81–0.86" hides that the hybrid is the worst retraining variant. This needs
   a proper discussion and more than one seed before any conclusion is drawn.

3. **Report numbers do not match the result files.**
   The report's timing and memory columns (for example hybrid 7.4 µs/sample, HDDM-W 6.1 kB) differ from
   `results/benchmark_overall.csv` (3.87 µs, 4.4 kB). They probably come from a different run or machine.
   Regenerate all tables and figures from one run.

4. **The Paper 1 reproduction is only partial.**
   - It uses a home-made generator, not Paper 1's open-source one, with a single setting: 10,000 samples, 3 drifts,
     amplitude 0.12, width 500. Paper 1 varies stream size, number of drifts and drift width/amplitude.
   - The Friedman test runs on F1 only; Paper 1 also tests detection delay and detection time.
   - The result contradicts the paper: EDDM comes last here, but Paper 1 found it the most accurate. The report
     acknowledges this, but varying the parameters would make the claim "reproduce Paper 1" defensible.

5. **Small correctness fixes.**
   - `drift/adaptive.py`: drift scoring uses a 1500-sample width for all three drifts, but the drifts at 5000 and
     10000 are abrupt. That makes the tolerance window far wider than in the synthetic benchmark (301 samples).
   - `drift/adaptive.py`: the retraining window is 500 samples, but the pipeline only waits 250 after an alarm, so
     half of the retraining data comes from the old concept. Consider `WAIT >= WINDOW`, or report it as a design choice.
   - `drift/benchmark.py:51`: `k, N = piv.shape[1][0] if False else len(piv.columns), len(piv)` is dead code;
     simplify to `k, N = len(piv.columns), len(piv)`.

6. **Repo hygiene.**
   - Not a git repository yet: initialise one so the team can share and track work.
   - `.venv/` is a Windows virtualenv (`Scripts/`), so it does not work on macOS/Linux. Delete it and gitignore it,
     along with `__pycache__/` and `.DS_Store`.
   - `requirements.txt` is unpinned, while the report lists exact versions (River 0.26.1, LightGBM 4.7.0, ...). Pin them.
   - There are no tests. Minimal tests for `metrics.score` and `generator.make_stream` would be cheap and useful.
   - The proposal mentions AUC and "scikit-learn Adaptive Random Forest". AUC is never computed, and ARF comes from
     River, not scikit-learn. Fix the wording or the code.

## 4. What remains to do (by priority)

1. **IoTID20 (or CICIDS2017).** This is the biggest gap. Write a loader that streams the CSV, sorted so attack
   types change over time (as in Paper 3), inject drift at known points so F1 and delay can be measured, and rerun
   `drift/adaptive.py` and the hybrid on it. Without this, nothing in the project uses industrial data.
2. **Rerun the downstream study with several seeds** and report mean ± standard deviation.
3. **Widen the synthetic benchmark** (stream sizes, drift counts, widths, amplitudes) and run the statistical tests
   on delay and detection time as well as F1.
4. **Decide what to do with the Kafka / Grafana pipeline.** Either run it with `docker compose up -d` and capture a
   Grafana screenshot plus end-to-end latency, or present it as an optional extra. The report title currently
   leads with it, even though it never ran. If the course's "Track B" requires the pipeline, this moves up the list.
5. **Fix the report:** the hybrid change, the downstream finding, numbers that match the result files, and a new
   title if the pipeline is demoted.
6. **Spread understanding across the team.** One person built nearly everything; every member should be able to
   explain the generator, the scoring rule, the hybrid and the downstream study.

## 5. How the hybrid check was run

The check in section 2.1 wraps each detector with the same 500-sample refractory rule and scores it with
`drift.metrics.score` on the same streams that `drift.benchmark` uses (`make_stream`, 30 seeds × 3 drift types,
seeds 0–29 and 2000–2029). It was run with River 0.26.1. The HDDM-A, ADWIN and hybrid numbers match
`results/benchmark_overall.csv` and `results_heldout/benchmark_overall.csv` exactly.
