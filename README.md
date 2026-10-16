# Detection of Concept Drift in Industrial Data Streams

Benchmark of drift detectors (DDM, EDDM, HDDM-A, HDDM-W, ADWIN) reproducing Sakurai et al. (2023), a proposed
**hybrid trigger (HDDM-A OR ADWIN with a refractory period)**, a drift-triggered LightGBM retraining study on
simulated IIoT data, IoTID20 and CICIDS2017 (Yang & Shami style), and a live
Kafka → River → InfluxDB → Grafana pipeline. Report: `report/report.pdf`. Review and status: `REVIEW.md`.
Dashboard and presentation instructions: `TEAM_TASKS.md`.

## Layout
| Path | Purpose |
|---|---|
| `drift/generator.py` | Synthetic abrupt / gradual / incremental error streams (fixed or varied setting) + simulated IIoT stream |
| `drift/datasets.py` | IoTID20 / CICIDS2017 loader and injected drift (feature remapping at known positions) |
| `drift/detectors.py` | River detectors, the hybrid, and ablations (AND rule, EDDM pairs, refractory-only, one-sided ADWIN) |
| `drift/metrics.py` | Tolerance-window F1, delay, penalised delay, false alarms per 1000 samples |
| `drift/benchmark.py` | Detector benchmark + Friedman / Nemenyi + Holm-corrected Wilcoxon (F1, delay, time) |
| `drift/tune_hybrid.py` | Refractory-period sweep on seeds 1000–1019 |
| `drift/adaptive.py` | Static vs drift-triggered LightGBM vs River ARF / HAT, all datasets, 10 seeds |
| `drift/make_figures.py` | Report figures |
| `streaming/` | Producer, consumer, online pipeline, sinks, local (no-Kafka) runner, optional Spark job |
| `docker-compose.yml`, `grafana/` | Kafka (KRaft), InfluxDB 2.7, Grafana with a provisioned dashboard |
| `data/` | IoTID20 and CICIDS2017 samples published with Paper 2 ([MSANA repo](https://github.com/Western-OC2-Lab/MSANA-Online-Data-Stream-Analytics-And-Concept-Drift-Adaptation), MIT licence) |
| `results/`, `results_heldout/`, `results_varied/` | Outputs (CSV, PNG); see `results/RESULTS.md` |
| `tests/` | Unit tests (`pytest tests`) |
| `report/` | IEEE two-column report (`report.tex`, `report.pdf`, `figs/`) |

## Setup
```bash
python3 -m venv .venv-mac && source .venv-mac/bin/activate   # any name; .venv/ is an old Windows env
pip install -r requirements.txt                              # macOS: brew install libomp (for LightGBM)
pytest tests
```

## Run
```bash
python -m drift.benchmark --seeds 30                                    # fixed setting, seeds 0-29 -> results/
python -m drift.benchmark --seeds 30 --start 2000 --out results_heldout # unseen seeds
python -m drift.benchmark --seeds 100 --start 3000 --varied --out results_varied   # varied setting (300 streams)
python -m drift.tune_hybrid                                             # refractory sweep
python -m drift.adaptive                                                # downstream study (~10 min)
python -m streaming.run_local                                           # in-process pipeline -> results/stream_local.csv
python -m drift.make_figures                                            # report figures
cd report && tectonic report.tex                                        # or pdflatex

# live pipeline (Docker running)
docker compose up -d
python -m streaming.consumer --max 20000 &      # Kafka -> River -> InfluxDB; prints F1, throughput, latency
python -m streaming.producer --rate 1000        # replay the simulated stream
# Grafana: http://localhost:3000 (dashboard "Concept Drift Monitor")
```

## Method notes
* Scoring: the first alarm in `[onset, onset + width + 300)` (capped at the next onset) is a true positive;
  every other alarm is a false alarm. Penalised delay counts a miss as the whole window.
* The hybrid's pairing and R=500 were chosen on seeds 0–29 (optimistic); `results_heldout/` and `results_varied/`
  are confirmatory. Baselines use River defaults.
* Injected drift on real data: rows are shuffled (removes natural drift), then at 25/50/75% of the stream the five
  most important features swap columns with other features (abrupt, abrupt, gradual over 500 rows).
* LightGBM retraining: after an alarm, wait 300 samples, retrain on post-alarm data only (≤500), reset the detector.
* Live pipeline: the detector is reset and skips 300 records at the start and after each model reset.
  `Hybrid-OR(HDDM-A,ADWIN-up)` uses a one-sided ADWIN that ignores error *decreases*.

## Caveats
* Dataset files are the published samples (1% IoTID20, 0.5% CICIDS2017), not the full datasets.
* The Spark job (`streaming/spark_features.py`) was not run; it is not on the scoring path.
* The live run is one consumer on one machine (~820 records/s); timings vary by machine.
