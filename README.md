# Detection of Concept Drift in Industrial Data Streams (Track B)

Kafka -> (Spark windows) -> River online learner + drift detectors -> InfluxDB -> Grafana, plus an offline
benchmark reproducing Sakurai et al. (2023) and a drift-triggered LightGBM study (Yang & Shami style).

## Layout
| Path | Purpose |
|---|---|
| `drift/generator.py` | Synthetic abrupt / gradual / incremental drift streams + simulated IIoT stream with real concept drift |
| `drift/detectors.py` | DDM, EDDM, HDDM-A, HDDM-W, ADWIN (River) + the **hybrid dual-detector trigger** |
| `drift/metrics.py` | Tolerance-window F1, precision, recall, delay, false alarms per 1000 samples |
| `drift/benchmark.py` | 30 seeds x 3 drift types x 9 detectors, Friedman + Nemenyi, memory, latency |
| `drift/tune_hybrid.py` | Refractory-period sweep on separate seeds (1000-1019) |
| `drift/make_figures.py` | Report figures (architecture, benchmark, downstream) |
| `report/` | IEEE two-column report (`report.tex`, `report.pdf`, `figs/`) |
| `drift/adaptive.py` | Static vs drift-triggered LightGBM vs River ARF/HAT (prequential) |
| `streaming/` | Producer, consumer, Spark window job, local (no-Kafka) runner |
| `docker-compose.yml`, `grafana/` | Kafka (KRaft), InfluxDB 2.7, Grafana with provisioned dashboard |

## Run
```bash
pip install -r requirements.txt
python -m drift.benchmark --seeds 30      # -> results/benchmark_*.csv, friedman.txt, benchmark.png
python -m drift.benchmark --seeds 30 --start 2000 --out results_heldout   # confirmatory run
python -m drift.adaptive                  # -> results/downstream.csv/.png
python -m streaming.run_local             # in-process pipeline -> results/stream_local.csv

# live pipeline
docker compose up -d
python -m streaming.consumer &            # Kafka -> River -> InfluxDB
python -m streaming.producer --rate 200   # replay stream
# Grafana: http://localhost:3000  (dashboard "Concept Drift Monitor")
# optional: spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 streaming/spark_features.py
```

## Method notes
* Drift-detection scoring: the first alarm in `[onset, onset + width + 300)` is a true positive; every other alarm is a false alarm; undetected onsets are misses.
* Every synthetic drift is an error-rate *increase* (one-sided detectors are the intended use case).
* Hybrid = HDDM-A OR ADWIN with a 500-sample refractory period. The pair and R=500 were picked in exploratory runs on seeds 0-29 (so those numbers are optimistic); `tune_hybrid.py` (seeds 1000-1019) is a later sweep, and `results_heldout/` (seeds 2000-2029, `--start 2000`) is the confirmatory run. Baselines use River defaults (untuned).
* LightGBM is not online: after an alarm the pipeline waits 250 samples, then retrains on the latest 500 labelled samples.
* ADWIN is two-sided, so it can also fire when a learner *improves* (e.g. cold start in `run_local`).

## Not done / caveats
* **IoTID20 / CICIDS2017 are not included** (no dataset access here). Add a loader that streams the CSV
  (sorted so attack sub-types change over time, as in Paper 3) into `drift.adaptive` / `streaming.producer`; real datasets have no drift labels, so injected drift is required for F1/delay.
* Kafka, InfluxDB, Grafana and Spark files were written but **not executed** in the build environment.
* Downstream results are a single simulated stream (seed 0).
