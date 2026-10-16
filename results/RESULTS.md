# Results files

Regenerate with the commands in `README.md`. `results_heldout/` (seeds 2000–2029) and `results_varied/`
(varied setting, seeds 3000–3099) have the same benchmark files as `results/` (seeds 0–29).

| File | Contents |
|---|---|
| `benchmark_raw.csv` | One row per stream × detector (varied runs also hold the stream setting) |
| `benchmark_overall.csv`, `benchmark_by_kind.csv` | Mean F1, precision, recall, delay, penalised delay, false alarms / 1k, µs per sample, peak kB |
| `benchmark_by_amp.csv` (varied only) | F1 by drift amplitude |
| `friedman.txt` | Friedman / Nemenyi and Holm-corrected Wilcoxon (hybrid vs each) for F1, penalised delay, time |
| `friedman_ranks.csv`, `friedman_ranks_all.csv`, `wilcoxon_f1.csv` | The same as tables |
| `downstream*.csv` | Downstream summary per dataset (mean over 10 seeds; `_natural` = one run in file order) |
| `downstream_curves*.csv` | Per-sample correctness, alarms, retrains, true drifts per strategy (seed 0) |
| `downstream_runs.csv` | Every downstream run |
| `downstream.png` | Rolling accuracy on the simulated, IoTID20 and CICIDS2017 streams |
| `stream_local.csv` | In-process online pipeline run (hybrid detector) |
| `stream_kafka.csv` | The Kafka → InfluxDB live runs, exported from InfluxDB (two detectors, with end-to-end latency) |
