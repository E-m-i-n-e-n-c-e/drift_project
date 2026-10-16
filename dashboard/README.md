# Demo dashboard

Offline Streamlit replay of the final project results. No Docker, Kafka, InfluxDB or Grafana needed.

```bash
pip install -r dashboard/requirements.txt      # plus the main requirements.txt for Page 4
streamlit run dashboard/app.py                 # run from the repo root (picks up .streamlit/config.toml)
```

Opens at http://localhost:8501. Navigation is the bar at the top.

| Page | Reads | Shows |
|---|---|---|
| 1. Live stream | `results/stream_kafka.csv` (falls back to `stream_local.csv`) | Replay of the real Kafka -> InfluxDB run: rolling error, true drifts, alarms (filled = detection, hollow = false alarm), delay. Toggle two-sided vs one-sided hybrid; throughput and latency come from the file |
| 2. Does retraining help? | `results/downstream_curves*.csv`, `results/downstream*.csv` | Five datasets; static vs drift-triggered LightGBM (window 300), trigger selector, retrain points, optional River HAT / ARF overlay, summary with mean ± std |
| 3. Detector comparison | `results/`, `results_heldout/`, `results_varied/` benchmark files | Fixed / unseen seeds / varied setting; F1 and delay by drift type; overall table; Holm-corrected Wilcoxon verdicts; Nemenyi ranks; F1 by drift size |
| 4. How a detector works | `drift.generator.make_stream` | What a detector receives: the 0/1 error stream and its true error probability |

Captions that make claims ("the hybrid is not the best trigger here", "not significantly different from ADWIN")
are computed from the files, so they stay correct if results are regenerated.

## Demo script (about 3 minutes)
1. Page 1, **Two-sided hybrid**, press **Play** (speed 400 replays the stream in about 13 s). Point at the drift
   lines, the filled triangles and the delay labels. At the end: 2 of 3 drifts, 3 false alarms, F1 0.50.
2. Switch to **One-sided hybrid**, press **Show all**: 3 of 3 drifts, 1 false alarm, F1 0.86. The false alarms the
   two-sided version raised came from ADWIN reacting to the error going *down*.
3. Page 2 opens on **CICIDS2017, natural order**: the static model collapses to near 0% twice; the retrained one
   recovers within a few hundred samples. Switch the trigger to DDM, then add River ARF as an overlay.
4. Optional: Page 3, switch the setting to **Varied** to show the hybrid is tied with ADWIN there.

## Before the talk
- Run it once on the presentation laptop with Wi-Fi off, at the projector resolution (tested at 1280x720 and 1440x900).
- Record a backup screen video of Pages 1 and 2.
- Colours: blue = our hybrid (dark blue = one-sided variant), orange = HDDM-A, aqua = ADWIN, violet = HDDM-W,
  yellow = DDM, pink = EDDM, grey = static model and failed pairings, purple / brown dash-dot = River HAT / ARF.
  True drift = dashed black, alarm = red, on every page.
