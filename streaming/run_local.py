"""Run the full online pipeline WITHOUT Kafka/Influx (in-process producer -> pipeline -> CSV). Good for demos/tests."""
import argparse
from drift.metrics import score
from .producer import records
from .pipeline import OnlinePipeline
from .sinks import CsvSink

ONSETS, WIDTHS = [5000, 10000, 15000], [1, 1, 1500]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--detector", default="Hybrid-OR(HDDM-A,ADWIN)")
    ap.add_argument("--out", default="results/stream_local.csv")
    a = ap.parse_args()
    pipe, sink, alarms, n = OnlinePipeline(a.detector), CsvSink(a.out), [], 0
    for r in records():
        out = pipe.process(r); sink.write(out); n += 1
        if out["alarm"]: alarms.append(out["t"])
    sink.close()
    sc = score(alarms, ONSETS, WIDTHS, n)
    print(f"true drifts: {ONSETS}  alarms: {alarms}\nF1={sc['f1']:.3f} delay={sc['delay']:.0f} false alarms={sc['fp']}")
