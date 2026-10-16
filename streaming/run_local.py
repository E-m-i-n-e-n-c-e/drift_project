"""Run the full online pipeline WITHOUT Kafka/Influx (in-process producer -> pipeline -> CSV). Good for demos/tests."""
import argparse
from .producer import records
from .pipeline import OnlinePipeline
from .sinks import CsvSink

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--detector", default="Hybrid-OR(HDDM-A,ADWIN)")
    a = ap.parse_args()
    pipe, sink, alarms = OnlinePipeline(a.detector), CsvSink("results/stream_local.csv"), []
    for r in records():
        out = pipe.process(r); sink.write(out)
        if out["alarm"]: alarms.append(out["t"])
    sink.close(); print("true drifts: [5000, 10000, 15000]  alarms:", alarms)
