"""Kafka consumer: sensors topic -> OnlinePipeline (River + drift detector) -> InfluxDB (Grafana reads it).
With --max N it stops after N records and prints alarms, drift F1, throughput and end-to-end latency."""
import argparse, json, time
import numpy as np
from kafka import KafkaConsumer
from drift.metrics import score
from .pipeline import OnlinePipeline
from .sinks import InfluxSink
from .run_local import ONSETS, WIDTHS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:9092"); ap.add_argument("--topic", default="sensors")
    ap.add_argument("--detector", default="Hybrid-OR(HDDM-A,ADWIN)"); ap.add_argument("--run", default="run1")
    ap.add_argument("--max", type=int, default=0, help="stop after this many records (0 = run forever)")
    a = ap.parse_args()
    c = KafkaConsumer(a.topic, bootstrap_servers=a.bootstrap, auto_offset_reset="earliest",
                      value_deserializer=lambda b: json.loads(b))
    pipe, sink = OnlinePipeline(a.detector), InfluxSink(run=a.run)
    alarms, lat, t0 = [], [], None
    for msg in c:
        rec = msg.value
        t0 = t0 or time.time()
        rec["features"] = {int(k): v for k, v in rec["features"].items()}
        out = pipe.process(rec)
        sink.write(out)
        lat.append(out.get("latency_ms", np.nan))
        if out["alarm"]:
            alarms.append(out["t"]); print(f"DRIFT ALARM at t={out['t']}", flush=True)
        if a.max and len(lat) >= a.max:
            break
    sink.close()
    if a.max:
        sc, dt = score(alarms, ONSETS, WIDTHS, len(lat)), time.time() - t0
        print(f"records={len(lat)} alarms={alarms} F1={sc['f1']:.3f} delay={sc['delay']:.0f} false_alarms={sc['fp']}\n"
              f"throughput={len(lat) / dt:.0f} rec/s  latency_ms p50={np.nanpercentile(lat, 50):.1f} "
              f"p95={np.nanpercentile(lat, 95):.1f} max={np.nanmax(lat):.1f}")
