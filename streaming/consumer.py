"""Kafka consumer: sensors topic -> OnlinePipeline (River + drift detector) -> InfluxDB (Grafana reads it)."""
import argparse, json
from kafka import KafkaConsumer
from .pipeline import OnlinePipeline
from .sinks import InfluxSink

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:9092"); ap.add_argument("--topic", default="sensors")
    ap.add_argument("--detector", default="Hybrid-OR(HDDM-A,ADWIN)"); ap.add_argument("--run", default="run1")
    a = ap.parse_args()
    c = KafkaConsumer(a.topic, bootstrap_servers=a.bootstrap, auto_offset_reset="earliest",
                      value_deserializer=lambda b: json.loads(b))
    pipe, sink = OnlinePipeline(a.detector), InfluxSink(run=a.run)
    for msg in c:
        rec = msg.value
        rec["features"] = {int(k): v for k, v in rec["features"].items()}
        out = pipe.process(rec)
        sink.write(out)
        if out["alarm"]:
            print(f"DRIFT ALARM at t={out['t']}")
