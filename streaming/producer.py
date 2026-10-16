"""Replay the simulated IIoT stream into Kafka topic `sensors` (ground-truth drift flag included)."""
import argparse, json, time
import numpy as np
from drift.generator import make_industrial_stream


def records(n=20000, seed=0):
    X, y, meta = make_industrial_stream(n=n, seed=seed)
    onsets = set(meta["onsets"])
    for i in range(n):
        yield dict(t=i, ts=time.time(), features={j: float(v) for j, v in enumerate(X[i])},
                   label=int(y[i]), true_drift=int(i in onsets))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:9092"); ap.add_argument("--topic", default="sensors")
    ap.add_argument("--rate", type=float, default=200, help="records per second")
    a = ap.parse_args()
    from kafka import KafkaProducer
    p = KafkaProducer(bootstrap_servers=a.bootstrap, value_serializer=lambda v: json.dumps(v).encode())
    for r in records():
        p.send(a.topic, r); time.sleep(1 / a.rate)
    p.flush()
