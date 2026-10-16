"""Tune the hybrid's refractory period on held-out seeds (1000+), disjoint from benchmark seeds (0-29)."""
import pandas as pd
from river import drift
from river.drift import binary
from .detectors import Wrap, Hybrid
from .generator import make_stream
from .metrics import score

rows = []
for ref in [100, 200, 300, 500, 800]:
    r = []
    for kind in ["abrupt", "gradual", "incremental"]:
        for seed in range(1000, 1020):
            s = make_stream(kind, seed=seed)
            d = Hybrid(Wrap(binary.HDDMA()), Wrap(drift.ADWIN()), "or", refractory=ref)
            a = [i for i, e in enumerate(s.errors) if d.update(int(e))]
            r.append(score(a, s.onsets, s.width, len(s.errors)))
    df = pd.DataFrame(r)
    rows.append(dict(refractory=ref, F1=df.f1.mean(), delay=df.delay.mean(), FA_per_1k=df.far.mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
