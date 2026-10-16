"""Online scoring core shared by the local runner and the Kafka consumer.
Per record: predict -> compare to label (prequential) -> update drift detector -> on alarm, reset the model.
At the start and after every reset the detector is reset and skips a warm-up (GRACE samples): otherwise the
two-sided ADWIN fires
when the fresh learner's error rate *falls*, which is improvement, not drift."""
import time
from collections import deque
from river import tree
from drift.detectors import REGISTRY

GRACE = 300


class OnlinePipeline:
    def __init__(self, detector="Hybrid-OR(HDDM-A,ADWIN)"):
        self.name = detector
        self.model = tree.HoeffdingAdaptiveTreeClassifier(seed=1)
        self.det = REGISTRY[detector]()
        self.recent = deque(maxlen=500)
        self.t, self.since_reset = 0, 0   # warm-up also covers the cold start

    def process(self, rec):
        x, y = rec["features"], rec["label"]
        pred = self.model.predict_one(x)
        err = int(pred != y)
        self.model.learn_one(x, y)
        self.since_reset += 1
        alarm = self.det.update(err) if self.since_reset > GRACE else False
        if alarm:  # adapt: start a fresh learner (and detector) so it forgets the old concept
            self.model = tree.HoeffdingAdaptiveTreeClassifier(seed=1)
            self.model.learn_one(x, y)
            self.det, self.since_reset = REGISTRY[self.name](), 0
        self.recent.append(err)
        self.t += 1
        out = dict(t=rec.get("t", self.t), err=err, rolling_err=sum(self.recent) / len(self.recent),
                   alarm=int(alarm), true_drift=int(rec.get("true_drift", 0)), pred=pred)
        if "ts" in rec:   # producer timestamp -> end-to-end latency (producer -> Kafka -> scored)
            out["latency_ms"] = (time.time() - rec["ts"]) * 1e3
        return out
