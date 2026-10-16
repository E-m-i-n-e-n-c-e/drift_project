"""Online scoring core shared by the local runner and the Kafka consumer.
Per record: predict -> compare to label (prequential) -> update drift detector -> on alarm, reset the model."""
from collections import deque
from river import tree
from drift.detectors import REGISTRY


class OnlinePipeline:
    def __init__(self, detector="Hybrid-OR(HDDM-A,ADWIN)"):
        self.model = tree.HoeffdingAdaptiveTreeClassifier(seed=1)
        self.det = REGISTRY[detector]()
        self.recent = deque(maxlen=500)
        self.t = 0

    def process(self, rec):
        x, y = rec["features"], rec["label"]
        pred = self.model.predict_one(x)
        err = int(pred != y)
        self.model.learn_one(x, y)
        alarm = self.det.update(err)
        if alarm:  # adapt: start a fresh learner so it forgets the old concept
            self.model = tree.HoeffdingAdaptiveTreeClassifier(seed=1)
            self.model.learn_one(x, y)
        self.recent.append(err)
        self.t += 1
        return dict(t=rec.get("t", self.t), err=err, rolling_err=sum(self.recent) / len(self.recent),
                    alarm=int(alarm), true_drift=int(rec.get("true_drift", 0)), pred=pred)
