"""Detector registry: River baselines + the proposed hybrid dual-detector trigger."""
from river import drift
from river.drift import binary


class Wrap:
    """Uniform interface: update(x) -> bool."""
    def __init__(self, det):
        self.det = det

    def update(self, x):
        self.det.update(x)
        return bool(self.det.drift_detected)


class OneSided:
    """ADWIN that only reports increases of the error rate. ADWIN is two-sided; on a detection it drops the older part
    of its window, so the window mean rising across the update means the error went up."""
    def __init__(self, det):
        self.det = det

    def update(self, x):
        before = self.det.estimation
        self.det.update(x)
        return bool(self.det.drift_detected) and self.det.estimation > before


class Refractory:
    """Single detector with the hybrid's refractory period (ablation: is the gain only from merging alarm bursts?)."""
    def __init__(self, det, refractory=500):
        self.det, self.ref, self.t, self.last = det, refractory, 0, -10**9

    def update(self, x):
        self.t += 1
        if self.det.update(x) and self.t - self.last > self.ref:
            self.last = self.t
            return True
        return False


class Hybrid:
    """Fast + slow detector fused by a rule.
    mode='or'  : alarm if either fires.
    mode='and' : alarm only if both fire within k samples of each other (confirmation).
    A refractory period merges bursts of alarms into one."""
    def __init__(self, fast, slow, mode="and", k=300, refractory=200):
        self.fast, self.slow, self.mode, self.k, self.ref = fast, slow, mode, k, refractory
        self.t, self.lf, self.ls, self.last = 0, -10**9, -10**9, -10**9

    def update(self, x):
        self.t += 1
        f, s = self.fast.update(x), self.slow.update(x)
        if f:
            self.lf = self.t
        if s:
            self.ls = self.t
        if self.mode == "or":
            fire = f or s
        else:
            fire = (f and self.t - self.ls <= self.k) or (s and self.t - self.lf <= self.k)
        if fire and self.t - self.last > self.ref:
            self.last = self.t
            return True
        return False


REGISTRY = {
    "DDM":     lambda: Wrap(binary.DDM()),
    "EDDM":    lambda: Wrap(binary.EDDM()),
    "HDDM-A":  lambda: Wrap(binary.HDDMA()),
    "HDDM-W":  lambda: Wrap(binary.HDDMW()),
    "ADWIN":   lambda: Wrap(drift.ADWIN()),
    # proposed detector: fast error-rate test (HDDM-A) OR window-mean test (ADWIN), refractory tuned on held-out seeds
    "Hybrid-OR(HDDM-A,ADWIN)": lambda: Hybrid(Wrap(binary.HDDMA()), Wrap(drift.ADWIN()), "or", refractory=500),
    # ablations
    "Hybrid-OR(HDDM-W,EDDM)":  lambda: Hybrid(Wrap(binary.HDDMW()), Wrap(binary.EDDM()), "or"),
    "Hybrid-AND(HDDM-W,EDDM)": lambda: Hybrid(Wrap(binary.HDDMW()), Wrap(binary.EDDM()), "and"),
    "Hybrid-AND(HDDM-W,ADWIN)": lambda: Hybrid(Wrap(binary.HDDMW()), Wrap(drift.ADWIN()), "and"),
    "Hybrid-OR(ADWIN,EDDM)":   lambda: Hybrid(Wrap(drift.ADWIN()), Wrap(binary.EDDM()), "or", refractory=500),  # Paper 2's pair
    "Hybrid-OR(HDDM-A,ADWIN-up)": lambda: Hybrid(Wrap(binary.HDDMA()), OneSided(drift.ADWIN()), "or", refractory=500),
    "HDDM-A+R500":             lambda: Refractory(Wrap(binary.HDDMA()), 500),   # refractory period without fusion
}
