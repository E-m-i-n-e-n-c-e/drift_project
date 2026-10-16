from drift.detectors import REGISTRY, Hybrid, Refractory
from drift.generator import make_stream


class Fixed:
    """Fake detector that fires at given steps."""
    def __init__(self, at): self.at, self.t = set(at), 0
    def update(self, x): self.t += 1; return self.t in self.at


def run(d, n):
    return [t for t in range(1, n + 1) if d.update(0)]


def test_or_rule_with_refractory():
    assert run(Hybrid(Fixed([10, 50]), Fixed([20, 700]), "or", refractory=500), 1000) == [10, 700]


def test_and_rule_needs_both_within_k():
    assert run(Hybrid(Fixed([10, 600]), Fixed([20, 1000]), "and", k=300, refractory=0), 1100) == [20]


def test_refractory_wrapper():
    assert run(Refractory(Fixed([5, 100, 600]), 500), 700) == [5, 600]


def test_every_registered_detector_runs():
    s = make_stream("abrupt", n=3000, seed=0)
    for name, make in REGISTRY.items():
        d = make()
        assert all(isinstance(d.update(int(e)), bool) for e in s.errors), name
