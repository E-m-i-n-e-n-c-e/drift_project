"""Synthetic drift streams (Paper 1 style) and a simulated IIoT concept-drift stream."""
from dataclasses import dataclass
import numpy as np

DEFAULT_WIDTH = {"abrupt": 1, "gradual": 500, "incremental": 500}


@dataclass
class DriftStream:
    errors: np.ndarray      # binary drift-indicator stream (1 = error)
    onsets: list            # ground-truth drift start indices
    width: int              # drift duration in samples (1 = abrupt)
    kind: str


def make_stream(kind="abrupt", n=10000, n_drifts=3, width=None,
                p_low=0.10, amp=0.12, seed=0) -> DriftStream:
    """Bernoulli error stream whose error probability rises by `amp` at each drift (p_low -> p_low+amp -> ...).
    abrupt: instant switch. incremental: linear ramp of p over `width`.
    gradual: each sample drawn from the new concept with linearly rising probability."""
    rng = np.random.default_rng(seed)
    width = width or DEFAULT_WIDTH[kind]
    onsets = np.linspace(0, n, n_drifts + 2)[1:-1].astype(int)
    levels = p_low + amp * np.arange(n_drifts + 1)   # every drift is an error-rate increase
    idx = np.arange(n)
    seg = np.searchsorted(onsets, idx, side="right")
    target = levels[seg]
    prev = np.maximum(seg - 1, 0)
    old = levels[prev]
    frac = np.where(seg == 0, 1.0, np.clip((idx - onsets[prev]) / max(width, 1), 0, 1))
    if kind == "incremental":
        p = old + (target - old) * frac
    elif kind == "gradual":
        p = np.where(rng.random(n) < frac, target, old)
    elif kind == "abrupt":
        p = target
    else:
        raise ValueError(kind)
    return DriftStream((rng.random(n) < p).astype(int), onsets.tolist(), int(width), kind)


SENSORS = ["temperature", "vibration", "pressure", "current", "rpm", "flow", "torque", "voltage"]


def make_industrial_stream(n=20000, d=8, seed=0):
    """Simulated IIoT classification stream (normal vs fault/attack) with REAL concept drift:
    the decision boundary changes at 5000 (abrupt), 10000 (abrupt), 15000 (gradual over 1500)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    W = [rng.normal(size=d) for _ in range(4)]
    idx = np.zeros(n, int)
    idx[5000:] = 1
    idx[10000:] = 2
    ramp = np.clip((np.arange(n) - 15000) / 1500, 0, 1)
    idx[15000:] = np.where(rng.random(n - 15000) < ramp[15000:], 3, 2)
    S = np.stack([X @ w for w in W], 1)
    s = S[np.arange(n), idx] + 0.4 * X[:, 0] * X[:, 1]
    y = (s + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y, dict(onsets=[5000, 10000, 15000], width=1500, names=SENSORS[:d])
