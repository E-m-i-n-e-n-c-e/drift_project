import numpy as np
from drift.datasets import load, inject_drift


def test_load_and_inject():
    X, y, cols = load("iotid20")
    assert X.shape[0] == len(y) == 6252 and "Flow_ID" not in cols and np.isfinite(X).all()
    Xd, yd, meta = inject_drift(X, y, init=1000, seed=0)
    on = meta["onsets"]
    assert (Xd[:on[0]] == X[:on[0]]).all()            # untouched before the first drift
    assert not (Xd[on[0]:on[1]] == X[on[0]:on[1]]).all()
    assert (np.sort(Xd, axis=1) == np.sort(X, axis=1)).all()   # rows only permuted, values unchanged
