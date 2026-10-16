import numpy as np
from drift.generator import make_stream, make_varied_stream, make_industrial_stream


def test_error_rate_rises_at_each_drift():
    s = make_stream("abrupt", n=40000, n_drifts=3, seed=0)
    b = [0] + s.onsets + [40000]
    rates = [s.errors[b[i]:b[i + 1]].mean() for i in range(4)]
    assert np.allclose(rates, [0.10, 0.22, 0.34, 0.46], atol=0.02)


def test_kinds_and_determinism():
    for kind in ["abrupt", "gradual", "incremental"]:
        a, b = make_stream(kind, seed=3), make_stream(kind, seed=3)
        assert (a.errors == b.errors).all() and set(np.unique(a.errors)) <= {0, 1}


def test_varied_streams_keep_drifts_apart():
    for kind in ["abrupt", "gradual", "incremental"]:
        for seed in range(50):
            s, cfg = make_varied_stream(kind, seed)
            gaps = np.diff([0] + s.onsets + [cfg["n"]])
            assert len(s.errors) == cfg["n"] and (gaps >= 2 * max(s.width, 500)).all()


def test_industrial_stream_shape():
    X, y, meta = make_industrial_stream(n=20000, seed=0)
    assert X.shape == (20000, 8) and set(np.unique(y)) == {0, 1} and meta["onsets"] == [5000, 10000, 15000]
