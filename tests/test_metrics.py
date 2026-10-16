import numpy as np
from drift.metrics import score


def test_perfect_detection():
    s = score([110, 1150], [100, 1000], width=1, n=2000)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 0, 0) and s["f1"] == 1.0 and s["delay"] == 80


def test_only_first_alarm_in_window_counts():
    s = score([105, 120], [100], width=1, n=1000)
    assert (s["tp"], s["fp"]) == (1, 1)


def test_alarm_before_onset_or_after_window_is_false():
    s = score([99, 100 + 1 + 300], [100], width=1, n=1000)
    assert (s["tp"], s["fp"], s["fn"]) == (0, 2, 1) and s["f1"] == 0.0


def test_window_capped_at_next_onset():
    s = score([250], [100, 200], width=500, n=1000)   # 250 is past the first window's cap (200)
    assert s["tp"] == 1 and s["delay"] == 50


def test_per_onset_widths_and_penalised_delay():
    s = score([1700], [100, 1000], width=[1, 1000], n=3000)
    assert s["tp"] == 1 and s["delay"] == 700
    assert s["pen_delay"] == np.mean([301, 700])   # the miss costs its whole window
