"""Drift-detection scoring with a tolerance window.
A drift onset t is detected (TP) if the first alarm falls in [t, t + width + margin) (and before the next onset).
Any alarm that is not a TP's first alarm counts as a false alarm; an undetected onset is a miss."""
import numpy as np


def score(alarms, onsets, width, n, margin=300):
    alarms = sorted(alarms)
    tp, delays = 0, []
    for i, t in enumerate(onsets):
        end = t + width + margin
        if i + 1 < len(onsets):
            end = min(end, onsets[i + 1])
        end = min(end, n)
        hit = next((a for a in alarms if t <= a < end), None)
        if hit is not None:
            tp += 1
            delays.append(hit - t)
    fp = len(alarms) - tp
    fn = len(onsets) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                delay=float(np.mean(delays)) if delays else np.nan,
                far=fp / n * 1000)   # false alarms per 1000 samples
