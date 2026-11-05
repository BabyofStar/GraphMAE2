"""Pure scheduling math, independent of graph features, labels, and RNG."""
import math


def probability_schedule(epochs,warmup_ratio=.2,ramp_end_ratio=.8):
    if type(epochs) is not int or epochs<=0:
        raise ValueError('epochs must be a positive integer')
    if (not math.isfinite(warmup_ratio) or not math.isfinite(ramp_end_ratio)
            or not 0<=warmup_ratio<ramp_end_ratio<=1):
        raise ValueError('Require finite 0 <= warmup < ramp_end <= 1')
    return [min(max((t/max(epochs-1,1)-warmup_ratio)/(ramp_end_ratio-warmup_ratio),0.),1.)
            for t in range(epochs)]
