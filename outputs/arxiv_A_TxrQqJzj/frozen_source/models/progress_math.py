"""Fixed-rank selection weights: shape does not depend on score magnitudes."""
import math

FAST_BETA=0.9
SLOW_BETA=0.99
TIE_SEED=2026110504
SHUFFLE_SEED=2026110505


def update_ema(fast,slow,error,seen):
    if not all(math.isfinite(x) and x>=0 for x in (fast,slow,error)):
        raise ValueError('EMA values must be finite and nonnegative')
    return ((FAST_BETA*fast+(1-FAST_BETA)*error,
             SLOW_BETA*slow+(1-SLOW_BETA)*error) if seen else (error,error))


def progress_score(fast,slow):
    if not all(math.isfinite(x) and x>=0 for x in (fast,slow)):
        raise ValueError('EMA values must be finite and nonnegative')
    return max(slow-fast,0.)


def rank_weights(scores,tie_priority,alpha):
    n=len(scores)
    if not n or len(tie_priority)!=n or len(set(tie_priority))!=n:
        raise ValueError('Nonempty scores and unique tie priorities are required')
    if not math.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('alpha must be in [0,1]')
    if not all(math.isfinite(x) and x>=0 for x in scores):
        raise ValueError('Invalid score')
    order=sorted(range(n),key=lambda i:(scores[i],tie_priority[i]))
    template=[(1-alpha)/n+alpha*2*(r+1)/(n*(n+1)) for r in range(n)]
    result=[0.]*n
    for r,i in enumerate(order): result[i]=template[r]
    return result


def shuffle_seed(step,layer):
    if type(step) is not int or step<0 or type(layer) is not int or not 0<=layer<4:
        raise ValueError('Invalid progress step or layer')
    return SHUFFLE_SEED+4*step+layer
