"""Fixed-quota pair coupling; each node retains quota/group-size marginal."""
import math

SEMANTIC_SEED=2026110506
RANDOM_SEED=2026110507

def pairing(group,candidates,fallback_order):
    group=list(group); fallback_order=list(fallback_order)
    if len(set(group))!=len(group) or set(fallback_order)!=set(group) or len(fallback_order)!=len(group):
        raise ValueError('Invalid group/permutation')
    allowed=set(group); used=set(); pairs=[]; matched=[]
    for score,u,v in sorted(candidates,key=lambda x:(-x[0],x[1],x[2])):
        if not math.isfinite(score): raise ValueError('Nonfinite affinity')
        if u not in allowed or v not in allowed or u==v: raise ValueError('Invalid pair candidate')
        if score>0 and u not in used and v not in used:
            pairs.append((u,v)); matched.append(True); used.update((u,v))
    rest=[u for u in fallback_order if u not in used]
    for i in range(0,len(rest)-1,2): pairs.append((rest[i],rest[i+1])); matched.append(False)
    return pairs,(rest[-1] if len(rest)%2 else None),matched

def selected_pair_positions(pair_count,order,coins,remaining,coupling):
    if (type(pair_count) is not int or pair_count<0 or sorted(order)!=list(range(pair_count))
            or len(coins)!=pair_count or any(x not in (0,1) for x in coins)
            or type(remaining) is not int or not 0<=remaining<=2*pair_count):
        raise ValueError('Invalid pair sampling state')
    if coupling=='anti':
        result=[2*i+coins[i] for i in order[:min(remaining,pair_count)]]
        if remaining>pair_count: result += [2*i+1-coins[i] for i in order[:remaining-pair_count]]
    elif coupling=='co':
        double=remaining//2; result=[2*i+j for i in order[:double] for j in (0,1)]
        if remaining%2:
            i=order[double]; result.append(2*i+coins[i])
    else: raise ValueError('Unknown pair coupling')
    return result
