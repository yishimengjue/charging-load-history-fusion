"""Query-time historical memory; targets are read only after time selection."""

import numpy as np


DAY_NS = 24*3600*10**9
POOLS = ((7, False), (28, True), (84, True))
FIELDS = ('mean', 'q10', 'q50', 'q90', 'zero_fraction', 'observed_count',
          'effective_n', 'mean_age', 'observed_mass')
ARMS = ('S', 'R', 'M')
NAMES = [f'memory_{days}_{name}' for days, _ in POOLS for name in FIELDS]


def logsumexp(a):
    m = np.max(a)
    return float(m + np.log(np.exp(a-m).sum()))


def summarize_targets(targets, log_weights, ages):
    targets = np.asarray(targets, float)
    lw, ages = np.asarray(log_weights, float), np.asarray(ages, float)
    out = np.full((24, 9), np.nan)
    out[:, [5, 6, 8]] = 0
    if not len(targets):
        return out
    assert targets.shape == (len(lw),24) and len(ages) == len(lw)
    assert np.isfinite(lw).all()
    total_log_mass = logsumexp(lw)
    for h in range(24):
        valid = np.isfinite(targets[:,h])
        if not valid.any():
            continue
        y, logits = targets[valid,h], lw[valid]
        raw = np.exp(logits-logits.max())
        weights = raw/raw.sum()
        order = np.argsort(y, kind='stable')
        cumulative = np.cumsum(weights[order]); cumulative[-1] = 1.
        q = [y[order[np.searchsorted(cumulative,p,side='left')]] for p in (.1,.5,.9)]
        out[h] = [weights@y, *q, weights@(y==0), valid.sum(),
                  1./(weights@weights), weights@ages[valid],
                  np.exp(logsumexp(logits)-total_log_mass)]
    return out


def eligible_history(history):
    finite = np.isfinite(history)
    return (finite.sum(axis=1) >= 24) & finite[:,-24:].any(axis=1)


def candidate_selection(query, x, ids, origins_ns):
    clock = origins_ns[query]
    age = (clock-origins_ns)/DAY_NS
    eligible = ((ids==ids[query]) & (age>=1) & (age<=84) & eligible_history(x))
    positions = np.flatnonzero(eligible)
    if not len(positions):
        return positions, np.empty(0), np.empty(0), np.empty(0), np.empty(0, int), np.empty(0, int)
    q = x[query]
    common = np.isfinite(x[positions]) & np.isfinite(q)
    n24, n144 = common[:,-24:].sum(axis=1), common[:,:144].sum(axis=1)
    keep = (n24>=12) & (n144>=72)
    positions, common, n24, n144 = positions[keep], common[keep], n24[keep], n144[keep]
    finite = q[np.isfinite(q)]
    scale = max(1.4826*float(np.median(abs(finite-np.median(finite)))),1.) if len(finite) else 1.
    absolute = np.where(common,abs(x[positions]-q),0.)/scale
    d24 = absolute[:,-24:].sum(axis=1)/np.maximum(n24,1)
    d144 = absolute[:,:144].sum(axis=1)/np.maximum(n144,1)
    return positions, age[positions], d24, d144, n24, n144


def build_query(query, x, y, ids, origins_ns):
    positions, ages, d24, d144, n24, n144 = candidate_selection(query,x,ids,origins_ns)
    whole = (n24*d24+n144*d144)/np.maximum(n24+n144,1)
    dual = .5*d24+.5*d144
    features = {a:[] for a in ARMS}
    records = []
    for days, weekday_only in POOLS:
        keep = ages<=days
        if weekday_only:
            keep &= (origins_ns[positions]//DAY_NS)%7 == (origins_ns[query]//DAY_NS)%7
        selected = positions[keep]
        # This is the first access to candidate labels, after query-time filtering.
        target = y[selected]
        for arm, logits in (('S',np.zeros(keep.sum())),('R',-whole[keep]),('M',-dual[keep])):
            features[arm].append(summarize_targets(target,logits,ages[keep]))
        for index in np.flatnonzero(keep):
            records.append(dict(query=int(query),candidate=int(positions[index]),pool_days=days,
                age_days=float(ages[index]),d24=float(d24[index]),d144=float(d144[index]),
                n24=int(n24[index]),n144=int(n144[index])))
    return {a:np.concatenate(values,axis=1) for a,values in features.items()},records
