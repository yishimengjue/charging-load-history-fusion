"""Identity-only stratified row sampling and forecast-origin UTC weekdays."""

import numpy as np
import pandas as pd


SEEDS = (42, 43, 44)
WEEKDAY_COLUMNS = tuple(f'origin_weekday_{d}' for d in range(7))


def utc_origins(values):
    dates = pd.DatetimeIndex(values)
    if dates.tz is None or str(dates.tz) != 'UTC' or dates.hasnans:
        raise ValueError('Explicit nonmissing UTC origins required')
    return dates


def add_weekday(design, origins):
    dates = utc_origins(origins)
    if len(dates) != len(design) or any(c in design for c in WEEKDAY_COLUMNS):
        raise ValueError('Weekday design shape/columns differ')
    out = design.copy()
    for day, name in enumerate(WEEKDAY_COLUMNS):
        out[name] = (dates.dayofweek == day).astype(float)
    return out


def sample_budget(pool, complete, seed):
    if seed not in SEEDS: raise ValueError('Unfrozen sampling seed')
    keys = ['depot_id', 'origin_utc', 'step']
    for records in (pool, complete):
        utc_origins(records.origin_utc)
        if records.duplicated(keys).any() or not records.step.isin(range(1, 25)).all():
            raise ValueError('Duplicated or invalid training identity')
        if len(records) and records.depot_id.nunique() != 1:
            raise ValueError('Sampling must be within one ID')
        if not records[keys].reset_index(drop=True).equals(
                records.sort_values(['origin_utc', 'step'])[keys].reset_index(drop=True)):
            raise ValueError('Training pool must be chronological')
    if not pd.MultiIndex.from_frame(complete[keys]).isin(pd.MultiIndex.from_frame(pool[keys])).all():
        raise ValueError('Complete pool is not a subset')
    weekdays = utc_origins(pool.origin_utc).dayofweek.to_numpy()
    baseline_days = utc_origins(complete.origin_utc).dayofweek.to_numpy()
    rng = np.random.default_rng(seed)
    selected, quotas = [], []
    for step in range(1, 25):
        for day in range(7):
            candidates = np.flatnonzero((pool.step.to_numpy() == step) & (weekdays == day))
            quota = int(((complete.step.to_numpy() == step) & (baseline_days == day)).sum())
            if quota > len(candidates): raise ValueError('Infeasible stratum quota')
            if quota: selected.extend(rng.choice(candidates, size=quota, replace=False).tolist())
            quotas.append({'step': step, 'origin_weekday': day, 'quota': quota,
                           'pool_rows': len(candidates)})
    indices = np.sort(np.asarray(selected, dtype=int))
    if len(indices) != len(complete) or len(np.unique(indices)) != len(indices):
        raise ValueError('Row budget or uniqueness changed')
    return indices, pd.DataFrame(quotas)
