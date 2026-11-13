"""Fixed-budget zero-mass versus direct-quantile feasibility experiment."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/lcl_zero_distribution_v1'
MONTHS = ('2013-08', '2013-09', '2013-10')
LEVELS = np.array([.01, .025, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .99])
MID = 7
DAY = 86400 * 10**9
CONFIG = dict(num_leaves=15, max_depth=5, min_data_in_leaf=100, learning_rate=.05,
              lambda_l2=1., max_bin=63, seed=42, deterministic=True,
              force_col_wise=True, num_threads=2, verbosity=-1)
KEYS = ('S', 'y', 'origins', 'ids', 'memory_max_label_time')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read(path, keys=None):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in (data.files if keys is None else keys)}


def sources(month):
    if month not in MONTHS:
        raise ValueError('Month outside approved range')
    base = ROOT / 'artifacts/lcl_august_evaluation_v1' if month == MONTHS[0] else ROOT / 'artifacts/lcl_monthly_followup_v1' / month
    inputs = ROOT / 'artifacts/lcl_forward_inputs_v1' if month == MONTHS[0] else base / 'inputs'
    return dict(train=inputs / 'full_training.npz', query=base / 'queries.npz',
                reference=base / 'predictions.npz', thresholds=base / 'preflight.json',
                input_acceptance=inputs / 'acceptance.json', acceptance=base / 'acceptance.json')


def features(data, query=False):
    raw, ids, y = data['S'], data['ids'], data['y']
    if raw.shape != (len(ids), 24, 207) or y.shape != (len(ids), 24):
        raise ValueError('Wrong feature or target shape')
    if not np.isfinite(ids).all() or (ids != ids.astype(int)).any() or (ids < 0).any() or (ids >= 48).any():
        raise ValueError('Invalid entity ID')
    if query and not np.isnan(y).all():
        raise ValueError('Hide query targets before prediction')
    if np.isinf(raw).any():
        raise ValueError('Infinite feature')
    if (data['memory_max_label_time'] > data['origins']).any():
        raise ValueError('Future memory')
    return np.column_stack([raw.reshape(-1, 207), np.repeat(ids, 24)]).astype(np.float64)


def validate_training(data, cutoff):
    features(data)
    if not (data['origins'] + DAY < cutoff).all():
        raise ValueError('Unmatured training labels')
    if len(np.unique(np.column_stack([data['ids'], data['origins']]), axis=0)) != len(data['ids']):
        raise ValueError('Duplicate training origin')
    y = data['y'].reshape(-1)
    if np.isinf(y).any() or (y[np.isfinite(y)] < 0).any():
        raise ValueError('Invalid nonnegative labels')
    return np.isfinite(y)


def project_quantiles(raw):
    raw = np.asarray(raw, float)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('Invalid quantile matrix')
    return np.sort(np.maximum(raw, 0), axis=1)


def mixture_quantiles(p0, positive, levels, training_max):
    p0, positive, levels = np.asarray(p0), np.asarray(positive), np.asarray(levels)
    if (p0.ndim != 1 or positive.shape != (len(p0), len(levels)) or
        not np.isfinite(p0).all() or (p0 < 0).any() or (p0 > 1).any() or
        not np.isfinite(training_max) or training_max <= 0 or
        not np.isfinite(levels).all() or not (np.diff(levels) > 0).all() or
        levels[0] <= 0 or levels[-1] >= 1):
        raise ValueError('Invalid mixture inputs')
    positive = np.minimum(project_quantiles(positive), training_max)
    knots = np.r_[0., levels, 1.]
    values = np.column_stack([np.zeros(len(p0)), positive, np.full(len(p0), training_max)])
    u = np.maximum(levels[None, :] - p0[:, None], 0) / np.maximum(1 - p0[:, None], 1e-15)
    u = np.clip(u, 0, 1)
    left = np.clip(np.searchsorted(knots, u, side='right') - 1, 0, len(knots) - 2)
    lower = np.take_along_axis(values, left, axis=1)
    upper = np.take_along_axis(values, left + 1, axis=1)
    fraction = (u - knots[left]) / (knots[left + 1] - knots[left])
    result = lower + fraction * (upper - lower)
    return np.where(levels[None, :] <= p0[:, None], 0., result)


def pinball(y, q, levels=LEVELS):
    observed = np.isfinite(y)
    error = y[observed, None] - q[observed]
    return float(np.maximum(levels * error, (levels - 1) * error).mean())


def paired_interval(sums, counts, block, circular, repeats=2000):
    sums, counts = np.asarray(sums), np.asarray(counts)
    days = len(counts)
    if days < block or counts.sum() <= 0:
        raise ValueError('Not enough dates or labels')
    rng = np.random.default_rng(20261112)
    starts = rng.integers(0, days if circular else days - block + 1,
                          size=(repeats, int(np.ceil(days / block))))
    indices = (starts[:, :, None] + np.arange(block)).reshape(repeats, -1)[:, :days]
    if circular:
        indices %= days
    denominator = counts[indices].sum(1)
    if (denominator == 0).any():
        raise ValueError('Empty bootstrap sample')
    return np.quantile(sums[indices].sum(1) / denominator, [.025, .975]).tolist()


def adopt(candidate, references):
    return bool(np.isfinite(candidate) and all(candidate < value for value in references.values()))


def fit_tree(x, y, query, path, objective, rounds, alpha=None):
    if path.exists():
        raise FileExistsError(path)
    params = dict(CONFIG, objective=objective)
    if alpha is not None:
        params['alpha'] = float(alpha)
    start = time.perf_counter()
    dataset = lgb.Dataset(x, label=y, categorical_feature=[207], free_raw_data=True)
    model = lgb.train(params, dataset, num_boost_round=rounds)
    seconds = time.perf_counter() - start
    model.save_model(str(path))
    prediction = model.predict(query, num_threads=2)
    return prediction, dict(model=str(path.relative_to(OUT)) if path.is_relative_to(OUT) else path.name,
                            rows=len(y), requested_trees=rounds, actual_trees=model.num_trees(),
                            seconds=seconds, objective=objective, alpha=alpha, sha256=sha(path))


def freeze():
    files, inherited = {}, 0
    for month in MONTHS:
        paths = sources(month)
        upstream = {}
        for key in ('input_acceptance', 'acceptance'):
            record = json.loads(paths[key].read_text())
            assert record['passed']
            upstream.update(record['files'])
        for key, path in paths.items():
            relative = str(path.relative_to(ROOT))
            digest = sha(path)
            if key not in ('input_acceptance', 'acceptance'):
                assert upstream[relative] == digest, relative
                inherited += 1
            files[relative] = digest
    for name in ('src/lcl_zero_distribution_v1.py', 'tests/test_lcl_zero_distribution_v1.py',
                 'diagnostics/verify_lcl_zero_distribution_v1.py', 'experiments/lcl_zero_distribution_v1.md'):
        files[name] = sha(ROOT / name)
    OUT.mkdir(exist_ok=False)
    dump(OUT / 'freeze.json', dict(files=files, upstream_matches=inherited, config=CONFIG,
         levels=LEVELS.tolist(), months=MONTHS, budget_per_arm_month=2400,
         main_score='raw-unit observed-point median MAE', exploratory=True))


def protect():
    record = json.loads((OUT / 'freeze.json').read_text())
    assert record['config'] == CONFIG and record['levels'] == LEVELS.tolist()
    for name, digest in record['files'].items():
        assert sha(ROOT / name) == digest, name
    return record


def fit_month(month):
    protect(); paths = sources(month); dest = OUT / month
    dest.mkdir(exist_ok=False)
    train = read(paths['train'], KEYS)
    support = validate_training(train, pd.Timestamp(month + '-01').value)
    x = features(train)[support]; y = train['y'].reshape(-1)[support]
    query = read(paths['query'], KEYS)
    query['y'] = np.full_like(query['y'], np.nan)
    xq = features(query, query=True)
    if not ((query['origins'] >= pd.Timestamp(month + '-01').value) &
            (query['origins'] < (pd.Timestamp(month + '-01') + pd.offsets.MonthBegin()).value)).all():
        raise ValueError('Query outside frozen month')
    positive = y > 0
    assert positive.any() and (~positive).any()
    training_max = float(y[positive].max())
    records = []; matrices = {}
    start = time.perf_counter()
    for family in ('direct', 'positive'):
        predictions = []
        for i, level in enumerate(LEVELS):
            ix = np.ones(len(y), bool) if family == 'direct' else positive
            p, log = fit_tree(x[ix], y[ix], xq, dest / f'{family}_{i:02d}.txt',
                              'quantile', 160 if family == 'direct' else 150, float(level))
            predictions.append(p); records.append(dict(family=family, **log))
            print(f'{month} {family} q={level:g}: {log["seconds"]:.1f}s', flush=True)
        matrices[family + '_raw'] = np.column_stack(predictions)
    p0, log = fit_tree(x, (~positive).astype(float), xq, dest / 'zero.txt', 'binary', 150)
    records.append(dict(family='zero', **log))
    direct = project_quantiles(matrices['direct_raw'])
    pos = np.minimum(project_quantiles(matrices['positive_raw']), training_max)
    mixed = mixture_quantiles(p0, pos, LEVELS, training_max)
    np.savez_compressed(dest / 'predictions.npz', origins=query['origins'], ids=query['ids'],
                        direct=direct, positive=pos, mixture=mixed, p0=p0, **matrices)
    np.savez_compressed(dest / 'support.npz', observed_flat_indices=np.flatnonzero(support),
                        positive_in_observed=np.flatnonzero(positive),
                        origins=train['origins'], ids=train['ids'])
    dump(dest / 'fit.json', dict(month=month, config=CONFIG, records=records,
         labels=len(y), positive_labels=int(positive.sum()), zero_labels=int((~positive).sum()),
         latest_target_ns=int(train['origins'].max() + DAY), query_points=len(xq),
         training_max=training_max, seconds=time.perf_counter() - start,
         predictions_sha256=sha(dest / 'predictions.npz'), support_sha256=sha(dest / 'support.npz')))


def score_data(month):
    paths = sources(month); dest = OUT / month
    p = read(dest / 'predictions.npz'); ref = read(paths['reference'])
    for key in ('origins', 'ids'):
        np.testing.assert_array_equal(p[key], ref[key])
    y = ref['y'].reshape(-1)
    errors = dict(zero=abs(y), S=abs(ref['S'].reshape(-1) - y),
        full_direct=np.mean([abs(ref[f'full_direct_{s}'].reshape(-1) - y) for s in (42, 43, 44)], axis=0),
        direct=abs(p['direct'][:, MID] - y), mixture=abs(p['mixture'][:, MID] - y))
    thresholds = json.loads(paths['thresholds'].read_text())['thresholds']
    ids = np.repeat(p['ids'], 24); high = np.zeros(len(y), bool)
    for row in thresholds:
        if row['positive_p90'] is not None:
            high |= (ids == row['id']) & (y > 0) & (y >= row['positive_p90'])
    origin = np.repeat(p['origins'], 24)
    return dict(y=y, errors=errors, high=high, ids=ids, origins=origin, p=p)


def summarize(data, start, end):
    y, errors = data['y'], data['errors']; valid = np.isfinite(y)
    masks = dict(all=valid, zero=valid & (y == 0), positive=valid & (y > 0),
                 high=valid & data['high'], ordinary_positive=valid & (y > 0) & ~data['high'])
    metrics = [dict(method=method, group=group, n=int(mask.sum()),
               mae=float(error[mask].mean()) if mask.any() else None)
               for group, mask in masks.items() for method, error in errors.items()]
    day = ((data['origins'] - start) // DAY).astype(int); days = int((end - start) // DAY)
    daily = []; entities = []; pairs = []
    counts = np.bincount(day[valid], minlength=days)
    sums = {k: np.bincount(day[valid], weights=v[valid], minlength=days) for k, v in errors.items()}
    for k in errors:
        for d in range(days):
            daily.append(dict(date=str(pd.Timestamp(start + d * DAY).date()), method=k,
                              n=int(counts[d]), absolute_error_sum=float(sums[k][d])))
        for identifier in range(48):
            mask = valid & (data['ids'] == identifier)
            entities.append(dict(id=identifier, method=k, n=int(mask.sum()),
                            mae=float(errors[k][mask].mean()) if mask.any() else None))
    for candidate in ('direct', 'mixture'):
        for reference in ('zero', 'full_direct', 'direct'):
            if candidate == reference:
                continue
            delta = sums[reference] - sums[candidate]
            pairs.append(dict(reference=reference, candidate=candidate,
                gain=float((errors[reference][valid] - errors[candidate][valid]).mean()),
                intervals={f'{"circular" if circular else "noncircular"}_{block}':
                    paired_interval(delta, counts, block, circular)
                    for circular in (False, True) for block in (7, 14)}))
    scores = {k: float(v[valid].mean()) for k, v in errors.items()}
    return dict(metrics=metrics, daily=daily, entities=entities, pairs=pairs,
                scores=scores, observed=int(valid.sum()), issued=len(y),
                qualifies_for_op=adopt(scores['mixture'], {k:scores[k] for k in ('zero', 'full_direct', 'direct')}))


def probability_scores(y, p):
    valid = np.isfinite(y); result = {}
    for method in ('direct', 'mixture'):
        q = p[method]; low, upper = q[:, 3], q[:, 11]
        result[method] = dict(mean_pinball=pinball(y, q),
            coverage_10_90=float(((y[valid] >= low[valid]) & (y[valid] <= upper[valid])).mean()),
            width_10_90=float((upper[valid] - low[valid]).mean()))
    p0 = p['p0'][valid]; z = (y[valid] == 0).astype(float); safe = np.clip(p0, 1e-15, 1 - 1e-15)
    bins = []
    for i in range(10):
        mask = (p0 >= i / 10) & ((p0 < (i + 1) / 10) if i < 9 else (p0 <= 1))
        bins.append(dict(low=i / 10, n=int(mask.sum()),
            probability=float(p0[mask].mean()) if mask.any() else None,
            zero_fraction=float(z[mask].mean()) if mask.any() else None))
    pos = p['p0'] < .5
    u = (.5 - p['p0'][pos]) / (1 - p['p0'][pos])
    result['zero_classifier'] = dict(brier=float(((p0-z)**2).mean()),
         log_loss=float(-(z*np.log(safe)+(1-z)*np.log(1-safe)).mean()), bins=bins)
    result['median_interpolation'] = dict(positive_outputs=int(pos.sum()),
        outside_knots=int(((u < LEVELS[0]) | (u > LEVELS[-1])).sum()))
    result['projection'] = {k:dict(crossing_rows=int((np.diff(p[k+'_raw'],axis=1)<0).any(1).sum()),
        median_changed_rows=int((np.maximum(p[k+'_raw'][:,MID],0)!=p[k][:,MID]).sum())) for k in ('direct','positive')}
    return result


def score():
    protect(); monthly = []; all_data = []
    for month in MONTHS:
        info = json.loads((OUT / month / 'fit.json').read_text())
        assert sha(OUT / month / 'predictions.npz') == info['predictions_sha256']
        d = score_data(month); start = pd.Timestamp(month + '-01'); end = start + pd.offsets.MonthBegin()
        summary = summarize(d, start.value, end.value)
        summary['probability'] = probability_scores(d['y'], d['p'])
        summary['month'] = month; monthly.append(summary); all_data.append(d)
    merged = {k: np.concatenate([d[k] for d in all_data]) for k in ('y','high','ids','origins')}
    merged['errors'] = {k:np.concatenate([d['errors'][k] for d in all_data]) for k in all_data[0]['errors']}
    p = {k:np.concatenate([d['p'][k] for d in all_data]) for k in all_data[0]['p']}
    pooled = summarize(merged, pd.Timestamp('2013-08-01').value, pd.Timestamp('2013-11-01').value)
    pooled['probability'] = probability_scores(merged['y'], p)
    assert pooled['observed'] == 96526 and pooled['issued'] == 98520
    dump(OUT / 'results.json', dict(monthly=monthly, pooled=pooled, unit='source_raw',
         main_score='median MAE', independent_confirmation=False,
         decision='qualifies_for_separate_OP_protocol' if pooled['qualifies_for_op'] else 'close_fixed_configuration'))
    print(json.dumps(dict(scores=pooled['scores'], qualifies_for_op=pooled['qualifies_for_op'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'fit', 'score'))
    args = parser.parse_args()
    if args.stage == 'freeze': freeze()
    elif args.stage == 'fit':
        for month in MONTHS: fit_month(month)
    else: score()
