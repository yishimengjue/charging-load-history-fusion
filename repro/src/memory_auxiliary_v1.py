"""Train-only selection of asymmetric real/zero-summary supervision."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from safetensors.torch import load_file

from src import memory_dual_view_v1 as d

h, tx, f, m, b, local = d.h, d.tx, d.f, d.m, d.b, d.local
ROOT = d.ROOT
OUT = ROOT / 'artifacts/memory_auxiliary_v1'
ALPHAS = (0., .05, .15)
FOLDS = (('2021-12-04', '2021-12-18'), ('2021-12-18', '2022-01-01'))
CONFIG = dict(h.CONFIG, batch_size=64, selection_seed=42)
NAMES = ('FA', 'LA', 'BA', 'FAzero', 'LAzero', 'BAzero')


def ns(day):
    return pd.Timestamp(day, tz='UTC').value


def split(data, start, end):
    mature = data['origins'] + tx.DAY
    train = mature <= start
    val = (data['origins'] >= start) & (mature <= end)
    assert train.any() and val.any() and not (train & val).any()
    return tuple({k: v[keep] for k, v in data.items()} for keep in (train, val))


def weighted_loss(real, zero, target, mask, scales, valid_per_origin, scale_reference, alpha):
    if alpha not in ALPHAS:
        raise ValueError('Unplanned auxiliary weight')
    a = tx.masked_loss(real, target, mask, scales, valid_per_origin, scale_reference)
    if alpha == 0:
        return a, a, None
    z = tx.masked_loss(zero, target, mask, scales, valid_per_origin, scale_reference)
    return (1-alpha)*a + alpha*z, a, z


def source_paths():
    paths = [Path(__file__), ROOT / 'experiments/memory_auxiliary_v1.md',
        ROOT / 'tests/test_memory_auxiliary_v1.py', ROOT / 'diagnostics/verify_memory_auxiliary_v1.py',
        d.OUT / 'acceptance.json', d.OUT / 'predictions.parquet', d.OUT / 'results.json']
    return paths


def freeze():
    d.protect()
    previous = json.loads((d.OUT / 'acceptance.json').read_text())
    assert previous['passed']
    for path, digest in previous['files'].items():
        assert tx.sha(path) == digest, path
    OUT.mkdir(exist_ok=False)
    tx.dump(OUT / 'freeze.json', dict(date='2026-11-04', config=CONFIG, alphas=ALPHAS, folds=FOLDS,
        selection_encoder_fits=6, selection_head_fits=6, maximum_final_encoder_fits=12,
        maximum_final_head_fits=12, final_if_selected_alpha_positive=True,
        no_new_data=True, previously_seen_evaluation=True,
        files={str(p): tx.sha(p) for p in source_paths()}))
    data = m.read(h.OUT / 'prepared/2022-01_train.npz')
    checks = []
    for fold, (start, end) in enumerate(FOLDS):
        tr, val = split(data, ns(start), ns(end))
        assert set(tr['ids']) == set(tx.IDS)
        for name, part in (('train', tr), ('val', val)):
            path = OUT / f'prepared/{fold}_{name}.npz'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, **part)
        checks.append(dict(fold=fold, start=start, end=end, train_origins=len(tr['origins']),
            training_labels=int(np.isfinite(tr['y']).sum()), validation_origins=len(val['origins']),
            validation_labels=int(np.isfinite(val['y']).sum()),
            latest_training_horizon_ns=int(tr['origins'].max()+tx.DAY),
            latest_validation_horizon_ns=int(val['origins'].max()+tx.DAY)))
    tx.dump(OUT / 'preparation.json', dict(checks=checks,
        files={str(p): tx.sha(p) for p in (OUT / 'prepared').glob('*.npz')}))
    print(json.dumps(checks), flush=True)


def protect():
    d.protect()
    lock = json.loads((OUT / 'freeze.json').read_text())
    assert lock['config'] == CONFIG and tuple(lock['alphas']) == ALPHAS
    assert tuple(map(tuple, lock['folds'])) == FOLDS
    preparation = json.loads((OUT / 'preparation.json').read_text())
    for manifest in (lock['files'], preparation['files']):
        for path, digest in manifest.items():
            assert tx.sha(path) == digest, path


def fit_stage(train, seed, alpha, folder, decoder=None, zero_cache=None):
    folder.mkdir(parents=True, exist_ok=False)
    t = tx.configure()
    t.manual_seed(seed)
    is_head = decoder is not None
    stage = 'head' if is_head else 'encoder'
    epochs = CONFIG[stage+'_epochs']
    if is_head:
        for key in ('base', 'ids', 'y', 'mask', 'scale', 'origins'):
            np.testing.assert_array_equal(train[key], zero_cache[key])
        model = local.head_bank(decoder, True)
        for cache in (train, zero_cache):
            np.testing.assert_allclose(local.predict(model, cache), cache['original']*cache['scale'][:, None],
                                       atol=5e-5, rtol=1e-6)
        e = train
        data = {k: t.from_numpy(e[k]) for k in ('hidden', 'base', 'ids', 'y', 'mask', 'scale')}
        hz = t.from_numpy(zero_cache['hidden'])
        scales = None
    else:
        scales = f.scales_from(train, 'p90')
        e = h.encode(train, scales, 'F1')
        model = h.make_model('F1')
        data = {k: t.from_numpy(e[k]) for k in ('x', 'memory', 'future', 'y', 'mask', 'scale')}
    opt = t.optim.AdamW(model.parameters(), lr=CONFIG[stage+'_lr'], weight_decay=.0001)
    rng = np.random.default_rng(seed)
    before, steps, trace = time.perf_counter(), 0, []
    for epoch in range(1, epochs+1):
        model.train()
        order = rng.permutation(len(train['origins']))
        totals = np.zeros(3)
        for start in range(0, len(order), CONFIG['batch_size']):
            idx = order[start:start+CONFIG['batch_size']]
            opt.zero_grad(set_to_none=True)
            if is_head:
                real = data['base'][idx]+model(data['hidden'][idx], data['ids'][idx])
                zero = data['base'][idx]+model(hz[idx], data['ids'][idx]) if alpha else None
            else:
                args = [data[k][idx] for k in ('x', 'memory', 'future')]
                if alpha:
                    real, zero = d.paired_forward(model, *args)
                else:
                    real, zero = model(*args), None
            losses = weighted_loss(real, zero, data['y'][idx], data['mask'][idx], data['scale'][idx],
                float(e['valid_per_origin']), float(e['scale_reference']), alpha)
            if not all(t.isfinite(v) for v in losses if v is not None):
                raise ValueError('Nonfinite loss')
            losses[0].backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            steps += 1
            totals += np.array([float(v.detach()) if v is not None else 0. for v in losses])*len(idx)
        if epoch % (5 if is_head else 20) == 0:
            row = dict(epoch=epoch, optimizer_steps=steps, seconds=time.perf_counter()-before,
                weighted_loss=totals[0]/len(order), real_loss=totals[1]/len(order),
                zero_loss=totals[2]/len(order) if alpha else None)
            trace.append(row)
            tx.dump(folder / 'trace.json', trace)
            if not is_head:
                print(folder.parent.name+' '+json.dumps(row), flush=True)
    d.save_model(model, folder / ('head.safetensors' if is_head else 'model.safetensors'))
    info = dict(stage=stage, config=CONFIG, seed=seed, alpha=alpha, scales=scales,
        origins=len(train['origins']), labels=int(e['mask'].sum()), optimizer_steps=steps,
        latest_training_horizon_ns=int(train['origins'].max()+tx.DAY), validation_used_in_fit=False,
        parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-before)
    tx.dump(folder / 'fit.json', info)
    return model, info


def load_or_fit(train, seed, alpha, folder):
    if (folder / 'encoder/fit.json').exists():
        info = json.loads((folder / 'encoder/fit.json').read_text())
        assert info['config'] == CONFIG and info['alpha'] == alpha and info['seed'] == seed
        model = h.make_model('F1')
        model.load_state_dict(load_file(str(folder / 'encoder/model.safetensors')))
    else:
        model, info = fit_stage(train, seed, alpha, folder / 'encoder')
    if (folder / 'head/fit.json').exists():
        hi = json.loads((folder / 'head/fit.json').read_text())
        assert hi['config'] == CONFIG and hi['alpha'] == alpha and hi['seed'] == seed
        head = local.head_bank(model.decoder, True)
        head.load_state_dict(load_file(str(folder / 'head/head.safetensors')))
    else:
        real, zero = [h.feature_cache(model, train, info['scales'], arm) for arm in ('F1', 'F0')]
        head, _ = fit_stage(real, seed, alpha, folder / 'head', model.decoder, zero)
    return model, head, info


def validation_forecasts(model, head, query, scales):
    assert not np.isfinite(query['y']).any()
    output = {}
    for arm, name in (('F1', 'real'), ('F0', 'zero')):
        cache = h.feature_cache(model, query, scales, arm)
        output[name] = np.maximum(local.predict(head, cache), 0.)
    e = h.encode(query, scales, 'F1')
    output['available_pools'] = d.available_pools(e['memory'])
    return output


def validation_score(y, prediction):
    mask = np.isfinite(y)
    return dict(points=int(mask.sum()), abs_sum=float(np.abs(prediction[mask]-y[mask]).sum()),
                mae_kw=float(np.abs(prediction[mask]-y[mask]).mean()))


def choose(records):
    options = []
    for alpha in ALPHAS:
        rows = [r for r in records if r['alpha'] == alpha]
        assert len(rows) == len(FOLDS)
        points = sum(r['real']['points'] for r in rows)
        options.append(dict(alpha=alpha, points=points,
            mae_kw=sum(r['real']['abs_sum'] for r in rows)/points))
    return dict(best=min(options, key=lambda r: (r['mae_kw'], r['alpha'])), options=options)


def select():
    protect()
    records = []
    for fold in range(len(FOLDS)):
        train = m.read(OUT / f'prepared/{fold}_train.npz')
        val = m.read(OUT / f'prepared/{fold}_val.npz')
        query = dict(val, y=np.full_like(val['y'], np.nan))
        for alpha in ALPHAS:
            folder = OUT / f'selection/fold{fold}_a{alpha}'
            model, head, info = load_or_fit(train, 42, alpha, folder)
            predictions = validation_forecasts(model, head, query, info['scales'])
            np.savez_compressed(folder / 'prediction.npz', **predictions)
            row = dict(fold=fold, alpha=alpha,
                real=validation_score(val['y'], predictions['real']),
                zero=validation_score(val['y'], predictions['zero']))
            records.append(row)
            tx.dump(OUT / 'selection_results.json', records)
            print('Validation '+json.dumps(row), flush=True)
    chosen = choose(records)
    chosen.update(latest_selection_label_ns=ns(FOLDS[-1][1]),
                  selection_results_sha256=tx.sha(OUT / 'selection_results.json'),
                  files={str(p): tx.sha(p) for p in (OUT / 'selection').rglob('*') if p.is_file()})
    tx.dump(OUT / 'chosen.json', chosen)
    print('Chosen '+json.dumps(chosen['best']), flush=True)


def check_selection():
    protect()
    chosen = json.loads((OUT / 'chosen.json').read_text())
    assert chosen['selection_results_sha256'] == tx.sha(OUT / 'selection_results.json')
    actual = choose(json.loads((OUT / 'selection_results.json').read_text()))
    assert chosen['best'] == actual['best'] and chosen['options'] == actual['options']
    for path, digest in chosen['files'].items():
        assert tx.sha(path) == digest, path
    return chosen


def run():
    alpha = check_selection()['best']['alpha']
    if alpha == 0:
        tx.dump(OUT / 'status.json', dict(status='selection_rejected_auxiliary_pending_verification',
            alpha=0, final_fits=0, retained='original_L_B'))
        print('Zero auxiliary weight won validation; retain original L/B, no final fits.', flush=True)
        return
    completed = []
    for month in h.MONTHS:
        train = m.read(h.OUT / f'prepared/{month}_train.npz')
        query = m.read(h.OUT / f'prepared/{month}_query.npz')
        assert (train['origins']+tx.DAY <= h.cutoff(month)).all()
        assert not np.isfinite(query['y']).any()
        for seed in tx.SEEDS:
            folder = OUT / f'final/{month}_{seed}'
            model, head, info = load_or_fit(train, seed, alpha, folder)
            original = d.forecasts(model, head, query, info['scales'])
            predictions = {k.replace('D', 'A'): v for k, v in original.items()}
            np.savez_compressed(folder / 'prediction.npz', **predictions)
            tx.dump(folder / 'complete.json', dict(month=month, seed=seed, alpha=alpha,
                chosen_sha256=tx.sha(OUT / 'chosen.json'), query_labels_hidden=True))
            completed.append(folder.name)
            tx.dump(OUT / 'training_status.json', dict(completed=completed, expected=12))
            print(f'Final {folder.name}: {len(completed)}/12', flush=True)


def score():
    chosen = check_selection()
    if chosen['best']['alpha'] == 0:
        return
    frame = pd.read_parquet(d.OUT / 'predictions.parquet')
    arms = dict(S=['S'], **{a: [f'{a}_{s}' for s in tx.SEEDS] for a in ('L1', 'B1', 'LD', 'BD', *NAMES)})
    for seed in tx.SEEDS:
        rows, arrays = [], {k: [] for k in NAMES}
        for month in h.MONTHS:
            query = m.read(h.OUT / f'prepared/{month}_query.npz')
            p = m.read(OUT / f'final/{month}_{seed}/prediction.npz')
            rows.append(query['rows'].ravel())
            np.testing.assert_array_equal(p['available_pools'].ravel(), frame.iloc[rows[-1]].available_pools)
            for name in NAMES:
                arrays[name].append(p[name].ravel())
        for name in NAMES:
            b.put_predictions(frame, np.concatenate(rows), f'{name}_{seed}', np.concatenate(arrays[name]))
    results = dict(chosen=chosen['best'], arms=arms, periods={}, previously_seen_evaluation=True)
    for period in (*h.MONTHS, 'pooled_Jan_Mar'):
        part = frame.loc[frame.month.isin(b.MONTHS) if period == 'pooled_Jan_Mar' else frame.month == period]
        record = dict(all={a: m.metrics(part, cols) for a, cols in arms.items()}, groups=[],
            seeds={c: m.metrics(part, [c]) for cols in arms.values() for c in cols}, pairs=[])
        for axis, label, group in d.groups_for(part):
            record['groups'].append(dict(axis=axis, label=label,
                models={a: m.metrics(group, cols) for a, cols in arms.items()}))
        for first, second in (('L1', 'LA'), ('B1', 'BA'), ('BD', 'BA'), ('S', 'BA')):
            pair, daily = m.compare(part, arms[first], arms[second])
            record['pairs'].append(dict(pair, first=first, second=second))
            daily.to_parquet(OUT / f'daily_{period}_{first}_{second}.parquet', index=False)
        results['periods'][period] = record
        print(period+' '+json.dumps({a: v['mae_kw'] for a, v in record['all'].items()}), flush=True)
    results['retain_candidate'] = all(
        results['periods'][p]['all']['BA']['mae_kw'] < results['periods'][p]['all']['B1']['mae_kw']
        for p in ('pooled_Jan_Mar', '2022-07'))
    frame.to_parquet(OUT / 'predictions.parquet', index=False)
    tx.dump(OUT / 'results.json', results)
    tx.dump(OUT / 'status.json', dict(status='completed_pending_verification'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'select', 'run', 'score'))
    args = parser.parse_args()
    {'freeze': freeze, 'select': select, 'run': run, 'score': score}[args.stage]()
