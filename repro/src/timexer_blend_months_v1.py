"""Fixed-config retrospective S/TimeXer blend on three admitted months."""

import argparse
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from safetensors.torch import load_file

from src import s_curve_residual_v1 as r

local, f, tx, m = r.local, r.f, r.tx, r.m
ROOT = tx.ROOT
OUT = ROOT / 'artifacts/timexer_blend_months_v1'
MONTHS = ('2022-01', '2022-02', '2022-03')
CONFIG = dict(encoder_epochs=80, encoder_lr=.0003, scaling='p90',
              head_epochs=20, head_lr=.001, blend_weight=.5)
PREFIX = dict(F1='TXF1', L='TXL', B='BL50')


def dump(path, value):
    tx.dump(path, value)


def cutoff(month):
    if month not in MONTHS:
        raise ValueError('Unplanned month')
    return pd.Timestamp(month+'-01', tz='UTC').value


def training_before(data, stop):
    keep = data['origins']+tx.DAY <= stop
    return {k: v[keep] for k, v in data.items()}


def join_rows(frame, ids, origins):
    index = pd.MultiIndex.from_arrays([frame.depot_id.astype(str), frame.origin_utc.astype('int64'), frame.step])
    if not index.is_unique:
        raise ValueError('Duplicate reference identity')
    keys = pd.MultiIndex.from_arrays([np.repeat(ids, 24), np.repeat(origins, 24),
                                      np.tile(np.arange(1, 25), len(ids))])
    rows = index.get_indexer(keys)
    if (rows < 0).any() or len(np.unique(rows)) != len(rows):
        raise ValueError('Missing or duplicate query identity')
    return rows.reshape(-1, 24)


def fixed_blend(s, neural):
    s, neural = np.asarray(s), np.asarray(neural)
    if s.shape != neural.shape or not np.isfinite(s).all() or not np.isfinite(neural).all():
        raise ValueError('Invalid blend inputs')
    return .5*s+.5*neural


def put_predictions(frame, rows, name, values):
    if name in frame:
        raise ValueError('Refusing to replace existing output column: '+name)
    flat = np.asarray(values).ravel()
    if len(flat) != np.asarray(rows).size or not np.isfinite(flat).all():
        raise ValueError('Prediction alignment/nonfinite output')
    frame[name] = np.nan
    frame.loc[np.asarray(rows).ravel(), name] = flat


def state_frames(frame):
    return [('zero', frame.loc[frame.actual_kw == 0]),
        ('ordinary', frame.loc[(frame.actual_kw > 0) & (frame.actual_kw < frame.high_threshold)]),
        ('high', frame.loc[frame.actual_kw >= frame.high_threshold])]


def freeze():
    r.protect()
    OUT.mkdir(exist_ok=False)
    paths = [tx.OUT / 'training.npz', m.OUT / 'tree_predictions.parquet',
             r.OUT / 'chosen.json', r.OUT / 'results.json', r.OUT / 'acceptance.json',
             local.OUT / 'chosen.json', f.OUT / 'fusion_chosen.json',
             ROOT / 'src/timexer_local_head_v1.py', ROOT / 'src/timexer_feature_fusion_v1.py',
             ROOT / 'src/memory_evidence_v1.py', ROOT / 'src/chronos_timexer_v1.py']
    for depot in tx.IDS:
        paths += [m.DEV / f'prepared/{depot}.npz', m.OUT / f'prepared/{depot}.npz']
        paths += [m.DEV / f'models/{depot}/{month}/S/model.txt' for month in MONTHS]
    code = [Path(__file__), ROOT / 'experiments/timexer_blend_months_v1.md',
            ROOT / 'tests/test_timexer_blend_months_v1.py', ROOT / 'diagnostics/verify_timexer_blend_months_v1.py']
    assert json.loads((r.OUT / 'chosen.json').read_text())['models']['B']['best']['strength'] == .5
    assert json.loads((local.OUT / 'chosen.json').read_text())['best']['epoch'] == 20
    dump(OUT / 'freeze.json', dict(date='2026-10-27', files={str(p): tx.sha(p) for p in paths+code},
        months=MONTHS, config=CONFIG, seeds=tx.SEEDS, planned_encoder_fits=9, planned_head_fits=9,
        hyperparameters_selected_after_test_months=True, new_data=False, retrospective=True))
    print('Frozen three-month fixed blend', flush=True)


def protect():
    r.protect()
    lock = json.loads((OUT / 'freeze.json').read_text())
    assert lock['config'] == CONFIG and tuple(lock['months']) == MONTHS
    for path, digest in lock['files'].items():
        if tx.sha(path) != digest:
            raise ValueError('Protected file changed: '+path)


def prepare():
    protect()
    (OUT / 'prepared').mkdir()
    source = pd.read_parquet(m.OUT / 'tree_predictions.parquet')
    frame = source.loc[source.month.isin(MONTHS)].reset_index(drop=True)
    frame.to_parquet(OUT / 'reference.parquet', index=False)
    original = m.read(tx.OUT / 'training.npz')
    checks = []
    for month in MONTHS:
        train = training_before(original, cutoff(month))
        train_memory = m.memory_for(train)
        queries = []
        for depot in tx.IDS:
            d = m.read(m.DEV / f'prepared/{depot}.npz')
            ti, qi = d['train_'+month], d['eval_'+month]
            tr = train['ids'] == depot
            for a, b in ((train['origins'][tr], d['origin_ns'][ti]),
                         (train['history'][tr], d['base'][ti, 0, :168]),
                         (train['y'][tr], d['truth'][ti]),
                         (train['source'][tr], d['target_sources'][ti]),
                         (train_memory[tr], d['S'][ti, :, 180:])):
                np.testing.assert_array_equal(a, b)
            q = dict(history=d['base'][qi, 0, :168], origins=d['origin_ns'][qi],
                     ids=np.full(len(qi), depot), y=np.full_like(d['truth'][qi], np.nan))
            rows = join_rows(frame, q['ids'], q['origins'])
            old = frame.iloc[rows.ravel()]
            np.testing.assert_array_equal(old.actual_kw.to_numpy().reshape(-1, 24), d['truth'][qi])
            np.testing.assert_array_equal(old.source_row.to_numpy().reshape(-1, 24), d['target_sources'][qi])
            np.testing.assert_array_equal(m.memory_for(q), d['S'][qi, :, 180:])
            model = lgb.Booster(model_file=str(m.DEV / f'models/{depot}/{month}/S/model.txt'))
            s = np.maximum(model.predict(d['S'][qi].reshape(-1, 207), num_threads=2), 0.).reshape(-1, 24)
            np.testing.assert_array_equal(s.ravel(), old.S.to_numpy())
            observed = train['y'][tr][np.isfinite(train['y'][tr])]
            threshold = max(float(np.quantile(observed, .9)), 1.)
            np.testing.assert_array_equal(old.high_threshold.to_numpy(), np.full(len(old), threshold))
            queries.append(dict(**q, rows=rows, baseline=s))
            checks.append(dict(month=month, depot_id=depot, training_origins=int(tr.sum()),
                training_labels=len(observed), query_origins=len(qi), matched_sources=True,
                S_replay_exact=True, threshold_kw=threshold))
        query = {k: np.concatenate([q[k] for q in queries]) for k in queries[0]}
        expected = frame.index[(frame.month == month) & frame.S.notna()].to_numpy()
        np.testing.assert_array_equal(np.sort(query['rows'].ravel()), expected)
        assert (train['origins']+tx.DAY <= cutoff(month)).all()
        assert (query['origins'] >= cutoff(month)).all()
        assert not np.isfinite(query['y']).any()
        np.savez_compressed(OUT / f'prepared/{month}_train.npz', **train)
        np.savez_compressed(OUT / f'prepared/{month}_query.npz', **query)
        print(f'{month}: {len(train["origins"])} training origins / {np.isfinite(train["y"]).sum()} labels; '
              f'{len(query["origins"])} queries; S exact', flush=True)
    files = [OUT / 'reference.parquet']+list((OUT / 'prepared').glob('*.npz'))
    dump(OUT / 'preparation.json', dict(checks=checks, files={str(p): tx.sha(p) for p in files}))


def check_prepared():
    protect()
    for path, digest in json.loads((OUT / 'preparation.json').read_text())['files'].items():
        if tx.sha(path) != digest:
            raise ValueError('Prepared data changed: '+path)


def train_models(month=None):
    check_prepared()
    for period in MONTHS if month is None else (month,):
        train = m.read(OUT / f'prepared/{period}_train.npz')
        query = m.read(OUT / f'prepared/{period}_query.npz')
        assert not np.isfinite(query['y']).any()
        for seed in tx.SEEDS:
            folder = OUT / f'models/{period}_{seed}'
            print(f'Start fixed {period} seed{seed}', flush=True)
            encoder, info, _ = f.fit(train, None, 'F1', CONFIG['encoder_lr'], 'p90',
                CONFIG['encoder_epochs'], seed, folder / 'encoder', (20, 40, 60, 80))
            assert info['latest_training_horizon_ns'] <= cutoff(period)
            ct, cq = local.extract(encoder, train, info['scales']), local.extract(encoder, query, info['scales'])
            np.savez_compressed(folder / 'query_features.npz', **cq)
            head, _ = local.fit_head(ct, None, encoder.decoder, 'L', CONFIG['head_lr'],
                                     CONFIG['head_epochs'], seed, folder / 'head')
            pf = np.maximum(cq['original']*cq['scale'][:, None], 0.)
            pl = np.maximum(local.predict(head, cq), 0.)
            pb = fixed_blend(query['baseline'], pl)
            np.savez_compressed(folder / 'prediction.npz', F1=pf, L=pl, B=pb)
            fresh = f.make_model('F1')
            fresh.load_state_dict(load_file(str(folder / 'encoder/model.safetensors')))
            q_replay = local.extract(fresh, query, info['scales'])
            np.testing.assert_array_equal(q_replay['original'], cq['original'])
            bank = local.head_bank(fresh.decoder, True)
            bank.load_state_dict(load_file(str(folder / 'head/head.safetensors')))
            np.testing.assert_array_equal(np.maximum(local.predict(bank, q_replay), 0.), pl)
            dump(folder / 'complete.json', dict(month=period, seed=seed, config=CONFIG,
                 labels=int(np.isfinite(train['y']).sum()), query_origins=len(query['origins']),
                 no_query_truth_used=True, full_replay=True))
            print(f'Finished {period} seed{seed}; encoder/head replay exact', flush=True)
    dump(OUT / 'training_status.json', dict(completed=[str(p.relative_to(OUT)) for p in (OUT / 'models').glob('*/complete.json')]))


def groups_for(frame):
    result = [('id', str(i), g) for i, g in frame.groupby('depot_id')]
    result += [('leave_out_id', i, frame.loc[frame.depot_id != i]) for i in tx.IDS]
    result += [('state', name, part) for name, part in state_frames(frame)]
    result += [('lead', str(i), frame.loc[frame.step.between(i, i+5)]) for i in (1, 7, 13, 19)]
    return result


def period_result(frame, arms):
    result = dict(all={a: m.metrics(frame, cols) for a, cols in arms.items()}, groups=[], pairs=[],
                  seeds={c: m.metrics(frame, [c]) for a in ('L', 'B') for c in arms[a]})
    for axis, label, part in groups_for(frame):
        result['groups'].append(dict(axis=axis, label=label, models={a: m.metrics(part, cols) for a, cols in arms.items()}))
    for a, b in (('C', 'B'), ('S', 'L'), ('S', 'B'), ('L', 'B'), ('F1', 'L')):
        pair, _ = m.compare(frame, arms[a], arms[b])
        result['pairs'].append(dict(a=a, b=b, **pair))
    issued = frame.S.notna()
    result['coverage'] = dict(candidate_origins=len(frame)//24, issued_origins=int(issued.sum())//24,
        observed=int((issued & frame.actual_kw.notna()).sum()), missing=int((issued & frame.actual_kw.isna()).sum()))
    return result


def score():
    check_prepared()
    frame = pd.read_parquet(OUT / 'reference.parquet')
    arms = {a: [a] for a in ('C', 'S')}
    for arm in PREFIX:
        arms[arm] = []
        for seed in tx.SEEDS:
            all_rows, all_predictions = [], []
            for month in MONTHS:
                folder = OUT / f'models/{month}_{seed}'
                complete = json.loads((folder / 'complete.json').read_text())
                assert complete['config'] == CONFIG
                q = m.read(OUT / f'prepared/{month}_query.npz')
                p = m.read(folder / 'prediction.npz')
                all_rows.append(q['rows'].ravel()); all_predictions.append(p[arm].ravel())
            name = f'{PREFIX[arm]}_{seed}'
            put_predictions(frame, np.concatenate(all_rows), name, np.concatenate(all_predictions))
            np.testing.assert_array_equal(frame[name].notna(), frame.S.notna())
            arms[arm].append(name)
    result = dict(config=CONFIG, arms=arms, periods={}, retrospective=True,
        july_reference=json.loads((r.OUT / 'results.json').read_text())['all'])
    for period in (*MONTHS, 'pooled_Jan_Mar'):
        part = frame if period == 'pooled_Jan_Mar' else frame.loc[frame.month == period]
        result['periods'][period] = period_result(part, arms)
        _, daily = m.compare(part, arms['S'], arms['B'])
        daily.to_parquet(OUT / f'daily_S_B_{period}.parquet', index=False)
        print(period+' '+json.dumps(result['periods'][period]['all']), flush=True)
    frame.to_parquet(OUT / 'predictions.parquet', index=False)
    dump(OUT / 'results.json', result)
    dump(OUT / 'status.json', dict(status='completed_pending_verification', encoder_fits=9, head_fits=9))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'prepare', 'train', 'score'))
    parser.add_argument('--month', choices=MONTHS)
    args = parser.parse_args()
    if args.stage == 'train':
        train_models(args.month)
    else:
        if args.month is not None:
            parser.error('--month is only valid for train')
        globals()[args.stage]()
