"""Matched history-readout controls for the existing TimeXer L/B pipeline."""

import argparse
import json
from pathlib import Path
import time
from types import MethodType

import numpy as np
import pandas as pd
from safetensors.torch import load_file, save_file

from src import timexer_blend_months_v1 as b

f, m, tx, local = b.f, b.m, b.tx, b.local
ROOT = b.ROOT
OUT = ROOT / 'artifacts/history_readout_v1'
MONTHS = (*b.MONTHS, '2022-07')
ARMS = ('F0', 'FU', 'F1')
CONFIG = dict(b.CONFIG)
PREFIX = {'F0': ('F0', 'E0', 'L0', 'B0'), 'FU': ('FU', 'EU', 'LU', 'BU'),
          'F1': ('F1', 'E1', 'L1', 'B1')}
PAIRS = (('F0', 'FU'), ('FU', 'F1'), ('E0', 'EU'), ('EU', 'E1'),
         ('L0', 'LU'), ('LU', 'L1'), ('B0', 'BU'), ('BU', 'B1'),
         ('S', 'B1'), ('SG', 'B1'))


def dump(path, value):
    tx.dump(path, value)


def cutoff(month):
    if month not in MONTHS:
        raise ValueError('Unplanned month')
    return pd.Timestamp(month+'-01', tz='UTC').value


def read_memory(self, query, memory):
    t = tx.configure()
    values = memory[:, :, :27].reshape(-1, 24, 3, 9)
    masks = memory[:, :, 27:].reshape(-1, 24, 3, 9)
    tokens = self.memory_embed(t.cat([values, masks], dim=-1))+self.pool_embed
    support = values[:, :, :, 5]
    valid = (support > 0) & (masks[:, :, :, 5] > 0)
    if self._drop_pool is not None:
        valid = valid.clone()
        if self._drop_pool == 'all':
            valid[:] = False
        else:
            valid[:, :, self._drop_pool] = False
    if self._uniform_readout:
        weights = valid.to(tokens.dtype)
    else:
        logits = (self.query_key(query).unsqueeze(-2)*self.memory_key(tokens)).sum(-1)/8.
        weights = t.softmax(logits.masked_fill(~valid, -1e9), dim=-1)*valid
    weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
    context = (weights.unsqueeze(-1)*self.memory_value(tokens)).sum(-2)
    return context, support, weights


def make_model(arm, drop=None, force_uniform=False):
    if arm not in ARMS or drop not in (None, 0, 1, 2, 'all'):
        raise ValueError('Unknown readout intervention')
    model = f.make_model(arm if arm != 'FU' else 'F1')
    model._uniform_readout = arm == 'FU' or force_uniform
    model._drop_pool = drop
    model.read_memory = MethodType(read_memory, model)
    return model


def encode(data, scales, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return f.encode(data, scales, 'F0' if arm == 'F0' else 'F1')


def feature_cache(model, data, scales, arm):
    t = tx.configure()
    model.eval()
    e = encode(data, scales, arm)
    base, hidden, original = [], [], []
    with t.no_grad():
        for start in range(0, len(e['x']), 128):
            args = [t.from_numpy(e[k][start:start+128]) for k in ('x', 'memory', 'future')]
            v, h = local.feature_batch(model, *args)
            p = v+model.decoder(h).squeeze(-1)
            np.testing.assert_array_equal(p.numpy(), model(*args).numpy())
            base.append(v.numpy()); hidden.append(h.numpy()); original.append(p.numpy())
    result = dict(base=np.concatenate(base), hidden=np.concatenate(hidden), original=np.concatenate(original),
        scale=e['scale'], y=e['y'], mask=e['mask'], ids=np.asarray(data['ids'], int)-1,
        origins=data['origins'], valid_per_origin=np.array(e['valid_per_origin']),
        scale_reference=np.array(e['scale_reference']), raw_y=data['y'])
    if 'source' in data:
        result['source'] = data['source']
    return result


def old_encoder(month, arm, seed):
    if arm == 'F1' and month in b.MONTHS:
        return b.OUT / f'models/{month}_{seed}/encoder'
    if month == '2022-07' and arm in ('F0', 'F1'):
        return f.OUT / f'final/{arm}_{seed}'
    return None


def old_head(month, seed):
    return (b.OUT / f'models/{month}_{seed}/head' if month in b.MONTHS
            else local.OUT / f'final/L_{seed}')


def protected_sources():
    paths = [b.OUT / 'predictions.parquet', local.OUT / 'predictions.parquet',
        tx.OUT / 'training.npz', tx.OUT / 'query.npz',
        ROOT / 'artifacts/global_local_forecast_v1/predictions.parquet',
        ROOT / 'artifacts/global_shared_july_v1/predictions.parquet',
        m.OUT / 'memory_membership.parquet',
        ROOT / 'src/timexer_blend_months_v1.py', ROOT / 'src/timexer_feature_fusion_v1.py',
        ROOT / 'src/timexer_local_head_v1.py', ROOT / 'src/memory_evidence_v1.py',
        ROOT / 'src/chronos_timexer_v1.py', ROOT / 'run_chronos_python.sh']
    paths += sorted((b.OUT / 'prepared').glob('*.npz'))
    paths += sorted((m.OUT / 'prepared').glob('*.npz'))
    for month in MONTHS:
        for seed in tx.SEEDS:
            for arm in ARMS:
                folder = old_encoder(month, arm, seed)
                if folder is not None:
                    paths += [folder / 'model.safetensors', folder / 'fit.json']
            folder = old_head(month, seed)
            paths += [folder / 'head.safetensors', folder / 'fit.json']
    return list(dict.fromkeys(paths))


def freeze():
    b.check_prepared()
    OUT.mkdir(exist_ok=False)
    code = [Path(__file__), ROOT / 'experiments/history_readout_v1.md',
        ROOT / 'tests/test_history_readout_v1.py', ROOT / 'diagnostics/verify_history_readout_v1.py']
    dump(OUT / 'freeze.json', dict(date='2026-11-04', config=CONFIG, months=MONTHS, seeds=tx.SEEDS,
        new_encoder_fits=21, new_head_fits=24, no_new_data=True, previously_seen_evaluation=True,
        files={str(p): tx.sha(p) for p in protected_sources()+code}))


def protect(prepared=True):
    lock = json.loads((OUT / 'freeze.json').read_text())
    assert lock['config'] == CONFIG and tuple(lock['months']) == MONTHS
    manifests = [lock['files']]
    if prepared:
        manifests.append(json.loads((OUT / 'preparation.json').read_text())['files'])
    for manifest in manifests:
        for path, digest in manifest.items():
            if tx.sha(path) != digest:
                raise ValueError('Protected file changed: '+path)


def prepare():
    protect(False)
    (OUT / 'prepared').mkdir()
    parts = []
    for monthly in (True, False):
        frame = pd.read_parquet(b.OUT / 'predictions.parquet' if monthly else local.OUT / 'predictions.parquet')
        if not monthly:
            frame = frame.loc[frame.month == '2022-07'].reset_index(drop=True)
        part = frame[['month', *tx.KEYS, 'actual_kw', 'high_threshold', 'C', 'S']].copy()
        for seed in tx.SEEDS:
            part[f'REF_F1_{seed}'] = frame[f'TXF1_{seed}' if monthly else f'F1_{seed}']
            part[f'REF_L_{seed}'] = frame[f'TXL_{seed}' if monthly else f'HL_{seed}']
            part[f'REF_B_{seed}'] = .5*part.S+.5*part[f'REF_L_{seed}']
            part[f'REF_F0_{seed}'] = np.nan if monthly else frame[f'F0_{seed}']
        path = ROOT / ('artifacts/global_local_forecast_v1/predictions.parquet' if monthly
                       else 'artifacts/global_shared_july_v1/predictions.parquet')
        global_tree = pd.read_parquet(path)[[*tx.KEYS, 'S', 'actual_kw', 'G350']]
        part = part.merge(global_tree, on=tx.KEYS, how='left', validate='one_to_one', suffixes=('', '_check'))
        np.testing.assert_array_equal(part.S, part.S_check)
        np.testing.assert_array_equal(part.actual_kw, part.actual_kw_check)
        part = part.drop(columns=['S_check', 'actual_kw_check'])
        part['SG'] = .5*part.S+.5*part.G350
        parts.append(part)
    reference = pd.concat(parts, ignore_index=True)
    reference.to_parquet(OUT / 'reference.parquet', index=False)
    membership = pd.read_parquet(m.OUT / 'memory_membership.parquet')
    assert (membership.candidate_ns+tx.DAY <= membership.query_ns).all()
    checks = []
    for month in MONTHS:
        train = m.read(b.OUT / f'prepared/{month}_train.npz' if month in b.MONTHS else tx.OUT / 'training.npz')
        query = m.read(b.OUT / f'prepared/{month}_query.npz' if month in b.MONTHS else tx.OUT / 'query.npz')
        query['y'] = np.full_like(query['y'], np.nan)
        query['rows'] = b.join_rows(reference, query['ids'], query['origins'])
        query['baseline'] = reference.iloc[query['rows'].ravel()].S.to_numpy().reshape(-1, 24)
        assert np.isfinite(query['baseline']).all()
        assert (train['origins']+tx.DAY <= cutoff(month)).all()
        assert (query['origins'] >= cutoff(month)).all()
        assert not np.isfinite(query['y']).any()
        expected = reference.index[(reference.month == month) & reference.S.notna()].to_numpy()
        np.testing.assert_array_equal(np.sort(query['rows'].ravel()), expected)
        np.savez_compressed(OUT / f'prepared/{month}_train.npz', **train)
        np.savez_compressed(OUT / f'prepared/{month}_query.npz', **query)
        checks.append(dict(month=month, train_origins=len(train['origins']),
            train_labels=int(np.isfinite(train['y']).sum()), query_origins=len(query['origins']),
            mature_training=True, query_labels_hidden=True))
    paths = [OUT / 'reference.parquet']+sorted((OUT / 'prepared').glob('*.npz'))
    dump(OUT / 'preparation.json', dict(checks=checks, memory_membership_checked=len(membership),
        files={str(p): tx.sha(p) for p in paths}))
    print(json.dumps(checks), flush=True)


def fit_encoder(train, arm, seed, folder):
    folder.mkdir(parents=True, exist_ok=False)
    t = tx.configure()
    t.manual_seed(seed)
    scales = f.scales_from(train, 'p90')
    tr = encode(train, scales, arm)
    model = make_model(arm)
    tensors = {k: t.from_numpy(tr[k]) for k in ('x', 'memory', 'future', 'y', 'mask', 'scale')}
    optimizer = t.optim.AdamW(model.parameters(), lr=CONFIG['encoder_lr'], weight_decay=.0001)
    rng = np.random.default_rng(seed)
    trace = []
    before = time.perf_counter()
    for epoch in range(1, CONFIG['encoder_epochs']+1):
        model.train()
        running = 0.
        order = rng.permutation(len(tr['x']))
        for start in range(0, len(order), 64):
            idx = order[start:start+64]
            optimizer.zero_grad(set_to_none=True)
            p = model(*[tensors[k][idx] for k in ('x', 'memory', 'future')])
            loss = tx.masked_loss(p, tensors['y'][idx], tensors['mask'][idx], tensors['scale'][idx],
                                  tr['valid_per_origin'], tr['scale_reference'])
            if not t.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            running += float(loss.detach())*len(idx)
        if epoch % 20 == 0:
            row = dict(epoch=epoch, training_batch_loss=running/len(tr['x']), seconds=time.perf_counter()-before)
            trace.append(row)
            dump(folder / 'trace.json', trace)
            print(f'{folder.parent.name} epoch{epoch}: '+json.dumps(row), flush=True)
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(folder / 'model.safetensors'))
    info = dict(arm=arm, learning_rate=CONFIG['encoder_lr'], scaling='p90', epochs=80,
        seed=seed, scales=scales, labels=int(tr['mask'].sum()), origins=len(train['origins']),
        latest_training_horizon_ns=int(train['origins'].max()+tx.DAY), validation_used=False,
        parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-before,
        unused_query_key_parameters=(8192 if arm == 'FU' else 0))
    dump(folder / 'fit.json', info)
    return model, info


def inspect_readout(model, data, scales):
    t = tx.configure()
    e = encode(data, scales, 'F1')
    model.eval()
    values = {'weights': [], 'valid': [], 'gate_mean': []}
    with t.no_grad():
        for start in range(0, len(e['x']), 128):
            x, memory, future = [t.from_numpy(e[k][start:start+128]) for k in ('x', 'memory', 'future')]
            _, z = f.latent_forward(model.base.backbone, x)
            const = x[:, :1, 1:11].expand(-1, 24, -1)
            query = model.query_norm(model.recent(z.flatten(1)).unsqueeze(1)+
                                     model.known(t.cat([const, future], -1))+model.lead)
            context, support, weights = model.read_memory(query, memory)
            gate = t.sigmoid(model.gate(t.cat([query, context, support], -1)))
            values['weights'].append(weights.numpy())
            values['valid'].append((support > 0).numpy())
            values['gate_mean'].append(gate.mean(-1).numpy())
    return {k: np.concatenate(v) for k, v in values.items()}


def run_models():
    protect()
    reference = pd.read_parquet(OUT / 'reference.parquet')
    completed = []
    for month in MONTHS:
        train = m.read(OUT / f'prepared/{month}_train.npz')
        query = m.read(OUT / f'prepared/{month}_query.npz')
        for seed in tx.SEEDS:
            for arm in ARMS:
                folder = OUT / f'models/{month}_{arm}_{seed}'
                if (folder / 'complete.json').exists():
                    completed.append(folder.name)
                    continue
                print('Start '+folder.name, flush=True)
                folder.mkdir(parents=True, exist_ok=True)
                source = old_encoder(month, arm, seed)
                encoder_folder = source or folder / 'encoder'
                if (encoder_folder / 'fit.json').exists():
                    info = json.loads((encoder_folder / 'fit.json').read_text())
                    model = make_model(arm)
                    model.load_state_dict(load_file(str(encoder_folder / 'model.safetensors')))
                else:
                    model, info = fit_encoder(train, arm, seed, encoder_folder)
                assert (info['epochs'], info['learning_rate'], info['scaling'], info['seed']) == (80, .0003, 'p90', seed)
                assert info['latest_training_horizon_ns'] <= cutoff(month)
                assert info['scales'] == f.scales_from(train, 'p90')
                assert info['labels'] == int(np.isfinite(train['y']).sum())
                assert not info['validation_used']
                cq = feature_cache(model, query, info['scales'], arm)
                if arm == 'F1':
                    head_folder = old_head(month, seed)
                else:
                    head_folder = folder / 'head'
                    if not (head_folder / 'fit.json').exists():
                        ct = feature_cache(model, train, info['scales'], arm)
                        local.fit_head(ct, None, model.decoder, 'L', CONFIG['head_lr'], CONFIG['head_epochs'], seed, head_folder)
                head = local.head_bank(model.decoder, True)
                head.load_state_dict(load_file(str(head_folder / 'head.safetensors')))
                hinfo = json.loads((head_folder / 'fit.json').read_text())
                assert (hinfo['epochs'], hinfo['rate'], hinfo['seed']) == (20, .001, seed)
                assert hinfo['max_training_horizon_ns'] <= cutoff(month)
                pf = np.maximum(cq['original']*cq['scale'][:, None], 0.)
                pl = np.maximum(local.predict(head, cq), 0.)
                predictions = dict(F=pf, E=b.fixed_blend(query['baseline'], pf),
                    L=pl, B=b.fixed_blend(query['baseline'], pl))
                rowdata = reference.iloc[query['rows'].ravel()]
                if arm == 'F1':
                    for name, target in (('F', 'F1'), ('L', 'L'), ('B', 'B')):
                        np.testing.assert_array_equal(predictions[name].ravel(), rowdata[f'REF_{target}_{seed}'])
                    observations = inspect_readout(model, query, info['scales'])
                    for tag, drop, uniform in (('drop0', 0, False), ('drop1', 1, False),
                            ('drop2', 2, False), ('dropall', 'all', False), ('uniform', None, True)):
                        model._drop_pool, model._uniform_readout = drop, uniform
                        cache = feature_cache(model, query, info['scales'], arm)
                        p = np.maximum(local.predict(head, cache), 0.)
                        observations['L_'+tag] = p
                        observations['B_'+tag] = b.fixed_blend(query['baseline'], p)
                    model._drop_pool, model._uniform_readout = None, False
                    np.savez_compressed(folder / 'readout.npz', **observations)
                if arm == 'F0' and month == '2022-07':
                    np.testing.assert_array_equal(pf.ravel(), rowdata[f'REF_F0_{seed}'])
                np.savez_compressed(folder / 'prediction.npz', **predictions)
                dump(folder / 'complete.json', dict(month=month, arm=arm, seed=seed,
                    encoder_path=str(encoder_folder), head_path=str(head_folder), config=CONFIG,
                    reused_encoder=source is not None, reused_head=arm == 'F1', query_truth_hidden=True))
                completed.append(folder.name)
                dump(OUT / 'training_status.json', dict(completed=completed, expected_runs=36))
                print(f'Finished {folder.name}: {len(completed)}/36', flush=True)


def score():
    protect()
    frame = pd.read_parquet(OUT / 'reference.parquet')
    arms = {name: [name] for name in ('C', 'S', 'G350', 'SG')}
    diag = []
    for arm in ARMS:
        for kind, name in zip(('F', 'E', 'L', 'B'), PREFIX[arm]):
            arms[name] = [f'{name}_{seed}' for seed in tx.SEEDS]
            for seed in tx.SEEDS:
                rows, values = [], []
                for month in MONTHS:
                    folder = OUT / f'models/{month}_{arm}_{seed}'
                    assert json.loads((folder / 'complete.json').read_text())['config'] == CONFIG
                    q = m.read(OUT / f'prepared/{month}_query.npz')
                    rows.append(q['rows'].ravel())
                    values.append(m.read(folder / 'prediction.npz')[kind].ravel())
                b.put_predictions(frame, np.concatenate(rows), f'{name}_{seed}', np.concatenate(values))
    for month in MONTHS:
        q = m.read(OUT / f'prepared/{month}_query.npz')
        for seed in tx.SEEDS:
            obs = m.read(OUT / f'models/{month}_F1_{seed}/readout.npz')
            rows = frame.iloc[q['rows'].ravel()][['month', *tx.KEYS, 'actual_kw', 'high_threshold', 'S']].copy()
            rows['seed'] = seed
            rows['L1'] = frame.iloc[q['rows'].ravel()][f'L1_{seed}'].to_numpy()
            rows['B1'] = frame.iloc[q['rows'].ravel()][f'B1_{seed}'].to_numpy()
            for k in ('weights', 'valid'):
                for pool in range(3):
                    rows[f'{k}_{pool}'] = obs[k][:, :, pool].ravel()
            rows['gate_mean'] = obs['gate_mean'].ravel()
            for tag in ('drop0', 'drop1', 'drop2', 'dropall', 'uniform'):
                for name in ('L', 'B'):
                    rows[name+'_'+tag] = obs[name+'_'+tag].ravel()
            diag.append(rows)
    diagnostic = pd.concat(diag, ignore_index=True)
    result = dict(config=CONFIG, arms=arms, periods={}, diagnostics={},
        new_encoder_fits=21, new_head_fits=24, previously_seen_evaluation=True)
    for period in (*MONTHS, 'pooled_Jan_Mar'):
        part = frame.loc[frame.month.isin(b.MONTHS) if period == 'pooled_Jan_Mar' else frame.month == period]
        record = dict(all={a: m.metrics(part, columns) for a, columns in arms.items()},
            groups=[], seeds={c: m.metrics(part, [c]) for columns in arms.values() for c in columns}, pairs=[])
        for axis, label, group in b.groups_for(part):
            record['groups'].append(dict(axis=axis, label=label,
                models={a: m.metrics(group, columns) for a, columns in arms.items()}))
        for first, second in PAIRS:
            pair, daily = m.compare(part, arms[first], arms[second])
            record['pairs'].append(dict(first=first, second=second, **{k:v for k,v in pair.items() if k not in ('first','second')}))
            daily.to_parquet(OUT / f'daily_{period}_{first}_{second}.parquet', index=False)
        record['coverage'] = dict(candidate_origins=len(part)//24, issued_origins=int(part.S.notna().sum())//24,
            observed=int((part.S.notna() & part.actual_kw.notna()).sum()))
        result['periods'][period] = record
        d = diagnostic.loc[diagnostic.month.isin(b.MONTHS) if period == 'pooled_Jan_Mar' else diagnostic.month == period]
        dr = dict(weights=d[[f'weights_{i}' for i in range(3)]].mean().to_dict(),
            valid_fraction=d[[f'valid_{i}' for i in range(3)]].mean().to_dict(),
            mean_gate=float(d.gate_mean.mean()), interventions={}, state=[])
        for state, subset in [('all', d), *b.state_frames(d)]:
            scores = {name: m.metrics(subset, [name]) for name in ['L1', 'B1']+
                [n+'_'+tag for n in ('L','B') for tag in ('drop0','drop1','drop2','dropall','uniform')]}
            # Stacked seeds have the right MAE; calculate RMSE per seed before averaging.
            for name in scores:
                seed_scores = [m.metrics(g, [name]) for _, g in subset.groupby('seed')]
                if scores[name]['points']:
                    scores[name]['rmse_kw'] = float(np.mean([v['rmse_kw'] for v in seed_scores]))
                    scores[name]['points'] //= len(tx.SEEDS)
            if state == 'all':
                dr['interventions'] = scores
            else:
                dr['state'].append(dict(state=state, models=scores))
        result['diagnostics'][period] = dr
        print(period+' '+json.dumps({a: round(v['mae_kw'],6) for a,v in record['all'].items()}), flush=True)
    frame.to_parquet(OUT / 'predictions.parquet', index=False)
    diagnostic.to_parquet(OUT / 'readout_diagnostics.parquet', index=False)
    dump(OUT / 'results.json', result)
    dump(OUT / 'status.json', dict(status='completed_pending_verification', encoder_fits=21, head_fits=24))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'prepare', 'run', 'score'])
    args = parser.parse_args()
    {'freeze': freeze, 'prepare': prepare, 'run': run_models, 'score': score}[args.stage]()
