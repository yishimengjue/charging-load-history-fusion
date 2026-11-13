"""Paired real/zero-summary supervision with one unchanged TimeXer model."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from safetensors.torch import load_file, save_file

from src import history_readout_v1 as h

tx, f, m, b, local = h.tx, h.f, h.m, h.b, h.local
ROOT = h.ROOT
OUT = ROOT / 'artifacts/memory_dual_view_v1'
MONTHS = h.MONTHS
CONFIG = dict(h.CONFIG, real_view_weight=.5, zero_view_weight=.5, batch_size=64,
              shared_backbone_dropout=True, paired_head_supervision=True)
NAMES = ('FD', 'LD', 'BD', 'FDzero', 'LDzero', 'BDzero')
PAIRS = (('F1', 'FD'), ('L1', 'LD'), ('B1', 'BD'), ('B0', 'BD'), ('S', 'BD'))


def paired_features(model, x, memory, future):
    t = tx.configure()
    base, z = f.latent_forward(model.base.backbone, x)
    const = x[:, :1, 1:11].expand(-1, 24, -1)
    base = base + model.base.head(t.cat([
        base.unsqueeze(-1), t.zeros_like(memory), const, future], -1)).squeeze(-1)
    query = model.query_norm(model.recent(z.flatten(1)).unsqueeze(1) +
                             model.known(t.cat([const, future], -1)) + model.lead)
    context, support, _ = model.read_memory(query, memory)
    gate = t.sigmoid(model.gate(t.cat([query, context, support], -1)))
    real = model.fusion_norm(query + gate * context)
    # Zeroed values AND masks imply exactly zero memory context in F1.
    zero = model.fusion_norm(query)
    return base, real, zero


def paired_forward(model, x, memory, future):
    base, real, zero = paired_features(model, x, memory, future)
    return (base + model.decoder(real).squeeze(-1),
            base + model.decoder(zero).squeeze(-1))


def paired_loss(real, zero, target, mask, scales, valid_per_origin, scale_reference):
    a = tx.masked_loss(real, target, mask, scales, valid_per_origin, scale_reference)
    z = tx.masked_loss(zero, target, mask, scales, valid_per_origin, scale_reference)
    return .5 * a + .5 * z, a, z


def available_pools(memory):
    values = memory[:, :, :27].reshape(-1, 24, 3, 9)
    masks = memory[:, :, 27:].reshape(-1, 24, 3, 9)
    return ((values[:, :, :, 5] > 0) & (masks[:, :, :, 5] > 0)).sum(-1)


def sources():
    paths = [h.OUT / n for n in ('freeze.json', 'preparation.json', 'acceptance.json',
                                 'predictions.parquet', 'readout_diagnostics.parquet', 'results.json')]
    paths += list((h.OUT / 'prepared').glob('*.npz'))
    paths += [ROOT / p for p in ('src/history_readout_v1.py',
        'src/memory_dual_view_v1.py', 'experiments/memory_dual_view_v1.md',
        'tests/test_memory_dual_view_v1.py', 'diagnostics/verify_memory_dual_view_v1.py')]
    return paths


def freeze():
    h.protect()
    assert json.loads((h.OUT / 'acceptance.json').read_text())['passed']
    OUT.mkdir(exist_ok=False)
    tx.dump(OUT / 'freeze.json', dict(date='2026-11-04', config=CONFIG,
        months=MONTHS, seeds=tx.SEEDS, encoder_fits=12, head_fits=12,
        previously_seen_evaluation=True, new_data=False,
        files={str(p): tx.sha(p) for p in sources()}))
    print('Frozen one paired-view candidate: 12 encoders and 12 heads.', flush=True)


def protect():
    h.protect()
    lock = json.loads((OUT / 'freeze.json').read_text())
    assert lock['config'] == CONFIG and tuple(lock['months']) == MONTHS
    for path, digest in lock['files'].items():
        assert tx.sha(path) == digest, path


def save_model(model, path):
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(path))


def fit_encoder(train, seed, folder):
    folder.mkdir(parents=True, exist_ok=False)
    t = tx.configure()
    t.manual_seed(seed)
    scales = f.scales_from(train, 'p90')
    e = h.encode(train, scales, 'F1')
    model = h.make_model('F1')
    data = {k: t.from_numpy(e[k]) for k in ('x', 'memory', 'future', 'y', 'mask', 'scale')}
    opt = t.optim.AdamW(model.parameters(), lr=CONFIG['encoder_lr'], weight_decay=.0001)
    rng = np.random.default_rng(seed)
    trace, steps = [], 0
    before = time.perf_counter()
    for epoch in range(1, CONFIG['encoder_epochs'] + 1):
        model.train()
        order = rng.permutation(len(e['x']))
        totals = np.zeros(3)
        for start in range(0, len(order), CONFIG['batch_size']):
            idx = order[start:start + CONFIG['batch_size']]
            opt.zero_grad(set_to_none=True)
            real, zero = paired_forward(model, *[data[k][idx] for k in ('x', 'memory', 'future')])
            losses = paired_loss(real, zero, data['y'][idx], data['mask'][idx], data['scale'][idx],
                                 e['valid_per_origin'], e['scale_reference'])
            if not all(t.isfinite(v) for v in losses):
                raise ValueError('Nonfinite paired encoder loss')
            losses[0].backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            steps += 1
            totals += np.array([float(v.detach()) for v in losses]) * len(idx)
        if epoch % 20 == 0:
            row = dict(epoch=epoch, optimizer_steps=steps, seconds=time.perf_counter()-before,
                       **dict(zip(('paired_loss', 'real_loss', 'zero_loss'), totals/len(order))))
            trace.append(row)
            tx.dump(folder / 'trace.json', trace)
            print(folder.parent.name + ' ' + json.dumps(row), flush=True)
    save_model(model, folder / 'model.safetensors')
    info = dict(seed=seed, config=CONFIG, scales=scales, labels=int(e['mask'].sum()),
        origins=len(train['origins']), optimizer_steps=steps, validation_used=False,
        latest_training_horizon_ns=int(train['origins'].max()+tx.DAY),
        parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-before)
    tx.dump(folder / 'fit.json', info)
    return model, info


def fit_head(real, zero, decoder, seed, folder):
    folder.mkdir(parents=True, exist_ok=False)
    t = tx.configure()
    t.manual_seed(seed)
    for key in ('base', 'ids', 'y', 'mask', 'scale', 'origins'):
        np.testing.assert_array_equal(real[key], zero[key])
    model = local.head_bank(decoder, True)
    for cache in (real, zero):
        np.testing.assert_allclose(local.predict(model, cache),
            cache['original'] * cache['scale'][:, None], atol=5e-5, rtol=1e-6)
    save_model(model, folder / 'initial.safetensors')
    data = {k: t.from_numpy(real[k]) for k in ('hidden', 'base', 'ids', 'y', 'mask', 'scale')}
    hz = t.from_numpy(zero['hidden'])
    opt = t.optim.AdamW(model.parameters(), lr=CONFIG['head_lr'], weight_decay=.0001)
    rng = np.random.default_rng(seed)
    before, steps, trace = time.perf_counter(), 0, []
    for epoch in range(1, CONFIG['head_epochs'] + 1):
        model.train()
        order = rng.permutation(len(real['ids']))
        totals = np.zeros(3)
        for start in range(0, len(order), CONFIG['batch_size']):
            idx = order[start:start + CONFIG['batch_size']]
            opt.zero_grad(set_to_none=True)
            pr = data['base'][idx] + model(data['hidden'][idx], data['ids'][idx])
            pz = data['base'][idx] + model(hz[idx], data['ids'][idx])
            losses = paired_loss(pr, pz, data['y'][idx], data['mask'][idx], data['scale'][idx],
                                 float(real['valid_per_origin']), float(real['scale_reference']))
            if not all(t.isfinite(v) for v in losses):
                raise ValueError('Nonfinite paired head loss')
            losses[0].backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            steps += 1
            totals += np.array([float(v.detach()) for v in losses]) * len(idx)
        if epoch % 5 == 0:
            trace.append(dict(epoch=epoch, optimizer_steps=steps,
                **dict(zip(('paired_loss', 'real_loss', 'zero_loss'), totals/len(order)))))
            tx.dump(folder / 'trace.json', trace)
    save_model(model, folder / 'head.safetensors')
    tx.dump(folder / 'fit.json', dict(seed=seed, config=CONFIG, labels=int(real['mask'].sum()),
        origins=len(real['ids']), optimizer_steps=steps, validation_used=False,
        latest_training_horizon_ns=int(real['origins'].max()+tx.DAY),
        parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-before,
        frozen_encoder=True))
    return model


def forecasts(model, head, query, scales):
    output = {}
    for arm, suffix in (('F1', ''), ('F0', 'zero')):
        cache = h.feature_cache(model, query, scales, arm)
        pf = np.maximum(cache['original'] * cache['scale'][:, None], 0.)
        pl = np.maximum(local.predict(head, cache), 0.)
        output.update({'FD'+suffix: pf, 'LD'+suffix: pl,
                       'BD'+suffix: b.fixed_blend(query['baseline'], pl)})
    e = h.encode(query, scales, 'F1')
    output['available_pools'] = available_pools(e['memory'])
    no_pool = output['available_pools'] == 0
    for name in ('FD', 'LD', 'BD'):
        np.testing.assert_array_equal(output[name][no_pool], output[name+'zero'][no_pool])
    return output


def run():
    protect()
    completed = []
    for month in MONTHS:
        train = m.read(h.OUT / f'prepared/{month}_train.npz')
        query = m.read(h.OUT / f'prepared/{month}_query.npz')
        assert (train['origins']+tx.DAY <= h.cutoff(month)).all()
        assert not np.isfinite(query['y']).any()
        for seed in tx.SEEDS:
            folder = OUT / f'models/{month}_{seed}'
            if (folder / 'complete.json').exists():
                completed.append(folder.name)
                continue
            print('Start ' + folder.name, flush=True)
            if (folder / 'encoder/fit.json').exists():
                info = json.loads((folder / 'encoder/fit.json').read_text())
                assert info['config'] == CONFIG
                model = h.make_model('F1')
                model.load_state_dict(load_file(str(folder / 'encoder/model.safetensors')))
            else:
                model, info = fit_encoder(train, seed, folder / 'encoder')
            if (folder / 'head/fit.json').exists():
                head = local.head_bank(model.decoder, True)
                head.load_state_dict(load_file(str(folder / 'head/head.safetensors')))
            else:
                caches = [h.feature_cache(model, train, info['scales'], arm) for arm in ('F1', 'F0')]
                head = fit_head(*caches, model.decoder, seed, folder / 'head')
            np.savez_compressed(folder / 'prediction.npz', **forecasts(model, head, query, info['scales']))
            tx.dump(folder / 'complete.json', dict(month=month, seed=seed, config=CONFIG,
                query_labels_hidden=True, inference_uses_available_summary=True))
            completed.append(folder.name)
            tx.dump(OUT / 'training_status.json', dict(completed=completed, expected=12))
            print(f'Finished {folder.name}: {len(completed)}/12', flush=True)
    protect()


def groups_for(frame):
    yield from b.groups_for(frame)
    for name, mask in (('none', frame.available_pools == 0),
                       ('partial', frame.available_pools.between(1, 2)),
                       ('all', frame.available_pools == 3)):
        yield 'history_availability', name, frame.loc[mask]


def score():
    protect()
    frame = pd.read_parquet(h.OUT / 'predictions.parquet')
    arms = dict(S=['S'], **{a: [f'{a}_{s}' for s in tx.SEEDS] for a in ('F1', 'L1', 'B1', 'L0', 'B0', *NAMES)})
    frame['available_pools'] = np.nan
    for seed in tx.SEEDS:
        rows, arrays = [], {k: [] for k in NAMES}
        for month in MONTHS:
            folder = OUT / f'models/{month}_{seed}'
            assert json.loads((folder / 'complete.json').read_text())['config'] == CONFIG
            query = m.read(h.OUT / f'prepared/{month}_query.npz')
            p = m.read(folder / 'prediction.npz')
            r = query['rows'].ravel()
            rows.append(r)
            for name in NAMES:
                arrays[name].append(p[name].ravel())
            if seed == tx.SEEDS[0]:
                frame.loc[r, 'available_pools'] = p['available_pools'].ravel()
            else:
                np.testing.assert_array_equal(frame.loc[r, 'available_pools'], p['available_pools'].ravel())
        for name in NAMES:
            b.put_predictions(frame, np.concatenate(rows), f'{name}_{seed}', np.concatenate(arrays[name]))
    results = dict(config=CONFIG, arms=arms, periods={}, previously_seen_evaluation=True)
    for period in (*MONTHS, 'pooled_Jan_Mar'):
        part = frame.loc[frame.month.isin(b.MONTHS) if period == 'pooled_Jan_Mar' else frame.month == period]
        record = dict(all={a: m.metrics(part, cols) for a, cols in arms.items()},
            seeds={c: m.metrics(part, [c]) for cols in arms.values() for c in cols}, groups=[], pairs=[])
        for axis, label, group in groups_for(part):
            record['groups'].append(dict(axis=axis, label=label,
                models={a: m.metrics(group, cols) for a, cols in arms.items()}))
        for first, second in PAIRS:
            pair, daily = m.compare(part, arms[first], arms[second])
            record['pairs'].append(dict(pair, first=first, second=second))
            daily.to_parquet(OUT / f'daily_{period}_{first}_{second}.parquet', index=False)
        results['periods'][period] = record
        print(period + ' ' + json.dumps({a: round(v['mae_kw'], 6) for a, v in record['all'].items()}), flush=True)
    frame.to_parquet(OUT / 'predictions.parquet', index=False)
    tx.dump(OUT / 'results.json', results)
    tx.dump(OUT / 'status.json', dict(status='completed_pending_verification'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'run', 'score'))
    args = parser.parse_args()
    {'freeze': freeze, 'run': run, 'score': score}[args.stage]()
