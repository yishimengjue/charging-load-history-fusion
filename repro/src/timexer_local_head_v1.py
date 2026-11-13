"""Train shared or ID-specific heads on frozen F1 representations."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from safetensors.torch import load_file, save_file

from src import timexer_feature_fusion_v1 as f

tx, m = f.tx, f.m
ROOT = f.ROOT
OUT = ROOT / 'artifacts/timexer_local_head_v1'
RATES = (.0003, .001)
EPOCHS = (0, 5, 10, 20, 40)
ARMS = ('G', 'L')


def dump(path, value):
    tx.dump(path, value)


def head_bank(decoder, local):
    t = tx.configure()

    class Heads(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.local = local
            self.heads = t.nn.ModuleList([copy.deepcopy(decoder) for _ in range(9 if local else 1)])

        def forward(self, hidden, ids):
            if not self.local:
                return self.heads[0](hidden).squeeze(-1)
            if not bool(((ids >= 0) & (ids < 9)).all()):
                raise ValueError('Unsupported ID in local decoder')
            output = hidden.new_zeros(hidden.shape[:2])
            for index, head in enumerate(self.heads):
                rows = (ids == index).nonzero(as_tuple=True)[0]
                if len(rows):
                    output[rows] = head(hidden[rows]).squeeze(-1)
            return output

    return Heads()


def feature_batch(model, x, memory, future):
    t = tx.configure()
    base, z = f.latent_forward(model.base.backbone, x)
    const = x[:, :1, 1:11].expand(-1, 24, -1)
    base = base+model.base.head(t.cat([base.unsqueeze(-1), t.zeros_like(memory), const, future], -1)).squeeze(-1)
    recent = model.recent(z.flatten(1)).unsqueeze(1)
    query = model.query_norm(recent+model.known(t.cat([const, future], -1))+model.lead)
    context, support, _ = model.read_memory(query, memory)
    gate = t.sigmoid(model.gate(t.cat([query, context, support], -1)))
    hidden = model.fusion_norm(query+gate*context)
    return base, hidden


def extract(model, data, scales):
    t = tx.configure()
    model.eval()
    # NumPy caches and no_grad freeze the encoder without changing inference kernels.
    e = f.encode(data, scales, 'F1')
    base, hidden, original = [], [], []
    with t.no_grad():
        for start in range(0, len(e['x']), 128):
            args = [t.from_numpy(e[k][start:start+128]) for k in ('x', 'memory', 'future')]
            b, h = feature_batch(model, *args)
            prediction = b+model.decoder(h).squeeze(-1)
            np.testing.assert_array_equal(prediction.numpy(), model(*args).numpy())
            base.append(b.numpy()); hidden.append(h.numpy()); original.append(prediction.numpy())
    result = dict(base=np.concatenate(base), hidden=np.concatenate(hidden), original=np.concatenate(original),
        scale=e['scale'], y=e['y'], mask=e['mask'], ids=np.asarray(data['ids'], int)-1,
        origins=data['origins'], valid_per_origin=np.array(e['valid_per_origin']),
        scale_reference=np.array(e['scale_reference']), raw_y=data['y'])
    if 'source' in data:
        result['source'] = data['source']
    return result


def predict(model, cache):
    t = tx.configure()
    model.eval()
    outputs = []
    with t.no_grad():
        for start in range(0, len(cache['ids']), 128):
            h = t.from_numpy(cache['hidden'][start:start+128])
            ids = t.from_numpy(cache['ids'][start:start+128])
            p = model(h, ids).numpy()+cache['base'][start:start+128]
            outputs.append(p*cache['scale'][start:start+128, None])
    result = np.concatenate(outputs)
    assert np.isfinite(result).all()
    return result


def fit_head(train, val, decoder, arm, rate, epochs, seed, folder):
    folder.mkdir(parents=True, exist_ok=False)
    t = tx.configure()
    t.manual_seed(seed)
    model = head_bank(decoder, arm == 'L')
    for p in model.parameters():
        p.requires_grad_(True)
    initial = predict(model, train)
    expected = train['original']*train['scale'][:, None]
    delta = float(np.max(np.abs(initial-expected)))
    np.testing.assert_allclose(initial, expected, rtol=1e-6, atol=5e-5)
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(folder / 'initial.safetensors'))
    opt = t.optim.AdamW(model.parameters(), lr=rate, weight_decay=.0001)
    data = {k: t.from_numpy(train[k]) for k in ('hidden', 'base', 'y', 'mask', 'scale', 'ids')}
    rng = np.random.default_rng(seed)
    trace = []
    before = time.perf_counter()

    def record(epoch, loss=None):
        row = dict(epoch=epoch, seconds=time.perf_counter()-before, training_batch_loss=loss)
        if val is not None:
            p = np.maximum(predict(model, val), 0.)
            mask = np.isfinite(val['raw_y'])
            total = float(np.abs(p[mask]-val['raw_y'][mask]).sum())
            row.update(validation_abs_sum=total, validation_points=int(mask.sum()), validation_mae=total/int(mask.sum()))
        trace.append(row)
        dump(folder / 'trace.json', trace)
        print(folder.name+' '+json.dumps(row), flush=True)

    record(0)
    for epoch in range(1, epochs+1):
        model.train()
        order = rng.permutation(len(train['ids']))
        running = 0.
        for start in range(0, len(order), 64):
            idx = order[start:start+64]
            opt.zero_grad(set_to_none=True)
            p = data['base'][idx]+model(data['hidden'][idx], data['ids'][idx])
            loss = tx.masked_loss(p, data['y'][idx], data['mask'][idx], data['scale'][idx],
                                 float(train['valid_per_origin']), float(train['scale_reference']))
            if not t.isfinite(loss):
                raise ValueError('Nonfinite head loss')
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            running += float(loss.detach())*len(idx)
        if epoch in EPOCHS or epoch == epochs:
            record(epoch, running/len(order))
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(folder / 'head.safetensors'))
    dump(folder / 'fit.json', dict(arm=arm, rate=rate, epochs=epochs, seed=seed, labels=int(train['mask'].sum()),
        origins=len(train['ids']), trainable_parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter()-before, initial_max_abs_delta_kw=delta, frozen_feature_training=True,
        max_training_horizon_ns=int(train['origins'].max()+tx.DAY)))
    return model, trace


def freeze():
    f.protect()
    OUT.mkdir(exist_ok=False)
    (OUT / 'cache').mkdir()
    inputs = [tx.OUT / 'training.npz', tx.OUT / 'query.npz', tx.OUT / 'validation_0.npz', tx.OUT / 'validation_1.npz',
        f.OUT / 'predictions.parquet', f.OUT / 'fusion_chosen.json', f.OUT / 'fusion_grid.json', f.OUT / 'freeze.json']
    inputs += [f.OUT / f'final/F1_{s}/{name}' for s in tx.SEEDS for name in ('model.safetensors', 'fit.json')]
    code = [Path(__file__), ROOT / 'experiments/timexer_local_head_v1.md',
        ROOT / 'tests/test_timexer_local_head_v1.py', ROOT / 'diagnostics/verify_timexer_local_head_v1.py']
    dump(OUT / 'freeze.json', dict(date='2026-10-27', files={str(p): tx.sha(p) for p in inputs+code},
        head_rates=RATES, head_epochs=EPOCHS, encoder_rebuilds=2, inner_head_runs=8, final_head_runs=6,
        new_data=False, hard_upper_bound=False, evaluation_previously_seen=True))
    print('Frozen local-head comparison', flush=True)


def protect():
    f.protect()
    lock = json.loads((OUT / 'freeze.json').read_text())
    for path, value in lock['files'].items():
        if tx.sha(path) != value:
            raise ValueError('Changed protected file: '+path)


def save_cache(name, data):
    np.savez_compressed(OUT / 'cache' / (name+'.npz'), **data)


def prepare():
    protect()
    checks = []
    grid = json.loads((f.OUT / 'fusion_grid.json').read_text())
    for fold, train, val in f.folds():
        folder = OUT / 'encoders' / f'fold{fold}'
        if (folder / 'fit.json').exists():
            info = json.loads((folder / 'fit.json').read_text())
            assert (info['epochs'], info['learning_rate'], info['scaling'], info['seed']) == (80, .0003, 'p90', 42)
            assert info['labels'] == int(np.isfinite(train['y']).sum())
            assert info['latest_training_horizon_ns'] == int(train['origins'].max()+tx.DAY)
            model = f.make_model('F1')
            model.load_state_dict(load_file(str(folder / 'model.safetensors')))
        else:
            model, info, _ = f.fit(train, None, 'F1', .0003, 'p90', 80, 42, folder, (20, 40, 60, 80))
        tr, va = extract(model, train, info['scales']), extract(model, val, info['scales'])
        p = np.maximum(va['original']*va['scale'][:, None], 0.)
        mask = np.isfinite(val['y'])
        mae = float(np.abs(p[mask]-val['y'][mask]).mean())
        expected = next(r['validation_mae'] for r in grid if r['fold'] == fold and r['arm'] == 'F1' and r['epoch'] == 80)
        np.testing.assert_allclose(mae, expected, atol=1e-10, rtol=0)
        save_cache(f'fold{fold}_train', tr); save_cache(f'fold{fold}_validation', va)
        checks.append(dict(fold=fold, reconstructed_validation_mae=mae, original_validation_mae=expected,
                           labels=int(tr['mask'].sum()), uses_post_cutoff_encoder=False))
    train, query = m.read(tx.OUT / 'training.npz'), m.read(tx.OUT / 'query.npz')
    query['y'] = np.full_like(query['y'], np.nan)
    reference = pd.read_parquet(f.OUT / 'predictions.parquet')
    for seed in tx.SEEDS:
        folder = f.OUT / 'final' / f'F1_{seed}'
        info = json.loads((folder / 'fit.json').read_text())
        model = f.make_model('F1')
        model.load_state_dict(load_file(str(folder / 'model.safetensors')))
        tr, qe = extract(model, train, info['scales']), extract(model, query, info['scales'])
        assert int(tr['mask'].sum()) == 74296
        p = np.maximum(qe['original']*qe['scale'][:, None], 0.)
        np.testing.assert_array_equal(p.ravel(), reference.iloc[query['rows'].ravel()][f'F1_{seed}'].to_numpy())
        save_cache(f'seed{seed}_train', tr); save_cache(f'seed{seed}_query', qe)
        checks.append(dict(seed=seed, labels=74296, initial_full_F1_replay=True, frozen_model_parameters=226490))
    manifest = {str(p): tx.sha(p) for p in (OUT / 'cache').glob('*.npz')}
    manifest.update({str(p): tx.sha(p) for p in (OUT / 'encoders').glob('*/*') if p.is_file()})
    dump(OUT / 'preparation.json', dict(checks=checks, files=manifest))


def check_cache():
    protect()
    info = json.loads((OUT / 'preparation.json').read_text())
    for path, digest in info['files'].items():
        assert tx.sha(path) == digest, path


def decoder_from(path):
    model = f.make_model('F1')
    model.load_state_dict(load_file(str(path)))
    return model.decoder


def select():
    check_cache()
    records = []
    for fold in (0, 1):
        train = m.read(OUT / 'cache' / f'fold{fold}_train.npz')
        val = m.read(OUT / 'cache' / f'fold{fold}_validation.npz')
        decoder = decoder_from(OUT / 'encoders' / f'fold{fold}/model.safetensors')
        for rate in RATES:
            for arm in ARMS:
                _, trace = fit_head(train, val, decoder, arm, rate, 40, 42,
                    OUT / 'selection' / f'fold{fold}_{arm}_{rate}')
                records += [dict(fold=fold, arm=arm, rate=rate, **r) for r in trace]
                dump(OUT / 'grid.json', records)
    options = []
    for rate in RATES:
        for epoch in EPOCHS:
            part = [r for r in records if r['rate'] == rate and r['epoch'] == epoch]
            assert len(part) == 4
            options.append(dict(rate=rate, epoch=epoch,
                validation_mae=sum(r['validation_abs_sum'] for r in part)/sum(r['validation_points'] for r in part)))
    best = min(options, key=lambda r: (r['validation_mae'], r['epoch'], r['rate']))
    dump(OUT / 'chosen.json', dict(best=best, options=options, grid_sha256=tx.sha(OUT / 'grid.json'), uses_july=False))
    print('Shared head configuration '+json.dumps(best), flush=True)


def final():
    check_cache()
    chosen = json.loads((OUT / 'chosen.json').read_text())
    assert chosen['grid_sha256'] == tx.sha(OUT / 'grid.json')
    setting = chosen['best']
    for seed in tx.SEEDS:
        train = m.read(OUT / 'cache' / f'seed{seed}_train.npz')
        query = m.read(OUT / 'cache' / f'seed{seed}_query.npz')
        decoder = decoder_from(f.OUT / 'final' / f'F1_{seed}/model.safetensors')
        for arm in ARMS:
            folder = OUT / 'final' / f'{arm}_{seed}'
            model, _ = fit_head(train, None, decoder, arm, setting['rate'], setting['epoch'], seed, folder)
            before = time.perf_counter()
            raw = predict(model, query)
            elapsed = time.perf_counter()-before
            replay = head_bank(decoder, arm == 'L')
            replay.load_state_dict(load_file(str(folder / 'head.safetensors')))
            np.testing.assert_array_equal(raw, predict(replay, query))
            np.savez_compressed(folder / 'prediction.npz', raw=raw, p50=np.maximum(raw, 0.))
            dump(folder / 'replay.json', dict(exact=True, cached_prediction_seconds=elapsed, curves=217))
    print('Six final head runs complete', flush=True)


def score():
    check_cache()
    frame = pd.read_parquet(f.OUT / 'predictions.parquet')
    query = m.read(tx.OUT / 'query.npz')
    arms = {a: [a] for a in ('C', 'S', 'P')}
    arms.update({a: [f'{a}_{s}' for s in tx.SEEDS] for a in ('F1',)})
    for arm in ARMS:
        names = []
        for seed in tx.SEEDS:
            name = f'H{arm}_{seed}'
            frame[name] = np.nan
            frame.loc[query['rows'].ravel(), name] = m.read(OUT / f'final/{arm}_{seed}/prediction.npz')['p50'].ravel()
            np.testing.assert_array_equal(frame[name].notna(), frame.S.notna())
            names.append(name)
        arms[arm] = names
    result = dict(all={a: m.metrics(frame, v) for a, v in arms.items()}, groups=[], pairs=[],
        seeds={name: m.metrics(frame, [name]) for arm in ARMS for name in arms[arm]})
    groups = [('id', str(i), g) for i, g in frame.groupby('depot_id')]
    groups += [('leave_out_id', i, frame.loc[frame.depot_id != i]) for i in tx.IDS]
    groups += [('state', 'zero', frame.loc[frame.actual_kw == 0]),
        ('state', 'positive_below_p90', frame.loc[(frame.actual_kw > 0) & (frame.actual_kw < frame.high_threshold)]),
        ('state', 'high', frame.loc[frame.actual_kw >= frame.high_threshold])]
    groups += [('lead', str(i), frame.loc[frame.step.between(i, i+5)]) for i in (1, 7, 13, 19)]
    for axis, label, part in groups:
        result['groups'].append(dict(axis=axis, label=label, models={a: m.metrics(part, v) for a, v in arms.items()}))
    for a, b in (('F1', 'G'), ('F1', 'L'), ('G', 'L'), ('S', 'L'), ('P', 'L')):
        comp, daily = m.compare(frame, arms[a], arms[b])
        result['pairs'].append(dict(a=a, b=b, **comp))
        daily.to_parquet(OUT / f'daily_{a}_{b}.parquet', index=False)
    result['coverage'] = dict(candidate_origins=len(frame)//24, issued_origins=int(frame.S.notna().sum())//24,
        observed=int((frame.S.notna() & frame.actual_kw.notna()).sum()),
        missing=int((frame.S.notna() & frame.actual_kw.isna()).sum()))
    frame.to_parquet(OUT / 'predictions.parquet', index=False)
    dump(OUT / 'results.json', result)
    dump(OUT / 'status.json', dict(status='completed_pending_verification', encoder_rebuilds=2,
        inner_head_runs=8, final_head_runs=6))
    print(json.dumps(result['all'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'prepare', 'select', 'final', 'score'))
    globals()[parser.parse_args().stage]()
