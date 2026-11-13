"""Bounded train-only optimization and recent-conditioned memory feature fusion."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from safetensors.torch import load_file, save_file

from src import memory_evidence_v1 as m

tx = m.tx
ROOT = m.ROOT
OUT = ROOT / 'artifacts/timexer_feature_fusion_v1'
CHECKPOINTS = (20, 40, 60, 70, 80, 100)
LRS = (.001, .0003)
SCALINGS = ('p90', 'rms')
ARMS = ('B', 'F0', 'F1')


def dump(path, value):
    tx.dump(path, value)


def scales_from(data, mode):
    result = {}
    for depot in tx.IDS:
        values = data['y'][data['ids'] == depot]
        values = values[np.isfinite(values)]
        if not len(values) or np.any(values < 0):
            raise ValueError('Invalid training labels for scaling')
        if mode == 'p90':
            scale = float(np.quantile(values, .9))
        elif mode == 'rms':
            scale = float(np.sqrt(np.mean(values*values)))
        else:
            raise ValueError(mode)
        result[depot] = max(scale, 1.)
    return result


def encode(data, scales, arm):
    e = m.encoded(data, scales)
    if arm != 'F1':
        e['memory'] = np.zeros_like(e['memory'])
    return e


def latent_forward(backbone, x):
    en, nvars = backbone.en_embedding(x[:, :, -1:].permute(0, 2, 1))
    ex = backbone.ex_embedding(x[:, :, :-1], None)
    z = backbone.encoder(en, ex)
    view = z.reshape(-1, nvars, z.shape[-2], z.shape[-1]).permute(0, 1, 3, 2)
    prediction = backbone.head(view).permute(0, 2, 1).squeeze(-1)
    return prediction, z


def make_model(arm):
    if arm == 'B':
        return m.make_model()
    t = tx.configure()

    class FeatureFusion(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = m.make_model()
            self.recent = t.nn.Linear(15*64, 64)
            self.known = t.nn.Linear(18, 64)
            self.lead = t.nn.Parameter(t.randn(1, 24, 64)*.02)
            self.query_norm = t.nn.LayerNorm(64)
            self.memory_embed = t.nn.Linear(18, 64)
            self.pool_embed = t.nn.Parameter(t.randn(1, 1, 3, 64)*.02)
            self.query_key = t.nn.Linear(64, 64, bias=False)
            self.memory_key = t.nn.Linear(64, 64, bias=False)
            self.memory_value = t.nn.Linear(64, 64, bias=False)
            self.gate = t.nn.Linear(131, 64)
            self.fusion_norm = t.nn.LayerNorm(64)
            self.decoder = t.nn.Sequential(t.nn.Linear(64, 32), t.nn.GELU(), t.nn.Linear(32, 1))
            t.nn.init.zeros_(self.decoder[-1].weight)
            t.nn.init.zeros_(self.decoder[-1].bias)

        def read_memory(self, query, memory):
            values = memory[:, :, :27].reshape(-1, 24, 3, 9)
            masks = memory[:, :, 27:].reshape(-1, 24, 3, 9)
            tokens = self.memory_embed(t.cat([values, masks], dim=-1))+self.pool_embed
            support = values[:, :, :, 5]
            valid = (support > 0) & (masks[:, :, :, 5] > 0)
            logits = (self.query_key(query).unsqueeze(-2)*self.memory_key(tokens)).sum(-1)/8.
            # Renormalizing after masking handles the all-missing case without NaNs.
            weights = t.softmax(logits.masked_fill(~valid, -1e9), dim=-1)*valid
            weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
            context = (weights.unsqueeze(-1)*self.memory_value(tokens)).sum(-2)
            return context, support, weights

        def forward(self, x, memory, future):
            base, z = latent_forward(self.base.backbone, x)
            const = x[:, :1, 1:11].expand(-1, 24, -1)
            # The original B head receives no memory in either matched arm.
            base = base+self.base.head(t.cat([base.unsqueeze(-1), t.zeros_like(memory), const, future], -1)).squeeze(-1)
            recent = self.recent(z.flatten(1)).unsqueeze(1)
            query = self.query_norm(recent+self.known(t.cat([const, future], -1))+self.lead)
            context, support, _ = self.read_memory(query, memory)
            gate = t.sigmoid(self.gate(t.cat([query, context, support], -1)))
            hidden = self.fusion_norm(query+gate*context)
            return base+self.decoder(hidden).squeeze(-1)

    return FeatureFusion()


def predict(model, data):
    return m.neural_predict(model, data)


def fit(train, validation, arm, rate, scaling, epochs, seed, folder, checkpoint_epochs):
    folder.mkdir(parents=True, exist_ok=False)
    t = tx.configure()
    t.manual_seed(seed)
    scales = scales_from(train, scaling)
    tr = encode(train, scales, arm)
    val = encode(validation, scales, arm) if validation is not None else None
    model = make_model(arm)
    tensors = {k: t.from_numpy(tr[k]) for k in ('x', 'memory', 'future', 'y', 'mask', 'scale')}
    optimizer = t.optim.AdamW(model.parameters(), lr=rate, weight_decay=.0001)
    rng = np.random.default_rng(seed)
    trace = []
    before = time.perf_counter()
    for epoch in range(1, epochs+1):
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
                raise ValueError('Nonfinite training loss')
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            running += float(loss.detach())*len(idx)
        if epoch in checkpoint_epochs or epoch == epochs:
            row = dict(epoch=epoch, training_batch_loss=running/len(tr['x']), seconds=time.perf_counter()-before)
            if val is not None:
                p = np.maximum(predict(model, val), 0.)
                mask = np.isfinite(validation['y'])
                row.update(validation_abs_sum=float(np.abs(p[mask]-validation['y'][mask]).sum()),
                           validation_points=int(mask.sum()))
                row['validation_mae'] = row['validation_abs_sum']/row['validation_points']
            trace.append(row)
            dump(folder / 'trace.json', trace)
            print(f'{folder.name} epoch{epoch}: '+json.dumps(row), flush=True)
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(folder / 'model.safetensors'))
    info = dict(arm=arm, learning_rate=rate, scaling=scaling, epochs=epochs, seed=seed, scales=scales,
        labels=int(tr['mask'].sum()), origins=len(train['origins']),
        latest_training_horizon_ns=int(train['origins'].max()+tx.DAY),
        parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-before,
        validation_used=validation is not None)
    dump(folder / 'fit.json', info)
    return model, info, trace


def freeze():
    m.protect()
    OUT.mkdir(exist_ok=False)
    inputs = [tx.OUT / 'training.npz', tx.OUT / 'query.npz',
        tx.OUT / 'validation_0.npz', tx.OUT / 'validation_1.npz',
        ROOT / 'artifacts/memory_head_control_v1/predictions.parquet',
        ROOT / 'artifacts/memory_head_control_v1/results.json', m.OUT / 'results.json',
        m.OUT / 'freeze.json']
    inputs += [m.OUT / 'prepared' / f'{i}.npz' for i in tx.IDS]
    inputs += [tx.INNER / 'prepared' / f'{f}_{i}.npz' for f in (0, 1) for i in tx.IDS]
    code = [Path(__file__), ROOT / 'tests/test_timexer_feature_fusion_v1.py',
        ROOT / 'experiments/timexer_feature_fusion_v1.md',
        ROOT / 'diagnostics/verify_timexer_feature_fusion_v1.py']
    dump(OUT / 'freeze.json', dict(date='2026-10-27', files={str(p): tx.sha(p) for p in inputs+code},
        planned_inner_fits=12, planned_final_fits=9, evaluation_already_seen=True,
        rates=LRS, scalings=SCALINGS, checkpoints=CHECKPOINTS, no_new_data=True))
    train = m.read(tx.OUT / 'training.npz')
    assert len(train['origins']) == 3278 and np.isfinite(train['y']).sum() == 74296
    assert (train['origins']+tx.DAY <= tx.CUTOFF).all()
    counts = []
    for fold, (start, end) in enumerate(tx.FOLDS):
        val = m.read(tx.OUT / f'validation_{fold}.npz')
        observed = np.isfinite(val['y'])
        assert ((val['origins'] >= start) & (val['origins'] < end)).all()
        memory = m.memory_for(val)
        for depot in tx.IDS:
            old = m.read(tx.INNER / 'prepared' / f'{fold}_{depot}.npz')
            subset = memory[val['ids'] == depot]
            np.testing.assert_array_equal(subset, old['query_raw_x'].reshape(-1, 24, 207)[:, :, 180:])
        keep = train['origins']+tx.DAY <= start
        counts.append(dict(fold=fold, training_origins=int(keep.sum()),
            training_labels=int(np.isfinite(train['y'][keep]).sum()), validation_labels=int(observed.sum()),
            cutoff_ns=int(start), memory_matches_original=True))
    dump(OUT / 'input_checks.json', dict(passed=True, folds=counts, labels=74296))
    print('Frozen and June memory/input checks passed', flush=True)


def protect():
    m.protect()
    lock = json.loads((OUT / 'freeze.json').read_text())
    for path, expected in lock['files'].items():
        if tx.sha(path) != expected:
            raise ValueError('Protected file changed: '+path)


def folds():
    train = m.read(tx.OUT / 'training.npz')
    for fold, (start, _) in enumerate(tx.FOLDS):
        keep = train['origins']+tx.DAY <= start
        yield fold, {k: v[keep] for k, v in train.items()}, m.read(tx.OUT / f'validation_{fold}.npz')


def choose(records, config_keys):
    options = []
    configurations = sorted({tuple(r[k] for k in config_keys) for r in records})
    for configuration in configurations:
        selected = [r for r in records if tuple(r[k] for k in config_keys) == configuration]
        value = sum(r['validation_abs_sum'] for r in selected)/sum(r['validation_points'] for r in selected)
        options.append(dict(zip(config_keys, configuration), validation_mae=value))
    best = min(options, key=lambda r: (r['validation_mae'], r['epoch'],
        r.get('scaling') != 'p90', r.get('learning_rate', .001) != .001))
    return dict(best=best, options=options)


def select_base():
    protect()
    records = []
    for fold, train, val in folds():
        for scaling in SCALINGS:
            for rate in LRS:
                folder = OUT / 'base_selection' / f'f{fold}_{scaling}_lr{rate}'
                _, _, trace = fit(train, val, 'B', rate, scaling, 100, 42, folder, CHECKPOINTS)
                records += [dict(fold=fold, scaling=scaling, learning_rate=rate, **r) for r in trace]
                dump(OUT / 'base_grid.json', records)
    selection = choose(records, ('learning_rate', 'scaling', 'epoch'))
    dump(OUT / 'base_chosen.json', dict(**selection, grid_sha256=tx.sha(OUT / 'base_grid.json'), uses_july=False))
    print('B selected '+json.dumps(selection['best']), flush=True)


def select_fusion():
    protect()
    chosen = json.loads((OUT / 'base_chosen.json').read_text())
    assert chosen['grid_sha256'] == tx.sha(OUT / 'base_grid.json')
    config = chosen['best']
    records = []
    for fold, train, val in folds():
        for arm in ('F0', 'F1'):
            _, _, trace = fit(train, val, arm, config['learning_rate'], config['scaling'], 100, 42,
                              OUT / 'fusion_selection' / f'f{fold}_{arm}', CHECKPOINTS)
            records += [dict(fold=fold, arm=arm, **r) for r in trace]
            dump(OUT / 'fusion_grid.json', records)
    selection = choose(records, ('epoch',))
    dump(OUT / 'fusion_chosen.json', dict(**selection, learning_rate=config['learning_rate'], scaling=config['scaling'],
        grid_sha256=tx.sha(OUT / 'fusion_grid.json'), base_selection_sha256=tx.sha(OUT / 'base_chosen.json'),
        shared_epoch=True, uses_july=False))
    print('Matched F0/F1 selected '+json.dumps(selection['best']), flush=True)


def final():
    protect()
    base = json.loads((OUT / 'base_chosen.json').read_text())
    fusion = json.loads((OUT / 'fusion_chosen.json').read_text())
    assert base['grid_sha256'] == tx.sha(OUT / 'base_grid.json')
    assert fusion['grid_sha256'] == tx.sha(OUT / 'fusion_grid.json')
    assert fusion['base_selection_sha256'] == tx.sha(OUT / 'base_chosen.json')
    dump(OUT / 'final_selection.json', dict(base=base, fusion=fusion))
    train, query = m.read(tx.OUT / 'training.npz'), m.read(tx.OUT / 'query.npz')
    query['y'] = np.full_like(query['y'], np.nan)
    for arm in ARMS:
        epochs = base['best']['epoch'] if arm == 'B' else fusion['best']['epoch']
        for seed in tx.SEEDS:
            folder = OUT / 'final' / f'{arm}_{seed}'
            model, info, _ = fit(train, None, arm, base['best']['learning_rate'], base['best']['scaling'],
                                 epochs, seed, folder, CHECKPOINTS)
            e = encode(query, info['scales'], arm)
            before = time.perf_counter()
            raw = predict(model, e)
            seconds = time.perf_counter()-before
            replay = make_model(arm)
            replay.load_state_dict(load_file(str(folder / 'model.safetensors')))
            np.testing.assert_array_equal(raw, predict(replay, e))
            np.savez_compressed(folder / 'prediction.npz', raw=raw, p50=np.maximum(raw, 0.))
            dump(folder / 'replay.json', dict(exact=True, curves=len(raw), cached_prediction_seconds=seconds))
    print('Nine final models complete; July not yet scored', flush=True)


def score():
    protect()
    frame = pd.read_parquet(ROOT / 'artifacts/memory_head_control_v1/predictions.parquet')
    q = m.read(tx.OUT / 'query.npz')
    arms = {a: [a] for a in ('C', 'S', 'P')}
    arms.update({a: [f'{a}_{s}' for s in tx.SEEDS] for a in ('Tcal', 'Tzero', 'Tmem')})
    for arm in ARMS:
        arms[arm] = []
        for seed in tx.SEEDS:
            name = f'{arm}_{seed}'
            frame[name] = np.nan
            frame.loc[q['rows'].ravel(), name] = m.read(OUT / f'final/{name}/prediction.npz')['p50'].ravel()
            np.testing.assert_array_equal(frame[name].notna(), frame.S.notna())
            arms[arm].append(name)
    frame.to_parquet(OUT / 'predictions.parquet', index=False)
    results = dict(all={a: m.metrics(frame, v) for a, v in arms.items()}, groups=[], pairs=[],
        seeds={f'{a}_{s}': m.metrics(frame, [f'{a}_{s}']) for a in ARMS for s in tx.SEEDS})
    groups = [('id', str(i), g) for i, g in frame.groupby('depot_id')]
    groups += [('leave_out_id', i, frame.loc[frame.depot_id != i]) for i in tx.IDS]
    groups += [('state', 'zero', frame.loc[frame.actual_kw == 0]),
        ('state', 'positive_below_p90', frame.loc[(frame.actual_kw > 0) & (frame.actual_kw < frame.high_threshold)]),
        ('state', 'high', frame.loc[frame.actual_kw >= frame.high_threshold])]
    groups += [('lead', str(i), frame.loc[frame.step.between(i, i+5)]) for i in (1, 7, 13, 19)]
    for axis, label, f in groups:
        results['groups'].append(dict(axis=axis, label=label, models={a: m.metrics(f, v) for a, v in arms.items()}))
    for a, b in (('Tzero', 'B'), ('B', 'F0'), ('F0', 'F1'), ('Tmem', 'F1'),
                 ('S', 'B'), ('S', 'F0'), ('S', 'F1'), ('P', 'F1')):
        comp, daily = m.compare(frame, arms[a], arms[b])
        results['pairs'].append(dict(a=a, b=b, **comp))
        daily.to_parquet(OUT / f'daily_{a}_{b}.parquet', index=False)
    results['coverage'] = dict(candidate_origins=len(frame)//24, issued_origins=int(frame.S.notna().sum())//24,
        observed=int((frame.S.notna() & frame.actual_kw.notna()).sum()),
        missing=int((frame.S.notna() & frame.actual_kw.isna()).sum()))
    dump(OUT / 'results.json', results)
    dump(OUT / 'status.json', dict(status='completed_pending_verification', inner_fits=12, final_fits=9))
    print(json.dumps(results['all'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'select_base', 'select_fusion', 'final', 'score'))
    globals()[parser.parse_args().stage]()
