"""Forward-S anchored residual curves with tree and TimeXer controls."""

import argparse
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from safetensors.torch import load_file, save_file

from src import timexer_local_head_v1 as local
from src.optimise_structure_features_v1 import aligned_features
from src.optimise_budget_weekday_v1 import add_weekday

f, tx, m = local.f, local.tx, local.m
ROOT = tx.ROOT
OUT = ROOT / 'artifacts/s_curve_residual_v1'
EPOCHS = (0, 20, 40, 60, 80, 100)
STRENGTHS = (0., .25, .5, .75, 1.)
RATE = .0003


def dump(path, value):
    tx.dump(path, value)


def take(data, keep):
    return {k: v[keep] for k, v in data.items()}


def mature(origins, cutoff):
    return np.asarray(origins, np.int64)+tx.DAY <= cutoff


def block_starts(origins):
    anchor = tx.FOLDS[0][0]
    step = 14*tx.DAY
    first = anchor-((anchor-int(np.min(origins))+step-1)//step)*step
    return range(first, tx.CUTOFF, step)


def raw_features(data):
    base, _ = aligned_features(data['history'])
    base = add_weekday(base, pd.to_datetime(np.repeat(data['origins'], 24), utc=True))
    return np.concatenate([base.to_numpy().reshape(-1, 24, 180), m.memory_for(data)], axis=-1)


def tree_features(raw, baseline):
    if raw.shape != (len(baseline), 24, 207) or baseline.shape[1:] != (24,):
        raise ValueError('Unaligned S feature/curve shape')
    if not np.isfinite(baseline).all():
        raise ValueError('Forward S unavailable')
    return np.concatenate([raw, np.broadcast_to(baseline[:, None, :], (len(baseline), 24, 24)),
                           baseline[:, :, None]], axis=-1)


def combine(baseline, residual, strength=1.):
    if baseline.shape != residual.shape or not np.isfinite(baseline).all() or not np.isfinite(residual).all():
        raise ValueError('Invalid baseline/residual')
    return np.maximum(baseline+strength*residual, 0.)


def validation_row(y, p, **fields):
    mask = np.isfinite(y)
    if not np.isfinite(p).all() or not mask.any():
        raise ValueError('Nonfinite forecast or empty validation')
    total = float(np.abs(y[mask]-p[mask]).sum())
    return dict(**fields, validation_abs_sum=total, validation_points=int(mask.sum()),
                validation_mae=total/int(mask.sum()))


def make_model():
    t = tx.configure()

    class ResidualCurve(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = tx.load_tx(64)
            del self.backbone.head
            self.recent = t.nn.Linear(15*64, 64)
            self.known = t.nn.Linear(18, 64)
            self.curve = t.nn.Linear(24, 64)
            self.level = t.nn.Linear(1, 64)
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
            tokens = self.memory_embed(t.cat([values, masks], -1))+self.pool_embed
            support = values[:, :, :, 5]
            valid = (support > 0) & (masks[:, :, :, 5] > 0)
            logits = (self.query_key(query).unsqueeze(-2)*self.memory_key(tokens)).sum(-1)/8.
            weights = t.softmax(logits.masked_fill(~valid, -1e9), dim=-1)*valid
            weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
            return (weights.unsqueeze(-1)*self.memory_value(tokens)).sum(-2), support

        def forward(self, x, memory, future, s):
            en, _ = self.backbone.en_embedding(x[:, :, -1:].permute(0, 2, 1))
            ex = self.backbone.ex_embedding(x[:, :, :-1], None)
            z = self.backbone.encoder(en, ex)
            const = x[:, :1, 1:11].expand(-1, 24, -1)
            query = self.query_norm(self.recent(z.flatten(1)).unsqueeze(1)
                +self.known(t.cat([const, future], -1))+self.lead
                +self.curve(s).unsqueeze(1)+self.level(s.unsqueeze(-1)))
            context, support = self.read_memory(query, memory)
            gate = t.sigmoid(self.gate(t.cat([query, context, support], -1)))
            return self.decoder(self.fusion_norm(query+gate*context)).squeeze(-1)

    return ResidualCurve()


def encoded(data, scales):
    e = m.encoded(data, scales)
    e['s'] = (data['baseline']/e['scale'][:, None]).astype(np.float32)
    if not np.isfinite(e['s']).all():
        raise ValueError('Missing S prediction entered residual learner')
    return e


def neural_predict(model, e, baseline):
    t = tx.configure()
    model.eval()
    result = []
    with t.no_grad():
        for start in range(0, len(e['x']), 128):
            args = [t.from_numpy(e[k][start:start+128]) for k in ('x', 'memory', 'future', 's')]
            residual = model(*args).numpy()*e['scale'][start:start+128, None]
            result.append(combine(baseline[start:start+128], residual))
    return np.concatenate(result)


def fit_neural(train, val, scales, epochs, seed, folder):
    folder.mkdir(parents=True, exist_ok=False)
    t = tx.configure()
    t.manual_seed(seed)
    tr = encoded(train, scales)
    va = encoded(val, scales) if val is not None else None
    tensors = {k: t.from_numpy(tr[k]) for k in ('x', 'memory', 'future', 's', 'y', 'mask', 'scale')}
    model = make_model()
    opt = t.optim.AdamW(model.parameters(), lr=RATE, weight_decay=.0001)
    rng = np.random.default_rng(seed)
    before = time.perf_counter()
    trace = []

    def record(epoch, loss=None):
        row = dict(epoch=epoch, training_loss=loss, seconds=time.perf_counter()-before)
        if val is not None:
            p = neural_predict(model, va, val['baseline'])
            row.update(validation_row(val['y'], p))
            np.savez_compressed(folder / f'prediction_{epoch}.npz', p50=p)
        trace.append(row)
        dump(folder / 'trace.json', trace)
        save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()},
                  str(folder / f'epoch_{epoch}.safetensors'))
        print(folder.name+' '+json.dumps(row), flush=True)

    record(0)
    for epoch in range(1, epochs+1):
        model.train()
        running = 0.
        order = rng.permutation(len(tr['x']))
        for start in range(0, len(order), 64):
            idx = order[start:start+64]
            opt.zero_grad(set_to_none=True)
            residual = model(*[tensors[k][idx] for k in ('x', 'memory', 'future', 's')])
            prediction = tensors['s'][idx]+residual
            loss = tx.masked_loss(prediction, tensors['y'][idx], tensors['mask'][idx], tensors['scale'][idx],
                                  tr['valid_per_origin'], tr['scale_reference'])
            if not t.isfinite(loss):
                raise ValueError('Nonfinite residual loss')
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            running += float(loss.detach())*len(idx)
        if epoch in EPOCHS or epoch == epochs:
            record(epoch, running/len(order))
    info = dict(seed=seed, epochs=epochs, rate=RATE, scales=scales, labels=int(tr['mask'].sum()),
        origins=len(train['origins']), latest_training_horizon_ns=int(train['origins'].max()+tx.DAY),
        parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-before,
        validation_used=val is not None)
    dump(folder / 'fit.json', info)
    return model, trace


def freeze():
    local.check_cache()
    OUT.mkdir(exist_ok=False)
    inputs = [tx.OUT / (p+'.npz') for p in ('training', 'query', 'validation_0', 'validation_1')]
    inputs += [tx.INNER / 'inner_predictions.parquet', local.OUT / 'predictions.parquet',
               local.OUT / 'chosen.json', local.OUT / 'preparation.json']
    for i in tx.IDS:
        inputs += [m.OLD / f'prepared/{i}.npz', m.OLD / f'models/S_{i}.txt', m.OUT / f'prepared/{i}.npz']
    for fold in (0, 1):
        inputs += [local.OUT / f'cache/fold{fold}_{p}.npz' for p in ('train', 'validation')]
        inputs += [local.OUT / f'encoders/fold{fold}/model.safetensors']
    code = [Path(__file__), ROOT / 'experiments/s_curve_residual_v1.md',
        ROOT / 'tests/test_s_curve_residual_v1.py', ROOT / 'diagnostics/verify_s_curve_residual_v1.py']
    dump(OUT / 'freeze.json', dict(date='2026-10-27', files={str(p): tx.sha(p) for p in inputs+code},
        epochs=EPOCHS, rate=RATE, strengths=STRENGTHS, seeds=tx.SEEDS, forward_block_days=14,
        no_new_data=True, july_previously_seen=True, inner_neural_fits=2, final_neural_fits=3))
    print('Frozen S-anchored comparison', flush=True)


def protect():
    local.check_cache()
    for path, digest in json.loads((OUT / 'freeze.json').read_text())['files'].items():
        if tx.sha(path) != digest:
            raise ValueError('Changed frozen file: '+path)


def prepare():
    protect()
    before = time.perf_counter()
    (OUT / 'prepared').mkdir()
    (OUT / 'forward_models').mkdir()
    data = m.read(tx.OUT / 'training.npz')
    raw = raw_features(data)
    baseline = np.full_like(data['y'], np.nan)
    cutoff_by_origin = np.full(len(data['origins']), -1, np.int64)
    logs = []
    for depot in tx.IDS:
        local_mask = data['ids'] == depot
        old = m.read(m.OLD / f'prepared/{depot}.npz')
        valid = np.isfinite(data['y'][local_mask])
        np.testing.assert_array_equal(raw[local_mask][valid], old['train_S'])
        np.testing.assert_array_equal(data['y'][local_mask][valid], old['train_y'])
        for start in block_starts(data['origins']):
            tr = local_mask & mature(data['origins'], start)
            qe = local_mask & (data['origins'] >= start) & (data['origins'] < start+14*tx.DAY)
            complete = int(np.isfinite(data['y'][tr]).all(axis=1).sum())
            if not qe.any() or complete < 30:
                continue
            observed = np.isfinite(data['y'][tr])
            path = OUT / 'forward_models' / f'{depot}_{start}.txt'
            begun = time.perf_counter()
            model = lgb.LGBMRegressor(**m.CONFIG, n_estimators=300).fit(raw[tr][observed], data['y'][tr][observed]).booster_
            model.save_model(str(path))
            prediction = np.maximum(model.predict(raw[qe].reshape(-1, 207), num_threads=2), 0.).reshape(-1, 24)
            baseline[qe] = prediction
            cutoff_by_origin[qe] = start
            logs.append(dict(depot_id=depot, cutoff_ns=int(start), complete_origins=complete,
                training_origins=int(tr.sum()), labels=int(observed.sum()), query_origins=int(qe.sum()),
                max_training_horizon_ns=int(data['origins'][tr].max()+tx.DAY),
                min_query_ns=int(data['origins'][qe].min()), seconds=time.perf_counter()-begun,
                model_path=str(path)))
        print(f'Forward S ID{depot}: {int(np.isfinite(baseline[local_mask]).all(axis=1).sum())} origins', flush=True)
    train = dict(**data, baseline=baseline, raw=raw, baseline_cutoff=cutoff_by_origin)
    np.savez_compressed(OUT / 'prepared/train.npz', **train)
    inner = pd.read_parquet(tx.INNER / 'inner_predictions.parquet')
    for fold, (start, _) in enumerate(tx.FOLDS):
        val = m.read(tx.OUT / f'validation_{fold}.npz')
        lookup = {(str(i), int(o)): j for j, (i, o) in enumerate(zip(data['ids'], data['origins']))}
        idx = np.asarray([lookup[(str(i), int(o))] for i, o in zip(val['ids'], val['origins'])])
        for k in data:
            np.testing.assert_array_equal(val[k], data[k][idx])
        assert (cutoff_by_origin[idx] == start).all()
        va = dict(**val, baseline=baseline[idx], raw=raw[idx], baseline_cutoff=cutoff_by_origin[idx])
        ref = inner.loc[inner.fold == fold].sort_values(['depot_id', 'origin_utc', 'step'])
        np.testing.assert_array_equal(va['baseline'].ravel(), ref.S.to_numpy())
        np.savez_compressed(OUT / f'prepared/validation_{fold}.npz', **va)
    query = m.read(tx.OUT / 'query.npz')
    query['y'] = np.full_like(query['y'], np.nan)
    raw_q = raw_features(query)
    s_q = np.full_like(query['y'], np.nan)
    reference = pd.read_parquet(local.OUT / 'predictions.parquet')
    for depot in tx.IDS:
        qe = query['ids'] == depot
        old = m.read(m.OLD / f'prepared/{depot}.npz')
        np.testing.assert_array_equal(raw_q[qe].reshape(-1, 207), old['query_S'])
        model = lgb.Booster(model_file=str(m.OLD / f'models/S_{depot}.txt'))
        s_q[qe] = np.maximum(model.predict(raw_q[qe].reshape(-1, 207), num_threads=2), 0.).reshape(-1, 24)
    np.testing.assert_array_equal(s_q.ravel(), reference.iloc[query['rows'].ravel()].S.to_numpy())
    np.savez_compressed(OUT / 'prepared/query.npz', **query, baseline=s_q, raw=raw_q,
                        baseline_cutoff=np.full(len(s_q), tx.CUTOFF, np.int64))
    supported = np.isfinite(baseline).all(axis=1)
    assert (cutoff_by_origin[supported] <= data['origins'][supported]).all()
    manifest = {str(p): tx.sha(p) for p in (OUT / 'prepared').glob('*.npz')}
    manifest.update({r['model_path']: tx.sha(r['model_path']) for r in logs})
    dump(OUT / 'forward_log.json', logs)
    dump(OUT / 'preparation.json', dict(files=manifest, forward_fits=len(logs),
        total_labels=int(np.isfinite(data['y']).sum()), residual_labels=int(np.isfinite(data['y'][supported]).sum()),
        residual_origins=int(supported.sum()), excluded_warmup_origins=int((~supported).sum()),
        june_S_replay_exact=True, july_S_replay_exact=True, seconds=time.perf_counter()-before))


def check_prepared():
    protect()
    for path, digest in json.loads((OUT / 'preparation.json').read_text())['files'].items():
        if tx.sha(path) != digest:
            raise ValueError('Changed prepared input: '+path)


def training(cutoff):
    all_data = m.read(OUT / 'prepared/train.npz')
    allowed = mature(all_data['origins'], cutoff)
    scales = tx.scales_from(take(all_data, allowed))
    keep = allowed & np.isfinite(all_data['baseline']).all(axis=1)
    train = take(all_data, keep)
    assert (train['baseline_cutoff'] <= train['origins']).all()
    return train, scales


def fit_tree(train, query, folder):
    folder.mkdir(parents=True, exist_ok=False)
    result = np.full_like(query['baseline'], np.nan)
    logs = []
    for depot in tx.IDS:
        tr, qe = train['ids'] == depot, query['ids'] == depot
        x = tree_features(train['raw'][tr], train['baseline'][tr])
        y = train['y'][tr]-train['baseline'][tr]
        valid = np.isfinite(y)
        begun = time.perf_counter()
        model = lgb.LGBMRegressor(**m.CONFIG, n_estimators=300).fit(x[valid], y[valid]).booster_
        model.save_model(str(folder / f'{depot}.txt'))
        q = tree_features(query['raw'][qe], query['baseline'][qe]).reshape(-1, 232)
        result[qe] = model.predict(q, num_threads=2).reshape(-1, 24)
        logs.append(dict(depot_id=depot, labels=int(valid.sum()), features=232, seconds=time.perf_counter()-begun))
    np.savez_compressed(folder / 'residual.npz', residual=result)
    dump(folder / 'fit.json', logs)
    return result


def select():
    check_prepared()
    blend_records, tree_records, neural_records = [], [], []
    for fold, (start, _) in enumerate(tx.FOLDS):
        train, scales = training(start)
        val = m.read(OUT / f'prepared/validation_{fold}.npz')
        cache = m.read(local.OUT / f'cache/fold{fold}_train.npz')
        vc = m.read(local.OUT / f'cache/fold{fold}_validation.npz')
        decoder = local.decoder_from(local.OUT / f'encoders/fold{fold}/model.safetensors')
        head, _ = local.fit_head(cache, vc, decoder, 'L', .001, 20, 42, OUT / f'selection/L_fold{fold}')
        p_local = np.maximum(local.predict(head, vc), 0.)
        np.savez_compressed(OUT / f'selection/L_fold{fold}/prediction.npz', p50=p_local)
        for alpha in STRENGTHS:
            p = (1-alpha)*val['baseline']+alpha*p_local
            blend_records.append(validation_row(val['y'], p, fold=fold, strength=alpha))
        residual = fit_tree(train, val, OUT / f'selection/R_fold{fold}')
        for strength in STRENGTHS:
            tree_records.append(validation_row(val['y'], combine(val['baseline'], residual, strength),
                                                fold=fold, strength=strength))
        _, trace = fit_neural(train, val, scales, 100, 42, OUT / f'selection/N_fold{fold}')
        neural_records += [dict(fold=fold, **r) for r in trace]
        dump(OUT / 'selection_grid.json', dict(blend=blend_records, tree=tree_records, neural=neural_records))
    chosen = {}
    for arm, records, key in (('B', blend_records, 'strength'), ('R', tree_records, 'strength'),
                               ('N', neural_records, 'epoch')):
        options = []
        for value in sorted({r[key] for r in records}):
            part = [r for r in records if r[key] == value]
            assert len(part) == 2
            options.append({key: value, 'validation_mae': sum(r['validation_abs_sum'] for r in part)
                /sum(r['validation_points'] for r in part)})
        chosen[arm] = dict(best=min(options, key=lambda r: (r['validation_mae'], r[key])), options=options)
    dump(OUT / 'chosen.json', dict(models=chosen, uses_july=False, grid_sha256=tx.sha(OUT / 'selection_grid.json')))
    print('June chosen '+json.dumps(chosen), flush=True)


def final():
    check_prepared()
    chosen = json.loads((OUT / 'chosen.json').read_text())
    assert chosen['grid_sha256'] == tx.sha(OUT / 'selection_grid.json')
    train, scales = training(tx.CUTOFF)
    query = m.read(OUT / 'prepared/query.npz')
    fit_tree(train, query, OUT / 'final/R')
    epochs = chosen['models']['N']['best']['epoch']
    for seed in tx.SEEDS:
        folder = OUT / f'final/N_{seed}'
        model, _ = fit_neural(train, None, scales, epochs, seed, folder)
        p = neural_predict(model, encoded(query, scales), query['baseline'])
        replay = make_model()
        replay.load_state_dict(load_file(str(folder / f'epoch_{epochs}.safetensors')))
        np.testing.assert_array_equal(p, neural_predict(replay, encoded(query, scales), query['baseline']))
        np.savez_compressed(folder / 'prediction.npz', p50=p)
    print('Final models complete; no July scoring in fitting', flush=True)


def score():
    check_prepared()
    chosen = json.loads((OUT / 'chosen.json').read_text())['models']
    frame = pd.read_parquet(local.OUT / 'predictions.parquet')
    query = m.read(OUT / 'prepared/query.npz')
    rows = query['rows'].ravel()
    arms = {a: [a] for a in ('C', 'S', 'P')}
    arms['L'] = [f'HL_{s}' for s in tx.SEEDS]
    arms['B'], arms['N'] = [], []
    for seed in tx.SEEDS:
        alpha = chosen['B']['best']['strength']
        assert f'SB_{seed}' not in frame and f'N_{seed}' not in frame
        frame[f'SB_{seed}'] = (1-alpha)*frame.S+alpha*frame[f'HL_{seed}']
        frame[f'N_{seed}'] = np.nan
        frame.loc[rows, f'N_{seed}'] = m.read(OUT / f'final/N_{seed}/prediction.npz')['p50'].ravel()
        arms['B'].append(f'SB_{seed}'); arms['N'].append(f'N_{seed}')
    assert 'R' not in frame
    frame['R'] = np.nan
    residual = m.read(OUT / 'final/R/residual.npz')['residual']
    frame.loc[rows, 'R'] = combine(query['baseline'], residual, chosen['R']['best']['strength']).ravel()
    arms['R'] = ['R']
    for names in arms.values():
        for name in names:
            np.testing.assert_array_equal(frame[name].notna(), frame.S.notna())
    result = dict(all={a: m.metrics(frame, v) for a, v in arms.items()}, arms=arms, groups=[], pairs=[],
                  seeds={n: m.metrics(frame, [n]) for a in ('B', 'N') for n in arms[a]})
    groups = [('id', str(i), part) for i, part in frame.groupby('depot_id')]
    groups += [('leave_out_id', i, frame.loc[frame.depot_id != i]) for i in tx.IDS]
    groups += [('state', 'zero', frame.loc[frame.actual_kw == 0]),
        ('state', 'positive_below_p90', frame.loc[(frame.actual_kw > 0) & (frame.actual_kw < frame.high_threshold)]),
        ('state', 'high', frame.loc[frame.actual_kw >= frame.high_threshold])]
    groups += [('lead', str(i), frame.loc[frame.step.between(i, i+5)]) for i in (1, 7, 13, 19)]
    for axis, label, part in groups:
        result['groups'].append(dict(axis=axis, label=label, models={a: m.metrics(part, v) for a, v in arms.items()}))
    for a, b in (('S', 'B'), ('P', 'B'), ('S', 'R'), ('S', 'N'), ('P', 'N'), ('B', 'N'), ('R', 'N')):
        pair, daily = m.compare(frame, arms[a], arms[b])
        result['pairs'].append(dict(a=a, b=b, **pair))
        daily.to_parquet(OUT / f'daily_{a}_{b}.parquet', index=False)
    result['coverage'] = dict(candidate_origins=len(frame)//24, issued_origins=int(frame.S.notna().sum())//24,
        observed=int((frame.S.notna() & frame.actual_kw.notna()).sum()),
        missing=int((frame.S.notna() & frame.actual_kw.isna()).sum()))
    frame.to_parquet(OUT / 'predictions.parquet', index=False)
    dump(OUT / 'results.json', result)
    dump(OUT / 'status.json', dict(status='completed_pending_verification'))
    print(json.dumps(result['all'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'prepare', 'select', 'final', 'score'))
    globals()[parser.parse_args().stage]()
