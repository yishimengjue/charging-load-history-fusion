"""Matched-history representations and a bounded memory-informed TimeXer."""

import argparse
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import chronos_timexer_v1 as tx
from src.global_local_forecast_v1 import CONFIG
from src.historical_memory_v1 import DAY_NS, candidate_selection, summarize_targets
from src.global_shared_july_v1 import uniform_memory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/memory_evidence_v1'
OLD = ROOT / 'artifacts/global_shared_july_v1'
DEV = ROOT / 'artifacts/historical_memory_v1'
AGES = np.array(list(range(1, 8)) + list(range(14, 85, 7)))
MONTHS = ('2022-01', '2022-02', '2022-03', '2022-07')
IDS = tx.IDS


def dump(name, value):
    tx.dump(OUT / name, value)


def read(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def protect():
    lock = json.loads((OUT / 'freeze.json').read_text())
    for key in ('inputs', 'implementation'):
        for path, expected in lock[key].items():
            if tx.sha(path) != expected:
                raise ValueError('Protected input changed: ' + path)
    return lock


def pack_memory(query, x, y, ids, clock):
    positions, ages, *_ = candidate_selection(query, x, ids, clock)
    keep = np.isin(ages, AGES)
    positions, ages = positions[keep], ages[keep]
    if len(positions) and (clock[positions] + DAY_NS > clock[query]).any():
        raise ValueError('Immature reference target')
    if len(set(ages)) != len(ages):
        raise ValueError('Duplicate reference age')
    raw = np.full((24, len(AGES)), np.nan)
    eligible = np.zeros_like(raw)
    slots = np.searchsorted(AGES, ages)
    raw[:, slots] = y[positions].T
    eligible[:, slots] = 1
    one = summarize_targets(y[positions], np.zeros(len(positions)), ages)
    multi = summary_from_raw(raw, eligible)
    return dict(L=np.concatenate([raw, eligible], axis=1), U=one, S=multi), positions


def summary_from_raw(raw, eligible):
    if not np.array_equal(eligible, np.broadcast_to(eligible[0], eligible.shape)):
        raise ValueError('Candidate eligibility must not depend on a future lead')
    parts = []
    for days, weekday in ((7, False), (28, True), (84, True)):
        keep = (AGES <= days) & (eligible[0] == 1)
        if weekday:
            keep &= AGES % 7 == 0
        # Preserve the original chronological reduction order, including rounding.
        slots = np.flatnonzero(keep)[::-1]
        parts.append(summarize_targets(raw[:, slots].T, np.zeros(len(slots)), AGES[slots]))
    return np.concatenate(parts, axis=1)


def freeze():
    OUT.mkdir(exist_ok=False)
    for name in ('prepared', 'models', 'neural'):
        (OUT / name).mkdir()
    inputs = [ROOT / 'artifacts/optimise_final_v1/released_snapshots.json',
              ROOT / 'artifacts/optimise_july_v1/windows.npz',
              DEV / 'predictions.parquet', OLD / 'predictions.parquet',
              tx.OUT / 'predictions.parquet', tx.OUT / 'training.npz', tx.OUT / 'query.npz',
              ROOT / 'artifacts/optimise_july_preflight_v1/fit_log.json',
              ROOT / 'experiments/forecast_external_replication_v1.md',
              ROOT / 'data/forecast_source_followup_2026-10-22/SOURCE_NOTES.md']
    for depot in IDS:
        inputs += [OLD / 'prepared' / f'{depot}.npz', DEV / 'prepared' / f'{depot}.npz',
                   OLD / 'models' / f'S_{depot}.txt']
        for month in MONTHS[:3]:
            inputs += [DEV / 'models' / depot / month / 'S/model.txt',
                       ROOT / 'artifacts/optimise_budget_weekday_v1/models' / depot / month / 'C_weekday.txt']
    oldlogs = json.loads((ROOT / 'artifacts/optimise_july_preflight_v1/fit_log.json').read_text())
    inputs += [Path(r['model_path']) for r in oldlogs if r.get('model') == 'C_weekday']
    for seed in tx.SEEDS:
        inputs += [tx.OUT / f'final/Tcal_{seed}/model.safetensors', tx.OUT / f'final/Tcal_{seed}/fit.json']
    inputs += [tx.VENDOR / p for p in tx.TX_FILES]
    implementation = [Path(__file__), ROOT / 'tests/test_memory_evidence_v1.py',
                      ROOT / 'diagnostics/verify_memory_evidence_v1.py',
                      ROOT / 'experiments/memory_evidence_v1.md',
                      ROOT / 'src/chronos_timexer_v1.py', ROOT / 'src/historical_memory_v1.py',
                      ROOT / 'src/global_shared_july_v1.py', ROOT / 'src/global_local_forecast_v1.py',
                      ROOT / 'src/optimise_final_data_v1.py', ROOT / 'src/optimise_numeric_data_v1.py',
                      ROOT / 'run_chronos_python.sh']
    dump('freeze.json', dict(inputs={str(p): tx.sha(p) for p in inputs},
        implementation={str(p): tx.sha(p) for p in implementation}, date='2026-10-26',
        new_data=False, all_evaluation_previously_seen=True, tree_fits=72, neural_fits=3,
        ages=AGES.tolist(), tree_config=dict(CONFIG, n_estimators=300), epochs=70, seeds=tx.SEEDS))
    print('Frozen matched-history experiment', flush=True)


def prepare():
    from src.optimise_final_data_v1 import build_extended_windows
    from src.optimise_numeric_data_v1 import TRAIN_START
    from src.optimise_missing_features_v1 import history_eligible
    protect()
    before = time.perf_counter()
    w = build_extended_windows(json.loads((ROOT / 'artifacts/optimise_final_v1/released_snapshots.json').read_text()),
                               TRAIN_START, pd.Timestamp('2022-07-01', tz='UTC'),
                               tuple(str(i) for i in range(1, 12)), allow_final=True)
    july = read(ROOT / 'artifacts/optimise_july_v1/windows.npz')
    rows = []
    for depot in IDS:
        h = np.flatnonzero(w.depot_ids == depot)
        q = np.flatnonzero(july['ids'] == depot)
        x = np.concatenate([w.X[h], july['histories'][q]])
        y = np.concatenate([w.Y[h], july['truth'][q]])
        clock = np.r_[w.origins[h].as_unit('ns').asi8, july['origin_ns'][q]]
        ids = np.full(len(x), depot)
        eligible = np.flatnonzero(history_eligible(x))
        data = dict(origins=clock[eligible], y=y[eligible])
        features = {a: [] for a in ('S', 'U', 'L')}
        started = time.perf_counter()
        for index in eligible:
            packed, members = pack_memory(index, x, y, ids, clock)
            old, _ = uniform_memory(index, x, y, ids, clock)
            np.testing.assert_array_equal(packed['S'], old)
            for arm in features:
                features[arm].append(packed[arm])
            rows.extend(dict(depot_id=depot, query_ns=int(clock[index]), candidate_ns=int(clock[i]),
                             age_days=int((clock[index]-clock[i])//DAY_NS)) for i in members)
        for arm in features:
            data[arm] = np.stack(features[arm])
        p = read(DEV / 'prepared' / f'{depot}.npz')
        index = np.searchsorted(data['origins'], p['origin_ns'])
        np.testing.assert_array_equal(data['origins'][index], p['origin_ns'])
        np.testing.assert_array_equal(data['S'][index], p['S'][:, :, 180:])
        np.testing.assert_array_equal(data['y'][index], p['truth'])
        p = read(OLD / 'prepared' / f'{depot}.npz')
        ti = np.searchsorted(data['origins'], p['train_origins'])
        np.testing.assert_array_equal(data['origins'][ti], p['train_origins'])
        np.testing.assert_array_equal(data['S'][ti, p['train_steps']-1], p['train_S'][:, 180:])
        np.testing.assert_array_equal(data['y'][ti, p['train_steps']-1], p['train_y'])
        data['prepare_seconds'] = np.array(time.perf_counter()-started)
        np.savez_compressed(OUT / 'prepared' / f'{depot}.npz', **data)
        print(f'Prepared {depot}: {len(eligible)} origins; exact old S reconstruction', flush=True)
    pd.DataFrame(rows).to_parquet(OUT / 'memory_membership.parquet', index=False)
    dump('preparation.json', dict(seconds=time.perf_counter()-before, old_S_exact=True,
                                 raw_to_S_exact=True, candidate_pairs=len(rows)))


def reference_frame():
    dev = pd.read_parquet(DEV / 'predictions.parquet')
    july = pd.read_parquet(tx.OUT / 'predictions.parquet')
    keys = ['depot_id', 'origin_utc', 'step', 'actual_kw', 'source_row', 'C', 'S']
    frame = pd.concat([dev[keys], july[keys]], ignore_index=True)
    frame['month'] = frame.origin_utc.dt.strftime('%Y-%m')
    for arm in ('P', 'Tcal_42', 'Tcal_43', 'Tcal_44'):
        frame[arm] = np.r_[np.full(len(dev), np.nan), july[arm].to_numpy()]
    frame['high_threshold'] = np.nan
    return frame


def tree_arrays(depot, month, frame):
    mem = read(OUT / 'prepared' / f'{depot}.npz')
    target = frame.loc[(frame.depot_id == depot) & (frame.month == month)]
    issued = target.S.notna()
    target = target.loc[issued]
    qo = target.origin_utc.astype('int64').to_numpy()
    qi = np.searchsorted(mem['origins'], qo)
    np.testing.assert_array_equal(mem['origins'][qi], qo)
    steps = target.step.to_numpy()-1
    oldlogs = json.loads((ROOT / 'artifacts/optimise_july_preflight_v1/fit_log.json').read_text())
    if month == '2022-07':
        d = read(OLD / 'prepared' / f'{depot}.npz')
        tr_orig, tr_steps, y = d['train_origins'], d['train_steps']-1, d['train_y']
        base, query = d['train_S'][:, :180], d['query_base']
        cpath = next(r['model_path'] for r in oldlogs if r.get('model') == 'C_weekday' and r['depot_id'] == depot)
        spath = OLD / 'models' / f'S_{depot}.txt'
    else:
        d = read(DEV / 'prepared' / f'{depot}.npz')
        t, q = d['train_'+month], d['eval_'+month]
        valid = np.isfinite(d['truth'][t])
        tr_orig = np.repeat(d['origin_ns'][t], 24)[valid.ravel()]
        tr_steps = np.tile(np.arange(24), len(t))[valid.ravel()]
        y = d['truth'][t][valid]
        base, query = d['base'][t][valid], d['base'][q].reshape(-1, 180)
        cpath = ROOT / 'artifacts/optimise_budget_weekday_v1/models' / depot / month / 'C_weekday.txt'
        spath = DEV / 'models' / depot / month / 'S/model.txt'
    cutoff = pd.Timestamp(month+'-01', tz='UTC').value
    assert (tr_orig+DAY_NS <= cutoff).all()
    ti = np.searchsorted(mem['origins'], tr_orig)
    np.testing.assert_array_equal(mem['origins'][ti], tr_orig)
    np.testing.assert_array_equal(mem['y'][ti, tr_steps], y)
    np.testing.assert_array_equal(mem['y'][qi, steps], target.actual_kw.to_numpy())
    train = {'C': base}
    queries = {'C': query}
    for arm in ('S', 'U', 'L'):
        train[arm] = np.column_stack([base, mem[arm][ti, tr_steps]])
        queries[arm] = np.column_stack([query, mem[arm][qi, steps]])
    return train, y, queries, target.index.to_numpy(), {'C': cpath, 'S': spath}


def trees():
    protect()
    frame = reference_frame()
    frame['U'] = np.nan
    frame['L'] = np.nan
    logs = []
    for month in MONTHS:
        for depot in IDS:
            x, y, q, rows, paths = tree_arrays(depot, month, frame)
            frame.loc[rows, 'high_threshold'] = max(float(np.quantile(y, .9)), 1.)
            for arm in ('C', 'S', 'U', 'L'):
                start = time.perf_counter()
                if arm in paths:
                    path = paths[arm]
                    model = lgb.Booster(model_file=str(path))
                else:
                    path = OUT / 'models' / f'{month}_{depot}_{arm}.txt'
                    if path.exists():
                        raise FileExistsError(path)
                    model = lgb.LGBMRegressor(**CONFIG, n_estimators=300).fit(x[arm], y).booster_
                    model.save_model(str(path))
                fit_seconds = time.perf_counter()-start
                start = time.perf_counter()
                raw = model.predict(q[arm], num_threads=2)
                infer_seconds = time.perf_counter()-start
                replay = lgb.Booster(model_file=str(path)).predict(q[arm], num_threads=2)
                np.testing.assert_array_equal(raw, replay)
                p = np.maximum(raw, 0.)
                if arm in paths:
                    np.testing.assert_array_equal(p, frame.loc[rows, arm].to_numpy())
                else:
                    frame.loc[rows, arm] = p
                logs.append(dict(month=month, depot_id=depot, arm=arm, labels=len(y), features=x[arm].shape[1],
                    path=str(path), sha256=tx.sha(path), fit_or_load_seconds=fit_seconds,
                    cached_prediction_seconds=infer_seconds, rows=len(rows), replay_exact=True,
                    new_fit=arm not in paths))
            print(f'Fitted matched trees {month} ID{depot}', flush=True)
    frame.to_parquet(OUT / 'tree_predictions.parquet', index=False)
    dump('tree_fit_log.json', logs)
    assert sum(r['new_fit'] for r in logs) == 72


def memory_for(data):
    memory = np.full((len(data['ids']), 24, 27), np.nan)
    for depot in IDS:
        keep = data['ids'] == depot
        d = read(OUT / 'prepared' / f'{depot}.npz')
        index = np.searchsorted(d['origins'], data['origins'][keep])
        np.testing.assert_array_equal(d['origins'][index], data['origins'][keep])
        memory[keep] = d['S'][index]
    return memory


def memory_encode(memory, scale):
    m = memory.copy().astype(np.float32)
    mask = np.isfinite(m).astype(np.float32)
    for start, days, count in ((0, 7, 7), (9, 28, 4), (18, 84, 12)):
        m[:, :, start:start+4] /= scale[:, None, None]
        m[:, :, start+5:start+7] /= count
        m[:, :, start+7] /= days
    return np.concatenate([np.nan_to_num(m, nan=0.), mask], axis=2)


def make_model():
    t = tx.configure()

    class MemoryTimeXer(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = tx.load_tx(64)
            self.head = t.nn.Sequential(t.nn.Linear(73, 32), t.nn.ReLU(), t.nn.Linear(32, 1))
            t.nn.init.zeros_(self.head[-1].weight)
            t.nn.init.zeros_(self.head[-1].bias)

        def forward(self, x, memory, future):
            base = self.backbone(x, None, None, None).squeeze(-1)
            const = x[:, :1, 1:11].expand(-1, 24, -1)
            v = t.cat([base.unsqueeze(-1), memory, const, future], dim=-1)
            return base + self.head(v).squeeze(-1)

    return MemoryTimeXer()


def encoded(data, scales):
    e = tx.encode(data, scales, True)
    e['memory'] = memory_encode(memory_for(data), e['scale'])
    e['future'] = np.concatenate([tx.weekday(data['origins'], np.arange(1, 25)),
        np.broadcast_to((np.arange(1, 25)/24).astype(np.float32)[None, :, None],
                        (len(data['origins']), 24, 1))], axis=2)
    assert np.isfinite(e['memory']).all()
    return e


def neural_predict(model, e):
    t = tx.configure()
    model.eval()
    values = []
    with t.no_grad():
        for start in range(0, len(e['x']), 128):
            p = model(*[t.from_numpy(e[k][start:start+128]) for k in ('x', 'memory', 'future')])
            values.append(p.numpy()*e['scale'][start:start+128, None])
    return np.concatenate(values)


def neural():
    from safetensors.torch import save_file, load_file
    protect()
    t = tx.configure()
    train, query = read(tx.OUT / 'training.npz'), read(tx.OUT / 'query.npz')
    query['y'] = np.full_like(query['y'], np.nan)
    scales = tx.scales_from(train)
    before = time.perf_counter()
    tr, qe = encoded(train, scales), encoded(query, scales)
    encode_seconds = time.perf_counter()-before
    assert int(tr['mask'].sum()) == 74296 and (train['origins']+DAY_NS <= tx.CUTOFF).all()
    np.savez_compressed(OUT / 'neural/encoded_query.npz', **qe)
    for seed in tx.SEEDS:
        folder = OUT / 'neural' / str(seed)
        folder.mkdir(exist_ok=False)
        t.manual_seed(seed)
        model = make_model()
        opt = t.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        tensors = {k: t.from_numpy(tr[k]) for k in ('x', 'memory', 'future', 'y', 'mask', 'scale')}
        rng = np.random.default_rng(seed)
        trace = []
        started = time.perf_counter()
        for epoch in range(1, 71):
            model.train()
            total = 0.
            for idx in np.array_split(rng.permutation(len(tr['x'])), np.arange(64, len(tr['x']), 64)):
                opt.zero_grad(set_to_none=True)
                p = model(*[tensors[k][idx] for k in ('x', 'memory', 'future')])
                loss = tx.masked_loss(p, tensors['y'][idx], tensors['mask'][idx], tensors['scale'][idx],
                                     tr['valid_per_origin'], tr['scale_reference'])
                if not t.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                total += float(loss.detach())*len(idx)
            if epoch % 10 == 0:
                trace.append(dict(epoch=epoch, training_batch_loss=total/len(tr['x'])))
                print(f'Tmem seed{seed} epoch{epoch}: training loss {total/len(tr["x"]):.4f}', flush=True)
        fit_seconds = time.perf_counter()-started
        save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(folder / 'model.safetensors'))
        started = time.perf_counter()
        raw = neural_predict(model, qe)
        inference_seconds = time.perf_counter()-started
        replay = make_model()
        replay.load_state_dict(load_file(str(folder / 'model.safetensors')))
        np.testing.assert_array_equal(raw, neural_predict(replay, qe))
        np.savez_compressed(folder / 'prediction.npz', raw=raw, p50=np.maximum(raw, 0.))
        tx.dump(folder / 'fit.json', dict(seed=seed, epochs=70, parameters=sum(p.numel() for p in model.parameters()),
            labels=74296, origins=3278, seconds=fit_seconds, cached_prediction_seconds=inference_seconds,
            shared_encoding_seconds=encode_seconds, trace=trace, replay_exact=True, scales=scales))
    print('Three memory-informed TimeXer models completed without July scoring', flush=True)


def metrics(frame, arms):
    y = frame.actual_kw.to_numpy()
    good = np.isfinite(y) & np.isfinite(frame[arms[0]])
    if not good.any():
        return dict(points=0, mae_kw=None, rmse_kw=None, bias_kw=None,
                    mean_under_kw=None, mean_over_kw=None, severe_under_rate=None)
    e = frame.loc[good, arms].to_numpy()-y[good, None]
    threshold = frame.loc[good, 'high_threshold'].to_numpy()[:, None]
    return dict(points=int(good.sum()), mae_kw=float(np.abs(e).mean()),
        rmse_kw=float(np.sqrt((e*e).mean(axis=0)).mean()), bias_kw=float(e.mean()),
        mean_under_kw=float(np.maximum(-e, 0).mean()), mean_over_kw=float(np.maximum(e, 0).mean()),
        severe_under_rate=float((-e > threshold).mean()))


def compare(frame, first, second):
    valid = frame.actual_kw.notna() & frame[first+second].notna().all(axis=1)
    issued = frame[first+second].notna().all(axis=1)
    y = frame.actual_kw.to_numpy()
    a, b = frame[first].to_numpy(), frame[second].to_numpy()
    gain = np.abs(a-y[:, None]).mean(axis=1)-np.abs(b-y[:, None]).mean(axis=1)
    grouped = pd.DataFrame(dict(date=frame.origin_utc.dt.floor('D'),
        gain=np.where(valid, gain, 0.), count=valid.astype(int))).groupby('date', sort=True)[['gain', 'count']].sum()
    rng = np.random.default_rng(42)
    ci = []
    for length in (7, 14):
        sums, counts = grouped.gain.to_numpy(), grouped['count'].to_numpy()
        values = []
        for _ in range(2000):
            starts = rng.integers(0, len(grouped)-length+1, size=int(np.ceil(len(grouped)/length)))
            positions = np.concatenate([np.arange(s, s+length) for s in starts])[:len(grouped)]
            values.append(float(sums[positions].sum()/counts[positions].sum()))
        ci.append(dict(block_days=length, low=float(np.quantile(values, .025)), high=float(np.quantile(values, .975))))
    unknown = issued & ~valid
    if len(first) == len(second):
        radius = float(np.abs(a[unknown]-b[unknown]).mean(axis=1).sum())
    elif len(first) == 1 or len(second) == 1:
        radius = float(np.abs(a[unknown]-b[unknown]).mean(axis=1).sum())
    else:
        raise ValueError('Undefined seed pairing')
    total = float(gain[valid].sum())
    n = int(issued.sum())
    return dict(first=first, second=second, gain_kw=total/int(valid.sum()),
        gain_percent=100*(1-metrics(frame, second)['mae_kw']/metrics(frame, first)['mae_kw']),
        intervals=ci, observed=int(valid.sum()), missing=int(unknown.sum()),
        all_issued_gain_outer_bounds_kw=[(total-radius)/n, (total+radius)/n]), grouped.reset_index()


def evaluate():
    protect()
    frame = pd.read_parquet(OUT / 'tree_predictions.parquet')
    july_rows = frame.index[frame.month == '2022-07'].to_numpy()
    q = read(tx.OUT / 'query.npz')
    for seed in tx.SEEDS:
        name = f'Tmem_{seed}'
        frame[name] = np.nan
        frame.loc[july_rows[q['rows'].ravel()], name] = read(OUT / f'neural/{seed}/prediction.npz')['p50'].ravel()
    frame.to_parquet(OUT / 'predictions.parquet', index=False)
    result = dict(periods={}, comparisons=[], coverage={}, exploratory=True, independent_validation=False)
    for month in MONTHS:
        part = frame.loc[frame.month == month]
        arms = {a: [a] for a in ('C', 'S', 'U', 'L')}
        if month == '2022-07':
            arms['P'] = ['P']
            for a in ('Tcal', 'Tmem'):
                arms[a] = [f'{a}_{s}' for s in tx.SEEDS]
        records = dict(all={a: metrics(part, v) for a, v in arms.items()}, groups=[])
        groups = [('id', str(i), g) for i, g in part.groupby('depot_id')]
        groups += [('leave_out_id', str(i), part.loc[part.depot_id != i]) for i in IDS]
        groups += [('state', 'zero', part.loc[part.actual_kw == 0]),
                   ('state', 'positive_below_p90', part.loc[(part.actual_kw > 0) & (part.actual_kw < part.high_threshold)]),
                   ('state', 'high', part.loc[part.actual_kw >= part.high_threshold])]
        groups += [('lead', f'{low}-{low+5}', part.loc[part.step.between(low, low+5)]) for low in (1, 7, 13, 19)]
        for axis, label, g in groups:
            records['groups'].append(dict(axis=axis, label=label, models={a: metrics(g, v) for a, v in arms.items()}))
        result['periods'][month] = records
        issued = part.C.notna()
        truth = part.actual_kw.notna() & issued
        result['coverage'][month] = dict(candidate_origins=len(part)//24, issued_origins=int(issued.sum())//24,
            issued_targets=int(issued.sum()), scored_targets=int(truth.sum()), missing_truth=int((issued & ~truth).sum()))
        pairs = [('C', 'S'), ('C', 'U'), ('C', 'L'), ('U', 'S'), ('L', 'S')]
        if month == '2022-07':
            pairs += [('Tcal', 'Tmem'), ('S', 'Tmem'), ('S', 'P')]
        for a, b in pairs:
            comparison, daily = compare(part, arms[a], arms[b])
            result['comparisons'].append(dict(month=month, a=a, b=b, **comparison))
            daily.to_parquet(OUT / f'daily_{month}_{a}_{b}.parquet', index=False)
    july = frame.loc[frame.month == '2022-07']
    result['neural_seeds'] = {f'{arm}_{s}': metrics(july, [f'{arm}_{s}']) for arm in ('Tcal', 'Tmem') for s in tx.SEEDS}
    result['development_pooled'] = {a: metrics(frame.loc[frame.month != '2022-07'], [a]) for a in ('C', 'S', 'U', 'L')}
    dump('results.json', result)
    dump('status.json', dict(status='completed_pending_verification', new_tree_fits=72, new_neural_fits=3))
    print(json.dumps({m: r['all'] for m, r in result['periods'].items()}, indent=2), flush=True)


def benchmark():
    """Single-process repeated online feature/prediction probes, not I/O latency."""
    from src.optimise_structure_features_v1 import aligned_features
    from src.optimise_budget_weekday_v1 import add_weekday
    from src.optimise_final_data_v1 import build_extended_windows
    from src.optimise_numeric_data_v1 import TRAIN_START
    protect()
    w = build_extended_windows(json.loads((ROOT / 'artifacts/optimise_final_v1/released_snapshots.json').read_text()),
        TRAIN_START, pd.Timestamp('2022-07-01', tz='UTC'), tuple(str(i) for i in range(1, 12)), allow_final=True)
    july = read(ROOT / 'artifacts/optimise_july_v1/windows.npz')
    logs = json.loads((OUT / 'tree_fit_log.json').read_text())
    result = []
    for depot in IDS:
        h = np.flatnonzero(w.depot_ids == depot)
        q = np.flatnonzero(july['ids'] == depot)
        x = np.concatenate([w.X[h], july['histories'][q]])
        y = np.concatenate([w.Y[h], july['truth'][q]])
        clock = np.r_[w.origins[h].as_unit('ns').asi8, july['origin_ns'][q]]
        ids = np.full(len(x), depot)
        supported = np.flatnonzero(np.isfinite(x[len(h):]).sum(axis=1) >= 24)
        supported = supported[np.isfinite(x[len(h)+supported, -24:]).any(axis=1)]
        if not len(supported):
            continue
        positions = np.unique(supported[[0, len(supported)//2, -1]])+len(h)
        models = {a: lgb.Booster(model_file=next(r['path'] for r in logs if r['month'] == '2022-07' and
                  r['depot_id'] == depot and r['arm'] == a)) for a in ('C', 'S')}
        for arm in ('C', 'S'):
            times = []
            for repeat in range(4):
                for index in positions:
                    start = time.perf_counter()
                    base, _ = aligned_features(x[index:index+1])
                    base = add_weekday(base, pd.to_datetime(np.repeat(clock[index], 24), utc=True)).to_numpy()
                    if arm == 'S':
                        extra, _ = uniform_memory(index, x, y, ids, clock)
                        base = np.column_stack([base, extra])
                    p = models[arm].predict(base, num_threads=2)
                    assert np.isfinite(p).all()
                    if repeat:
                        times.append(time.perf_counter()-start)
            result.append(dict(depot_id=depot, arm=arm, probes=len(times),
                median_seconds=float(np.median(times)), p90_seconds=float(np.quantile(times, .9))))
    dump('online_cost.json', dict(probes=result, excludes='I/O, model load, telemetry delivery, concurrent service',
                                  process='CPU, two LightGBM threads; one warmup, three measured repeats'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'prepare', 'trees', 'neural', 'evaluate', 'benchmark'))
    globals()[parser.parse_args().stage]()
