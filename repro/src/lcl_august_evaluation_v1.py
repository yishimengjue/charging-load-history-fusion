"""First monthly evaluation of fixed LCL models, with August-only numeric access."""
import argparse
import csv
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from safetensors.torch import load_file

from src import lcl_forward_inputs_v1 as forward
from src import lcl_neural_training_v1 as neural
from src import lcl_training_projection_v1 as projection
from diagnostics.intake_lcl_ev_metadata_v1 import POWER, parse_clock
from diagnostics.intake_lcl_ev_train_v1 import EXPECTED_HASH

ROOT = neural.ROOT
OUT = ROOT / 'artifacts/lcl_august_evaluation_v1'
START, END = datetime(2013,8,1), datetime(2013,9,1)
A, B = pd.Timestamp(START).value, pd.Timestamp(END).value
ARCHIVE = ROOT / 'data/independent_source_intake_2026-11-09/lcl_ev_load_profiles.zip'
FAMILIES = ('old_residual', 'new_residual', 'old_direct', 'new_direct', 'full_direct')
PAIRS = (('old_residual','new_residual'), ('old_direct','new_direct'),
         ('S','new_residual'), ('S','new_direct'), ('half_S','new_residual'),
         ('half_S','new_direct'), ('full_direct','new_residual'),
         ('full_direct','new_direct'), ('new_direct','new_residual'))


def bounded_value(stamp, value):
    if not START <= stamp < END:
        return np.nan
    try:
        number = float(value)
    except (ValueError, TypeError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def read_august(stream):
    reader = csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig'))
    assert POWER in reader.fieldnames
    values = np.full(31*24, np.nan)
    seen = set()
    for row in reader:
        try:
            stamp = parse_clock(row['ReadingDate'])
        except ValueError:
            continue
        if not START <= stamp < END or stamp.minute or stamp.second:
            continue
        index = int((stamp-START).total_seconds()//3600)
        if index in seen:
            raise ValueError('Duplicate exact-hour source record')
        seen.add(index)
        values[index] = bounded_value(stamp, row.get(POWER))
    return values


def queries(values, times, identifier, start=A, stop=B):
    slots = np.arange(168, len(times)-24, 24)
    history = values[slots[:,None]+np.arange(-167,1)]
    target = values[slots[:,None]+np.arange(1,25)]
    origins = times[slots].astype('datetime64[ns]').astype(np.int64)
    requested = (origins >= start) & (origins < stop)
    eligible = forward.history_eligible(history)
    negative = (np.isfinite(history) & (history < 0)).any(1)
    selected = np.flatnonzero(requested & eligible & ~negative)
    ids = np.full(len(slots), identifier)
    memories, ages = [], []
    for q in selected:
        memory, records = forward.uniform_memory(q, history, target, ids, origins)
        latest = max((r['candidate_ns']+forward.DAY_NS for r in records), default=-1)
        assert latest <= origins[q]
        memories.append(memory)
        ages.append(latest)
    h = history[selected]
    if len(h):
        base, _ = forward.aligned_features(h)
        days = pd.DatetimeIndex(origins[selected]).dayofweek.to_numpy()
        c = np.column_stack([base.to_numpy(), np.repeat(np.eye(7)[days],24,axis=0)]).reshape(-1,24,180)
    else:
        c = np.empty((0,24,180))
    memory = np.asarray(memories).reshape(-1,24,27)
    data = dict(history=h, y=target[selected], origins=origins[selected],
        ids=np.full(len(selected),identifier), memory=memory, C=c,
        S=np.concatenate([c,memory],-1), memory_max_label_time=np.asarray(ages,np.int64))
    coverage = dict(id=identifier, calendar_origins=int(requested.sum()), issued_origins=len(selected),
        insufficient_history=int((requested & ~eligible).sum()),
        negative_history=int((requested & eligible & negative).sum()))
    return data, coverage


def protect():
    projection.protect()
    for name, digest in json.loads((OUT/'freeze.json').read_text())['files'].items():
        assert neural.sha(ROOT/name) == digest, name


def freeze():
    projection.protect()
    acc = json.loads((projection.OUT/'acceptance.json').read_text())
    assert acc['passed']
    for name,digest in acc['files'].items():
        assert neural.sha(ROOT/name) == digest, name
    paths = [Path(__file__), ROOT/'tests/test_lcl_august_evaluation_v1.py',
             ROOT/'diagnostics/verify_lcl_august_evaluation_v1.py',
             ROOT/'experiments/lcl_august_evaluation_v1.md', projection.OUT/'acceptance.json']
    files = dict(acc['files'])
    files.update({str(p.relative_to(ROOT)):neural.sha(p) for p in paths})
    OUT.mkdir(exist_ok=False)
    (OUT/'trees').mkdir()
    neural.dump(OUT/'freeze.json',dict(files=files, families=FAMILIES, pairs=PAIRS,
        evaluation_start=START.isoformat(), evaluation_end_exclusive=END.isoformat(),
        bootstrap_seed=20261109, bootstrap_repeats=2000, blocks=[7,14],
        scoring='point-weighted MAE, seed-mean losses, shared observed support',
        unit='source_raw', last_target_outside_window='unscored NaN',
        monthly_phase=1, september_october_values_read=False))


def preflight():
    protect()
    grid = neural.read(forward.UP/'training_grid.npz')
    mapping = json.loads((neural.UP/'entity_mapping.json').read_text())
    rows, checks = [], 0
    for entity in mapping['selected']:
        identifier = entity['id']
        train = neural.read(neural.UP/'prepared'/f'entity_{identifier:02d}.npz')
        source_row = list(grid['members']).index(entity['member'])
        for stamp in (int(train['origins'][0]), int(train['origins'][-1])):
            q,_ = queries(grid['values'][source_row],grid['times'],identifier,stamp,stamp+1)
            ix = np.flatnonzero(train['origins']==stamp)
            for key in ('history','y','origins','ids','memory','C','S'):
                np.testing.assert_array_equal(q[key],train[key][ix])
            checks += 1
        assert int((train['origins']+forward.DAY_NS).max()) < A
        valid = np.isfinite(train['y'])
        complete = int(valid.all(1).sum())
        assert complete >= 30
        for name,width in (('C',180),('S',207)):
            path = OUT/'trees'/f'{identifier:02d}_{name}.txt'
            if not path.exists():
                model = lgb.LGBMRegressor(**forward.CONFIG,n_estimators=300).fit(
                    train[name][valid],train['y'][valid]).booster_
                model.save_model(str(path))
        labels = train['y'][valid]
        positive = labels[labels>0]
        rows.append(dict(id=identifier, complete_days=complete, labels=int(valid.sum()),
                         positive_p90=float(np.quantile(positive,.9)) if len(positive) else None))
    neural.dump(OUT/'preflight.json',dict(passed=True,feature_checks=checks,trees=96,
        thresholds=rows, test_values_read=False, files={str(p.relative_to(ROOT)):neural.sha(p)
        for p in (OUT/'trees').glob('*.txt')}))
    print('PASS: 96 feature reconstructions, 96 train-only trees; no August values read',flush=True)


def release():
    protect()
    pre = json.loads((OUT/'preflight.json').read_text()); assert pre['passed']
    for name,digest in pre['files'].items():
        assert neural.sha(ROOT/name)==digest
    if (OUT/'queries.npz').exists():
        raise ValueError('August already prepared; use predict and score stages')
    assert neural.sha(ARCHIVE)==EXPECTED_HASH
    with (OUT/'release.json').open('x') as stream:
        json.dump(dict(start=START.isoformat(),end_exclusive=END.isoformat(),
            frozen_protocol_sha256=neural.sha(OUT/'freeze.json')),stream,indent=2)
    grid = neural.read(forward.UP/'training_grid.npz')
    mapping = json.loads((neural.UP/'entity_mapping.json').read_text())
    aug = []
    with zipfile.ZipFile(ARCHIVE) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read('EV Residential trials.zip'))) as inner:
            for member in grid['members']:
                with inner.open(str(member)) as stream:
                    aug.append(read_august(stream))
    aug = np.stack(aug)
    np.savez_compressed(OUT/'august_grid.npz',values=aug,members=grid['members'])
    # The September 1 endpoint exists only as NaN; its source value is not parsed.
    values = np.concatenate([grid['values'],aug,np.full((len(aug),1),np.nan)],axis=1)
    times = np.arange(grid['times'][0],np.datetime64(END)+np.timedelta64(1,'h'),np.timedelta64(1,'h'))
    parts, coverage = [], []
    for entity in mapping['selected']:
        i = list(grid['members']).index(entity['member'])
        data, row = queries(values[i],times,entity['id'])
        row.update(observed_targets=int(np.isfinite(data['y']).sum()),
                   negative_targets=int((data['y']<0).sum()))
        parts.append(data); coverage.append(row)
    merged = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(OUT/'queries.npz',**merged)
    neural.dump(OUT/'coverage.json',dict(selected=coverage,unsupported=mapping['unsupported'],
        august_finite_by_source=[dict(member=str(m),finite=int(np.isfinite(v).sum()),
            negative=int((v<0).sum())) for m,v in zip(grid['members'],aug)],
        september_october_values_read=False,queries_sha256=neural.sha(OUT/'queries.npz')))
    print(f'Prepared August: {len(merged["origins"])} origins; {np.isfinite(merged["y"]).sum()} labels',flush=True)


def predict():
    protect()
    pre = json.loads((OUT/'preflight.json').read_text())
    coverage = json.loads((OUT/'coverage.json').read_text())
    assert neural.sha(OUT/'queries.npz')==coverage['queries_sha256']
    for name,digest in pre['files'].items(): assert neural.sha(ROOT/name)==digest
    data = neural.read(OUT/'queries.npz')
    result = {k:data[k] for k in ('origins','ids','y')}
    for name,width in (('C',180),('S',207)):
        p = np.full_like(data['y'],np.nan)
        for identifier in range(48):
            ix = data['ids']==identifier
            if ix.any():
                model = lgb.Booster(model_file=str(OUT/'trees'/f'{identifier:02d}_{name}.txt'))
                p[ix] = np.maximum(model.predict(data[name][ix].reshape(-1,width),num_threads=2),0).reshape(-1,24)
        result[name]=p
    result['zero']=np.zeros_like(result['S']); result['half_S']=.5*result['S']
    q = dict(data,anchor=result['S'],y=np.full_like(data['y'],np.nan))
    for family in FAMILIES:
        root = projection.OUT if family.startswith('new_') else neural.OUT
        arm = 'direct_full' if family=='full_direct' else ('residual_matched' if family.endswith('residual') else 'direct_matched')
        for seed in neural.SEEDS:
            folder=root/f'{arm}_{seed}'; info=json.loads((folder/'fit.json').read_text())
            model=neural.make_model(arm=='residual_matched')
            model.load_state_dict(load_file(str(folder/'encoder.safetensors')))
            heads=neural.make_heads(model.decoder)
            heads.load_state_dict(load_file(str(folder/'heads.safetensors')))
            pred=neural.predict(model,heads,q,{int(k):v for k,v in info['scales'].items()})
            result[f'{family}_{seed}']=pred['final']
            result[f'branch_{family}_{seed}']=pred['branch']
        print('Predicted '+family,flush=True)
    for key,value in result.items():
        if key not in ('origins','ids','y'): assert np.isfinite(value).all() and (value>=0).all(),key
    np.savez_compressed(OUT/'predictions.npz',**result)
    neural.dump(OUT/'prediction_manifest.json',dict(predictions_sha256=neural.sha(OUT/'predictions.npz'),
        queries_sha256=neural.sha(OUT/'queries.npz'),origin_count=len(data['origins']),
        september_october_values_read=False))


def loss_arrays(data):
    y=data['y']; errors={}; signed={}; squared={}
    for name in ('C','S','zero','half_S')+FAMILIES:
        keys=[f'{name}_{s}' for s in neural.SEEDS] if name in FAMILIES else [name]
        diff=np.stack([data[k]-y for k in keys])
        errors[name]=np.abs(diff).mean(0); signed[name]=diff.mean(0); squared[name]=(diff**2).mean(0)
    return errors,signed,squared


def interval(daily_gain,daily_count,block,repeats=2000):
    if daily_count.sum()==0:return None
    rng=np.random.default_rng(20261109); n=len(daily_count); estimates=[]
    for _ in range(repeats):
        starts=rng.integers(0,n-block+1,size=(n+block-1)//block)
        indices=np.concatenate([np.arange(s,s+block) for s in starts])[:n]
        count=daily_count[indices].sum()
        if count:estimates.append(daily_gain[indices].sum()/count)
    return dict(block=block,low=float(np.quantile(estimates,.025)),high=float(np.quantile(estimates,.975)),
                replicates=len(estimates)) if estimates else None


def score():
    protect()
    manifest=json.loads((OUT/'prediction_manifest.json').read_text())
    assert neural.sha(OUT/'predictions.npz')==manifest['predictions_sha256']
    data=neural.read(OUT/'predictions.npz');y=data['y'];valid=np.isfinite(y)
    thresholds=json.loads((OUT/'preflight.json').read_text())['thresholds']
    threshold={r['id']:r['positive_p90'] for r in thresholds}
    high=np.zeros_like(valid)
    for identifier,value in threshold.items():
        if value is not None: high |= (data['ids'][:,None]==identifier)&(y>=value)&(y>0)
    groups=dict(all=valid,positive=valid&(y>0),ordinary_positive=valid&(y>0)&~high,
                high=valid&high,zero=valid&(y==0),negative=valid&(y<0))
    errors,signed,squared=loss_arrays(data); rows=[];pairs=[]
    day=((data['origins']-A)//forward.DAY_NS).astype(int)
    for group,mask in groups.items():
        count=int(mask.sum())
        for name in errors:
            rows.append(dict(group=group,method=name,n=count,
                mae=float(errors[name][mask].mean()) if count else None,
                rmse=float(np.sqrt(squared[name][mask].mean())) if count else None,
                bias=float(signed[name][mask].mean()) if count else None))
        if not count:continue
        for before,after in PAIRS:
            delta=errors[before]-errors[after]
            gain=float(delta[mask].mean()); base=float(errors[before][mask].mean())
            daily_gain=np.array([np.where(mask&(day[:,None]==d),delta,0).sum() for d in range(31)])
            daily_count=np.array([(mask&(day[:,None]==d)).sum() for d in range(31)])
            pairs.append(dict(group=group,before=before,after=after,n=count,gain=gain,
                gain_percent=100*gain/base if base else None,
                intervals=[interval(daily_gain,daily_count,k) for k in (7,14)]))
    entity_rows=[];daily_rows=[];seed_rows=[]
    for identifier in range(48):
        mask=valid&(data['ids'][:,None]==identifier)
        for name in errors:
            entity_rows.append(dict(id=identifier,method=name,n=int(mask.sum()),
                mae=float(errors[name][mask].mean()) if mask.any() else None))
    for d in range(31):
        mask=valid&(day[:,None]==d)
        for name in errors:
            daily_rows.append(dict(day=d+1,method=name,n=int(mask.sum()),
                absolute_error_sum=float(errors[name][mask].sum())))
    for family in FAMILIES:
        for seed in neural.SEEDS:
            for group,mask in groups.items():
                seed_rows.append(dict(method=family,seed=seed,group=group,n=int(mask.sum()),
                    mae=float(np.abs(data[f'{family}_{seed}']-y)[mask].mean()) if mask.any() else None))
    neural.dump(OUT/'scores.json',dict(metrics=rows,pairs=pairs,entities=entity_rows,days=daily_rows,
        seeds=seed_rows,seed_aggregation='mean loss, not ensemble forecast',unit='source_raw',
        september_october_values_read=False))
    print(json.dumps([r for r in rows if r['group'] in ('all','positive','high')],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','preflight','release','predict','score'])
    args=parser.parse_args();globals()[args.stage]()
