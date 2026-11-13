"""Frozen September/October expanding-window continuation of the LCL comparison."""
import argparse
import csv
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from types import FunctionType

import lightgbm as lgb
import numpy as np
import pandas as pd
from safetensors.torch import load_file

from src import lcl_august_evaluation_v1 as august
from src import lcl_neural_training_v1 as neural
from src import lcl_training_projection_v1 as projection

ROOT=neural.ROOT
OUT=ROOT/'artifacts/lcl_monthly_followup_v1'
MONTHS=('2013-09','2013-10')
FAMILIES=august.FAMILIES


def bounds(month):
    start=pd.Timestamp(month+'-01')
    end=start+pd.offsets.MonthBegin(1)
    return start,end


def folder(month):
    assert month in MONTHS
    return OUT/month


def check_files(files):
    for name,digest in files.items():assert neural.sha(ROOT/name)==digest,name


def protect(month=None):
    august.protect()
    check_files(json.loads((OUT/'freeze.json').read_text())['files'])
    if month:
        path=folder(month)/'inputs/acceptance.json'
        if path.exists():check_files(json.loads(path.read_text())['files'])


def freeze():
    august.protect()
    acc=json.loads((august.OUT/'acceptance.json').read_text());assert acc['passed']
    check_files(acc['files'])
    paths=[Path(__file__),ROOT/'src/lcl_monthly_scoring_v1.py',
           ROOT/'tests/test_lcl_monthly_followup_v1.py',
           ROOT/'diagnostics/verify_lcl_monthly_followup_v1.py',
           ROOT/'experiments/lcl_monthly_followup_v1.md',august.OUT/'acceptance.json']
    files=dict(acc['files']);files.update({str(p.relative_to(ROOT)):neural.sha(p) for p in paths})
    OUT.mkdir(exist_ok=False)
    neural.dump(OUT/'freeze.json',dict(months=MONTHS,config=neural.CONFIG,
        families=FAMILIES,files=files,forward_block_anchor=august.A,
        stages='prepare,train,verify_fit,release,predict,score,verify',
        no_monthly_hyperparameter_selection=True))


def known_grid(month):
    grid=neural.read(august.forward.UP/'training_grid.npz')
    aug=neural.read(august.OUT/'august_grid.npz')
    np.testing.assert_array_equal(grid['members'],aug['members'])
    parts=[grid['values'],aug['values']]
    if month=='2013-10':
        previous=folder('2013-09')
        acc=json.loads((previous/'acceptance.json').read_text());assert acc['passed']
        check_files(acc['files'])
        sep=neural.read(previous/'evaluation_grid.npz')
        np.testing.assert_array_equal(grid['members'],sep['members'])
        parts.append(sep['values'])
    start,_=bounds(month)
    times=np.arange(grid['times'][0],np.datetime64(start),np.timedelta64(1,'h'))
    values=np.concatenate(parts,axis=1);assert values.shape==(55,len(times))
    return dict(values=values,times=times,members=grid['members'])


def block_starts(origins,cutoff):
    step=august.forward.STEP
    first=august.A-((august.A-int(origins.min())+step-1)//step)*step
    return range(first,cutoff,step)


def prepare(month):
    protect(); dest=folder(month); up=dest/'inputs'
    if (up/'acceptance.json').exists():protect(month);return
    up.mkdir(parents=True,exist_ok=True)
    for name in ('inputs/prepared','inputs/forward_models','trees','models/old','models/new'):
        (dest/name).mkdir(parents=True,exist_ok=True)
    grid=known_grid(month);start,_=bounds(month);cutoff=start.value
    base_mapping=json.loads((neural.UP/'entity_mapping.json').read_text())
    parts=[];mapping=[];logs=[];thresholds=[];prefix_checks=0
    for entity in base_mapping['selected']:
        identifier=entity['id'];row=list(grid['members']).index(entity['member'])
        q,_=august.queries(grid['values'][row],grid['times'],identifier,
                           pd.Timestamp('2013-02-01').value,cutoff)
        keep=np.isfinite(q['y']).any(1)
        data={k:v[keep] for k,v in q.items()}
        assert (data['origins']+august.forward.DAY_NS<cutoff).all()
        assert not (data['y'][np.isfinite(data['y'])]<0).any(), 'Negative training targets require review'
        old=neural.read(neural.UP/'prepared'/f'entity_{identifier:02d}.npz')
        prefix=data['origins']<=old['origins'].max()
        for key in ('history','y','origins','ids','memory','C','S'):
            np.testing.assert_array_equal(data[key][prefix],old[key])
        prefix_checks+=1
        anchor=np.full_like(data['y'],np.nan);cutoffs=np.full(len(anchor),-1,np.int64)
        for block in block_starts(data['origins'],cutoff):
            tr,qe=august.forward.block_masks(data['origins'],block)
            observed=np.isfinite(data['y'][tr]);complete=int(observed.all(1).sum())
            if not qe.any() or complete<30:continue
            path=up/'forward_models'/f'{identifier:02d}_{block}.txt'
            if not path.exists():
                model=lgb.LGBMRegressor(**august.forward.CONFIG,n_estimators=300).fit(
                    data['S'][tr][observed],data['y'][tr][observed]).booster_
                model.save_model(str(path))
            model=lgb.Booster(model_file=str(path))
            anchor[qe]=np.maximum(model.predict(data['S'][qe].reshape(-1,207),num_threads=2),0).reshape(-1,24)
            cutoffs[qe]=block
            latest=int((data['origins'][tr]+august.forward.DAY_NS).max());assert latest<=block
            logs.append(dict(id=identifier,cutoff_ns=block,latest_training_target=latest,
                train_indices=np.flatnonzero(tr).tolist(),query_indices=np.flatnonzero(qe).tolist(),
                labels=int(observed.sum()),model_path=str(path.relative_to(ROOT))))
        np.testing.assert_array_equal(anchor[prefix],old['anchor'])
        data.update(anchor=anchor,anchor_cutoff=cutoffs)
        np.savez_compressed(up/'prepared'/f'entity_{identifier:02d}.npz',**data)
        valid=np.isfinite(data['y']);labels=data['y'][valid];positive=labels[labels>0]
        complete=int(valid.all(1).sum());assert complete>=30
        mapping.append(dict(entity,scale=max(float(np.quantile(labels,.9)),1.),
                            full_origins=len(anchor),matched_origins=int(np.isfinite(anchor).all(1).sum())))
        thresholds.append(dict(id=identifier,complete_days=complete,labels=int(valid.sum()),
            positive_p90=float(np.quantile(positive,.9)) if len(positive) else None))
        for name in ('C','S'):
            path=dest/'trees'/f'{identifier:02d}_{name}.txt'
            if not path.exists():
                model=lgb.LGBMRegressor(**august.forward.CONFIG,n_estimators=300).fit(
                    data[name][valid],labels).booster_
                model.save_model(str(path))
        parts.append(data)
        print(f'{month} input {identifier+1}/48: full={len(anchor)}, matched={np.isfinite(anchor).all(1).sum()}',flush=True)
    merged={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    matched=np.isfinite(merged['anchor']).all(1)
    np.savez_compressed(up/'full_training.npz',**merged)
    np.savez_compressed(up/'matched_training.npz',**{k:v[matched] for k,v in merged.items()})
    neural.dump(up/'entity_mapping.json',dict(selected=mapping,unsupported=base_mapping['unsupported']))
    neural.dump(up/'forward_log.json',logs)
    neural.dump(dest/'preflight.json',dict(passed=True,thresholds=thresholds,trees=96,
        prediction_month=month,latest_training_target=int((merged['origins']+august.forward.DAY_NS).max()),
        files={str(p.relative_to(ROOT)):neural.sha(p) for p in (dest/'trees').glob('*.txt')}))
    paths=[p for p in up.rglob('*') if p.is_file() and p.name!='acceptance.json']
    paths+=[dest/'preflight.json']+list((dest/'trees').glob('*.txt'))
    neural.dump(up/'acceptance.json',dict(passed=True,prefix_checks=prefix_checks,
        full_origins=len(matched),matched_origins=int(matched.sum()),
        full_labels=int(np.isfinite(merged['y']).sum()),matched_labels=int(np.isfinite(merged['y'][matched]).sum()),
        forward_trees=len(logs),prediction_month_values_read=False,
        files={str(p.relative_to(ROOT)):neural.sha(p) for p in paths}))


def fitted_trainer(month,kind):
    assert kind in ('old','new')
    ns=dict(neural.fit.__globals__,UP=folder(month)/'inputs',OUT=folder(month)/'models'/kind,
        protect=lambda:protect(month),loss=neural.loss if kind=='old' else projection.loss)
    return FunctionType(neural.fit.__code__,ns,name='monthly_fit')


def train(month):
    protect(month)
    for kind,arms in (('old',neural.ARMS),('new',projection.ARMS)):
        for arm in arms:
            for seed in neural.SEEDS:
                print(f'{month}/{kind}/{arm}/{seed}',flush=True)
                fitted_trainer(month,kind)(arm,seed)


def verify_fit(month):
    protect(month);dest=folder(month);infos=[];replayed=0
    for kind,arms in (('old',neural.ARMS),('new',projection.ARMS)):
        for arm in arms:
            data=neural.read(dest/'inputs'/('full_training.npz' if arm=='direct_full' else 'matched_training.npz'))
            for seed in neural.SEEDS:
                path=dest/'models'/kind/f'{arm}_{seed}';info=json.loads((path/'fit.json').read_text())
                assert info['config']==neural.CONFIG and info['latest_target_ns']<bounds(month)[0].value
                assert info['labels']==int(np.isfinite(data['y']).sum()) and info['origins']==len(data['origins'])
                model,heads=load_model(path,arm)
                probe=neural.read(path/'probe.npz');q={k:v[probe['indices']] for k,v in data.items()}
                q['y']=np.full_like(q['y'],np.nan)
                output=neural.predict(model,heads,q,{int(k):v for k,v in info['scales'].items()})
                for key,value in output.items():np.testing.assert_array_equal(value,probe[key])
                replayed+=output['final'].size
                if kind=='new':
                    before=json.loads((dest/'models/old'/f'{arm}_{seed}'/'fit.json').read_text())
                    for key in ('config','origins','labels','scales','latest_target_ns','encoder_parameters','head_parameters'):
                        assert info[key]==before[key],key
                infos.append(dict(kind=kind,**info))
    neural.dump(dest/'fit_acceptance.json',dict(passed=True,model_runs=15,probe_predictions=replayed,
        records=infos,prediction_month_values_read=False,
        files={str(p.relative_to(ROOT)):neural.sha(p) for p in (dest/'models').rglob('*') if p.is_file()}))
    print(f'{month}: 15 fits and {replayed} training probes verified',flush=True)


def load_model(path,arm):
    model=neural.make_model(arm=='residual_matched')
    model.load_state_dict(load_file(str(path/'encoder.safetensors')))
    heads=neural.make_heads(model.decoder)
    heads.load_state_dict(load_file(str(path/'heads.safetensors')))
    return model,heads


def bounded_value(stamp,raw,start,end):
    if not start<=stamp<end:return np.nan
    try:value=float(raw)
    except (ValueError,TypeError):return np.nan
    return value if np.isfinite(value) else np.nan


def read_month(stream,start,end):
    n=int((end-start).total_seconds()//3600);values=np.full(n,np.nan);seen=set()
    reader=csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig'));assert august.POWER in reader.fieldnames
    for row in reader:
        try:stamp=august.parse_clock(row['ReadingDate'])
        except ValueError:continue
        if not start<=stamp<end or stamp.minute or stamp.second:continue
        ix=int((stamp-start).total_seconds()//3600)
        if ix in seen:raise ValueError('Duplicate exact-hour record')
        seen.add(ix);values[ix]=bounded_value(stamp,row.get(august.POWER),start,end)
    return values


def release(month):
    protect(month);dest=folder(month)
    fit=json.loads((dest/'fit_acceptance.json').read_text());assert fit['passed'];check_files(fit['files'])
    if (dest/'queries.npz').exists():return
    start,end=bounds(month);known=known_grid(month)
    assert neural.sha(august.ARCHIVE)==august.EXPECTED_HASH
    with (dest/'release.json').open('x') as f:
        json.dump(dict(month=month,start=start.isoformat(),end_exclusive=end.isoformat(),
            freeze_sha256=neural.sha(OUT/'freeze.json'),fit_acceptance_sha256=neural.sha(dest/'fit_acceptance.json')),f,indent=2)
    rows=[]
    with zipfile.ZipFile(august.ARCHIVE) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read('EV Residential trials.zip'))) as inner:
            for member in known['members']:
                with inner.open(str(member)) as stream:
                    rows.append(read_month(stream,start.to_pydatetime(),end.to_pydatetime()))
    future=np.stack(rows)
    np.savez_compressed(dest/'evaluation_grid.npz',values=future,members=known['members'])
    values=np.concatenate([known['values'],future,np.full((55,1),np.nan)],axis=1)
    times=np.arange(known['times'][0],np.datetime64(end)+np.timedelta64(1,'h'),np.timedelta64(1,'h'))
    mapping=json.loads((dest/'inputs/entity_mapping.json').read_text());parts=[];coverage=[]
    for entity in mapping['selected']:
        row=list(known['members']).index(entity['member'])
        q,record=august.queries(values[row],times,entity['id'],start.value,end.value)
        record.update(observed_targets=int(np.isfinite(q['y']).sum()),negative_targets=int((q['y']<0).sum()))
        parts.append(q);coverage.append(record)
    merged={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(dest/'queries.npz',**merged)
    neural.dump(dest/'coverage.json',dict(month=month,selected=coverage,unsupported=mapping['unsupported'],
        source_inventory=[dict(member=str(m),finite=int(np.isfinite(v).sum()),negative=int((v<0).sum()))
                          for m,v in zip(known['members'],future)],
        future_months_values_read=False,queries_sha256=neural.sha(dest/'queries.npz')))
    print(f'{month}: {len(merged["origins"])} origins, {np.isfinite(merged["y"]).sum()} observed labels',flush=True)


def predict(month):
    protect(month);dest=folder(month)
    check_files(json.loads((dest/'fit_acceptance.json').read_text())['files'])
    assert neural.sha(dest/'queries.npz')==json.loads((dest/'coverage.json').read_text())['queries_sha256']
    data=neural.read(dest/'queries.npz');result={k:data[k] for k in ('origins','ids','y')}
    for name,width in (('C',180),('S',207)):
        p=np.full_like(data['y'],np.nan)
        for identifier in range(48):
            mask=data['ids']==identifier
            if mask.any():
                model=lgb.Booster(model_file=str(dest/'trees'/f'{identifier:02d}_{name}.txt'))
                p[mask]=np.maximum(model.predict(data[name][mask].reshape(-1,width),num_threads=2),0).reshape(-1,24)
        result[name]=p
    result['zero']=np.zeros_like(result['S']);result['half_S']=.5*result['S']
    blind=dict(data,anchor=result['S'],y=np.full_like(data['y'],np.nan))
    for family in FAMILIES:
        kind='new' if family.startswith('new_') else 'old'
        arm='direct_full' if family=='full_direct' else ('residual_matched' if family.endswith('residual') else 'direct_matched')
        for seed in neural.SEEDS:
            path=dest/'models'/kind/f'{arm}_{seed}';info=json.loads((path/'fit.json').read_text())
            model,heads=load_model(path,arm)
            pred=neural.predict(model,heads,blind,{int(k):v for k,v in info['scales'].items()})
            result[f'{family}_{seed}']=pred['final'];result[f'branch_{family}_{seed}']=pred['branch']
    for key,value in result.items():
        if key not in ('origins','ids','y'):assert np.isfinite(value).all() and (value>=0).all(),key
    np.savez_compressed(dest/'predictions.npz',**result)
    neural.dump(dest/'prediction_manifest.json',dict(predictions_sha256=neural.sha(dest/'predictions.npz'),
        queries_sha256=neural.sha(dest/'queries.npz'),month=month,future_months_values_read=False))


def run():
    from src.lcl_monthly_scoring_v1 import score_month,aggregate
    from diagnostics.verify_lcl_monthly_followup_v1 import verify_month,verify_scores
    for month in MONTHS:
        if (folder(month)/'acceptance.json').exists():
            check_files(json.loads((folder(month)/'acceptance.json').read_text())['files']);continue
        prepare(month);train(month);verify_fit(month);release(month);predict(month)
        score_month(month);verify_month(month)
        print(f'{month}: evaluation accepted',flush=True)
    aggregate();verify_scores(OUT/'aggregate')
    print('September and October complete; three-month pooled scores verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run'])
    args=p.parse_args();freeze() if args.stage=='freeze' else run()
