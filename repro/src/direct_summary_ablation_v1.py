"""Local encoded-summary ablation of frozen, source-specific direct trainers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
from safetensors.torch import load_file

from src import journal_evidence_v1 as op
from src import ams_fixed_comparison_v1 as ams
from src import lcl_neural_training_v1 as lcl
from src import lcl_monthly_followup_v1 as lcl_old

ROOT=op.ROOT
OUT=ROOT/'artifacts/direct_summary_ablation_v1'
PERIODS={'op':('2022-01','2022-02','2022-03','2022-07'),
         'ams':('2019-11','2019-12','2020-01'),
         'lcl':('2013-11','2013-12','2014-01')}
SEEDS=(42,43,44)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def read(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}


def no_summary_encoder(encoder):
    def encode(*args,**kwargs):
        result=encoder(*args,**kwargs)
        memory=result['memory']
        if memory.ndim!=3 or memory.shape[1:]!=(24,54):
            raise ValueError('Expected54 encoded summary value/mask channels')
        return dict(result,memory=np.zeros_like(memory))
    return encode


def clone_function(function,**bindings):
    result=FunctionType(function.__code__,dict(function.__globals__,**bindings),
                        function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


def proxy(module,**bindings):return SimpleNamespace(**dict(vars(module),**bindings))


def seed_mae(y,predictions):
    p=np.asarray(predictions);mask=np.isfinite(y)
    if p.ndim!=y.ndim+1 or p.shape[1:]!=y.shape or not mask.any():
        raise ValueError('Mismatched prediction support')
    if not np.isfinite(p[:,mask]).all():raise ValueError('Missing prediction')
    return float(np.abs(p-y)[...,mask].mean())


def adapter(source,removed):
    if source=='op':
        enc=no_summary_encoder(op.h.encode) if removed else op.h.encode
        h=proxy(op.h,encode=enc)
        cache=clone_function(op.cache,h=h)
        return SimpleNamespace(encode=enc,fit=clone_function(op.fit_encoder,h=h),cache=cache,
                               predict=clone_function(op.predict,cache=cache))
    if source=='ams':
        enc=no_summary_encoder(ams.pilot.encoded) if removed else ams.pilot.encoded
        pilot=proxy(ams.pilot,encoded=enc)
        cache=clone_function(ams.cache,pilot=pilot)
        return SimpleNamespace(encode=enc,fit=clone_function(ams.train_model,pilot=pilot,cache=cache),
                               cache=cache,predict=clone_function(ams.predict,cache=cache))
    if source=='lcl':
        enc=no_summary_encoder(lcl.encode) if removed else lcl.encode
        predict=clone_function(lcl.predict,encode=enc)
        return SimpleNamespace(encode=enc,predict=predict)
    raise ValueError(source)


def locations(source,month):
    if source not in PERIODS or month not in PERIODS[source]:raise ValueError('Unplanned case')
    if source=='op':
        return op.h.OUT/'prepared',op.OUT/'models',None
    if source=='ams':
        return ams.OUT/'prepared',ams.OUT/'models'/month,None
    base=ROOT/'artifacts/lcl_next_period_v1'/month
    return base/'inputs',base/'neural',base


def case_inputs(source,month):
    prepared,models,base=locations(source,month)
    if source=='lcl':
        train=read(prepared/'full_training.npz');q=read(base/'queries.npz')
        p=read(base/'predictions.npz');truth=p['y'];s=p['S']
        np.testing.assert_array_equal(q['origins'],p['origins'])
        np.testing.assert_array_equal(q['ids'],p['ids'])
        high=read(base/'high.npz')['high']
        scales={x['id']:x['scale'] for x in json.loads((prepared/'entity_mapping.json').read_text())['selected']}
        q=dict(q,anchor=s,y=np.full_like(truth,np.nan))
        expected={seed:p[f'branch_{seed}'] for seed in SEEDS}
    else:
        train=read(prepared/f'{month}_train.npz');q=read(prepared/f'{month}_query.npz')
        if source=='op':
            frame=pd.read_parquet(op.OUT/'predictions.parquet').loc[q['rows'].ravel()]
            truth=frame.actual_kw.to_numpy().reshape(-1,24);s=q['baseline']
            np.testing.assert_array_equal(frame.origin_utc.astype('int64').to_numpy().reshape(-1,24)[:,0],q['origins'])
            np.testing.assert_array_equal(frame.S.to_numpy().reshape(-1,24),s)
            high=truth>=frame.high_threshold.to_numpy().reshape(-1,24)
            expected={seed:read(models/f'{month}_{seed}'/'prediction.npz')['neural'] for seed in SEEDS}
            scales=json.loads((models/f'{month}_42'/'encoder/fit.json').read_text())['scales']
        else:
            truth_data=read(prepared/f'{month}_truth.npz');truth=truth_data['y']
            np.testing.assert_array_equal(truth_data['origins'],q['origins'])
            s=np.load(models/'statistics.npy')
            scales=json.loads((models/'concat_42/fit.json').read_text())['scale']
            high=truth>=scales
            expected={seed:np.load(models/f'concat_{seed}'/'prediction.npy') for seed in SEEDS}
    if np.isfinite(q['y']).any():raise ValueError('Query labels visible')
    cutoff=pd.Timestamp(month+'-01').value
    if not (train['origins']+op.tx.DAY<=cutoff).all():raise ValueError('Training horizon crosses cutoff')
    return train,q,truth,s,high,scales,expected


def encode_case(ad,source,data,scales):
    if source=='op':return ad.encode(data,scales,'F1')
    if source=='ams':return ad.encode(data,scales)
    return ad.encode(data,scales,False)


def load_models(source,folder):
    if source=='op':
        info=json.loads((folder/'encoder/fit.json').read_text());model=op.make_model()
        model.load_state_dict(load_file(str(folder/'encoder/model.safetensors')))
        head=op.local.head_bank(model.decoder,True)
        head.load_state_dict(load_file(str(folder/'head/head.safetensors')))
        return model,head,info['scales']
    if source=='ams':
        model,head,info=ams.load_model(folder,'concat');return model,head,info['scale']
    model,head=lcl_old.load_model(folder,'direct_full')
    info=json.loads((folder/'fit.json').read_text())
    return model,head,{int(k):v for k,v in info['scales'].items()}


def neural_prediction(ad,source,model,head,q,scales):
    out=ad.predict(model,head,q,scales)
    return out['neural'] if source=='op' else out['branch'] if source=='lcl' else out


def reference_folder(source,month,seed):
    _,models,_=locations(source,month)
    return models/(f'{month}_{seed}' if source=='op' else f'concat_{seed}' if source=='ams' else f'direct_full_{seed}')


def freeze():
    if OUT.exists():raise FileExistsError(OUT)
    paths=list((ROOT/'src').glob('*.py'))
    paths += [ROOT/'experiments/direct_summary_ablation_v1.md',ROOT/'tests/test_direct_summary_ablation_v1.py',
              ROOT/'run_chronos_python.sh']
    for source,months in PERIODS.items():
        for month in months:
            prepared,models,base=locations(source,month)
            paths += list(prepared.glob('*'))
            for seed in SEEDS:paths+=list(reference_folder(source,month,seed).rglob('*'))
            if source=='ams':paths += [models/'statistics.npy']
            if source=='lcl':paths += [base/'queries.npz',base/'predictions.npz',base/'high.npz',base/'acceptance.json']
    paths += list((op.m.OUT/'prepared').glob('*.npz'))
    paths += [op.OUT/'predictions.parquet',op.OUT/'acceptance.json',ams.OUT/'acceptance.json']
    paths += [op.tx.VENDOR/x for x in op.tx.TX_FILES]
    paths=sorted({p for p in paths if p.is_file()})
    # Existing acceptance records bind the saved training/query/model inputs.
    for path in (op.OUT/'acceptance.json',ams.OUT/'acceptance.json'):
        record=json.loads(path.read_text());assert record['passed']
        for name,digest in record['files'].items():assert sha(name)==digest,name
    OUT.mkdir()
    dump(OUT/'freeze.json',dict(periods=PERIODS,seeds=SEEDS,planned_fits=30,workers=2,
         no_new_data=True,retrospective=True,files={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    print(f'Frozen {len(paths)} inputs and30 planned fits',flush=True)


def protect():
    record=json.loads((OUT/'freeze.json').read_text())
    for path,digest in record['files'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Frozen input changed: '+path)


def run_case(source,month):
    protect();dest=OUT/source/month
    if (dest/'acceptance.json').exists():
        for name,digest in json.loads((dest/'acceptance.json').read_text())['files'].items():
            assert sha(ROOT/name)==digest
        print(f'Reused {source}/{month}',flush=True);return
    dest.mkdir(parents=True,exist_ok=True)
    train,q,y,s,high,scales,expected=case_inputs(source,month)
    full,empty=adapter(source,False),adapter(source,True)
    for data in (train,q):
        actual=encode_case(full,source,data,scales);zero=encode_case(empty,source,data,scales)
        for key in actual:
            if key!='memory':np.testing.assert_array_equal(actual[key],zero[key])
        assert not zero['memory'].any()
    dump(dest/'preflight.json',dict(source=source,month=month,training_origins=len(train['origins']),
         training_labels=int(np.isfinite(train['y']).sum()),issued=int(y.size),observed=int(np.isfinite(y).sum()),
         summary_only_intervention=True,encoder_bytecode_reused=True,scales=scales))
    result=dict(y=y,S=s,origins=q['origins'],ids=q['ids'],high=high)
    for seed in SEEDS:
        old=reference_folder(source,month,seed)
        model,head,scale=load_models(source,old)
        original=neural_prediction(full,source,model,head,q,scale)
        np.testing.assert_array_equal(original,expected[seed])
        result[f'N_with_{seed}']=original;result[f'F_with_{seed}']=.5*(s+original)
        folder=dest/'models'/(f'direct_full_{seed}' if source=='lcl' else str(seed))
        if source=='op':
            if not (folder/'encoder/fit.json').exists():
                model,info=empty.fit(train,seed,folder/'encoder')
            else:
                info=json.loads((folder/'encoder/fit.json').read_text());model=op.make_model()
                model.load_state_dict(load_file(str(folder/'encoder/model.safetensors')))
            if not (folder/'head/fit.json').exists():
                real=empty.cache(model,train,info['scales']);zero=empty.cache(model,train,info['scales'],zero=True)
                np.testing.assert_array_equal(real['hidden'],zero['hidden'])
                op.a.fit_stage(real,seed,.05,folder/'head',model.decoder,zero)
        elif source=='ams':
            if not (folder/'fit.json').exists():empty.fit(train,seed,'concat',folder)
        else:
            folder.parent.mkdir(exist_ok=True)
            fn=clone_function(lcl.fit,UP=locations(source,month)[0],OUT=folder.parent,
                              protect=protect,encode=empty.encode,predict=empty.predict)
            fn('direct_full',seed)
        model,head,scale=load_models(source,folder)
        prediction=neural_prediction(empty,source,model,head,q,scale)
        # Removing the encoded summary must also remove any effect of its raw values at inference.
        if 'memory' in q:
            changed=dict(q,memory=np.full_like(q['memory'],9.))
            np.testing.assert_array_equal(prediction,neural_prediction(empty,source,model,head,changed,scale))
        model2,head2,scale2=load_models(source,folder)
        np.testing.assert_array_equal(prediction,neural_prediction(empty,source,model2,head2,q,scale2))
        if not np.isfinite(prediction).all() or (prediction<0).any():raise ValueError('Invalid prediction')
        result[f'N_without_{seed}']=prediction;result[f'F_without_{seed}']=.5*(s+prediction)
        if source=='op':
            for stage in ('encoder','head'):
                a=json.loads((folder/stage/'fit.json').read_text());b=json.loads((old/stage/'fit.json').read_text())
                for key in ('origins','labels','optimizer_steps','parameters'):assert a[key]==b[key],key
        else:
            a=json.loads((folder/'fit.json').read_text());b=json.loads((old/'fit.json').read_text())
            keys=('origins','labels','encoder_parameters','head_parameters')
            if source=='ams':keys+=('encoder_steps','head_steps')
            for key in keys:assert a[key]==b[key],key
        print(f'Completed {source}/{month}/{seed}: models replayed',flush=True)
    np.savez_compressed(dest/'predictions.npz',**result)
    metrics={key:seed_mae(y,np.array([result[f'{key}_{seed}'] for seed in SEEDS]))
             for key in ('N_with','N_without','F_with','F_without')}
    dump(dest/'scores.json',dict(mae=metrics,observed=int(np.isfinite(y).sum()),
         unit={'op':'kW','ams':'kWh','lcl':'source_raw'}[source]))
    protect()
    dump(dest/'acceptance.json',dict(passed=True,old_models_replayed=3,new_models_replayed=3,
         files={str(p.relative_to(ROOT)):sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='run.log'}))
    print(f'SCORED {source}/{month}: '+json.dumps(metrics),flush=True)


def run_all():
    protect()
    cases=[(s,m) for s,months in PERIODS.items() for m in months]
    # LCL is longest; interleave it with short OP/AMS cases to bound resource use.
    cases=[cases[7],*cases[:4],cases[8],*cases[4:7],cases[9]]
    def work(case):
        source,month=case;folder=OUT/source/month;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'run.log').open('a') as stream:
            p=subprocess.run([sys.executable,'-B','-m','src.direct_summary_ablation_v1','case',source,month],
                             cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(f'{source}/{month} failed; inspect run.log')
        print(f'Case finished: {source}/{month}',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in pool.map(work,cases):pass
    protect();print('All30 no-summary fits and replays completed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run','case'))
    parser.add_argument('source',nargs='?');parser.add_argument('month',nargs='?');args=parser.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='run':run_all()
    else:run_case(args.source,args.month)
