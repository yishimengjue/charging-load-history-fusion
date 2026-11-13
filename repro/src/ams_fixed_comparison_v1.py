"""Fixed three-way comparison on the already released AMS period; no release API."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from safetensors.torch import load_file

from src import ams_history_fusion_transfer_v1 as pilot
from src import journal_evidence_v1 as concat
from src import ams_final_data as windows

ROOT=pilot.ROOT
OUT=ROOT/'artifacts/ams_fixed_comparison_v1'
SOURCE=ROOT/'artifacts/ams_final_test_v1/released_hourly.parquet'
SOURCE_SHA='b19d74388e093cd6945e53e2fbb6d2eb9dce3680e77f6585bade3f7c1eef53be'
MONTHS=('2019-11','2019-12','2020-01')
SEEDS=(42,43,44)
ARMS=('conditional','concat')
CONFIG=dict(pilot.CONFIG,training_min_origins=80,primary_score='observed_issued_hours',
    primary_pair=['plain','conditional_blend'],primary_block=7,independent_validation=False)
tx,a,local=pilot.tx,pilot.a,pilot.local


def model_factory(arm):
    if arm=='conditional':return pilot.h.make_model('F1')
    if arm=='concat':return concat.make_model()
    raise ValueError('Unknown fixed arm')


def features(model,x,memory,future):
    if hasattr(model,'concat_projection'):return model.features(x,memory,future)
    return local.feature_batch(model,x,memory,future)


def paired(model,x,memory,future):
    if hasattr(model,'concat_projection'):return concat.paired_forward(model,x,memory,future)
    return a.d.paired_forward(model,x,memory,future)


def with_memory(w):
    """Only mature complete candidate days enter each historical library."""
    origins=w.origins.as_unit('ns').asi8
    complete=np.isfinite(w.Y).all(1)
    summaries=[]
    for i,origin in enumerate(origins):
        refs=np.flatnonzero(complete & (origins+tx.DAY<=origin))
        x=np.concatenate([w.X[refs],w.X[i:i+1]])
        y=np.concatenate([w.Y[refs],np.full((1,24),np.nan)])
        clock=np.r_[origins[refs],origin]
        result,positions=pilot.m.pack_memory(len(refs),x,y,np.ones(len(clock),int),clock)
        assert (clock[positions]+tx.DAY<=origin).all()
        summaries.append(result['S'])
    return dict(history=w.X.copy(),y=w.Y.copy(),origins=origins,ids=np.full(len(origins),'1'),
        memory=np.asarray(summaries).reshape(-1,24,27))


def split(data,cutoff):
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is not None or cutoff!=cutoff.normalize() or cutoff.strftime('%Y-%m') not in MONTHS or cutoff.day!=1:
        raise ValueError('Only the three declared month starts may fit')
    complete=np.isfinite(data['y']).all(1)
    trainmask=complete & (data['origins']+tx.DAY<=cutoff.value)
    stop=(cutoff+pd.offsets.MonthBegin()).value
    querymask=(data['origins']>=cutoff.value)&(data['origins']<stop)
    train={k:v[trainmask].copy() for k,v in data.items()}
    query={k:v[querymask].copy() for k,v in data.items()}
    truth=query['y'].copy();query['y'][:]=np.nan
    return train,query,truth


def cache(model,data,scale,zero=False):
    t=tx.configure();model.eval();e=pilot.encoded(data,scale,zero)
    bases=[];hidden=[];original=[]
    with t.no_grad():
        for start in range(0,len(data['origins']),128):
            args=[t.from_numpy(e[k][start:start+128]) for k in ('x','memory','future')]
            base,z=features(model,*args)
            pred=base+model.decoder(z).squeeze(-1)
            np.testing.assert_array_equal(pred.numpy(),model(*args).numpy())
            bases.append(base.numpy());hidden.append(z.numpy());original.append(pred.numpy())
    return dict(e,base=np.concatenate(bases),hidden=np.concatenate(hidden),original=np.concatenate(original),
        ids=np.zeros(len(data['origins']),np.int64))


def predict(model,head,query,scale):
    if not np.isnan(query['y']).all():raise ValueError('Query truth must be hidden')
    if not len(query['origins']):return np.empty((0,24))
    value=np.maximum(local.predict(head,cache(model,query,scale)),0.)
    if not np.isfinite(value).all():raise ValueError('Nonfinite forecast; do not silently drop targets')
    return value


def train_model(data,seed,arm,folder):
    folder.mkdir(parents=True,exist_ok=False)
    t=tx.configure();t.manual_seed(seed)
    scale=max(float(np.quantile(data['y'],.9)),1.)
    model=model_factory(arm);e=pilot.encoded(data,scale)
    tensors={k:t.from_numpy(e[k]) for k in ('x','memory','future','y','mask','scale')}
    opt=t.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    rng=np.random.default_rng(seed);started=time.perf_counter();trace=[]
    for epoch in range(1,81):
        model.train();order=rng.permutation(len(data['origins']));total=0.
        for start in range(0,len(order),64):
            idx=order[start:start+64];opt.zero_grad(set_to_none=True)
            pr,pz=paired(model,*[tensors[k][idx] for k in ('x','memory','future')])
            loss,_,_=a.weighted_loss(pr,pz,tensors['y'][idx],tensors['mask'][idx],tensors['scale'][idx],
                e['valid_per_origin'],e['scale_reference'],.05)
            assert t.isfinite(loss)
            loss.backward();t.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(idx)
        if epoch%20==0:
            trace.append(dict(epoch=epoch,loss=total/len(order),seconds=time.perf_counter()-started))
            tx.dump(folder/'trace.json',trace)
            print(f'{folder.parent.name}/{folder.name} '+json.dumps(trace[-1]),flush=True)
    encoder_seconds=time.perf_counter()-started
    real,zero=cache(model,data,scale),cache(model,data,scale,True)
    t.manual_seed(seed);head=local.head_bank(model.decoder,False)
    np.testing.assert_allclose(local.predict(head,real),real['original']*real['scale'][:,None],atol=5e-5,rtol=1e-6)
    np.testing.assert_array_equal(real['base'],zero['base'])
    opt=t.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    tensors={k:t.from_numpy(real[k]) for k in ('hidden','base','ids','y','mask','scale')};hz=t.from_numpy(zero['hidden'])
    rng=np.random.default_rng(seed);started=time.perf_counter()
    for _ in range(20):
        head.train();order=rng.permutation(len(data['origins']))
        for start in range(0,len(order),64):
            idx=order[start:start+64];opt.zero_grad(set_to_none=True)
            pr=tensors['base'][idx]+head(tensors['hidden'][idx],tensors['ids'][idx])
            pz=tensors['base'][idx]+head(hz[idx],tensors['ids'][idx])
            loss,_,_=a.weighted_loss(pr,pz,tensors['y'][idx],tensors['mask'][idx],tensors['scale'][idx],
                e['valid_per_origin'],e['scale_reference'],.05)
            assert t.isfinite(loss)
            loss.backward();t.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
    a.d.save_model(model,folder/'encoder.safetensors');a.d.save_model(head,folder/'head.safetensors')
    info=dict(config=CONFIG,arm=arm,seed=seed,scale=scale,origins=len(order),labels=int(np.isfinite(data['y']).sum()),
        latest_training_end_ns=int(data['origins'].max()+tx.DAY),encoder_steps=80*int(np.ceil(len(order)/64)),
        head_steps=20*int(np.ceil(len(order)/64)),encoder_parameters=sum(p.numel() for p in model.parameters()),
        head_parameters=sum(p.numel() for p in head.parameters()),encoder_seconds=encoder_seconds,
        head_seconds=time.perf_counter()-started)
    tx.dump(folder/'fit.json',info)
    return model,head,info


def freeze():
    concat.protect();pilot.protect()
    acceptance=ROOT/'artifacts/ams_final_acceptance_v1/summary.json'
    accepted=json.loads(acceptance.read_text())
    assert accepted['passed'] and accepted['final_test_is_now_seen']
    assert accepted['immutable_inputs_sha256'][str(SOURCE)]==SOURCE_SHA
    assert tx.sha(SOURCE)==SOURCE_SHA
    paths=[Path(__file__),ROOT/'experiments/ams_fixed_comparison_v1.md',
        ROOT/'tests/test_ams_fixed_comparison_v1.py',ROOT/'diagnostics/verify_ams_fixed_comparison_v1.py',
        SOURCE,acceptance,ROOT/'artifacts/ams_final_test_release_v1.json',
        ROOT/'data/public_ams_v1/manifest.json',pilot.MASKED,ROOT/'run_chronos_python.sh']
    paths += [Path(mod.__file__).resolve() for mod in list(sys.modules.values())
        if getattr(mod,'__file__',None) and Path(mod.__file__).resolve().is_relative_to(ROOT/'src')]
    paths += [tx.VENDOR / name for name in tx.TX_FILES]
    OUT.mkdir(exist_ok=False)
    tx.dump(OUT/'freeze.json',dict(config=CONFIG,months=MONTHS,seeds=SEEDS,neural_fits=18,tree_fits=6,
        independent_validation=False,source_previously_released=True,raw_csv_read=False,
        packages={n:importlib.metadata.version(n) for n in ('numpy','pandas','torch','lightgbm','safetensors','pyarrow')},
        files={str(p):tx.sha(p) for p in sorted(set(paths))}))
    print('Frozen three-way comparison on previously seen AMS months; no new data release.',flush=True)


def protect():
    lock=json.loads((OUT/'freeze.json').read_text());assert lock['config']==CONFIG
    for p,digest in lock['files'].items():assert tx.sha(p)==digest,p
    for name,version in lock['packages'].items():assert importlib.metadata.version(name)==version


def prepare():
    protect();frame=pd.read_parquet(SOURCE)
    w=windows.build_final_windows(frame);data=with_memory(w)
    dest=OUT/'prepared';dest.mkdir(exist_ok=False)
    w.origin_audit.to_csv(dest/'origin_support.csv',index=False);log=[]
    for month in MONTHS:
        train,query,truth=split(data,month+'-01')
        assert len(train['origins'])>=80
        np.savez_compressed(dest/f'{month}_train.npz',**train)
        np.savez_compressed(dest/f'{month}_query.npz',**query)
        np.savez_compressed(dest/f'{month}_truth.npz',y=truth,origins=query['origins'])
        log.append(dict(month=month,training_origins=len(train['origins']),issued_origins=len(query['origins']),
            observed_hours=int(np.isfinite(truth).sum()),complete_days=int(np.isfinite(truth).all(1).sum())))
    tx.dump(OUT/'preparation.json',dict(log=log,files={str(p):tx.sha(p) for p in dest.glob('*')}))
    print(json.dumps(log),flush=True)


def check_prepared():
    protect()
    for p,digest in json.loads((OUT/'preparation.json').read_text())['files'].items():assert tx.sha(p)==digest,p


def load_model(folder,arm):
    info=json.loads((folder/'fit.json').read_text());assert info['config']==CONFIG and info['arm']==arm
    model=model_factory(arm);model.load_state_dict(load_file(str(folder/'encoder.safetensors')))
    head=local.head_bank(model.decoder,False);head.load_state_dict(load_file(str(folder/'head.safetensors')))
    return model,head,info


def run():
    check_prepared();done=[]
    for month in MONTHS:
        train=pilot.m.read(OUT/f'prepared/{month}_train.npz');q=pilot.m.read(OUT/f'prepared/{month}_query.npz')
        assert (train['origins']+tx.DAY<=pd.Timestamp(month+'-01').value).all()
        folder=OUT/'models'/month;folder.mkdir(parents=True,exist_ok=True)
        for name,summary in [('plain',False),('statistics',True)]:
            file=folder/f'{name}.txt'
            if not file.exists():
                model=lgb.LGBMRegressor(n_estimators=300,**pilot.TREE).fit(pilot.tree_features(train,summary),train['y'].ravel()).booster_
                model.save_model(str(file))
            model=lgb.Booster(model_file=str(file))
            value=np.maximum(model.predict(pilot.tree_features(q,summary),num_threads=2),0.).reshape(-1,24)
            assert np.isfinite(value).all();np.save(folder/f'{name}.npy',value)
        for arm in ARMS:
            for seed in SEEDS:
                dest=folder/f'{arm}_{seed}'
                if (dest/'fit.json').exists():model,head,info=load_model(dest,arm)
                else:model,head,info=train_model(train,seed,arm,dest)
                np.save(dest/'prediction.npy',predict(model,head,q,info['scale']))
                done.append(f'{month}/{arm}_{seed}')
                tx.dump(OUT/'training_status.json',dict(completed=done,expected=18))
                print(f'Completed {len(done)}/18',flush=True)


def metrics(frame,cols):
    good=frame.actual_kwh.notna()
    if not np.isfinite(frame[cols]).all().all():raise ValueError('All issued predictions must be finite')
    return pilot.metrics(frame.loc[good],cols)


def paired_intervals(frame,first,second,months):
    pieces=[];daily_rows=[]
    for month in months:
        part=frame.loc[frame.month==month]
        good=part.actual_kwh.notna()
        delta=np.abs(part[first].to_numpy()-part.actual_kwh.to_numpy()[:,None]).mean(1)-np.abs(part[second].to_numpy()-part.actual_kwh.to_numpy()[:,None]).mean(1)
        daily=pd.DataFrame(dict(origin=part.origin_local,gain=np.where(good,delta,0.),count=good.astype(int)))
        daily=daily.groupby('origin')[['gain','count']].sum()
        calendar=pd.date_range(month+'-01',pd.Timestamp(month+'-01')+pd.offsets.MonthBegin(),inclusive='left')
        daily=daily.reindex(calendar,fill_value=0);pieces.append(daily.to_numpy());daily_rows.append(daily)
    rng=np.random.default_rng(42);intervals=[]
    for block in (7,14):
        values=[]
        for _ in range(2000):
            sums=np.zeros(2)
            for piece in pieces:
                starts=rng.integers(0,len(piece)-block+1,size=int(np.ceil(len(piece)/block)))
                positions=(starts[:,None]+np.arange(block)).ravel()[:len(piece)]
                sums+=piece[positions].sum(0)
            if sums[1]>0:values.append(sums[0]/sums[1])
        limits=np.quantile(values,[.025,.975]).tolist() if values else [None,None]
        intervals.append(dict(block_days=block,low=limits[0],high=limits[1],valid_draws=len(values)))
    daily=pd.concat(daily_rows);count=float(daily['count'].sum())
    return dict(gain_kwh=float(daily.gain.sum()/count) if count else None,intervals=intervals),daily


def score():
    check_prepared();pieces=[]
    for month in MONTHS:
        q=pilot.m.read(OUT/f'prepared/{month}_query.npz');truth=pilot.m.read(OUT/f'prepared/{month}_truth.npz')
        train=pilot.m.read(OUT/f'prepared/{month}_train.npz')
        frame=pd.DataFrame(dict(month=month,origin_local=np.repeat(pd.to_datetime(q['origins']),24),
            step=np.tile(np.arange(1,25),len(q['origins'])),actual_kwh=truth['y'].ravel(),
            complete_target=np.repeat(np.isfinite(truth['y']).all(1),24),train_p90=max(float(np.quantile(train['y'],.9)),1.),
            yesterday=q['history'][:,-24:].ravel(),last_week=q['history'][:,:24].ravel()))
        for name in ('plain','statistics'):frame[name]=np.load(OUT/f'models/{month}/{name}.npy').ravel()
        for arm in ARMS:
            for seed in SEEDS:
                frame[f'{arm}_{seed}']=np.load(OUT/f'models/{month}/{arm}_{seed}/prediction.npy').ravel()
                frame[f'{arm}_blend_{seed}']=.5*frame.statistics+.5*frame[f'{arm}_{seed}']
        pieces.append(frame)
    frame=pd.concat(pieces,ignore_index=True)
    arms={k:[k] for k in ('plain','statistics','yesterday','last_week')}
    arms.update({k:[f'{k}_{s}' for s in SEEDS] for k in ('conditional','concat','conditional_blend','concat_blend')})
    periods={}
    for period in (*MONTHS,'pooled'):
        months=MONTHS if period=='pooled' else (period,);part=frame.loc[frame.month.isin(months)]
        record=dict(all={k:metrics(part,c) for k,c in arms.items()},
            complete_days={k:metrics(part.loc[part.complete_target],c) for k,c in arms.items()},
            seeds={k:[metrics(part,[c]) for c in cols] for k,cols in arms.items()},pairs=[],groups={})
        for first,second in [('plain','conditional_blend'),('plain','concat_blend'),('concat_blend','conditional_blend')]:
            pair,daily=paired_intervals(part,arms[first],arms[second],months)
            pair.update(first=first,second=second);record['pairs'].append(pair)
            daily.to_csv(OUT/f'daily_{period}_{first}_{second}.csv',index_label='origin')
        for name,mask in [('zero',part.actual_kwh==0),('ordinary',(part.actual_kwh>0)&(part.actual_kwh<part.train_p90)),('high',part.actual_kwh>=part.train_p90)]:
            record['groups'][name]={k:metrics(part.loc[mask],cols) for k,cols in arms.items()}
        record['support']=dict(issued_hours=len(part),observed_hours=int(part.actual_kwh.notna().sum()),
            complete_days=int(part.loc[part.complete_target,'origin_local'].nunique()))
        periods[period]=record
    frame.to_parquet(OUT/'predictions.parquet',index=False)
    tx.dump(OUT/'results.json',dict(config=CONFIG,arms=arms,periods=periods,independent_validation=False,
        previously_seen_period=True,source_release_date='2026-09-11',new_raw_release=False))
    print(json.dumps({p:{k:v['mae_kwh'] for k,v in row['all'].items()} for p,row in periods.items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','prepare','run','score'))
    globals()[parser.parse_args().stage]()
