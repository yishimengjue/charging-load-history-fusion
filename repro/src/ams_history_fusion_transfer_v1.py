"""Transfer the locked two-branch forecast recipe to measured AMS energy."""
import argparse
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from safetensors.torch import load_file

from src import ams_forecast_data as ams
from src import memory_auxiliary_v1 as a
from src.optimise_structure_features_v1 import aligned_features
from src.global_local_forecast_v1 import CONFIG as TREE

ROOT=a.ROOT
OUT=ROOT/'artifacts/ams_history_fusion_transfer_v1'
MASKED=ROOT/'artifacts/public_ams_v1/masked_hourly.parquet'
MONTHS=('2019-09','2019-10')
SEEDS=(42,43,44)
CONFIG=dict(context=168,horizon=24,encoder_epochs=80,head_epochs=20,
    encoder_lr=.0003,head_lr=.001,batch=64,auxiliary=.05,blend=.5,unit='kWh')
tx,h,m,local=a.tx,a.h,a.m,a.local


def civil_weekday(origins,offsets):
    clock=np.asarray(origins,np.int64)[:,None]+np.asarray(offsets)[None,:]*tx.HOUR
    days=pd.DatetimeIndex(clock.ravel()).dayofweek.to_numpy().reshape(clock.shape)
    return np.eye(7,dtype=np.float32)[days]


def memories(history,y,origins):
    ids=np.ones(len(origins),dtype=int)
    output=[]
    for i in range(len(origins)):
        values,positions=m.pack_memory(i,history,y,ids,origins)
        assert (origins[positions]+tx.DAY<=origins[i]).all()
        output.append(values['S'])
    return np.stack(output)


def tree_features(data,summary):
    design,_=aligned_features(data['history'])
    calendar=civil_weekday(data['origins'],np.arange(24)).reshape(-1,7)
    parts=[design.to_numpy(),calendar]
    if summary:parts.append(data['memory'].reshape(-1,27))
    values=np.concatenate(parts,axis=1)
    assert values.shape==(len(data['origins'])*24,207 if summary else 180)
    return values


def encoded(data,scale,zero=False):
    result=tx.encode(data,{'1':scale},True)
    result['x'][:,:,11:18]=civil_weekday(data['origins'],np.arange(-168,0))
    memory=m.memory_encode(data['memory'],result['scale'])
    result['memory']=np.zeros_like(memory) if zero else memory
    result['future']=np.concatenate([civil_weekday(data['origins'],np.arange(24)),
        np.broadcast_to((np.arange(1,25)/24).astype(np.float32)[None,:,None],(len(data['origins']),24,1))],axis=2)
    return result


def cache(model,data,scale,zero=False):
    t=tx.configure(); model.eval()
    e=encoded(data,scale,zero)
    hidden=[];base=[];original=[]
    with t.no_grad():
        for start in range(0,len(data['origins']),128):
            args=[t.from_numpy(e[k][start:start+128]) for k in ('x','memory','future')]
            b,z=local.feature_batch(model,*args)
            pred=b+model.decoder(z).squeeze(-1)
            np.testing.assert_array_equal(pred.numpy(),model(*args).numpy())
            hidden.append(z.numpy());base.append(b.numpy());original.append(pred.numpy())
    return dict(e,hidden=np.concatenate(hidden),base=np.concatenate(base),original=np.concatenate(original),
                ids=np.zeros(len(data['origins']),np.int64))


def train_model(data,seed,folder):
    t=tx.configure();t.manual_seed(seed)
    scale=max(float(np.quantile(data['y'][np.isfinite(data['y'])],.9)),1.)
    model=h.make_model('F1')
    e=encoded(data,scale)
    tensors={k:t.from_numpy(e[k]) for k in ('x','memory','future','y','mask','scale')}
    optimizer=t.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    rng=np.random.default_rng(seed); started=time.perf_counter();trace=[]
    for epoch in range(1,81):
        model.train(); order=rng.permutation(len(data['origins']));loss_sum=0.
        for start in range(0,len(order),64):
            idx=order[start:start+64];optimizer.zero_grad(set_to_none=True)
            real,zero=a.d.paired_forward(model,*[tensors[k][idx] for k in ('x','memory','future')])
            loss,_,_=a.weighted_loss(real,zero,tensors['y'][idx],tensors['mask'][idx],tensors['scale'][idx],e['valid_per_origin'],e['scale_reference'],.05)
            assert t.isfinite(loss)
            loss.backward();t.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            loss_sum+=float(loss.detach())*len(idx)
        if epoch%20==0:
            trace.append(dict(epoch=epoch,loss=loss_sum/len(order),seconds=time.perf_counter()-started))
            print(folder.name+' '+json.dumps(trace[-1]),flush=True)
    encoder_seconds=time.perf_counter()-started
    real,zero=cache(model,data,scale),cache(model,data,scale,True)
    t.manual_seed(seed);head=local.head_bank(model.decoder,False)
    np.testing.assert_allclose(local.predict(head,real),real['original']*real['scale'][:,None],atol=5e-5,rtol=1e-6)
    optimizer=t.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    tensors={k:t.from_numpy(real[k]) for k in ('hidden','base','ids','y','mask','scale')}
    hz=t.from_numpy(zero['hidden'])
    np.testing.assert_array_equal(real['base'],zero['base'])
    rng=np.random.default_rng(seed);started=time.perf_counter()
    for epoch in range(20):
        head.train();order=rng.permutation(len(data['origins']))
        for start in range(0,len(order),64):
            idx=order[start:start+64];optimizer.zero_grad(set_to_none=True)
            pr=tensors['base'][idx]+head(tensors['hidden'][idx],tensors['ids'][idx])
            pz=tensors['base'][idx]+head(hz[idx],tensors['ids'][idx])
            loss,_,_=a.weighted_loss(pr,pz,tensors['y'][idx],tensors['mask'][idx],tensors['scale'][idx],e['valid_per_origin'],e['scale_reference'],.05)
            assert t.isfinite(loss)
            loss.backward();t.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
    info=dict(config=CONFIG,seed=seed,scale=scale,origins=len(data['origins']),labels=int(np.isfinite(data['y']).sum()),
        latest_training_end_ns=int(data['origins'].max()+tx.DAY),encoder_seconds=encoder_seconds,
        head_seconds=time.perf_counter()-started,encoder_steps=80*int(np.ceil(len(order)/64)),
        head_steps=20*int(np.ceil(len(order)/64)),encoder_parameters=sum(p.numel() for p in model.parameters()),
        active_head_parameters=sum(p.numel() for p in head.parameters()),trace=trace)
    a.d.save_model(model,folder/'encoder.safetensors');a.d.save_model(head,folder/'head.safetensors')
    tx.dump(folder/'fit.json',info)
    return model,head,info


def predict(model,head,query,scale):
    assert np.isnan(query['y']).all()
    return np.maximum(local.predict(head,cache(model,query,scale)),0.)


def freeze():
    a.check_selection()
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'tests/test_ams_history_fusion_transfer_v1.py',ROOT/'experiments/ams_history_fusion_transfer_v1.md',
        MASKED,ROOT/'artifacts/public_ams_v1/source_semantics.json',ROOT/'data/public_ams_v1/manifest.json',
        ROOT/'outputs/research_atlas_2026-11-04/route-lock.json',ROOT/'src/ams_forecast_data.py',
        ROOT/'src/optimise_structure_features_v1.py',ROOT/'src/historical_memory_v1.py',
        ROOT/'src/optimise_missing_features_v1.py',ROOT/'src/public_ams.py']
    tx.dump(OUT/'freeze.json',dict(date='2026-11-05',config=CONFIG,months=MONTHS,seeds=SEEDS,
        models=6,tree_models=4,source='10.17632/jbks2rcwyj.1',source_previously_seen=True,
        files={str(p):tx.sha(p) for p in files}))


def protect():
    a.check_selection()
    for p,digest in json.loads((OUT/'freeze.json').read_text())['files'].items():assert tx.sha(p)==digest,p


def prepare():
    protect()
    manifest=json.loads((ROOT/'data/public_ams_v1/manifest.json').read_text())
    assert tx.sha(MASKED)==manifest['artifact_files']['masked_hourly.parquet']['sha256']
    frame=pd.read_parquet(MASKED)
    assert frame.loc[frame.partition=='final_test','ams_kwh'].isna().all()
    windows=ams.build_windows(frame,'2019-01-01','2019-11-01',context=168,horizon=24)
    origins=windows.origins.as_unit('ns').asi8
    data=dict(history=windows.X,y=windows.Y,origins=origins,ids=np.full(len(origins),'1'))
    data['memory']=memories(data['history'],data['y'],origins)
    folder=OUT/'prepared';folder.mkdir(exist_ok=False)
    windows.origin_audit.to_csv(folder/'origin_audit.csv',index=False)
    log=[]
    for month in MONTHS:
        cutoff=pd.Timestamp(month+'-01').value
        stop=(pd.Timestamp(month+'-01')+pd.offsets.MonthBegin()).value
        tr=origins+tx.DAY<=cutoff;qe=(origins>=cutoff)&(origins<stop)
        train={k:v[tr] for k,v in data.items()};query={k:v[qe] for k,v in data.items()}
        truth=query['y'].copy();query['y']=np.full_like(truth,np.nan)
        np.savez_compressed(folder/f'{month}_train.npz',**train)
        np.savez_compressed(folder/f'{month}_query.npz',**query)
        np.savez_compressed(folder/f'{month}_truth.npz',y=truth,origins=query['origins'])
        log.append(dict(month=month,train_origins=int(tr.sum()),query_origins=int(qe.sum()),
            training_labels=int(train['y'].size),evaluation_labels=int(truth.size),
            latest_training_end_ns=int(train['origins'].max()+tx.DAY)))
    assert sum(r['query_origins'] for r in log)==48 and log[0]['train_origins']==210
    tx.dump(OUT/'preparation.json',dict(windows=log,files={str(p):tx.sha(p) for p in folder.glob('*')}))
    print(json.dumps(log),flush=True)


def check_prepared():
    protect()
    for p,digest in json.loads((OUT/'preparation.json').read_text())['files'].items():assert tx.sha(p)==digest,p


def run():
    check_prepared()
    for month in MONTHS:
        train=m.read(OUT/f'prepared/{month}_train.npz');query=m.read(OUT/f'prepared/{month}_query.npz')
        folder=OUT/'models'/month;folder.mkdir(parents=True,exist_ok=True)
        assert (train['origins']+tx.DAY<=pd.Timestamp(month+'-01').value).all()
        for key,summary in [('plain',False),('statistics',True)]:
            file=folder/f'{key}.txt'
            if not file.exists():
                model=lgb.LGBMRegressor(n_estimators=300,**TREE).fit(tree_features(train,summary),train['y'].ravel()).booster_
                model.save_model(str(file))
            model=lgb.Booster(model_file=str(file))
            pred=np.maximum(model.predict(tree_features(query,summary),num_threads=2),0.).reshape(-1,24)
            np.save(folder/f'{key}.npy',pred)
        for seed in SEEDS:
            dest=folder/str(seed)
            if (dest/'fit.json').exists():
                info=json.loads((dest/'fit.json').read_text()); assert info['config']==CONFIG
                model=h.make_model('F1');model.load_state_dict(load_file(str(dest/'encoder.safetensors')))
                head=local.head_bank(model.decoder,False);head.load_state_dict(load_file(str(dest/'head.safetensors')))
            else:
                dest.mkdir(exist_ok=False);model,head,info=train_model(train,seed,dest)
            prediction=predict(model,head,query,info['scale'])
            np.save(dest/'prediction.npy',prediction)
            print(f'Completed {month} seed{seed}',flush=True)


def metrics(frame,cols):
    errors=frame[cols].to_numpy()-frame.actual_kwh.to_numpy()[:,None]
    return dict(points=len(frame),mae_kwh=float(np.abs(errors).mean()),
        rmse_kwh=float(np.sqrt(np.mean(errors**2,axis=0)).mean()),bias_kwh=float(errors.mean()),
        mean_under_kwh=float(np.maximum(-errors,0).mean())) if len(frame) else dict(points=0,mae_kwh=None,rmse_kwh=None,bias_kwh=None,mean_under_kwh=None)


def intervals(frame,first,second):
    pieces=[]
    for month,part in frame.groupby('month'):
        gain=np.abs(part[first].to_numpy()-part.actual_kwh.to_numpy()[:,None]).mean(axis=1)-np.abs(part[second].to_numpy()-part.actual_kwh.to_numpy()[:,None]).mean(axis=1)
        daily=pd.DataFrame(dict(origin=part.origin_local,gain=gain,count=1)).groupby('origin')[['gain','count']].sum()
        dates=pd.date_range(month+'-01',pd.Timestamp(month+'-01')+pd.offsets.MonthBegin(),inclusive='left')
        pieces.append(daily.reindex(dates,fill_value=0).to_numpy())
    rng=np.random.default_rng(42);ci=[]
    for block in (7,14):
        draws=[]
        for _ in range(2000):
            sums=np.zeros(2)
            for p in pieces:
                starts=rng.integers(0,len(p)-block+1,size=int(np.ceil(len(p)/block)))
                idx=np.concatenate([np.arange(i,i+block) for i in starts])[:len(p)]
                sums+=p[idx].sum(axis=0)
            if sums[1]:draws.append(sums[0]/sums[1])
        ci.append(dict(block_days=block,low=float(np.quantile(draws,.025)),high=float(np.quantile(draws,.975))))
    return ci


def score():
    check_prepared();parts=[]
    for month in MONTHS:
        train=m.read(OUT/f'prepared/{month}_train.npz');q=m.read(OUT/f'prepared/{month}_query.npz');truth=m.read(OUT/f'prepared/{month}_truth.npz')
        frame=pd.DataFrame(dict(month=month,origin_local=np.repeat(pd.to_datetime(q['origins']),24),
            step=np.tile(np.arange(1,25),len(q['origins'])),actual_kwh=truth['y'].ravel(),
            train_p90=max(float(np.quantile(train['y'],.9)),1.),yesterday=q['history'][:,-24:].ravel(),last_week=q['history'][:,:24].ravel()))
        for key in ('plain','statistics'):frame[key]=np.load(OUT/f'models/{month}/{key}.npy').ravel()
        for seed in SEEDS:
            frame[f'fusion_{seed}']=np.load(OUT/f'models/{month}/{seed}/prediction.npy').ravel()
            frame[f'blend_{seed}']=.5*frame.statistics+.5*frame[f'fusion_{seed}']
        parts.append(frame)
    frame=pd.concat(parts,ignore_index=True);assert len(frame)==1152 and np.isfinite(frame.actual_kwh).all()
    arms={k:[k] for k in ('yesterday','last_week','plain','statistics')}
    arms.update({k:[f'{k}_{s}' for s in SEEDS] for k in ('fusion','blend')})
    periods={}
    for period in (*MONTHS,'pooled'):
        part=frame if period=='pooled' else frame.loc[frame.month==period]
        groups=[]
        for label,mask in [('zero',part.actual_kwh==0),('ordinary',(part.actual_kwh>0)&(part.actual_kwh<part.train_p90)),('high',part.actual_kwh>=part.train_p90)]:
            groups.append(dict(axis='state',label=label,all={k:metrics(part.loc[mask],c) for k,c in arms.items()}))
        for start in (1,7,13,19):groups.append(dict(axis='lead',label=str(start),all={k:metrics(part.loc[part.step.between(start,start+5)],c) for k,c in arms.items()}))
        all_metrics={k:metrics(part,c) for k,c in arms.items()}
        periods[period]=dict(all=all_metrics,groups=groups,seeds={str(s):{k:metrics(part,[f'{k}_{s}']) for k in ('fusion','blend')} for s in SEEDS},
            blend_vs_statistics=dict(gain_kwh=all_metrics['statistics']['mae_kwh']-all_metrics['blend']['mae_kwh'],intervals=intervals(part,arms['statistics'],arms['blend'])))
    frame.to_parquet(OUT/'predictions.parquet',index=False)
    tx.dump(OUT/'results.json',dict(config=CONFIG,arms=arms,periods=periods,garage_count=1,evaluation_origins=48,source_previously_seen=True))
    print(json.dumps(periods['pooled']['all']),flush=True)


def verify():
    check_prepared();saved=pd.read_parquet(OUT/'predictions.parquet');result=json.loads((OUT/'results.json').read_text())
    count=0
    for month in MONTHS:
        train=m.read(OUT/f'prepared/{month}_train.npz');query=m.read(OUT/f'prepared/{month}_query.npz')
        truth=m.read(OUT/f'prepared/{month}_truth.npz');part=saved.loc[saved.month==month]
        np.testing.assert_array_equal(part.actual_kwh.to_numpy(),truth['y'].ravel())
        for key,summary in [('plain',False),('statistics',True)]:
            model=lgb.Booster(model_file=str(OUT/f'models/{month}/{key}.txt'))
            pred=np.maximum(model.predict(tree_features(query,summary),num_threads=2),0.)
            np.testing.assert_array_equal(pred,part[key].to_numpy());count+=1
        for seed in SEEDS:
            folder=OUT/f'models/{month}/{seed}';info=json.loads((folder/'fit.json').read_text())
            assert info['scale']==max(float(np.quantile(train['y'],.9)),1.)
            assert info['latest_training_end_ns']<=pd.Timestamp(month+'-01').value
            assert info['config']==CONFIG and info['encoder_steps']==80*int(np.ceil(len(train['origins'])/64))
            model=h.make_model('F1');model.load_state_dict(load_file(str(folder/'encoder.safetensors')))
            head=local.head_bank(model.decoder,False);head.load_state_dict(load_file(str(folder/'head.safetensors')))
            pred=predict(model,head,query,info['scale']).ravel()
            np.testing.assert_array_equal(pred,part[f'fusion_{seed}'].to_numpy());count+=1
            np.testing.assert_array_equal(.5*part.statistics+.5*pred,part[f'blend_{seed}']);count+=1
    for period,entry in result['periods'].items():
        part=saved if period=='pooled' else saved.loc[saved.month==period]
        for key,cols in result['arms'].items():
            expected=sum(float(np.abs(part[c]-part.actual_kwh).sum()) for c in cols)/(len(part)*len(cols))
            np.testing.assert_allclose(expected,entry['all'][key]['mae_kwh'],atol=1e-12)
    check_prepared()
    tx.dump(OUT/'acceptance.json',dict(passed=True,exact_prediction_arrays=count,mae_checks=18,model_release_unchanged=True,
        files={str(p):tx.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='acceptance.json'}))
    print('AMS transfer: saved models and predictions replay exactly',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','prepare','run','score','verify'))
    args=parser.parse_args();globals()[args.stage]()
