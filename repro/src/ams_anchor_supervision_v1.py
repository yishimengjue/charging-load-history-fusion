"""Forward-anchor conditioned correction: branch vs deployed-output supervision."""
import argparse
import json
from pathlib import Path
from types import MethodType
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from safetensors.torch import load_file
from src import ams_fixed_comparison_v1 as old

ROOT=old.ROOT;OUT=ROOT/'artifacts/ams_anchor_supervision_v1'
tx,a,local=old.tx,old.a,old.local
ARMS=('branch','joint','joint_no_context');MONTHS=old.MONTHS;SEEDS=old.SEEDS
CONFIG=dict(epochs=80,head_epochs=20,lr=.0003,head_lr=.001,batch=64,alpha=.05,
    blend=.5,first_forward_month='2019-05',min_tree_days=80,independent_validation=False)


def feature_path(self,x,memory,future,paired=False):
    t=tx.configure();_,z=old.concat.f.latent_forward(self.base.backbone,x)
    const=x[:,:1,1:11].expand(-1,24,-1)
    anchor=future[:,:,8:]
    if not self.anchor_context:anchor=t.zeros_like(anchor)
    query=self.query_norm(self.recent(z.flatten(1)).unsqueeze(1)+
        self.known(t.cat([const,future[:,:,:8]],-1))+self.lead+self.anchor_embed(anchor))
    real=self.fusion_norm(self.concat_projection(t.cat([query,memory],-1)))
    base=future[:,:,-1]
    if not paired:return base,real
    zero=self.fusion_norm(self.concat_projection(t.cat([query,t.zeros_like(memory)],-1)))
    return base,real,zero


def make_model(context=True):
    t=tx.configure();model=old.concat.make_model()
    model.anchor_embed=t.nn.Linear(25,64,bias=False)
    model.anchor_context=context
    model.features=MethodType(feature_path,model)
    t.nn.init.zeros_(model.decoder[-1].weight);t.nn.init.zeros_(model.decoder[-1].bias)
    return model


def outputs(raw,anchor):
    t=tx.configure();branch=t.clamp(raw,min=0.)
    return branch,.5*anchor+.5*branch


def objective(raw,zero,e,idx,arm):
    anchor=e['future'][idx,:,-1]
    br,final=outputs(raw,anchor);bz,fz=outputs(zero,anchor)
    pred,pzero=(br,bz) if arm=='branch' else (final,fz)
    return a.weighted_loss(pred,pzero,e['y'][idx],e['mask'][idx],e['scale'][idx],
        e['valid_per_origin'],e['scale_reference'],.05)[0]


def encode(data,scale,zero=False):
    if not np.isfinite(data['anchor']).all():raise ValueError('Forward anchor missing')
    e=old.pilot.encoded(data,scale,zero)
    anchor=(data['anchor']/scale).astype(np.float32)
    curve=np.broadcast_to(anchor[:,None,:],(len(anchor),24,24))
    e['future']=np.concatenate([e['future'],curve,anchor[:,:,None]],axis=2)
    return e


def cache(model,data,scale,zero=False):
    t=tx.configure();e=encode(data,scale,zero);model.eval();bs=[];hs=[]
    with t.no_grad():
        for start in range(0,len(data['origins']),128):
            args=[t.from_numpy(e[k][start:start+128]) for k in ('x','memory','future')]
            base,h=model.features(*args);bs.append(base.numpy());hs.append(h.numpy())
    return dict(e,base=np.concatenate(bs),hidden=np.concatenate(hs),ids=np.zeros(len(data['origins']),int))


def predict(model,head,data,scale):
    if not np.isnan(data['y']).all():raise ValueError('Hide query truth')
    c=cache(model,data,scale)
    raw=local.predict(head,c)
    branch=np.maximum(raw,0.)
    return dict(branch=branch,final=.5*data['anchor']+.5*branch)


def fit(data,scale,arm,seed,folder):
    folder.mkdir(parents=True,exist_ok=False);t=tx.configure();t.manual_seed(seed)
    model=make_model(arm!='joint_no_context');encoded=encode(data,scale)
    e={k:t.from_numpy(encoded[k]) for k in ('x','memory','future','y','mask','scale')}
    e.update({k:encoded[k] for k in ('valid_per_origin','scale_reference')})
    opt=t.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    rng=np.random.default_rng(seed);started=time.perf_counter();trace=[];n=len(data['origins'])
    for epoch in range(1,81):
        model.train();order=rng.permutation(n);total=0.
        for start in range(0,n,64):
            idx=order[start:start+64];opt.zero_grad(set_to_none=True)
            pr,pz=old.concat.paired_forward(model,*[e[k][idx] for k in ('x','memory','future')])
            loss=objective(pr,pz,e,idx,arm);assert t.isfinite(loss)
            loss.backward();t.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(idx)
        if epoch%20==0:
            trace.append(dict(epoch=epoch,loss=total/n,seconds=time.perf_counter()-started))
            tx.dump(folder/'trace.json',trace)
    encoder_seconds=time.perf_counter()-started
    real,zero=cache(model,data,scale),cache(model,data,scale,True)
    np.testing.assert_array_equal(real['base'],zero['base'])
    t.manual_seed(seed);head=local.head_bank(model.decoder,False)
    opt=t.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    extra={k:t.from_numpy(real[k]) for k in ('hidden','base','ids')};hz=t.from_numpy(zero['hidden'])
    rng=np.random.default_rng(seed);started=time.perf_counter()
    for _ in range(20):
        head.train();order=rng.permutation(n)
        for start in range(0,n,64):
            idx=order[start:start+64];opt.zero_grad(set_to_none=True)
            pr=extra['base'][idx]+head(extra['hidden'][idx],extra['ids'][idx])
            pz=extra['base'][idx]+head(hz[idx],extra['ids'][idx])
            loss=objective(pr,pz,e,idx,arm);assert t.isfinite(loss)
            loss.backward();t.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
    a.d.save_model(model,folder/'encoder.safetensors');a.d.save_model(head,folder/'head.safetensors')
    info=dict(config=CONFIG,scale=scale,arm=arm,seed=seed,training_origins=n,labels=int(np.isfinite(data['y']).sum()),
        encoder_steps=80*int(np.ceil(n/64)),head_steps=20*int(np.ceil(n/64)),
        encoder_seconds=encoder_seconds,head_seconds=time.perf_counter()-started,
        latest_training_end_ns=int(data['origins'].max()+tx.DAY),parameters=sum(p.numel() for p in model.parameters()))
    tx.dump(folder/'fit.json',info)
    return model,head,info


def fold_masks(origins,cutoff):
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is not None or cutoff!=cutoff.normalize() or cutoff.day!=1:raise ValueError('Month start required')
    return origins+tx.DAY<=cutoff.value,(origins>=cutoff.value)&(origins<(cutoff+pd.offsets.MonthBegin()).value)


def freeze():
    old.check_prepared()
    acc=json.loads((old.OUT/'acceptance.json').read_text());assert acc['passed']
    for p,digest in acc['files'].items():assert tx.sha(p)==digest,p
    paths=[Path(__file__),ROOT/'experiments/ams_anchor_supervision_v1.md',ROOT/'tests/test_ams_anchor_supervision_v1.py',
        ROOT/'diagnostics/verify_ams_anchor_supervision_v1.py',old.OUT/'acceptance.json',old.OUT/'results.json',old.OUT/'predictions.parquet']
    paths+=list((old.OUT/'prepared').glob('*.npz'))
    paths+=list((old.OUT/'models').glob('*/statistics.txt'))
    paths+=list((old.OUT/'models').glob('*/statistics.npy'))
    paths+=list((ROOT/'src').glob('*.py'))
    OUT.mkdir(exist_ok=False)
    tx.dump(OUT/'freeze.json',dict(config=CONFIG,months=MONTHS,seeds=SEEDS,arms=ARMS,
        new_encoder_fits=27,forward_tree_fits=8,all_periods_previously_seen=True,
        files={str(p):tx.sha(p) for p in sorted(set(paths))}))
    print('Frozen matched anchor-supervision experiment',flush=True)


def protect():
    old.check_prepared();lock=json.loads((OUT/'freeze.json').read_text());assert lock['config']==CONFIG
    for p,digest in lock['files'].items():assert tx.sha(p)==digest,p


def prepare():
    protect();data=old.pilot.m.read(old.OUT/'prepared/2020-01_train.npz')
    dest=OUT/'prepared';dest.mkdir(exist_ok=False);tree_dir=OUT/'forward_trees';tree_dir.mkdir()
    anchor=np.full_like(data['y'],np.nan);cutoffs=np.full(len(anchor),-1,dtype=np.int64);logs=[]
    for cutoff in pd.date_range('2019-05-01','2019-12-01',freq='MS'):
        tr,qe=fold_masks(data['origins'],cutoff);assert tr.sum()>=80
        train={k:v[tr] for k,v in data.items()};query={k:v[qe] for k,v in data.items()}
        model=lgb.LGBMRegressor(n_estimators=300,**old.pilot.TREE).fit(old.pilot.tree_features(train,True),train['y'].ravel()).booster_
        model.save_model(str(tree_dir/(cutoff.strftime('%Y-%m')+'.txt')))
        anchor[qe]=np.maximum(model.predict(old.pilot.tree_features(query,True),num_threads=2),0.).reshape(-1,24)
        cutoffs[qe]=cutoff.value
        logs.append(dict(cutoff_ns=cutoff.value,training_indices=np.flatnonzero(tr).tolist(),query_indices=np.flatnonzero(qe).tolist(),
            latest_training_end_ns=int(data['origins'][tr].max()+tx.DAY)))
    np.savez_compressed(dest/'forward.npz',**data,anchor=anchor,anchor_cutoff=cutoffs)
    records=[]
    for month in MONTHS:
        full=old.pilot.m.read(old.OUT/f'prepared/{month}_train.npz')
        mask=(data['origins']+tx.DAY<=pd.Timestamp(month+'-01').value)&np.isfinite(anchor).all(1)
        train={k:v[mask] for k,v in data.items()};train.update(anchor=anchor[mask],anchor_cutoff=cutoffs[mask])
        assert (train['anchor_cutoff']<=train['origins']).all()
        assert np.isfinite(train['y']).all() and len(train['origins'])>=80
        q=old.pilot.m.read(old.OUT/f'prepared/{month}_query.npz')
        tree=lgb.Booster(model_file=str(old.OUT/f'models/{month}/statistics.txt'))
        qa=np.maximum(tree.predict(old.pilot.tree_features(q,True),num_threads=2),0.).reshape(-1,24)
        np.testing.assert_array_equal(qa,np.load(old.OUT/f'models/{month}/statistics.npy'))
        q.update(anchor=qa,anchor_cutoff=np.full(len(qa),pd.Timestamp(month+'-01').value))
        np.savez_compressed(dest/f'{month}_train.npz',**train);np.savez_compressed(dest/f'{month}_query.npz',**q)
        scale=max(float(np.quantile(full['y'],.9)),1.)
        records.append(dict(month=month,scale=scale,full_training_origins=len(full['origins']),
            correction_training_origins=len(train['origins']),query_origins=len(q['origins'])))
    tx.dump(OUT/'forward_log.json',logs)
    paths=list(dest.glob('*.npz'))+list(tree_dir.glob('*.txt'))+[OUT/'forward_log.json']
    tx.dump(OUT/'preparation.json',dict(records=records,files={str(p):tx.sha(p) for p in paths}))
    print(json.dumps(records),flush=True)


def check_prepared():
    protect()
    for p,digest in json.loads((OUT/'preparation.json').read_text())['files'].items():assert tx.sha(p)==digest,p


def load_model(folder):
    info=json.loads((folder/'fit.json').read_text());assert info['config']==CONFIG
    model=make_model(info['arm']!='joint_no_context');model.load_state_dict(load_file(str(folder/'encoder.safetensors')))
    head=local.head_bank(model.decoder,False);head.load_state_dict(load_file(str(folder/'head.safetensors')))
    return model,head,info


def run():
    check_prepared();done=[]
    records=json.loads((OUT/'preparation.json').read_text())['records']
    for row in records:
        month=row['month'];train=old.pilot.m.read(OUT/f'prepared/{month}_train.npz');q=old.pilot.m.read(OUT/f'prepared/{month}_query.npz')
        for arm in ARMS:
            for seed in SEEDS:
                folder=OUT/f'models/{month}/{arm}_{seed}'
                if (folder/'fit.json').exists():model,head,info=load_model(folder)
                else:model,head,info=fit(train,row['scale'],arm,seed,folder)
                np.savez_compressed(folder/'prediction.npz',**predict(model,head,q,row['scale']))
                done.append(f'{month}/{arm}_{seed}');tx.dump(OUT/'status.json',dict(completed=done,expected=27))
                print(f'Completed {len(done)}/27: {done[-1]}',flush=True)


def score():
    check_prepared();frame=pd.read_parquet(old.OUT/'predictions.parquet')
    arms={k:[k] for k in ('plain','statistics')}
    arms['old_concat']=[f'concat_blend_{s}' for s in SEEDS]
    for arm in ARMS:
        arms[arm]=[f'new_{arm}_{s}' for s in SEEDS]
        for seed in SEEDS:
            for month in MONTHS:
                p=old.pilot.m.read(OUT/f'models/{month}/{arm}_{seed}/prediction.npz')
                frame.loc[frame.month==month,f'new_{arm}_{seed}']=p['final'].ravel()
                if arm=='branch':frame.loc[frame.month==month,f'full_residual_{seed}']=p['branch'].ravel()
    arms['full_residual']=[f'full_residual_{s}' for s in SEEDS];periods={}
    for period in (*MONTHS,'pooled'):
        months=MONTHS if period=='pooled' else (period,);part=frame.loc[frame.month.isin(months)]
        rec=dict(all={k:old.metrics(part,c) for k,c in arms.items()},groups={},pairs=[],
            seeds={k:[old.metrics(part,[c]) for c in cols] for k,cols in arms.items()})
        for key in ('branch','joint_no_context','old_concat','plain','statistics','full_residual'):
            pair,daily=old.paired_intervals(part,arms[key],arms['joint'],months)
            pair.update(first=key,second='joint');rec['pairs'].append(pair)
            daily.to_csv(OUT/f'daily_{period}_{key}_joint.csv',index_label='origin')
        for name,mask in [('zero',part.actual_kwh==0),('ordinary',(part.actual_kwh>0)&(part.actual_kwh<part.train_p90)),('high',part.actual_kwh>=part.train_p90)]:
            rec['groups'][name]={k:old.metrics(part.loc[mask],c) for k,c in arms.items()}
        periods[period]=rec
    frame.to_parquet(OUT/'predictions.parquet',index=False)
    tx.dump(OUT/'results.json',dict(arms=arms,periods=periods,config=CONFIG,independent_validation=False))
    print(json.dumps({p:{k:v['mae_kwh'] for k,v in rec['all'].items()} for p,rec in periods.items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','prepare','run','score'))
    globals()[parser.parse_args().stage]()
