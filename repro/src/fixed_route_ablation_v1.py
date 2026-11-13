"""Component ablations of the locked S/LA forecast; only uniform LA is new."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from safetensors.torch import load_file

from src import memory_auxiliary_v1 as a

d,h,tx,f,m,b,local = a.d,a.h,a.tx,a.f,a.m,a.b,a.local
ROOT=a.ROOT
OUT=ROOT/'artifacts/fixed_route_ablation_v1'
CONFIG=dict(a.CONFIG,alpha=.05,readout='valid_pool_uniform')
TITLES={
    'full':'完整主线 S + L5',
    'no_L':'移除L，只用S',
    'no_S':'移除S，只用L5',
    'no_S_summary':'移除S的摘要：C + L5',
    'no_L_summary':'移除L的摘要：S + L0',
    'uniform':'历史池权重均分：S + LU5',
    'no_local_head':'移除编号头适配：S + F5',
    'no_aux':'移除无摘要辅助训练：原B',
}


def freeze():
    a.check_selection()
    assert json.loads((a.OUT/'acceptance.json').read_text())['passed']
    OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),ROOT/'experiments/fixed_route_ablation_v1.md',
        ROOT/'tests/test_fixed_route_ablation_v1.py',ROOT/'diagnostics/verify_fixed_route_ablation_v1.py',
        ROOT/'outputs/research_atlas_2026-11-04/route-lock.json',
        a.OUT/'acceptance.json',a.OUT/'predictions.parquet',a.OUT/'results.json',
        h.OUT/'predictions.parquet',h.OUT/'results.json']
    tx.dump(OUT/'freeze.json',dict(date='2026-11-04',config=CONFIG,new_encoder_fits=12,new_head_fits=12,
        no_new_data=True,previously_seen_evaluation=True,main_route_unchanged=True,
        files={str(p):tx.sha(p) for p in paths}))


def protect():
    a.check_selection()
    lock=json.loads((OUT/'freeze.json').read_text())
    assert lock['config']==CONFIG
    for p,digest in lock['files'].items():
        assert tx.sha(p)==digest,p


def fit_encoder(train,seed,folder):
    folder.mkdir(parents=True,exist_ok=False)
    t=tx.configure()
    t.manual_seed(seed)
    scales=f.scales_from(train,'p90')
    e=h.encode(train,scales,'F1')
    model=h.make_model('FU')
    data={k:t.from_numpy(e[k]) for k in ('x','memory','future','y','mask','scale')}
    opt=t.optim.AdamW(model.parameters(),lr=CONFIG['encoder_lr'],weight_decay=.0001)
    rng=np.random.default_rng(seed)
    before,steps,trace=time.perf_counter(),0,[]
    for epoch in range(1,81):
        model.train()
        order=rng.permutation(len(train['origins']))
        totals=np.zeros(3)
        for start in range(0,len(order),64):
            idx=order[start:start+64]
            opt.zero_grad(set_to_none=True)
            real,zero=d.paired_forward(model,*[data[k][idx] for k in ('x','memory','future')])
            losses=a.weighted_loss(real,zero,data['y'][idx],data['mask'][idx],data['scale'][idx],
                e['valid_per_origin'],e['scale_reference'],.05)
            if not all(t.isfinite(v) for v in losses):
                raise ValueError('Nonfinite uniform-readout loss')
            losses[0].backward()
            t.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            steps+=1
            totals+=np.array([float(v.detach()) for v in losses])*len(idx)
        if epoch%20==0:
            row=dict(epoch=epoch,optimizer_steps=steps,seconds=time.perf_counter()-before,
                weighted_loss=totals[0]/len(order),real_loss=totals[1]/len(order),zero_loss=totals[2]/len(order))
            trace.append(row)
            tx.dump(folder/'trace.json',trace)
            print(folder.parent.name+' '+json.dumps(row),flush=True)
    d.save_model(model,folder/'model.safetensors')
    info=dict(config=CONFIG,seed=seed,alpha=.05,scales=scales,origins=len(train['origins']),
        labels=int(e['mask'].sum()),optimizer_steps=steps,parameters=sum(p.numel() for p in model.parameters()),
        inactive_query_key_parameters=8192,latest_training_horizon_ns=int(train['origins'].max()+tx.DAY),
        validation_used_in_fit=False,seconds=time.perf_counter()-before)
    tx.dump(folder/'fit.json',info)
    return model,info


def predict(model,head,query,scales):
    assert not np.isfinite(query['y']).any()
    cache=h.feature_cache(model,query,scales,'FU')
    pf=np.maximum(cache['original']*cache['scale'][:,None],0.)
    pl=np.maximum(local.predict(head,cache),0.)
    return dict(FU5=pf,LU5=pl,BU5=b.fixed_blend(query['baseline'],pl))


def run():
    protect()
    done=[]
    for month in h.MONTHS:
        train=m.read(h.OUT/f'prepared/{month}_train.npz')
        query=m.read(h.OUT/f'prepared/{month}_query.npz')
        assert (train['origins']+tx.DAY<=h.cutoff(month)).all()
        assert not np.isfinite(query['y']).any()
        for seed in tx.SEEDS:
            folder=OUT/f'models/{month}_{seed}'
            if (folder/'encoder/fit.json').exists():
                info=json.loads((folder/'encoder/fit.json').read_text())
                assert info['config']==CONFIG and info['seed']==seed
                model=h.make_model('FU')
                model.load_state_dict(load_file(str(folder/'encoder/model.safetensors')))
            else:
                model,info=fit_encoder(train,seed,folder/'encoder')
            if (folder/'head/fit.json').exists():
                head=local.head_bank(model.decoder,True)
                head.load_state_dict(load_file(str(folder/'head/head.safetensors')))
            else:
                real=h.feature_cache(model,train,info['scales'],'FU')
                zero=h.feature_cache(model,train,info['scales'],'F0')
                head,_=a.fit_stage(real,seed,.05,folder/'head',model.decoder,zero)
            np.savez_compressed(folder/'prediction.npz',**predict(model,head,query,info['scales']))
            tx.dump(folder/'complete.json',dict(month=month,seed=seed,config=CONFIG,query_truth_hidden=True))
            done.append(folder.name)
            tx.dump(OUT/'training_status.json',dict(completed=done,expected=12))
            print(f'Uniform {folder.name}: {len(done)}/12',flush=True)


def score():
    protect()
    frame=pd.read_parquet(a.OUT/'predictions.parquet')
    for seed in tx.SEEDS:
        rows,values=[],[]
        for month in h.MONTHS:
            query=m.read(h.OUT/f'prepared/{month}_query.npz')
            p=m.read(OUT/f'models/{month}_{seed}/prediction.npz')
            rows.append(query['rows'].ravel());values.append(p['BU5'].ravel())
        b.put_predictions(frame,np.concatenate(rows),f'BU5_{seed}',np.concatenate(values))
        frame[f'C_LA_{seed}']=.5*frame.C+.5*frame[f'LA_{seed}']
        frame[f'S_FA_{seed}']=.5*frame.S+.5*frame[f'FA_{seed}']
    arms=dict(full=[f'BA_{s}' for s in tx.SEEDS],no_L=['S'],no_S=[f'LA_{s}' for s in tx.SEEDS],
        no_S_summary=[f'C_LA_{s}' for s in tx.SEEDS],no_L_summary=[f'B0_{s}' for s in tx.SEEDS],
        uniform=[f'BU5_{s}' for s in tx.SEEDS],no_local_head=[f'S_FA_{s}' for s in tx.SEEDS],
        no_aux=[f'B1_{s}' for s in tx.SEEDS])
    results=dict(config=CONFIG,arms=arms,titles=TITLES,periods={},new_fits=12)
    for period in (*h.MONTHS,'pooled_Jan_Mar'):
        part=frame.loc[frame.month.isin(b.MONTHS) if period=='pooled_Jan_Mar' else frame.month==period]
        record=dict(all={k:m.metrics(part,v) for k,v in arms.items()},groups=[],pairs=[])
        for axis,label,group in d.groups_for(part):
            record['groups'].append(dict(axis=axis,label=label,models={k:m.metrics(group,v) for k,v in arms.items()}))
        for key in arms:
            if key=='full':continue
            pair,daily=m.compare(part,arms[key],arms['full'])
            record['pairs'].append(dict(pair,first=key,second='full'))
            daily.to_parquet(OUT/f'daily_{period}_{key}.parquet',index=False)
        results['periods'][period]=record
        print(period+' '+json.dumps({k:v['mae_kw'] for k,v in record['all'].items()}),flush=True)
    results['web_rows']=[dict(method=TITLES[key],scope='一至三月 / 七月；MAE / kW',
        effect=f"{results['periods']['pooled_Jan_Mar']['all'][key]['mae_kw']:.4f} / {results['periods']['2022-07']['all'][key]['mae_kw']:.4f}",
        source='FIXED_ROUTE_ABLATION_RESULTS_2026-11-04.md') for key in arms]
    gains={p:results['periods'][p]['all']['uniform']['mae_kw']-results['periods'][p]['all']['full']['mae_kw']
           for p in ('pooled_Jan_Mar','2022-07')}
    results['web_conclusion']='学习历史池权重相对均匀读取：一至三月MAE差 '+f"{gains['pooled_Jan_Mar']:+.4f}"+'，七月 '+f"{gains['2022-07']:+.4f}"+' kW（正数表示完整主线较好）。均匀对照保留近期条件门控。'
    frame.to_parquet(OUT/'predictions.parquet',index=False)
    tx.dump(OUT/'results.json',results)
    tx.dump(OUT/'status.json',dict(status='completed_pending_verification'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','run','score'))
    args=parser.parse_args()
    {'freeze':freeze,'run':run,'score':score}[args.stage]()
