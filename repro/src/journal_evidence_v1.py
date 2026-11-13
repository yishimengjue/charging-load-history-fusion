"""Bounded direct-concat control and read-only statistical supplementation."""
import argparse
import json
from pathlib import Path
import time
from types import MethodType

import numpy as np
import pandas as pd
from safetensors.torch import load_file
from src import fixed_route_ablation_v1 as old

a, h, tx, f, m, b, local, d = old.a, old.h, old.tx, old.f, old.m, old.b, old.local, old.d
ROOT = old.ROOT
OUT = ROOT / 'artifacts/journal_evidence_v1'
CONFIG = dict(a.CONFIG, alpha=.05, readout='direct_concat_118_to_64', epochs=80)


def features(self, x, memory, future, paired=False):
    t = tx.configure()
    base, z = f.latent_forward(self.base.backbone, x)
    const = x[:, :1, 1:11].expand(-1, 24, -1)
    base = base + self.base.head(t.cat([base.unsqueeze(-1), t.zeros_like(memory), const, future], -1)).squeeze(-1)
    query = self.query_norm(self.recent(z.flatten(1)).unsqueeze(1) +
                            self.known(t.cat([const, future], -1)) + self.lead)
    real = self.fusion_norm(self.concat_projection(t.cat([query, memory], -1)))
    if not paired:return base, real
    zero = self.fusion_norm(self.concat_projection(t.cat([query, t.zeros_like(memory)], -1)))
    return base, real, zero


def forward(self, x, memory, future):
    base, hidden = self.features(x, memory, future)
    return base + self.decoder(hidden).squeeze(-1)


def make_model():
    t = tx.configure()
    model = h.make_model('F1')
    for name in ('memory_embed', 'pool_embed', 'query_key', 'memory_key', 'memory_value', 'gate'):
        delattr(model, name)
    model.concat_projection = t.nn.Linear(118, 64, bias=False)
    with t.no_grad():model.concat_projection.weight[:, :64].copy_(t.eye(64))
    model.features = MethodType(features, model)
    model.forward = MethodType(forward, model)
    return model


def paired_forward(model, x, memory, future):
    base, real, zero = model.features(x, memory, future, paired=True)
    return base + model.decoder(real).squeeze(-1), base + model.decoder(zero).squeeze(-1)


def cache(model, data, scales, zero=False):
    t = tx.configure()
    model.eval()
    e = h.encode(data, scales, 'F0' if zero else 'F1')
    base, hidden, original = [], [], []
    with t.no_grad():
        for start in range(0, len(e['x']), 128):
            args = [t.from_numpy(e[k][start:start+128]) for k in ('x', 'memory', 'future')]
            v, z = model.features(*args)
            p = v + model.decoder(z).squeeze(-1)
            np.testing.assert_array_equal(p.numpy(), model(*args).numpy())
            base.append(v.numpy());hidden.append(z.numpy());original.append(p.numpy())
    return dict(base=np.concatenate(base), hidden=np.concatenate(hidden), original=np.concatenate(original),
        scale=e['scale'], y=e['y'], mask=e['mask'], ids=np.asarray(data['ids'], int)-1,
        origins=data['origins'], valid_per_origin=np.array(e['valid_per_origin']),
        scale_reference=np.array(e['scale_reference']), raw_y=data['y'])


def freeze():
    assert not OUT.exists()
    paths = [Path(__file__), ROOT/'experiments/journal_evidence_v1.md',
             ROOT/'tests/test_journal_evidence_v1.py', ROOT/'diagnostics/verify_journal_evidence_v1.py',
             ROOT/'outputs/research_atlas_2026-11-04/route-lock.json']
    paths += [ROOT/'src'/name for name in ('fixed_route_ablation_v1.py', 'memory_auxiliary_v1.py',
        'memory_dual_view_v1.py', 'history_readout_v1.py', 'timexer_feature_fusion_v1.py',
        'timexer_local_head_v1.py', 'memory_evidence_v1.py', 'chronos_timexer_v1.py')]
    for folder in (a.OUT, old.OUT):
        acc = json.loads((folder/'acceptance.json').read_text())
        assert acc['passed']
        for p, digest in acc['files'].items():assert tx.sha(p) == digest, p
        paths += [folder/'predictions.parquet', folder/'results.json', folder/'acceptance.json']
    paths += list((h.OUT/'prepared').glob('*.npz'))
    OUT.mkdir()
    tx.dump(OUT/'freeze.json', dict(config=CONFIG, seeds=tx.SEEDS, months=h.MONTHS,
        new_encoder_fits=12, new_head_banks=12, independent_validation=False,
        files={str(p):tx.sha(p) for p in paths}))
    print('Frozen12 concat encoder/head runs; old results and held-out data unchanged.', flush=True)


def protect():
    lock = json.loads((OUT/'freeze.json').read_text())
    assert lock['config'] == CONFIG
    for p, digest in lock['files'].items():assert tx.sha(p) == digest, p


def fit_encoder(train, seed, folder):
    folder.mkdir(parents=True, exist_ok=False)
    t = tx.configure();t.manual_seed(seed)
    scales = f.scales_from(train, 'p90')
    e = h.encode(train, scales, 'F1')
    model = make_model()
    tensors = {k:t.from_numpy(e[k]) for k in ('x','memory','future','y','mask','scale')}
    opt = t.optim.AdamW(model.parameters(), lr=CONFIG['encoder_lr'], weight_decay=.0001)
    rng = np.random.default_rng(seed)
    before, steps, trace = time.perf_counter(), 0, []
    for epoch in range(1,81):
        model.train();order=rng.permutation(len(train['origins']));total=0.
        for start in range(0,len(order),64):
            idx=order[start:start+64];opt.zero_grad(set_to_none=True)
            real,zero=paired_forward(model,*[tensors[k][idx] for k in ('x','memory','future')])
            losses=a.weighted_loss(real,zero,tensors['y'][idx],tensors['mask'][idx],tensors['scale'][idx],
                e['valid_per_origin'],e['scale_reference'],.05)
            assert all(t.isfinite(v) for v in losses)
            losses[0].backward();t.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            steps+=1;total+=float(losses[0].detach())*len(idx)
        if epoch%20==0:
            row=dict(epoch=epoch,optimizer_steps=steps,loss=total/len(order),seconds=time.perf_counter()-before)
            trace.append(row);tx.dump(folder/'trace.json',trace)
            print(folder.parent.name+' '+json.dumps(row),flush=True)
    d.save_model(model,folder/'model.safetensors')
    info=dict(config=CONFIG,seed=seed,scales=scales,origins=len(train['origins']),labels=int(e['mask'].sum()),
        optimizer_steps=steps,parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter()-before,latest_training_horizon_ns=int(train['origins'].max()+tx.DAY))
    tx.dump(folder/'fit.json',info)
    return model,info


def predict(model,head,query,scales):
    assert not np.isfinite(query['y']).any(), 'Query truth must stay hidden'
    c=cache(model,query,scales)
    neural=np.maximum(local.predict(head,c),0.)
    return dict(neural=neural,blend=b.fixed_blend(query['baseline'],neural))


def run():
    protect();done=[]
    for month in h.MONTHS:
        train=m.read(h.OUT/f'prepared/{month}_train.npz');query=m.read(h.OUT/f'prepared/{month}_query.npz')
        assert (train['origins']+tx.DAY<=h.cutoff(month)).all()
        assert not np.isfinite(query['y']).any()
        for seed in tx.SEEDS:
            folder=OUT/f'models/{month}_{seed}'
            if (folder/'encoder/fit.json').exists():
                info=json.loads((folder/'encoder/fit.json').read_text());assert info['config']==CONFIG
                model=make_model();model.load_state_dict(load_file(str(folder/'encoder/model.safetensors')))
            else:model,info=fit_encoder(train,seed,folder/'encoder')
            if (folder/'head/fit.json').exists():
                head=local.head_bank(model.decoder,True);head.load_state_dict(load_file(str(folder/'head/head.safetensors')))
            else:
                real=cache(model,train,info['scales']);zero=cache(model,train,info['scales'],zero=True)
                head,_=a.fit_stage(real,seed,.05,folder/'head',model.decoder,zero)
            np.savez_compressed(folder/'prediction.npz',**predict(model,head,query,info['scales']))
            done.append(folder.name);tx.dump(OUT/'training_status.json',dict(completed=done,expected=12))
            print(f'Concat {folder.name}: {len(done)}/12',flush=True)


def score():
    protect()
    frame=pd.read_parquet(old.OUT/'predictions.parquet')
    for seed in tx.SEEDS:
        rows,neural,blend=[],[],[]
        for month in h.MONTHS:
            query=m.read(h.OUT/f'prepared/{month}_query.npz');p=m.read(OUT/f'models/{month}_{seed}/prediction.npz')
            rows.append(query['rows'].ravel());neural.append(p['neural'].ravel());blend.append(p['blend'].ravel())
        b.put_predictions(frame,np.concatenate(rows),f'LC_{seed}',np.concatenate(neural))
        b.put_predictions(frame,np.concatenate(rows),f'BC_{seed}',np.concatenate(blend))
    frame.to_parquet(OUT/'predictions.parquet',index=False)
    summarize(frame, 'results.json')


def summarize(frame, filename):
    arms={k:[k] for k in ('C','S','G350','SG')}
    arms.update({k:[f'{p}_{s}' for s in tx.SEEDS] for k,p in
        [('conditional','BA'),('conditional_neural','LA'),('uniform_gate','BU5'),('no_head','S_FA'),('no_aux','B1')]})
    if 'BC_42' in frame:
        arms.update(concat=[f'BC_{s}' for s in tx.SEEDS],concat_neural=[f'LC_{s}' for s in tx.SEEDS])
    result=dict(arms=arms,periods={},bootstrap=dict(draws=2000,seed=42,blocks=[7,14],percentiles=[2.5,97.5]),
                exploratory=True,independent_validation=False)
    for period in (*h.MONTHS,'pooled_Jan_Mar'):
        part=frame.loc[frame.month.isin(b.MONTHS) if period=='pooled_Jan_Mar' else frame.month==period]
        dates=part.origin_utc.dt.floor('D').drop_duplicates().sort_values()
        assert len(dates)==len(pd.date_range(dates.min(),dates.max(),freq='D'))
        record=dict(all={},seeds={},pairs=[])
        for key,cols in arms.items():
            record['all'][key]=m.metrics(part,cols)
            vals=[m.metrics(part,[c])['mae_kw'] for c in cols]
            record['seeds'][key]=dict(columns=cols,mae_kw=vals,sample_sd_kw=float(np.std(vals,ddof=1)) if len(vals)>1 else None)
        for key in ('C','S','G350','SG','uniform_gate','no_head','no_aux','concat'):
            if key not in arms:continue
            pair,daily=m.compare(part,arms[key],arms['conditional'])
            pair.update(first=key,second='conditional');record['pairs'].append(pair)
            daily.to_parquet(OUT/f'daily_{period}_{key}.parquet',index=False)
        record['support']=dict(candidate_rows=len(part),issued=int(part[arms['conditional']].notna().all(axis=1).sum()),
            observed=record['all']['conditional']['points'],dates=len(dates))
        result['periods'][period]=record
    costs=[]
    for month in h.MONTHS:
        for seed in tx.SEEDS:
            for arm,folder in [('conditional',a.OUT/f'final/{month}_{seed}'),
                ('uniform_gate',old.OUT/f'models/{month}_{seed}'),('concat',OUT/f'models/{month}_{seed}')]:
                if not (folder/'encoder/fit.json').exists():continue
                enc=json.loads((folder/'encoder/fit.json').read_text());head=json.loads((folder/'head/fit.json').read_text())
                costs.append(dict(arm=arm,month=month,seed=seed,encoder=enc,head=head))
    result['costs']=costs
    tx.dump(OUT/filename,result)
    print(json.dumps({p:{k:v['mae_kw'] for k,v in r['all'].items()} for p,r in result['periods'].items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','stats','run','score'))
    stage=parser.parse_args().stage
    if stage=='stats':protect();summarize(pd.read_parquet(old.OUT/'predictions.parquet'),'existing_statistics.json')
    else:globals()[stage]()
