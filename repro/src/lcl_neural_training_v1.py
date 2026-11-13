"""Expand only the entity interface of the fixed direct/residual model families."""
import argparse
import copy
import json
from pathlib import Path
from types import MethodType
import time

import numpy as np
import pandas as pd
from safetensors.torch import save_file, load_file

from src import journal_evidence_v1 as direct
from src import ams_anchor_supervision_v1 as residual
from src.lcl_forward_inputs_v1 import ROOT, sha, dump

tx = direct.tx
UP = ROOT / "artifacts/lcl_forward_inputs_v1"
OUT = ROOT / "artifacts/lcl_neural_training_v1"
ARMS = ("residual_matched", "direct_matched", "direct_full")
SEEDS = (42,43,44)
CONFIG = dict(entities=48, epochs=80, head_epochs=20, lr=.0003, head_lr=.001,
              batch=64, alpha=.05, blend=.5, weight_decay=.0001)


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def features(self, x, memory, future, paired=False):
    t = tx.configure()
    b, z = direct.f.latent_forward(self.base.backbone, x)
    const = x[:,:1,1:self.n_entities+2].expand(-1,24,-1)
    known = future[:,:,:8]
    q = self.recent(z.flatten(1)).unsqueeze(1)+self.known(t.cat([const,known],-1))+self.lead
    if self.is_residual:
        q = q+self.anchor_embed(future[:,:,8:])
        base = future[:,:,-1]
    else:
        base = b+self.base.head(t.cat([b.unsqueeze(-1),t.zeros_like(memory),const,known],-1)).squeeze(-1)
    query = self.query_norm(q)
    real = self.fusion_norm(self.concat_projection(t.cat([query,memory],-1)))
    if not paired:
        return base, real
    zero = self.fusion_norm(self.concat_projection(t.cat([query,t.zeros_like(memory)],-1)))
    return base, real, zero


def make_model(is_residual, n_entities=48):
    t = tx.configure()
    model = residual.make_model() if is_residual else direct.make_model()
    model.is_residual, model.n_entities = is_residual, n_entities
    if n_entities != 9:
        model.known = t.nn.Linear(n_entities+9,64)
        model.base.head[0] = t.nn.Linear(n_entities+64,32)
    model.features = MethodType(features,model)
    return model


def make_heads(decoder, n_entities=48):
    t = tx.configure()
    class Heads(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = t.nn.ModuleList([copy.deepcopy(decoder) for _ in range(n_entities)])
        def forward(self, hidden, ids):
            if not bool(((ids>=0)&(ids<n_entities)).all()):
                raise ValueError("Entity outside fixed mapping")
            output = hidden.new_zeros(hidden.shape[:2])
            for i, head in enumerate(self.heads):
                rows = (ids==i).nonzero(as_tuple=True)[0]
                if len(rows):
                    output[rows] = head(hidden[rows]).squeeze(-1)
            return output
    return Heads()


def calendar(origins, offsets):
    clock = np.asarray(origins)[:,None]+np.asarray(offsets)[None,:]*3600*10**9
    days = pd.DatetimeIndex(clock.ravel()).dayofweek.to_numpy().reshape(clock.shape)
    return np.eye(7,dtype=np.float32)[days]


def encode(data, scales, is_residual, n_entities=48):
    ids = np.asarray(data['ids'],int)
    if not ((ids>=0)&(ids<n_entities)).all():
        raise ValueError("Invalid entity encoding")
    scale = np.asarray([scales[int(i)] for i in ids],np.float32)
    if not np.isfinite(scale).all() or (scale<1).any():
        raise ValueError("Invalid fixed scale")
    h = data['history']
    if np.isinf(h).any() or (h[np.isfinite(h)]<0).any():
        raise ValueError("Invalid nonnegative-task history")
    observed = np.isfinite(h)
    load = np.where(observed,h/scale[:,None],0).astype(np.float32)
    const = np.column_stack([np.eye(n_entities,dtype=np.float32)[ids],np.log1p(scale)]).astype(np.float32)
    x = np.concatenate([observed[:,:,None].astype(np.float32),
        np.repeat(const[:,None,:],168,axis=1),calendar(data['origins'],np.arange(-167,1)),load[:,:,None]],-1)
    memory = direct.m.memory_encode(data['memory'],scale)
    known = np.concatenate([calendar(data['origins'],np.arange(1,25)),
        np.broadcast_to((np.arange(1,25)/24).astype(np.float32)[None,:,None],(len(ids),24,1))],-1)
    if is_residual:
        anchor = (data['anchor']/scale[:,None]).astype(np.float32)
        if not np.isfinite(anchor).all():
            raise ValueError("Residual training needs forward S")
        known = np.concatenate([known,np.broadcast_to(anchor[:,None,:],(len(ids),24,24)),anchor[:,:,None]],-1)
    y = data['y'].astype(np.float32)
    mask = np.isfinite(y)
    if (y[mask]<0).any():
        raise ValueError("Negative label cannot be flipped or clipped")
    return dict(x=x,memory=memory,future=known,y=np.where(mask,y/scale[:,None],0).astype(np.float32),
        mask=mask.astype(np.float32),scale=scale,ids=ids,
        valid_per_origin=float(mask.sum()/len(ids)),
        scale_reference=float((scale[:,None]*mask).sum()/mask.sum()) if mask.any() else 1.)


def loss(pr,pz,e,ix,arm):
    t = tx.configure()
    if arm != 'direct_full':
        pr,pz = t.clamp(pr,min=0),t.clamp(pz,min=0)
    return direct.a.weighted_loss(pr,pz,e['y'][ix],e['mask'][ix],e['scale'][ix],
        e['valid_per_origin'],e['scale_reference'],.05)[0]


def cache(model,e):
    t = tx.configure(); model.eval(); parts=[[],[],[]]
    with t.no_grad():
        for start in range(0,len(e['x']),128):
            vals=model.features(*[t.from_numpy(e[k][start:start+128]) for k in ('x','memory','future')],paired=True)
            for part,value in zip(parts,vals):
                part.append(value.numpy())
    return [np.concatenate(p) for p in parts]


def predict(model,heads,data,scales):
    if not np.isnan(data['y']).all():
        raise ValueError("Hide query labels before inference")
    t = tx.configure();e=encode(data,scales,model.is_residual,model.n_entities)
    base,h,_=cache(model,e);heads.eval()
    with t.no_grad():
        branch=np.maximum((base+heads(t.from_numpy(h),t.from_numpy(e['ids'])).numpy())*e['scale'][:,None],0)
    return dict(branch=branch,final=.5*data['anchor']+.5*branch)


def freeze():
    acc=json.loads((UP/'acceptance.json').read_text());assert acc['passed']
    for name,digest in acc['files'].items():assert sha(ROOT/name)==digest,name
    OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),ROOT/'experiments/lcl_neural_training_v1.md',
        ROOT/'tests/test_lcl_neural_training_v1.py',ROOT/'diagnostics/verify_lcl_neural_training_v1.py',UP/'acceptance.json',
        UP/'full_training.npz',UP/'matched_training.npz',UP/'entity_mapping.json']
    paths += list((ROOT/'src').glob('*.py'))
    paths += [tx.VENDOR/p for p in tx.TX_FILES]
    dump(OUT/'freeze.json',dict(config=CONFIG,arms=ARMS,seeds=SEEDS,
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))},test_values_read=False))


def protect():
    lock=json.loads((OUT/'freeze.json').read_text());assert lock['config']==CONFIG
    for name,digest in lock['files'].items():assert sha(ROOT/name)==digest,name


def fit(arm,seed):
    protect();folder=OUT/f'{arm}_{seed}'
    if (folder/'fit.json').exists():return
    folder.mkdir(exist_ok=False)
    data=read(UP/('full_training.npz' if arm=='direct_full' else 'matched_training.npz'))
    mapping=json.loads((UP/'entity_mapping.json').read_text())['selected']
    scales={r['id']:r['scale'] for r in mapping}
    encoded=encode(data,scales,arm=='residual_matched')
    t=tx.configure();t.manual_seed(seed);model=make_model(arm=='residual_matched')
    e={k:t.from_numpy(encoded[k]) for k in ('x','memory','future','y','mask','scale')}
    e.update({k:encoded[k] for k in ('valid_per_origin','scale_reference')})
    opt=t.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    rng=np.random.default_rng(seed);n=len(data['origins']);begun=time.perf_counter();trace=[]
    for epoch in range(1,81):
        model.train();order=rng.permutation(n);total=0.
        for start in range(0,n,64):
            ix=order[start:start+64];opt.zero_grad(set_to_none=True)
            base,hr,hz=model.features(*[e[k][ix] for k in ('x','memory','future')],paired=True)
            pr,pz=base+model.decoder(hr).squeeze(-1),base+model.decoder(hz).squeeze(-1)
            value=loss(pr,pz,e,ix,arm);assert t.isfinite(value)
            value.backward();t.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(value.detach())*len(ix)
        if epoch%20==0:
            trace.append(dict(epoch=epoch,training_loss=total/n,seconds=time.perf_counter()-begun))
            dump(folder/'trace.json',trace);print(f'{arm}/{seed}: epoch {epoch}/80',flush=True)
    encoder_seconds=time.perf_counter()-begun
    base,hr,hz=cache(model,encoded)
    t.manual_seed(seed);heads=make_heads(model.decoder)
    opt=t.optim.AdamW(heads.parameters(),lr=.001,weight_decay=.0001)
    bt,ht,zt,it=[t.from_numpy(v) for v in (base,hr,hz,encoded['ids'])]
    rng=np.random.default_rng(seed);begun=time.perf_counter()
    for _ in range(20):
        heads.train();order=rng.permutation(n)
        for start in range(0,n,64):
            ix=order[start:start+64];opt.zero_grad(set_to_none=True)
            value=loss(bt[ix]+heads(ht[ix],it[ix]),bt[ix]+heads(zt[ix],it[ix]),e,ix,arm)
            assert t.isfinite(value);value.backward();t.nn.utils.clip_grad_norm_(heads.parameters(),1.);opt.step()
    save_file({k:v.contiguous() for k,v in model.state_dict().items()},str(folder/'encoder.safetensors'))
    save_file({k:v.contiguous() for k,v in heads.state_dict().items()},str(folder/'heads.safetensors'))
    probe=np.array([np.flatnonzero((data['ids']==i)&np.isfinite(data['anchor']).all(1))[0] for i in range(48)])
    q={k:v[probe] for k,v in data.items()};q['y']=np.full_like(q['y'],np.nan)
    np.savez_compressed(folder/'probe.npz',indices=probe,**predict(model,heads,q,scales))
    dump(folder/'fit.json',dict(arm=arm,seed=seed,config=CONFIG,origins=n,labels=int(encoded['mask'].sum()),
        encoder_seconds=encoder_seconds,head_seconds=time.perf_counter()-begun,
        encoder_parameters=sum(p.numel() for p in model.parameters()),head_parameters=sum(p.numel() for p in heads.parameters()),
        scales=scales,latest_target_ns=int(data['origins'].max()+tx.DAY),test_values_read=False))
    print(f'Completed {arm}/{seed}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','train']);args=p.parse_args()
    if args.stage=='freeze':freeze()
    else:
        for arm in ARMS:
            for seed in SEEDS:fit(arm,seed)
