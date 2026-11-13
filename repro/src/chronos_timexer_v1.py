"""Bounded, source-pinned Chronos and TimeXer comparison on saved data."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/chronos_timexer_v1'
OLD = ROOT / 'artifacts/global_shared_july_v1'
INNER = ROOT / 'artifacts/sharing_regret_gate_v1'
JULY = ROOT / 'artifacts/tirex2_july_v1'
SMALL_REV = 'ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a'
BASE_REV = '29ec3766d36d6f73f0696f85560a422f50e8498c'
TX_REV = '4e938a1767106324dd753b2a44832bf870a0252e'
SMALL = ROOT / 'vendor' / f'chronos-2-small-{SMALL_REV}'
BASE = ROOT / 'vendor' / f'chronos-2-{BASE_REV}'
VENDOR = ROOT / 'vendor' / f'timexer-{TX_REV}'
REVIEW = ROOT / 'data/chronos_timexer_review_2026-10-26'
HOUR = 3600 * 10**9
DAY = HOUR * 24
CUTOFF = pd.Timestamp('2022-07-01', tz='UTC').value
FOLDS = [(pd.Timestamp('2022-06-03', tz='UTC').value, pd.Timestamp('2022-06-17', tz='UTC').value),
         (pd.Timestamp('2022-06-17', tz='UTC').value, CUTOFF)]
IDS = tuple(str(i) for i in range(1, 10))
SEEDS = (42, 43, 44)
WIDTHS = (32, 64)
EPOCHS = tuple(range(10, 81, 10))
TX_FILES = ('LICENSE', 'models/TimeXer.py', 'layers/Embed.py',
            'layers/SelfAttention_Family.py', 'utils/masking.py')
KEYS = ['depot_id', 'origin_utc', 'step']
_torch_ready = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def configure():
    global _torch_ready
    import torch
    if not _torch_ready:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        _torch_ready = True
    return torch


def weekday(origin_ns, offsets):
    origins = np.asarray(origin_ns, np.int64)
    clock = origins[:, None] + np.asarray(offsets)[None, :] * HOUR
    day = pd.to_datetime(clock.ravel(), utc=True).dayofweek.to_numpy().reshape(clock.shape)
    return np.eye(7, dtype=np.float32)[day]


def regroup(x, y, origins, steps, sources=None):
    """Recover partial 24-label vectors without inventing missing labels."""
    x, y = np.asarray(x), np.asarray(y)
    origins, steps = np.asarray(origins, np.int64), np.asarray(steps, np.int64)
    if x.ndim != 2 or x.shape[1] < 168 or not (len(x) == len(y) == len(origins) == len(steps)):
        raise ValueError('Unaligned training inputs')
    if not np.isin(steps, np.arange(1, 25)).all() or not np.isfinite(y).all():
        raise ValueError('Only finite observed labels and valid leads can be regrouped')
    dates, first, inverse = np.unique(origins, return_index=True, return_inverse=True)
    if len(set(zip(inverse.tolist(), steps.tolist()))) != len(y):
        raise ValueError('Duplicate origin/step')
    history = x[first, :168].copy()
    np.testing.assert_array_equal(history[inverse], x[:, :168])
    yy = np.full((len(dates), 24), np.nan)
    ss = np.full((len(dates), 24), -1, np.int64)
    yy[inverse, steps-1] = y
    if sources is not None:
        ss[inverse, steps-1] = sources
    assert np.isfinite(yy).sum() == len(y)
    return dict(history=history, y=yy, origins=dates, source=ss)


def scales_from(train):
    result = {}
    for depot in IDS:
        v = train['y'][train['ids'] == depot]
        finite = v[np.isfinite(v)]
        if not len(finite):
            raise ValueError('No training labels for '+depot)
        result[depot] = max(float(np.quantile(finite, .9)), 1.)
    return result


def encode(data, scales, calendar):
    h = data['history']
    scale = np.asarray([scales[str(i)] for i in data['ids']], np.float32)
    mask = np.isfinite(h).astype(np.float32)
    if np.isinf(h).any() or np.any(h[np.isfinite(h)] < 0):
        raise ValueError('Invalid history')
    load = np.where(np.isfinite(h), h / scale[:, None], 0.).astype(np.float32)
    ids = np.eye(9, dtype=np.float32)[np.asarray(data['ids'], int)-1]
    const = np.column_stack([ids, np.log1p(scale)]).astype(np.float32)
    cal = weekday(data['origins'], np.arange(-167, 1))
    if not calendar:
        cal[:] = 0
    # The target must be the last variate in official TimeXer MS mode.
    xx = np.concatenate([mask[..., None], np.repeat(const[:, None, :], 168, axis=1), cal, load[..., None]], axis=2)
    assert xx.shape == (len(h), 168, 19) and np.isfinite(xx).all()
    yy = np.asarray(data['y'], np.float32)
    ym = np.isfinite(yy)
    yn = np.where(ym, yy / scale[:, None], 0.).astype(np.float32)
    return dict(x=xx, y=yn, mask=ym.astype(np.float32), scale=scale,
                valid_per_origin=float(ym.sum()/len(ym)),
                scale_reference=float(np.sum(scale[:, None]*ym)/ym.sum()) if ym.sum() else 1.)


def masked_loss(pred, target, mask, scales, valid_per_origin, scale_reference):
    # Fixed population denominator preserves per-observed-label raw-kW L1.
    return ((pred-target).abs()*mask*scales[:, None]).sum() / (
        len(pred)*valid_per_origin*scale_reference)


def load_tx(width):
    """Execute only reviewed official class definitions; no optional imports."""
    torch = configure()
    ns = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, np=np,
              math=math, sqrt=math.sqrt, __name__='timexer_pinned_classes')
    selection = {
        'utils/masking.py': ['TriangularCausalMask'],
        'layers/SelfAttention_Family.py': ['FullAttention', 'AttentionLayer'],
        'layers/Embed.py': ['PositionalEmbedding', 'DataEmbedding_inverted'],
        'models/TimeXer.py': ['FlattenHead', 'EnEmbedding', 'Encoder', 'EncoderLayer', 'Model'],
    }
    for file, names in selection.items():
        parsed = ast.parse((VENDOR / file).read_text(), filename=str(VENDOR / file))
        nodes = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in names]
        assert [node.name for node in nodes] == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(VENDOR / file), 'exec'), ns)
    config = SimpleNamespace(task_name='short_term_forecast', features='MS', seq_len=168,
        pred_len=24, use_norm=False, patch_len=12, enc_in=19, d_model=width,
        dropout=.1, embed='fixed', freq='h', factor=5, n_heads=4,
        d_ff=2*width, activation='gelu', e_layers=2)
    return ns['Model'](config)


def chronos_load(folder):
    torch = configure()
    from chronos import Chronos2Pipeline
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        if os.environ.get(name) != '1':
            raise ValueError('Explicit offline flags required')
    model = Chronos2Pipeline.from_pretrained(str(folder), device_map='cpu', torch_dtype=torch.float32,
        local_files_only=True, trust_remote_code=False, attn_implementation='eager')
    model.model.eval()
    return model


def chronos_predict(model, history, origins, calendar):
    from src.chronos_adapter_v1 import validate_histories
    x = validate_histories(history)
    before = x.copy()
    if calendar:
        past = weekday(origins, np.arange(-167, 1))
        future = weekday(origins, np.arange(1, 25))
        inputs = [dict(target=x[i].copy(),
            past_covariates={f'weekday_{k}': past[i, :, k] for k in range(7)},
            future_covariates={f'weekday_{k}': future[i, :, k] for k in range(7)}) for i in range(len(x))]
    else:
        inputs = x[:, None, :]
    raw = model.predict(inputs, prediction_length=24, context_length=168,
                        batch_size=16, cross_learning=False, limit_prediction_length=True)
    np.testing.assert_array_equal(x, before)
    values = np.stack([q.detach().cpu().numpy()[0] for q in raw])
    assert values.shape == (len(x), len(model.quantiles), 24) and np.isfinite(values).all()
    return values, list(model.quantiles)


def protect():
    lock = json.loads((OUT / 'freeze.json').read_text())
    for name in ('inputs', 'implementation'):
        for file, expected in lock[name].items():
            if sha(file) != expected:
                raise ValueError('Protected file changed: '+file)
    return lock


def prepare():
    OUT.mkdir(exist_ok=False)
    paths = [JULY / 'input_histories.npz', JULY / 'predictions.parquet',
             INNER / 'predictions.parquet', INNER / 'inner_predictions.parquet',
             OLD / 'training_membership.parquet', REVIEW / 'timexer_tree.json', REVIEW / 'chronos2_tree.json']
    paths += [OLD / 'prepared' / f'{i}.npz' for i in IDS]
    paths += [INNER / 'prepared' / f'{f}_{i}.npz' for f in (0, 1) for i in IDS]
    paths += [VENDOR / f for f in TX_FILES]
    paths += [folder / f for folder in (SMALL, BASE) for f in ('config.json', 'model.safetensors')]
    git_tree = {r['path']: r['sha'] for r in json.loads((REVIEW / 'timexer_tree.json').read_text())['tree']}
    for file in TX_FILES:
        b = (VENDOR / file).read_bytes()
        blob = hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
        assert blob == git_tree[file]
    weights = next(r for r in json.loads((REVIEW/'chronos2_tree.json').read_text()) if r['path']=='model.safetensors')
    assert sha(BASE/'model.safetensors') == weights['lfs']['oid']
    assert (BASE/'model.safetensors').stat().st_size == weights['size']
    code = [Path(__file__), ROOT/'experiments/chronos_timexer_v1.md',
            ROOT/'tests/test_chronos_timexer_v1.py', ROOT/'diagnostics/verify_chronos_timexer_v1.py',
            ROOT/'src/chronos_adapter_v1.py', ROOT/'src/run_tirex2_july_v1.py', ROOT/'run_chronos_python.sh']
    dump(OUT/'freeze.json', dict(date='2026-10-26', revisions=dict(small=SMALL_REV,standard=BASE_REV,timexer=TX_REV),
         inputs={str(p):sha(p) for p in paths}, implementation={str(p):sha(p) for p in code},
         versions={p:importlib.metadata.version(p) for p in ('torch','numpy','pandas','lightgbm','chronos-forecasting','safetensors')},
         widths=WIDTHS, epochs=EPOCHS, seeds=SEEDS, new_data=False, july_previously_seen=True,
         planned_neural_fits=14, planned_tree_fits=2, pretraining_overlap_excluded=False))
    membership = pd.read_parquet(OLD/'training_membership.parquet')
    pieces = []
    for depot in IDS:
        d = np.load(OLD/'prepared'/f'{depot}.npz', allow_pickle=False)
        old = membership.loc[membership.depot_id.astype(str).eq(depot)]
        for a,b in ((d['train_y'],old.actual_kw), (d['train_steps'],old.step),
                    (d['train_origins'],old.origin_utc.astype('int64')), (d['train_source'],old.source_row)):
            np.testing.assert_array_equal(a,b)
        assert (d['train_origins']+DAY <= CUTOFF).all()
        data = regroup(d['train_S'], d['train_y'], d['train_origins'], d['train_steps'], d['train_source'])
        data['ids'] = np.full(len(data['origins']), depot)
        pieces.append(data)
    train = {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    assert len(train['origins']) == 3278 and np.isfinite(train['y']).sum() == 74296
    np.savez_compressed(OUT/'training.npz', **train)
    oof = pd.read_parquet(INNER/'inner_predictions.parquet')
    for fold,(start,end) in enumerate(FOLDS):
        pieces=[]
        for depot in IDS:
            d=np.load(INNER/'prepared'/f'{fold}_{depot}.npz',allow_pickle=False)
            f=oof.loc[(oof.fold==fold)&oof.depot_id.astype(str).eq(depot)].sort_values(['origin_utc','step'])
            x=d['query_raw_x'].reshape(-1,24,207)
            assert len(f)==len(x)*24
            history=x[:,0,:168]
            np.testing.assert_array_equal(np.repeat(history[:,None,:],24,axis=1),x[:,:,:168])
            origins=f.origin_utc.astype('int64').to_numpy().reshape(-1,24)[:,0]
            assert ((origins>=start)&(origins<end)).all()
            np.testing.assert_array_equal(f.step.to_numpy().reshape(-1,24),np.tile(np.arange(1,25),(len(origins),1)))
            pieces.append(dict(history=history,y=f.actual_kw.to_numpy().reshape(-1,24),origins=origins,
                               ids=np.full(len(origins),depot),source=f.source_row.to_numpy().reshape(-1,24)))
        val={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        np.savez_compressed(OUT/f'validation_{fold}.npz',**val)
    q=np.load(JULY/'input_histories.npz',allow_pickle=False)
    frame=pd.read_parquet(JULY/'predictions.parquet')
    other=pd.read_parquet(INNER/'predictions.parquet')
    for f in (frame,other): f['depot_id']=f.depot_id.astype(str)
    frame=frame.merge(other[KEYS+['P','G350']],on=KEYS,how='left',validate='one_to_one',sort=False)
    rows=q['frame_rows']
    assert np.isfinite(frame.loc[rows.ravel(),'C']).all()
    for i,r in enumerate(rows):
        np.testing.assert_array_equal(frame.loc[r,'step'],np.arange(1,25))
        assert frame.loc[r,'depot_id'].eq(q['ids'][i]).all()
        assert frame.loc[r,'origin_utc'].astype('int64').eq(q['origin_ns'][i]).all()
    query=dict(history=q['histories'],origins=q['origin_ns'],ids=q['ids'],rows=rows,
               y=frame.loc[rows.ravel(),'actual_kw'].to_numpy().reshape(-1,24))
    assert query['history'].shape==(217,168) and np.isfinite(query['y']).sum()==4775
    np.savez_compressed(OUT/'query.npz',**query)
    frame.to_parquet(OUT/'reference.parquet',index=False)
    prepared=[p for p in OUT.iterdir() if p.suffix in ('.npz','.parquet')]
    dump(OUT/'prepared_hashes.json',{str(p):sha(p) for p in prepared})
    dump(OUT/'status.json',dict(status='prepared',training_origins=3278,training_labels=74296,issued_origins=217))
    print('Prepared 3278 training origins / 74296 labels / 217 query origins',flush=True)


def read_data(name):
    with np.load(OUT/name,allow_pickle=False) as d:
        return {k:d[k].copy() for k in d.files}


def preflight():
    protect()
    folder=OUT/'preflight'; folder.mkdir(exist_ok=False)
    t=configure(); t.manual_seed(42)
    history=np.tile(np.arange(168,dtype=np.float32)%24,(2,1)); history[0,20:30]=np.nan
    origins=np.array([pd.Timestamp('2022-05-01',tz='UTC').value,pd.Timestamp('2022-05-02',tz='UTC').value])
    checks=[]
    for name,path in (('Small',SMALL),('Base',BASE)):
        model=chronos_load(path)
        for cal in (False,True):
            raw,q=chronos_predict(model,history,origins,cal)
            again,_=chronos_predict(model,history,origins,cal)
            np.testing.assert_array_equal(raw,again)
            changed=history.copy(); changed[1]+=100
            changed_prediction,_=chronos_predict(model,changed,origins,cal)
            np.testing.assert_array_equal(raw[0],changed_prediction[0])
            zero=np.nan_to_num(history,nan=0.)
            zp,_=chronos_predict(model,zero,origins,cal)
            assert not np.array_equal(raw[0],zp[0])
            checks.append(dict(model=name,calendar=cal,shape=list(raw.shape),repeat_equal=True,
                 isolation=True,nan_not_zero=True,quantiles=q,parameters=sum(p.numel() for p in model.model.parameters())))
        del model
    for width in WIDTHS:
        model=load_tx(width); model.eval()
        x=t.zeros((2,168,19)); x[:,:,-1]=t.arange(168)%24
        with t.no_grad(): p=model(x,None,None,None)
        assert p.shape==(2,24,1) and t.isfinite(p).all()
        checks.append(dict(model='TimeXer',width=width,shape=list(p.shape),parameters=sum(p.numel() for p in model.parameters())))
    dump(folder/'checks.json',checks)
    print('Synthetic model preflight passed',flush=True)


def tx_predict(model, encoded):
    t=configure(); model.eval(); result=[]
    with t.no_grad():
        for begin in range(0,len(encoded['x']),128):
            x=t.from_numpy(encoded['x'][begin:begin+128])
            p=model(x,None,None,None).squeeze(-1).cpu().numpy()
            result.append(p*encoded['scale'][begin:begin+128,None])
    value=np.concatenate(result)
    assert np.isfinite(value).all()
    return value


def train_tx(train, validation, width, calendar, seed, epochs, folder, checkpoints):
    from safetensors.torch import save_file
    t=configure(); t.manual_seed(seed)
    scales=scales_from(train)
    tr=encode(train,scales,calendar); val=encode(validation,scales,calendar)
    model=load_tx(width)
    opt=t.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    xx=t.from_numpy(tr['x']); yy=t.from_numpy(tr['y']); mm=t.from_numpy(tr['mask']); ss=t.from_numpy(tr['scale'])
    rng=np.random.default_rng(seed)
    trace=[]; before=time.monotonic(); folder.mkdir(parents=True,exist_ok=False)
    for epoch in range(1,epochs+1):
        order=rng.permutation(len(xx)); model.train()
        for begin in range(0,len(xx),64):
            idx=order[begin:begin+64]
            opt.zero_grad(set_to_none=True)
            prediction=model(xx[idx],None,None,None).squeeze(-1)
            loss=masked_loss(prediction,yy[idx],mm[idx],ss[idx],tr['valid_per_origin'],tr['scale_reference'])
            if not t.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); t.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        if epoch in checkpoints or epoch==epochs:
            vp=np.maximum(tx_predict(model,val),0.)
            mask=np.isfinite(validation['y'])
            total=float(np.abs(vp[mask]-validation['y'][mask]).sum())
            tp=tx_predict(model,tr); tm=np.isfinite(train['y'])
            training_mae=float(np.abs(tp[tm]-train['y'][tm]).mean())
            trace.append(dict(epoch=epoch,validation_abs_sum=total,validation_points=int(mask.sum()),
                              validation_mae=total/int(mask.sum()),training_mae=training_mae,
                              seconds=time.monotonic()-before))
            dump(folder/'trace.json',trace)
            print(f'TimeXer {folder.name} epoch {epoch}: train {training_mae:.4f} / val {total/int(mask.sum()):.4f}',flush=True)
    save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},str(folder/'model.safetensors'))
    dump(folder/'fit.json',dict(width=width,calendar=calendar,seed=seed,epochs=epochs,scales=scales,
         labels=int(np.isfinite(train['y']).sum()),origins=len(train['origins']),
         max_training_horizon_end=int(train['origins'].max()+DAY),
         parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-before,
         valid_per_origin=tr['valid_per_origin'],scale_reference=tr['scale_reference']))
    return model,scales,trace


def select():
    protect()
    assert (OUT/'preflight/checks.json').exists()
    folder=OUT/'selection'; folder.mkdir(exist_ok=False)
    train=read_data('training.npz'); records=[]
    for fold,(start,end) in enumerate(FOLDS):
        keep=train['origins']+DAY<=start
        part={k:v[keep] for k,v in train.items()}
        validation=read_data(f'validation_{fold}.npz')
        for width in WIDTHS:
            for calendar in (False,True):
                model,scales,trace=train_tx(part,validation,width,calendar,42,80,
                    folder/f'f{fold}_d{width}_cal{int(calendar)}',EPOCHS)
                records.extend(dict(fold=fold,width=width,calendar=calendar,**r) for r in trace)
                del model
                dump(folder/'grid.json',records)
    grid=[]
    for width in WIDTHS:
        for epoch in EPOCHS:
            selected=[r for r in records if r['width']==width and r['epoch']==epoch]
            assert len(selected)==4
            grid.append(dict(width=width,epoch=epoch,
                joint_validation_mae=sum(r['validation_abs_sum'] for r in selected)/sum(r['validation_points'] for r in selected)))
    best=min(grid,key=lambda r:(r['joint_validation_mae'],r['epoch'],r['width']))
    dump(folder/'chosen.json',dict(**best,grid=grid,selection_uses_july=False,
                                 same_configuration_for_both_arms=True,grid_sha256=sha(folder/'grid.json')))
    print('Frozen TimeXer selection: '+json.dumps(best),flush=True)


def run_chronos():
    protect()
    assert (OUT/'preflight/checks.json').exists()
    folder=OUT/'chronos'; folder.mkdir(exist_ok=False)
    q=read_data('query.npz'); logs=[]
    for name,path in (('Small',SMALL),('Base',BASE)):
        model=chronos_load(path)
        for cal in (False,True):
            arm=name+('_cal' if cal else '_raw')
            before=time.monotonic()
            raw,quantiles=chronos_predict(model,q['history'],q['origins'],cal)
            p50=raw[:,quantiles.index(.5)]
            np.savez_compressed(folder/f'{arm}.npz',raw_quantiles=raw,quantiles=quantiles,p50=np.maximum(p50,0.))
            logs.append(dict(arm=arm,seconds=time.monotonic()-before,negative_p50=int((p50<0).sum()),
                             adjacent_crossings=int((np.diff(raw,axis=1)<0).sum()),curves=len(raw)))
            print(f'Chronos {arm}: {len(raw)} curves saved',flush=True)
        del model
    old=pd.read_parquet(OUT/'reference.parquet').loc[q['rows'].ravel(),'Chronos_p50'].to_numpy().reshape(-1,24)
    new=np.load(folder/'Small_raw.npz')['p50']
    delta=float(np.max(np.abs(old-new)))
    np.testing.assert_allclose(new,old,rtol=1e-5,atol=1e-5)
    dump(folder/'run.json',dict(logs=logs,old_small_max_abs_difference_kw=delta,old_small_replayed=True))


def tree_design(data,scales,calendar):
    e=encode(data,scales,calendar)
    const=e['x'][:,0,1:11]
    dow=weekday(data['origins'],[0])[:,0] if calendar else np.zeros((len(data['origins']),7),np.float32)
    base=np.column_stack([e['x'][:,:,-1],e['x'][:,:,0],const,dow])
    x=np.column_stack([np.repeat(base,24,axis=0),np.tile(np.arange(1,25),len(base))]).astype(np.float32)
    return x,e


def fit_final():
    protect()
    chosen=json.loads((OUT/'selection/chosen.json').read_text())
    assert chosen['grid_sha256']==sha(OUT/'selection/grid.json')
    folder=OUT/'final'; folder.mkdir(exist_ok=False)
    train=read_data('training.npz'); q=read_data('query.npz')
    q['y'] = np.full_like(q['y'], np.nan)
    # Final fitting receives no July truth, even for training trace evaluation.
    training_only_validation={k:v.copy() for k,v in train.items()}
    for calendar in (False,True):
        arm='Tcal' if calendar else 'T0'
        for seed in SEEDS:
            model,scales,trace=train_tx(train,training_only_validation,chosen['width'],calendar,seed,chosen['epoch'],
                folder/f'{arm}_{seed}',tuple(range(10,chosen['epoch']+1,10)))
            pred=tx_predict(model,encode(q,scales,calendar))
            np.savez_compressed(folder/f'{arm}_{seed}/prediction.npz',raw=pred,p50=np.maximum(pred,0.))
            del model
    import lightgbm as lgb
    from safetensors.torch import load_file
    scales=scales_from(train)
    for calendar in (False,True):
        arm='Gcalinfo' if calendar else 'G0info'
        tx,tr=tree_design(train,scales,calendar); qx,qe=tree_design(q,scales,calendar)
        good=tr['mask'].ravel().astype(bool)
        weights=np.repeat(tr['scale'],24)[good]/tr['scale_reference']
        names=[f'f{i}' for i in range(tx.shape[1])]
        model=lgb.LGBMRegressor(objective='quantile',alpha=.5,n_estimators=300,num_leaves=15,
            min_child_samples=40,learning_rate=.05,random_state=42,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
        before=time.monotonic()
        model.fit(pd.DataFrame(tx[good],columns=names),tr['y'].ravel()[good],sample_weight=weights)
        raw=model.predict(pd.DataFrame(qx,columns=names)).reshape(-1,24)*qe['scale'][:,None]
        model.booster_.save_model(str(folder/f'{arm}.txt'))
        np.savez_compressed(folder/f'{arm}.npz',raw=raw,p50=np.maximum(raw,0.))
        dump(folder/f'{arm}.json',dict(labels=int(good.sum()),scales=scales,features=tx.shape[1],
            calendar=calendar,seconds=time.monotonic()-before,trees=model.booster_.num_trees()))
    dump(folder/'frozen_selection.json',chosen)
    print('All six final TimeXer and two shared-tree fits saved; July unscored',flush=True)


def score():
    from src.run_tirex2_july_v1 import paired_blocks
    protect()
    if (OUT/'results.json').exists(): raise FileExistsError('Already scored')
    frame=pd.read_parquet(OUT/'reference.parquet'); q=read_data('query.npz')
    names={a:[a] for a in ('C','S','P','G350','Chronos_p50')}
    for a in ('Small_raw','Small_cal','Base_raw','Base_cal'):
        frame[a]=np.nan
        frame.loc[q['rows'].ravel(),a]=np.load(OUT/f'chronos/{a}.npz')['p50'].ravel()
        names[a]=[a]
    for a in ('G0info','Gcalinfo'):
        frame[a]=np.nan
        frame.loc[q['rows'].ravel(),a]=np.load(OUT/f'final/{a}.npz')['p50'].ravel()
        names[a]=[a]
    for arm in ('T0','Tcal'):
        names[arm]=[]
        for seed in SEEDS:
            a=f'{arm}_{seed}'; names[arm].append(a)
            frame[a]=np.nan
            frame.loc[q['rows'].ravel(),a]=np.load(OUT/f'final/{a}/prediction.npz')['p50'].ravel()
            names[a]=[a]
    issued=frame.C.notna(); obs=issued & frame.actual_kw.notna()
    for columns in names.values():
        for col in columns: np.testing.assert_array_equal(frame[col].notna(),issued)
    errors={a:frame[cols].to_numpy()-frame.actual_kw.to_numpy()[:,None] for a,cols in names.items()}
    def metrics(index):
        result={}
        for a,e in errors.items():
            v=e[index]
            result[a]=dict(points=int(index.sum()),mae_kw=float(np.abs(v).mean()),
                rmse_kw=float(np.sqrt((v*v).mean(axis=0)).mean()),bias_kw=float(v.mean()),
                underprediction_kw=float(np.maximum(-v,0).mean()))
        return result
    main=metrics(obs.to_numpy()); groups=[]
    for axis in ('depot_id','state','step','week','history_complete'):
        for value in frame.loc[obs,axis].unique():
            ix=(obs & frame[axis].eq(value)).to_numpy()
            groups.append(dict(axis=axis,label=str(value),metrics=metrics(ix)))
    leave4=metrics((obs & frame.depot_id.ne('4')).to_numpy())
    dates=pd.date_range('2022-07-01','2022-07-25',tz='UTC')
    daily=pd.DataFrame(index=dates)
    daily['points']=frame.loc[obs].groupby('origin_utc').size().reindex(dates,fill_value=0)
    for a,e in errors.items():
        values=pd.Series(np.abs(e[obs]).mean(axis=1),index=frame.loc[obs].index)
        daily[a+'_abs_sum']=values.groupby(frame.loc[obs,'origin_utc']).sum().reindex(dates,fill_value=0)
    candidates=('Small_cal','Base_raw','Base_cal','T0','Tcal','G0info','Gcalinfo')
    pairs=[(a,b) for b in candidates for a in ('C','S','P')]
    pairs += [('Small_raw','Small_cal'),('Base_raw','Base_cal'),('Small_raw','Base_raw'),
              ('Small_cal','Base_cal'),('T0','Tcal'),('Gcalinfo','Tcal'),('G0info','Gcalinfo'),('G350','Tcal')]
    results=[]
    for a,b in pairs:
        daily['gain_sum']=daily[a+'_abs_sum']-daily[b+'_abs_sum']
        intervals=[paired_blocks(daily,block)[0] for block in (7,14)]
        unknown=issued & ~obs
        # Mean seed losses use the mean of pairwise prediction distances.
        pa=frame.loc[unknown,names[a]].to_numpy(); pb=frame.loc[unknown,names[b]].to_numpy()
        radius=float(np.abs(pa[:,:,None]-pb[:,None,:]).mean(axis=(1,2)).sum())
        known=float(daily.gain_sum.sum())
        results.append(dict(first=a,second=b,gain_kw=main[a]['mae_kw']-main[b]['mae_kw'],
            gain_percent=100*(1-main[b]['mae_kw']/main[a]['mae_kw']),intervals=intervals,
            missing_truth_bounds=dict(low_kw=(known-radius)/int(issued.sum()),high_kw=(known+radius)/int(issued.sum()),
                                      missing_points=int(unknown.sum()),issued_points=int(issued.sum()))))
    daily.drop(columns='gain_sum').to_parquet(OUT/'daily_losses.parquet')
    frame.to_parquet(OUT/'predictions.parquet',index=False)
    gates={}
    for a in candidates:
        idwins=sum(g['metrics'][a]['mae_kw']<g['metrics']['S']['mae_kw'] for g in groups if g['axis']=='depot_id')
        checks=dict(beats_S=main[a]['mae_kw']<main['S']['mae_kw'],beats_P=main[a]['mae_kw']<main['P']['mae_kw'],
                    majority_ids_vs_S=idwins>=5,rmse_within_1pct_S=main[a]['rmse_kw']<=1.01*main['S']['rmse_kw'])
        gates[a]=dict(checks=checks,ids_better_than_S=idwins,passed=all(checks.values()))
    result=dict(main=main,groups=groups,excluding_id4=leave4,pairs=results,working_gate=gates,
        seed_means_are_metric_means_not_ensembles=True,evaluation_previously_seen=True,
        pretraining_overlap_excluded=False,independent_validation=False,new_data=False,
        selection=json.loads((OUT/'selection/chosen.json').read_text()),
        references_retained=['C','S','P'],new_neural_fits=14,new_tree_fits=2)
    dump(OUT/'results.json',result)
    dump(OUT/'status.json',dict(status='completed_pending_verification',promising=[a for a,g in gates.items() if g['passed']]))
    print(json.dumps(main,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','preflight','select','chronos','fit','score'])
    args=parser.parse_args()
    try:
        {'prepare':prepare,'preflight':preflight,'select':select,'chronos':run_chronos,'fit':fit_final,'score':score}[args.stage]()
    except Exception:
        if OUT.exists():
            dump(OUT/f'failure_{args.stage}_{time.time_ns()}.json',dict(stage=args.stage,traceback=traceback.format_exc()))
        raise
