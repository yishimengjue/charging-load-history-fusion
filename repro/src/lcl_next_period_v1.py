"""Seven frozen methods; fit every model before opening a new month."""
import argparse
import json
import time
from types import FunctionType

import lightgbm as lgb
import numpy as np

from src import lcl_next_period_data_v1 as data
from src import global_local_forecast_v1 as shared

util=data.util
METHODS=('C','S','G','SG','F','zero','mixture')
TREE_METHODS=('C','S','G','SG','zero','mixture')


def scales(month):
    return {e['id']:e['scale'] for e in json.loads((data.folder(month)/'inputs/entity_mapping.json').read_text())['selected']}


def shared_features(d,scale):
    ids=np.asarray(d['ids'])
    if not ((ids>=0)&(ids<48)&(ids==ids.astype(int))).all():
        raise ValueError('Invalid fixed entity ID')
    s=np.array([scale[int(i)] for i in ids])
    if not np.isfinite(s).all() or (s<1).any():raise ValueError('Invalid scale')
    raw=np.asarray(d['S'],float).copy()
    if raw.shape!=(len(ids),24,207):raise ValueError('Wrong shared feature shape')
    raw[:,:,shared.POWER_COLUMNS]/=s[:,None,None]
    const=np.column_stack([np.log1p(s),np.eye(48)[ids.astype(int)]])
    return np.concatenate([raw,np.repeat(const[:,None,:],24,axis=1)],axis=-1).reshape(-1,256)


def fitted_trainer(month):
    path=data.folder(month)
    ns=dict(data.neural.fit.__globals__,UP=path/'inputs',OUT=path/'neural',protect=lambda:data.protect(month))
    return FunctionType(data.neural.fit.__code__,ns,name='next_period_direct_fit')


def tree_fit(x,y,path,config,rounds,weight=None,initial=None,categorical=None):
    if path.exists():raise FileExistsError(path)
    before=time.perf_counter()
    ds=lgb.Dataset(x,label=y,weight=weight,categorical_feature=categorical or [],free_raw_data=True)
    model=lgb.train(config,ds,num_boost_round=rounds,init_model=initial)
    if initial is not None:
        parent=lgb.Booster(model_file=initial);shared.prefix_unchanged(parent,model)
    model.save_model(str(path))
    return dict(file=path.name,rows=len(y),requested_rounds=rounds,actual_trees=model.num_trees(),
                seconds=time.perf_counter()-before,sha256=util.sha(path))


def train(month):
    data.protect(month);dest=data.folder(month)
    if (dest/'fit_acceptance.json').exists():
        data.check(json.loads((dest/'fit_acceptance.json').read_text())['files']);return
    if (dest/'tree_fit.json').exists() and all((dest/f'neural/direct_full_{s}/fit.json').exists() for s in data.SEEDS):
        return
    path=dest/'trees';path.mkdir(exist_ok=False)
    tr=util.read(dest/'inputs/full_training.npz');mask=np.isfinite(tr['y']);s=scales(month);logs=[]
    for i in range(48):
        rows=tr['ids']==i;observed=mask[rows];y=tr['y'][rows][observed]
        for name in ('C','S'):
            # Use the same sklearn wrapper as the existing local baseline.
            p=path/f'{i:02d}_{name}.txt';start=time.perf_counter()
            m=lgb.LGBMRegressor(**shared.CONFIG,n_estimators=300).fit(tr[name][rows][observed],y).booster_
            m.save_model(str(p));logs.append(dict(file=p.name,rows=len(y),requested_rounds=300,
                actual_trees=m.num_trees(),seconds=time.perf_counter()-start,sha256=util.sha(p)))
    x=shared_features(tr,s)[mask.ravel()]
    row_scale=np.repeat([s[int(i)] for i in tr['ids']],24)[mask.ravel()]
    target=tr['y'][mask]/row_scale;weight=row_scale/row_scale.mean()
    # Match the original300+50 sklearn continued shared fit.
    for name,rounds,parent in (('G300',300,None),('G',50,path/'G300.txt')):
        before=time.perf_counter();m=shared.fit_head(x,target,weight,rounds,str(parent) if parent else None)
        if parent:shared.prefix_unchanged(lgb.Booster(model_file=str(parent)),m)
        p=path/f'{name}.txt';m.save_model(str(p));logs.append(dict(file=p.name,rows=len(target),
            requested_rounds=rounds,actual_trees=m.num_trees(),seconds=time.perf_counter()-before,sha256=util.sha(p)))
    x=util.features(tr)[mask.ravel()];y=tr['y'][mask];positive=y>0
    assert positive.any() and (~positive).any()
    for index,tau in enumerate(util.LEVELS):
        config=dict(util.CONFIG,objective='quantile',alpha=float(tau))
        logs.append(tree_fit(x[positive],y[positive],path/f'positive_{index:02d}.txt',config,150,categorical=[207]))
    logs.append(tree_fit(x,(~positive).astype(float),path/'zero.txt',dict(util.CONFIG,objective='binary'),150,categorical=[207]))
    util.dump(dest/'tree_fit.json',dict(logs=logs,training_max=float(y[positive].max()),
        latest_target_ns=int(tr['origins'].max()+data.DAY),labels=len(y),shared_weight_reference=float(row_scale.mean())))
    (dest/'neural').mkdir()
    for seed in data.SEEDS:fitted_trainer(month)('direct_full',seed)


def load_tree(month,name):
    return lgb.Booster(model_file=str(data.folder(month)/'trees'/name))


def predict_arrays(month,q):
    if not np.isnan(q['y']).all():raise ValueError('Query truth must be hidden')
    result={};s=scales(month)
    for name in ('C','S'):
        p=np.empty_like(q['y'])
        for i in range(48):
            rows=q['ids']==i
            if rows.any():p[rows]=np.maximum(load_tree(month,f'{i:02d}_{name}.txt').predict(q[name][rows].reshape(-1,q[name].shape[-1]),num_threads=2),0).reshape(-1,24)
        result[name]=p
    normalized=load_tree(month,'G.txt').predict(shared_features(q,s),num_threads=2).reshape(-1,24)
    result['G']=np.maximum(normalized*np.array([s[int(i)] for i in q['ids']])[:,None],0)
    result['SG']=.5*(result['S']+result['G']);result['zero']=np.zeros_like(q['y'])
    x=util.features(q,query=True)
    raw=np.column_stack([load_tree(month,f'positive_{i:02d}.txt').predict(x,num_threads=2) for i in range(15)])
    p0=load_tree(month,'zero.txt').predict(x,num_threads=2)
    cap=json.loads((data.folder(month)/'tree_fit.json').read_text())['training_max']
    quantiles=util.mixture_quantiles(p0,raw,util.LEVELS,cap)
    result['mixture']=quantiles[:,7].reshape(-1,24)
    result.update(mixture_raw=raw,p0=p0,mixture_quantiles=quantiles)
    blind=dict(q,anchor=result['S'])
    for seed in data.SEEDS:
        model,heads=data.old.load_model(data.folder(month)/f'neural/direct_full_{seed}','direct_full')
        output=data.neural.predict(model,heads,blind,s)
        result[f'F_{seed}']=output['final'];result[f'branch_{seed}']=output['branch']
    for name,v in result.items():
        assert np.isfinite(v).all(),name
        if name!='mixture_raw':assert (v>=0).all(),name
    return result


def predict(month):
    data.protect(month);dest=data.folder(month)
    if data.completed(dest/'prediction_manifest.json',[dest/'predictions.npz']):return
    data.check(json.loads((dest/'fit_acceptance.json').read_text())['files'])
    data.check(json.loads((dest/'release_acceptance.json').read_text())['files'])
    q=util.read(dest/'queries.npz');blind=dict(q,y=np.full_like(q['y'],np.nan))
    output=predict_arrays(month,blind)
    np.savez_compressed(dest/'predictions.npz',origins=q['origins'],ids=q['ids'],y=q['y'],**output)
    util.dump(dest/'prediction_manifest.json',dict(files=data.hashes([dest/'predictions.npz',dest/'queries.npz'])))


def run():
    from diagnostics import verify_lcl_next_period_v1 as verify
    from src import lcl_next_period_scores_v1 as score
    data.protect()
    verify.verify_old()
    for month in data.MONTHS:
        dest=data.folder(month)
        if (dest/'acceptance.json').exists():
            data.check(json.loads((dest/'acceptance.json').read_text())['files']);continue
        data.prepare(month);train(month);verify.verify_fit(month)
        data.release(month);predict(month);score.score_month(month);verify.verify_month(month)
    score.aggregate();verify.verify_aggregate()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'))
    args=parser.parse_args()
    data.freeze() if args.stage=='freeze' else run()
