"""Checkpoint, chronology, mixture arithmetic and score verification."""
import json

import lightgbm as lgb
import numpy as np

from src import lcl_next_period_v1 as model
from src import lcl_next_period_scores_v1 as scoring
from diagnostics.verify_lcl_zero_distribution_v1 import scalar_mix,interval

data=model.data
util=data.util


def verify_old():
    data.protect();path=data.OUT/'old_replay.json'
    if path.exists():
        assert json.loads(path.read_text())['passed'];return
    prior=data.old.folder('2013-10');q=util.read(prior/'queries.npz');p=util.read(prior/'predictions.npz')
    for key in ('origins','ids','y'):np.testing.assert_array_equal(q[key],p[key])
    old_scales={e['id']:e['scale'] for e in json.loads((prior/'inputs/entity_mapping.json').read_text())['selected']}
    blind=dict(q,y=np.full_like(q['y'],np.nan),anchor=p['S'])
    for seed in data.SEEDS:
        m,h=data.old.load_model(prior/f'models/old/direct_full_{seed}','direct_full')
        replay=data.neural.predict(m,h,blind,old_scales)
        np.testing.assert_array_equal(replay['final'],p[f'full_direct_{seed}'])
        np.testing.assert_array_equal(replay['branch'],p[f'branch_full_direct_{seed}'])
    util.dump(path,dict(passed=True,old_month='2013-10',seeds=list(data.SEEDS),
                       prediction_values=q['y'].size*6,changed_old_files=False))
    print('Old October direct fusion replay passed',flush=True)


def verify_fit(month):
    data.protect(month);dest=data.folder(month)
    if (dest/'fit_acceptance.json').exists():
        data.check(json.loads((dest/'fit_acceptance.json').read_text())['files']);return
    if (dest/'all_method_probe.npz').exists():raise FileExistsError('Incomplete fit verification retained')
    assert not (dest/'release.json').exists()
    tr=util.read(dest/'inputs/full_training.npz');mask=np.isfinite(tr['y'])
    assert data.training_rows(tr,data.bounds(month)[0].value).all()
    assert (tr['memory_max_label_time']<=tr['origins']).all()
    np.testing.assert_array_equal(tr['anchor'],np.zeros_like(tr['y']))
    probe=np.array([np.flatnonzero(tr['ids']==i)[0] for i in range(48)])
    q={k:v[probe] for k,v in tr.items()};q['y']=np.full_like(q['y'],np.nan)
    for seed in data.SEEDS:
        path=dest/f'neural/direct_full_{seed}'
        info=json.loads((path/'fit.json').read_text())
        assert info['config']==data.neural.CONFIG and info['arm']=='direct_full'
        assert info['labels']==mask.sum() and info['origins']==len(tr['ids'])
        assert info['latest_target_ns']<data.bounds(month)[0].value
        for key,value in model.scales(month).items():assert info['scales'][str(key)]==value
        m,h=data.old.load_model(path,'direct_full');saved=util.read(path/'probe.npz')
        pq={k:v[saved['indices']] for k,v in tr.items()};pq['y']=np.full_like(pq['y'],np.nan)
        replay=data.neural.predict(m,h,pq,model.scales(month))
        for key,value in replay.items():np.testing.assert_array_equal(value,saved[key])
    logs=json.loads((dest/'tree_fit.json').read_text());assert len(logs['logs'])==114
    assert logs['labels']==mask.sum() and logs['latest_target_ns']<data.bounds(month)[0].value
    assert logs['training_max']==tr['y'][mask].max()
    for row in logs['logs']:
        p=dest/'trees'/row['file'];assert util.sha(p)==row['sha256']
        m=lgb.Booster(model_file=str(p));assert m.num_trees()==row['actual_trees']
    model.shared.prefix_unchanged(model.load_tree(month,'G300.txt'),model.load_tree(month,'G.txt'))
    values=model.predict_arrays(month,q)
    np.savez_compressed(dest/'all_method_probe.npz',indices=probe,**values)
    files=list((dest/'neural').rglob('*'))+list((dest/'trees').glob('*.txt'))
    files=[p for p in files if p.is_file()]+[dest/'tree_fit.json',dest/'all_method_probe.npz']
    util.dump(dest/'fit_acceptance.json',dict(passed=True,neural_runs=3,tree_checkpoints=114,
        latest_training_target=int(tr['origins'].max()+data.DAY),new_month_values_read=False,
        files=data.hashes(files)))
    print(month+' all fits verified before numerical release',flush=True)


def check_scores(path):
    p=util.read(path/'predictions.npz');y=p['y'];h=util.read(path/'high.npz')['high']
    result=json.loads((path/'scores.json').read_text());valid=np.isfinite(y)
    masks=dict(all=valid,zero=valid&(y==0),positive=valid&(y>0),
               ordinary_positive=valid&(y>0)&~h,high=valid&h,negative=valid&(y<0))
    error={k:abs(p[k]-y) for k in model.TREE_METHODS}
    error['F']=sum(abs(p[f'F_{s}']-y) for s in data.SEEDS)/3
    for group,mask in masks.items():
        assert result['counts'][group]==mask.sum()
        for name,e in error.items():
            if mask.any():np.testing.assert_allclose(result['metrics'][group][name],e[mask].mean(),rtol=1e-12)
            else:assert result['metrics'][group][name] is None
    n=(result['end_ns']-result['start_ns'])//data.DAY
    counts=[];sums={k:[] for k in error}
    for day in range(n):
        a=result['start_ns']+day*data.DAY
        mask=valid&(p['origins'][:,None]>=a)&(p['origins'][:,None]<a+data.DAY)
        counts.append(int(mask.sum()))
        for k,e in error.items():sums[k].append(float(e[mask].sum()))
    np.testing.assert_array_equal(counts,result['daily_counts'])
    for k in error:np.testing.assert_allclose(sums[k],result['daily_sums'][k],atol=1e-10)
    for row in result['pairs']:
        name=row['reference'];delta=np.array(sums[name])-np.array(sums['F'])
        np.testing.assert_allclose(row['gain'],(error[name][valid]-error['F'][valid]).mean(),atol=1e-12)
        for key,ci in row['intervals'].items():
            np.testing.assert_allclose(ci,interval(delta,counts,int(key.rsplit('_',1)[1]),key.startswith('circular')),atol=1e-12)
    for row in result['entities']:
        mask=valid&(p['ids'][:,None]==row['id'])
        assert row['observed']==mask.sum() and row['issued']==24*(p['ids']==row['id']).sum()
        for k,e in error.items():
            if mask.any():np.testing.assert_allclose(row['metrics'][k],e[mask].mean(),atol=1e-12)
            else:assert row['metrics'][k] is None
    for row in result['seeds']:
        e=abs(p[f'F_{row["seed"]}']-y)
        for group,mask in masks.items():
            if mask.any():np.testing.assert_allclose(row['metrics'][group],e[mask].mean(),atol=1e-12)
    primary=next(r for r in result['pairs'] if r['reference']=='SG')
    assert result['decision']==dict(lower_than_SG=bool(error['F'][valid].mean()<error['SG'][valid].mean()),
        lower_than_all_references=bool(all(error['F'][valid].mean()<error[k][valid].mean() for k in model.TREE_METHODS)),
        all_primary_intervals_positive=all(v[0]>0 for v in primary['intervals'].values()))
    assert result['unknown']==(~valid).sum() and result['issued']==y.size


def verify_month(month):
    dest=data.folder(month);data.protect(month)
    data.check(json.loads((dest/'fit_acceptance.json').read_text())['files'])
    data.check(json.loads((dest/'release_acceptance.json').read_text())['files'])
    data.check(json.loads((dest/'score_manifest.json').read_text())['files'])
    q=util.read(dest/'queries.npz');p=util.read(dest/'predictions.npz')
    for key in ('origins','ids','y'):np.testing.assert_array_equal(q[key],p[key])
    saved=util.read(dest/'all_method_probe.npz');train=util.read(dest/'inputs/full_training.npz')
    probe={k:v[saved['indices']] for k,v in train.items()};probe['y']=np.full_like(probe['y'],np.nan)
    for key,v in model.predict_arrays(month,probe).items():np.testing.assert_array_equal(v,saved[key])
    blind=dict(q,y=np.full_like(q['y'],np.nan))
    for key,v in model.predict_arrays(month,blind).items():np.testing.assert_array_equal(v,p[key])
    cap=json.loads((dest/'tree_fit.json').read_text())['training_max']
    independent=scalar_mix(p['p0'],p['mixture_raw'],util.LEVELS,cap)
    np.testing.assert_allclose(independent,p['mixture_quantiles'],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(independent[:,7].reshape(-1,24),p['mixture'],atol=1e-12,rtol=1e-12)
    grid=data.known_grid(month);new=util.read(dest/'evaluation_grid.npz')
    np.testing.assert_array_equal(grid['members'],new['members'])
    values=np.concatenate([grid['values'],new['values'],np.full((55,1),np.nan)],axis=1)
    a,b=data.bounds(month);times=np.arange(grid['times'][0],np.datetime64(b)+np.timedelta64(1,'h'),np.timedelta64(1,'h'))
    mapping=json.loads((dest/'inputs/entity_mapping.json').read_text())
    covers=json.loads((dest/'coverage.json').read_text())['selected']
    expected_high=np.zeros_like(q['y'],bool)
    for e in mapping['selected']:
        i=list(grid['members']).index(e['member']);replay,c=data.old.august.queries(values[i],times,e['id'],a.value,b.value)
        mask=q['ids']==e['id']
        for key,v in replay.items():np.testing.assert_array_equal(v,q[key][mask])
        for key,v in c.items():assert covers[e['id']][key]==v
        labels=train['y'][train['ids']==e['id']];positive=labels[np.isfinite(labels)&(labels>0)]
        threshold=json.loads((dest/'inputs/thresholds.json').read_text())[e['id']]['positive_p90']
        assert threshold==(float(np.quantile(positive,.9)) if len(positive) else None)
        if threshold is not None:
            expected_high[mask]=(q['y'][mask]>0)&(q['y'][mask]>=threshold)
    np.testing.assert_array_equal(expected_high,util.read(dest/'high.npz')['high'])
    check_scores(dest)
    files=[p for p in dest.rglob('*') if p.is_file()]
    util.dump(dest/'acceptance.json',dict(passed=True,neural_reloads=3,tree_checkpoints=114,
        scalar_mixture_values=independent.size,query_reconstruction_entities=48,intervals_verified=24,
        files=data.hashes(files)))
    print(month+' model, source-grid, score and interval verification passed',flush=True)


def verify_aggregate():
    data.protect();dest=data.OUT/'aggregate';p=util.read(dest/'predictions.npz')
    if (dest/'acceptance.json').exists():
        data.check(json.loads((dest/'acceptance.json').read_text())['files']);return
    data.check(json.loads((dest/'score_manifest.json').read_text())['files'])
    for key in p:
        combined=np.concatenate([util.read(data.folder(m)/'predictions.npz')[key] for m in data.MONTHS])
        np.testing.assert_array_equal(combined,p[key])
    np.testing.assert_array_equal(util.read(dest/'high.npz')['high'],
        np.concatenate([util.read(data.folder(m)/'high.npz')['high'] for m in data.MONTHS]))
    check_scores(dest)
    result=json.loads((dest/'scores.json').read_text());assert len(result['daily_counts'])==92
    for month in data.MONTHS:data.check(json.loads((data.folder(month)/'acceptance.json').read_text())['files'])
    util.dump(dest/'acceptance.json',dict(passed=True,months=list(data.MONTHS),calendar_days=92,
        neural_fits=9,tree_checkpoints=342,files=data.hashes(list(dest.iterdir()))))
    print('PASS: all three months, pooled scores and immutable artifacts verified',flush=True)


if __name__=='__main__':
    verify_aggregate()
