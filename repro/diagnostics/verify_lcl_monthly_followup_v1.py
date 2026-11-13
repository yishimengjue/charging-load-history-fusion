"""Independent score reductions and checkpoint replay for monthly continuation."""
import json
import math
import lightgbm as lgb
import numpy as np
from src import lcl_monthly_followup_v1 as n


def verify_scores(dest):
    data=n.neural.read(dest/'predictions.npz');y=data['y'];valid=np.isfinite(y)
    high=n.neural.read(dest/'scoring_masks.npz')['high'];s=json.loads((dest/'scores.json').read_text())
    masks=dict(all=valid,positive=valid&(y>0),ordinary_positive=valid&(y>0)&~high,
        high=valid&high,zero=valid&(y==0),negative=valid&(y<0))
    errors={};signed={};squares={}
    for name in ('C','S','zero','half_S')+n.FAMILIES:
        keys=[f'{name}_{seed}' for seed in n.neural.SEEDS] if name in n.FAMILIES else [name]
        errors[name]=sum(abs(data[k]-y) for k in keys)/len(keys)
        signed[name]=sum(data[k]-y for k in keys)/len(keys)
        squares[name]=sum((data[k]-y)**2 for k in keys)/len(keys)
    for row in s['metrics']:
        mask=masks[row['group']];count=int(mask.sum());assert row['n']==count
        if not count:assert row['mae'] is None;continue
        expected=dict(mae=float(errors[row['method']][mask].sum()/count),
            bias=float(signed[row['method']][mask].sum()/count),
            rmse=math.sqrt(float(squares[row['method']][mask].sum()/count)))
        for key,value in expected.items():assert math.isclose(row[key],value,rel_tol=1e-10,abs_tol=1e-12)
    for row in s['seeds']:
        mask=masks[row['group']]
        if mask.any():assert math.isclose(row['mae'],float(abs(data[f'{row["method"]}_{row["seed"]}']-y)[mask].mean()),abs_tol=1e-12)
    for row in s['entities']:
        mask=valid&(data['ids'][:,None]==row['id']);assert int(mask.sum())==row['n']
        if mask.any():assert math.isclose(float(errors[row['method']][mask].mean()),row['mae'],abs_tol=1e-12)
    for row in (r for r in s['metrics'] if r['group']=='all'):
        daily=[r for r in s['days'] if r['method']==row['method']]
        assert sum(r['n'] for r in daily)==row['n']
        assert math.isclose(sum(r['absolute_error_sum'] for r in daily)/row['n'],row['mae'],abs_tol=1e-12)
    day=((data['origins']-s['start_ns'])//n.august.forward.DAY_NS).astype(int)
    days=s['day_count'];intervals=0
    for pair in s['pairs']:
        mask=masks[pair['group']];diff=errors[pair['before']]-errors[pair['after']]
        gain=float(diff[mask].mean());assert math.isclose(gain,pair['gain'],rel_tol=1e-10,abs_tol=1e-12)
        base=float(errors[pair['before']][mask].mean())
        if base:assert math.isclose(pair['gain_percent'],100*gain/base,rel_tol=1e-10,abs_tol=1e-12)
        count=np.array([int(mask[day==d].sum()) for d in range(days)])
        sums=np.array([float(diff[day==d][mask[day==d]].sum()) for d in range(days)])
        for record in pair['intervals']:
            if record is None:continue
            block=record['block'];rng=np.random.default_rng(20261109)
            starts=rng.integers(0,days-block+1,size=(2000,(days+block-1)//block))
            ix=(starts[:,:,None]+np.arange(block)).reshape(2000,-1)[:,:days]
            den=count[ix].sum(1);values=sums[ix].sum(1)[den>0]/den[den>0]
            lo,hi=np.quantile(values,[.025,.975])
            assert math.isclose(lo,record['low'],abs_tol=1e-12) and math.isclose(hi,record['high'],abs_tol=1e-12)
            intervals+=1
    if 'monthly' in s:
        n.check_files(s['source_acceptance_hashes'])
        for row in s['metrics']:
            parts=[next(r for r in m['scores']['metrics'] if r['method']==row['method'] and r['group']==row['group']) for m in s['monthly']]
            total=sum(r['n'] for r in parts);assert total==row['n']
            if total:
                weighted=sum(r['mae']*r['n'] for r in parts if r['n'])/total
                assert math.isclose(row['mae'],weighted,abs_tol=1e-12)
        paths=[p for p in dest.rglob('*') if p.is_file() and p.name!='acceptance.json']
        n.neural.dump(dest/'acceptance.json',dict(passed=True,metric_groups=len(s['metrics']),interval_checks=intervals,
            monthly_weighting_verified=True,files={str(p.relative_to(n.ROOT)):n.neural.sha(p) for p in paths}))
    return dict(metric_groups=len(s['metrics']),interval_checks=intervals)


def verify_month(month):
    n.protect(month);dest=n.folder(month)
    n.check_files(json.loads((dest/'fit_acceptance.json').read_text())['files'])
    manifest=json.loads((dest/'prediction_manifest.json').read_text())
    assert n.neural.sha(dest/'predictions.npz')==manifest['predictions_sha256']
    assert n.neural.sha(dest/'queries.npz')==manifest['queries_sha256']
    data=n.neural.read(dest/'predictions.npz');q=n.neural.read(dest/'queries.npz')
    for key in ('origins','ids','y'):np.testing.assert_array_equal(data[key],q[key])
    known=n.known_grid(month);current=n.neural.read(dest/'evaluation_grid.npz')
    np.testing.assert_array_equal(known['members'],current['members'])
    source=np.concatenate([known['values'],current['values'],np.full((55,1),np.nan)],axis=1)
    start=int(known['times'][0].astype('datetime64[ns]').astype(np.int64))
    mapping=json.loads((dest/'inputs/entity_mapping.json').read_text())['selected']
    for entity in mapping:
        rows=q['ids']==entity['id'];ix=((q['origins'][rows]-start)//(3600*10**9)).astype(int)
        source_row=list(known['members']).index(entity['member'])
        np.testing.assert_array_equal(q['history'][rows],source[source_row][ix[:,None]+np.arange(-167,1)])
        np.testing.assert_array_equal(q['y'][rows],source[source_row][ix[:,None]+np.arange(1,25)])
    assert (q['memory_max_label_time']<=q['origins']).all()
    for name,width in (('C',180),('S',207)):
        for identifier in range(48):
            rows=q['ids']==identifier
            if not rows.any():continue
            model=lgb.Booster(model_file=str(dest/'trees'/f'{identifier:02d}_{name}.txt'))
            p=np.maximum(model.predict(q[name][rows].reshape(-1,width),num_threads=2),0).reshape(-1,24)
            np.testing.assert_array_equal(p,data[name][rows])
    np.testing.assert_array_equal(data['half_S'],.5*data['S']);assert not data['zero'].any()
    blind=dict(q,anchor=data['S'],y=np.full_like(q['y'],np.nan));replayed=0
    for family in n.FAMILIES:
        kind='new' if family.startswith('new_') else 'old'
        arm='direct_full' if family=='full_direct' else ('residual_matched' if family.endswith('residual') else 'direct_matched')
        for seed in n.neural.SEEDS:
            path=dest/'models'/kind/f'{arm}_{seed}';info=json.loads((path/'fit.json').read_text())
            model,heads=n.load_model(path,arm)
            output=n.neural.predict(model,heads,blind,{int(k):v for k,v in info['scales'].items()})
            for key,prefix in (('final',''),('branch','branch_')):
                np.testing.assert_array_equal(output[key],data[f'{prefix}{family}_{seed}'])
            replayed+=output['final'].size
    pre=json.loads((dest/'preflight.json').read_text());high=np.zeros_like(q['y'],bool)
    for row in pre['thresholds']:
        train=n.neural.read(dest/'inputs/prepared'/f'entity_{row["id"]:02d}.npz')
        positives=train['y'][train['y']>0]
        expected=float(np.quantile(positives,.9)) if len(positives) else None
        assert expected==row['positive_p90']
        if expected is not None:high |= (q['ids'][:,None]==row['id'])&(q['y']>0)&(q['y']>=expected)
    np.testing.assert_array_equal(high,n.neural.read(dest/'scoring_masks.npz')['high'])
    checks=verify_scores(dest)
    paths=[p for p in dest.rglob('*') if p.is_file() and p!=dest/'acceptance.json']
    n.neural.dump(dest/'acceptance.json',dict(passed=True,month=month,neural_replays=15,
        final_prediction_values=replayed,tree_replays=96,**checks,
        files={str(p.relative_to(n.ROOT)):n.neural.sha(p) for p in paths}))
    print(f'{month}: prediction replay and score checks passed',flush=True)
