"""Reload all models and recompute scores without the experiment's scoring path."""
import json
import math

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import lcl_zero_distribution_v1 as run


def interval(sums, counts, block, circular):
    n = len(counts); rng = np.random.default_rng(20261112)
    starts = rng.integers(n if circular else n-block+1, size=(2000, math.ceil(n/block)))
    samples = []
    for row in starts:
        indices = [int((start+j) % n if circular else start+j) for start in row for j in range(block)][:n]
        samples.append(sum(sums[i] for i in indices) / sum(counts[i] for i in indices))
    return np.percentile(samples, [2.5, 97.5])


def scalar_mix(p0, raw, levels, cap):
    result = np.empty_like(raw)
    knots = [0.] + list(levels) + [1.]
    for i, probability in enumerate(p0):
        values = [0.] + sorted(min(max(float(v), 0.), cap) for v in raw[i]) + [cap]
        for j, tau in enumerate(levels):
            if tau <= probability:
                result[i, j] = 0.
                continue
            u = (tau-probability)/(1-probability)
            for k in range(len(knots)-1):
                if knots[k] <= u <= knots[k+1]:
                    weight = (u-knots[k])/(knots[k+1]-knots[k])
                    result[i,j] = values[k]*(1-weight)+values[k+1]*weight
                    break
            else:
                raise AssertionError('Uncovered mixture probability')
    return result


def verify_scores(summary, y, p, errors, ids, origins, high, start, end):
    valid = np.isfinite(y); count = int(valid.sum())
    assert summary['observed'] == count and summary['issued'] == len(y)
    masks = dict(all=valid, zero=valid&(y==0), positive=valid&(y>0),
                 high=valid&high, ordinary_positive=valid&(y>0)&~high)
    for row in summary['metrics']:
        mask = masks[row['group']]; assert row['n'] == int(mask.sum())
        if mask.any():
            np.testing.assert_allclose(row['mae'], np.sum(errors[row['method']][mask])/mask.sum(), rtol=1e-12)
    for row in summary['entities']:
        mask=valid&(ids==row['id']); assert row['n']==int(mask.sum())
        if mask.any(): np.testing.assert_allclose(row['mae'], errors[row['method']][mask].mean(),rtol=1e-12)
    days = list(range(start, end, run.DAY)); sums = {k:[] for k in errors}; counts = []
    daily = {(r['date'],r['method']):r for r in summary['daily']}
    for d in days:
        mask = valid & (origins >= d) & (origins < d+run.DAY)
        counts.append(int(mask.sum()))
        for k in sums:
            sums[k].append(float(errors[k][mask].sum()))
            row=daily[(str(pd.Timestamp(d).date()),k)]
            assert row['n']==int(mask.sum())
            np.testing.assert_allclose(row['absolute_error_sum'],sums[k][-1],atol=1e-10)
    for pair in summary['pairs']:
        a,b = pair['reference'],pair['candidate']
        np.testing.assert_allclose(pair['gain'], (errors[a][valid]-errors[b][valid]).mean(), rtol=1e-12,atol=1e-15)
        delta=np.array(sums[a])-np.array(sums[b])
        for key,value in pair['intervals'].items():
            circular = key.startswith('circular'); block=int(key.split('_')[-1])
            np.testing.assert_allclose(value,interval(delta,counts,block,circular),atol=1e-12,rtol=1e-10)
    probability = summary['probability']
    for method in ('direct','mixture'):
        q=p[method][valid]; observed=y[valid]
        scores=[]
        for j,tau in enumerate(run.LEVELS):
            scores.append(np.where(observed>=q[:,j],tau*(observed-q[:,j]),(1-tau)*(q[:,j]-observed)).mean())
        np.testing.assert_allclose(probability[method]['mean_pinball'],np.mean(scores),rtol=1e-12)
        np.testing.assert_allclose(probability[method]['coverage_10_90'],np.mean((observed>=q[:,3])&(observed<=q[:,11])))
        np.testing.assert_allclose(probability[method]['width_10_90'],np.mean(q[:,11]-q[:,3]))
    zero=(y[valid]==0).astype(float); pp=p['p0'][valid]; safe=np.clip(pp,1e-15,1-1e-15)
    np.testing.assert_allclose(probability['zero_classifier']['brier'],np.mean((zero-pp)**2))
    np.testing.assert_allclose(probability['zero_classifier']['log_loss'],np.mean(np.where(zero>0,-np.log(safe),-np.log(1-safe))))
    for row in probability['zero_classifier']['bins']:
        mask=(pp>=row['low'])&((pp<row['low']+.1) if row['low']<.9 else (pp<=1))
        assert row['n']==int(mask.sum())
        if mask.any():
            np.testing.assert_allclose(row['probability'],pp[mask].mean())
            np.testing.assert_allclose(row['zero_fraction'],zero[mask].mean())
    scores={k:float(v[valid].mean()) for k,v in errors.items()}
    assert summary['qualifies_for_op'] == all(scores['mixture']<scores[k] for k in ('zero','full_direct','direct'))
    return len(summary['metrics'])+len(summary['entities'])


def main():
    lock=run.protect(); out=run.OUT; results=json.loads((out/'results.json').read_text())
    initial={str(p.relative_to(out)):run.sha(p) for p in out.rglob('*') if p.is_file()}
    parts=[]; models=0; raw_values=0; checked_metrics=0; mixture_values=0
    for month,summary in zip(run.MONTHS,results['monthly']):
        paths=run.sources(month); dest=out/month; info=json.loads((dest/'fit.json').read_text())
        data=run.read(paths['query'],run.KEYS); train=run.read(paths['train'],run.KEYS)
        p=run.read(dest/'predictions.npz'); ref=run.read(paths['reference'])
        support=run.read(dest/'support.npz'); yy=train['y'].ravel(); observed=np.isfinite(yy)
        np.testing.assert_array_equal(support['observed_flat_indices'],np.flatnonzero(observed))
        np.testing.assert_array_equal(support['positive_in_observed'],np.flatnonzero(yy[observed]>0))
        for k in ('ids','origins'):
            np.testing.assert_array_equal(support[k],train[k])
            np.testing.assert_array_equal(data[k],p[k]);np.testing.assert_array_equal(data[k],ref[k])
        np.testing.assert_array_equal(data['y'],ref['y'])
        assert info['labels']==observed.sum() and info['positive_labels']==(yy[observed]>0).sum()
        assert info['latest_target_ns']==int(train['origins'].max()+run.DAY)<pd.Timestamp(month+'-01').value
        assert (train['memory_max_label_time']<=train['origins']).all()
        assert (data['memory_max_label_time']<=data['origins']).all()
        x=np.column_stack([data['S'].reshape(-1,207),np.repeat(data['ids'],24)])
        assert info['training_max']==yy[observed].max()
        for item in info['records']:
            path=out/item['model'];assert run.sha(path)==item['sha256']
            model=lgb.Booster(model_file=str(path));prediction=model.predict(x,num_threads=2)
            assert model.num_trees()==item['actual_trees']
            if item['family']=='zero': expected=p['p0']
            else:
                i=list(run.LEVELS).index(item['alpha']);expected=p[item['family']+'_raw'][:,i]
            np.testing.assert_array_equal(prediction,expected)
            models+=1;raw_values+=len(prediction)
        assert sum(x['requested_trees'] for x in info['records'] if x['family']=='direct')==2400
        assert sum(x['requested_trees'] for x in info['records'] if x['family']!='direct')==2400
        np.testing.assert_array_equal(p['direct'],np.sort(np.clip(p['direct_raw'],0,None),axis=1))
        np.testing.assert_allclose(p['mixture'],scalar_mix(p['p0'],p['positive_raw'],run.LEVELS,info['training_max']),atol=1e-12,rtol=1e-12)
        mixture_values+=p['mixture'].size
        y=ref['y'].ravel();ids=np.repeat(data['ids'],24);origins=np.repeat(data['origins'],24)
        errors={k:abs(p[k][:,7]-y) for k in ('direct','mixture')}
        errors.update(zero=abs(y),S=abs(ref['S'].ravel()-y),full_direct=sum(abs(ref[f'full_direct_{s}'].ravel()-y) for s in (42,43,44))/3)
        high=np.zeros(len(y),bool)
        for row in json.loads(paths['thresholds'].read_text())['thresholds']:
            if row['positive_p90'] is not None:high|=(ids==row['id'])&(y>0)&(y>=row['positive_p90'])
        start=pd.Timestamp(month+'-01');end=start+pd.offsets.MonthBegin()
        checked_metrics+=verify_scores(summary,y,p,errors,ids,origins,high,start.value,end.value)
        parts.append(dict(y=y,p=p,errors=errors,ids=ids,origins=origins,high=high))
        print(f'{month}: 31 models and monthly scores independently replayed',flush=True)
    combined={k:np.concatenate([d[k] for d in parts]) for k in ('y','ids','origins','high')}
    for key in ('p','errors'):
        combined[key]={k:np.concatenate([d[key][k] for d in parts]) for k in parts[0][key]}
    checked_metrics+=verify_scores(results['pooled'],**combined,start=pd.Timestamp('2013-08-01').value,end=pd.Timestamp('2013-11-01').value)
    assert models==93
    run.protect()
    for path,digest in initial.items():assert run.sha(out/path)==digest,path
    run.dump(out/'verification.json',dict(passed=True,models_reloaded=models,raw_predictions_replayed=raw_values,
        scalar_mixture_quantiles=mixture_values,group_entity_metrics=checked_metrics,
        interval_recalculations=80,input_files_unchanged=len(lock['files']),output_hashes=initial,
        note='Independent arithmetic path by same executor; not independent data or external replication'))
    print('PASS: source hashes, model reloads, scalar mixture, masked scores and 80 intervals',flush=True)


if __name__=='__main__':main()
