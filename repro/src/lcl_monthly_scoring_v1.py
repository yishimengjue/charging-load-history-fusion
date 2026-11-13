"""Point-weighted monthly and pooled scoring without changing model selection."""
import json
import numpy as np
import pandas as pd
from src import lcl_monthly_followup_v1 as n


def high_mask(data,thresholds):
    high=np.zeros_like(data['y'],bool)
    for row in thresholds:
        value=row['positive_p90']
        if value is not None:
            high |= (data['ids'][:,None]==row['id'])&(data['y']>0)&(data['y']>=value)
    return high


def compute(data,high,start,end):
    y=data['y'];valid=np.isfinite(y)
    groups=dict(all=valid,positive=valid&(y>0),ordinary_positive=valid&(y>0)&~high,
                high=valid&high,zero=valid&(y==0),negative=valid&(y<0))
    errors,signed,squared=n.august.loss_arrays(data)
    day=((data['origins']-start)//n.august.forward.DAY_NS).astype(int)
    days=int((end-start)//n.august.forward.DAY_NS)
    assert ((day>=0)&(day<days)).all()
    metrics=[];pairs=[];entities=[];daily=[];seeds=[]
    for group,mask in groups.items():
        count=int(mask.sum())
        for name in errors:
            metrics.append(dict(group=group,method=name,n=count,
                mae=float(errors[name][mask].mean()) if count else None,
                rmse=float(np.sqrt(squared[name][mask].mean())) if count else None,
                bias=float(signed[name][mask].mean()) if count else None))
        if not count:continue
        for before,after in n.august.PAIRS:
            delta=errors[before]-errors[after];gain=float(delta[mask].mean());base=float(errors[before][mask].mean())
            daily_gain=np.array([np.where(mask&(day[:,None]==d),delta,0).sum() for d in range(days)])
            daily_count=np.array([(mask&(day[:,None]==d)).sum() for d in range(days)])
            pairs.append(dict(group=group,before=before,after=after,n=count,gain=gain,
                gain_percent=100*gain/base if base else None,
                intervals=[n.august.interval(daily_gain,daily_count,k) for k in (7,14)]))
    for identifier in range(48):
        mask=valid&(data['ids'][:,None]==identifier)
        for name in errors:
            entities.append(dict(id=identifier,method=name,n=int(mask.sum()),
                mae=float(errors[name][mask].mean()) if mask.any() else None))
    for d in range(days):
        mask=valid&(day[:,None]==d)
        for name in errors:
            daily.append(dict(day=d+1,date=str(pd.Timestamp(start+d*n.august.forward.DAY_NS).date()),
                method=name,n=int(mask.sum()),absolute_error_sum=float(errors[name][mask].sum())))
    for family in n.FAMILIES:
        for seed in n.neural.SEEDS:
            for group,mask in groups.items():
                seeds.append(dict(method=family,seed=seed,group=group,n=int(mask.sum()),
                    mae=float(abs(data[f'{family}_{seed}']-y)[mask].mean()) if mask.any() else None))
    return dict(metrics=metrics,pairs=pairs,entities=entities,days=daily,seeds=seeds,
                start_ns=start,end_ns=end,day_count=days,unit='source_raw',
                seed_aggregation='mean loss, not ensemble forecast')


def score_month(month):
    n.protect(month);dest=n.folder(month)
    manifest=json.loads((dest/'prediction_manifest.json').read_text())
    assert n.neural.sha(dest/'predictions.npz')==manifest['predictions_sha256']
    data=n.neural.read(dest/'predictions.npz')
    high=high_mask(data,json.loads((dest/'preflight.json').read_text())['thresholds'])
    np.savez_compressed(dest/'scoring_masks.npz',high=high)
    start,end=n.bounds(month)
    result=compute(data,high,start.value,end.value)
    n.neural.dump(dest/'scores.json',result)
    print(month+' '+json.dumps([r for r in result['metrics'] if r['group']=='all']),flush=True)


def aggregate():
    n.protect();dest=n.OUT/'aggregate';dest.mkdir(exist_ok=True)
    sources=[n.august.OUT]+[n.folder(m) for m in n.MONTHS]
    data=[];high=[];monthly=[];hashes={}
    for path in sources:
        acc=json.loads((path/'acceptance.json').read_text());assert acc['passed'];n.check_files(acc['files'])
        d=n.neural.read(path/'predictions.npz');data.append(d)
        high.append(high_mask(d,json.loads((path/'preflight.json').read_text())['thresholds']))
        monthly.append(dict(path=str(path.relative_to(n.ROOT)),scores=json.loads((path/'scores.json').read_text())))
        hashes[str((path/'acceptance.json').relative_to(n.ROOT))]=n.neural.sha(path/'acceptance.json')
    merged={k:np.concatenate([d[k] for d in data]) for k in data[0]}
    combined_high=np.concatenate(high)
    np.savez_compressed(dest/'predictions.npz',**merged)
    np.savez_compressed(dest/'scoring_masks.npz',high=combined_high)
    scores=compute(merged,combined_high,n.august.A,pd.Timestamp('2013-11-01').value)
    scores.update(monthly=monthly,source_acceptance_hashes=hashes,
                  missing_month_end_targets='retained, not filled after later release')
    n.neural.dump(dest/'scores.json',scores)
    print('POOLED '+json.dumps([r for r in scores['metrics'] if r['group']=='all']),flush=True)
