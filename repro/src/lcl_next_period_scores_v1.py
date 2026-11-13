"""Fixed point-weighted new-period comparisons and explicit full calendars."""
import json
import numpy as np

from src import lcl_next_period_v1 as model
from src import lcl_monthly_scoring_v1 as prior

data=model.data
util=data.util


def groups(y,high):
    valid=np.isfinite(y)
    return dict(all=valid,zero=valid&(y==0),positive=valid&(y>0),
                ordinary_positive=valid&(y>0)&~high,high=valid&high,negative=valid&(y<0))


def loss_arrays(p):
    errors={name:abs(p[name]-p['y']) for name in model.TREE_METHODS}
    errors['F']=np.mean([abs(p[f'F_{s}']-p['y']) for s in data.SEEDS],axis=0)
    return errors


def compute(p,high,start,end):
    masks=groups(p['y'],high);errors=loss_arrays(p);valid=masks['all']
    metrics={g:{name:float(e[m].mean()) if m.any() else None for name,e in errors.items()} for g,m in masks.items()}
    day=((p['origins']-start)//data.DAY).astype(int);ndays=(end-start)//data.DAY
    assert ((day>=0)&(day<ndays)).all()
    counts=[int((valid&(day[:,None]==i)).sum()) for i in range(ndays)]
    sums={k:[float(np.where(valid&(day[:,None]==i),v,0).sum()) for i in range(ndays)] for k,v in errors.items()}
    pairs=[]
    if valid.any():
        for reference in model.TREE_METHODS:
            delta=np.array(sums[reference])-np.array(sums['F'])
            pairs.append(dict(reference=reference,candidate='F',gain=metrics['all'][reference]-metrics['all']['F'],
                intervals={f'{"circular" if circle else "noncircular"}_{block}':util.paired_interval(delta,counts,block,circle)
                           for circle in (False,True) for block in (7,14)}))
    entities=[]
    for i in range(48):
        mask=valid&(p['ids'][:,None]==i)
        entities.append(dict(id=i,issued=int((p['ids']==i).sum()*24),observed=int(mask.sum()),
            metrics={k:float(v[mask].mean()) if mask.any() else None for k,v in errors.items()}))
    seeds=[dict(seed=s,metrics={g:float(abs(p[f'F_{s}']-p['y'])[m].mean()) if m.any() else None
                               for g,m in masks.items()}) for s in data.SEEDS]
    primary=next((p for p in pairs if p['reference']=='SG'),None)
    return dict(metrics=metrics,counts={k:int(v.sum()) for k,v in masks.items()},entities=entities,
        seeds=seeds,pairs=pairs,daily_sums=sums,daily_counts=counts,start_ns=start,end_ns=end,
        issued=int(p['y'].size),unknown=int((~valid).sum()),calendar_origins=48*int(ndays),
        unit='source_raw',seed_aggregation='mean losses, not mean predictions',
        decision=dict(lower_than_SG=metrics['all']['F']<metrics['all']['SG'] if valid.any() else False,
            lower_than_all_references=all(metrics['all']['F']<metrics['all'][a] for a in model.TREE_METHODS) if valid.any() else False,
            all_primary_intervals_positive=all(v[0]>0 for v in primary['intervals'].values()) if primary else False))


def score_month(month):
    dest=data.folder(month)
    if data.completed(dest/'score_manifest.json',[dest/'scores.json',dest/'high.npz']):return
    data.check(json.loads((dest/'prediction_manifest.json').read_text())['files'])
    p=util.read(dest/'predictions.npz');threshold=json.loads((dest/'inputs/thresholds.json').read_text())
    high=prior.high_mask(p,threshold);np.savez_compressed(dest/'high.npz',high=high)
    a,b=data.bounds(month);result=compute(p,high,a.value,b.value)
    util.dump(dest/'scores.json',result)
    util.dump(dest/'score_manifest.json',dict(files=data.hashes([dest/'scores.json',dest/'high.npz'])))
    print(month+' MAE '+json.dumps(result['metrics']['all']),flush=True)


def aggregate():
    data.protect();dest=data.OUT/'aggregate'
    if data.completed(dest/'score_manifest.json',[dest]):return
    dest.mkdir(exist_ok=False)
    parts=[];high=[];monthly=[]
    for month in data.MONTHS:
        path=data.folder(month);acc=json.loads((path/'acceptance.json').read_text());assert acc['passed'];data.check(acc['files'])
        p=util.read(path/'predictions.npz');parts.append({k:v for k,v in p.items() if k in ('origins','ids','y')+model.TREE_METHODS+tuple(f'F_{s}' for s in data.SEEDS)})
        high.append(util.read(path/'high.npz')['high']);monthly.append(dict(month=month,scores=json.loads((path/'scores.json').read_text())))
    p={k:np.concatenate([v[k] for v in parts]) for k in parts[0]};h=np.concatenate(high)
    np.savez_compressed(dest/'predictions.npz',**p);np.savez_compressed(dest/'high.npz',high=h)
    result=compute(p,h,data.bounds(data.MONTHS[0])[0].value,data.bounds(data.MONTHS[-1])[1].value)
    result['monthly']=monthly;result['original_month_end_unknowns_preserved']=True
    util.dump(dest/'scores.json',result)
    util.dump(dest/'score_manifest.json',dict(files=data.hashes(list(dest.iterdir()))))
    print('POOLED MAE '+json.dumps(result['metrics']['all']),flush=True)
