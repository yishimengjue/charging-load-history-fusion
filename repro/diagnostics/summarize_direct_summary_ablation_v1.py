"""Score only completed ablation artifacts; independently check scalar losses."""
import calendar
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/direct_summary_ablation_v1'
SEEDS=(42,43,44)
ARMS=('N_with','N_without','F_with','F_without')
SCOPES={'op_Jan_Mar':('op',('2022-01','2022-02','2022-03')),
        'op_July':('op',('2022-07',)),
        'ams':('ams',('2019-11','2019-12','2020-01')),
        'lcl':('lcl',('2013-11','2013-12','2014-01'))}
PAIRS=(('N_without','N_with'),('F_without','F_with'),('N_with','F_with'),('N_without','F_without'))


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def groups(y,high):
    valid=np.isfinite(y)
    return dict(all=valid,zero=valid&(y==0),positive=valid&(y>0),
                ordinary=valid&(y>0)&~high,high=valid&high)


def errors(p):
    y=p['y'];out={'S':np.abs(p['S']-y)}
    for key in ARMS:out[key]=np.mean([np.abs(p[f'{key}_{s}']-y) for s in SEEDS],axis=0)
    return out


def scalar_mae(y,predictions):
    mask=np.isfinite(y)
    return math.fsum(math.fsum(abs(float(a)-float(b)) for a,b in zip(pred[mask],y[mask]))
                     /int(mask.sum()) for pred in predictions)/len(predictions)


def metrics(p):
    masks=groups(p['y'],p['high']);e=errors(p)
    result={g:{k:float(value[m].mean()) if m.any() else None for k,value in e.items()} for g,m in masks.items()}
    checked=0
    for key in ARMS:
        scalar=scalar_mae(p['y'],np.array([p[f'{key}_{s}'] for s in SEEDS]))
        np.testing.assert_allclose(scalar,result['all'][key],rtol=0,atol=1e-12);checked+=1
    pairs=[]
    for ref,candidate in PAIRS:
        gain=result['all'][ref]-result['all'][candidate]
        contributions={g:float(np.where(masks[g],e[ref]-e[candidate],0).sum()/masks['all'].sum())
                       for g in ('zero','ordinary','high')}
        np.testing.assert_allclose(sum(contributions.values()),gain,atol=1e-12)
        pairs.append(dict(reference=ref,candidate=candidate,gain=gain,
                          gain_percent=100*gain/result['all'][ref],contributions=contributions))
    return dict(metrics=result,counts={g:int(m.sum()) for g,m in masks.items()},pairs=pairs,
                seed_mae={k:[float(abs(p[f'{k}_{s}']-p['y'])[masks['all']].mean()) for s in SEEDS] for k in ARMS},
                scalar_checks=checked)


def interval(parts,months,ref,candidate,circular,block):
    rng=np.random.default_rng(20261113)
    numerator=np.zeros(2000);denominator=np.zeros(2000)
    for p,month in zip(parts,months):
        y=p['y'];e=errors(p);valid=np.isfinite(y)
        start=np.datetime64(month+'-01','ns').astype(np.int64)
        ndays=25 if month=='2022-07' else calendar.monthrange(int(month[:4]),int(month[5:]))[1]
        days=(p['origins']-start)//(86400*10**9)
        assert ((days>=0)&(days<ndays)).all()
        sums=np.array([np.nansum((e[ref]-e[candidate])[days==d]) for d in range(ndays)])
        counts=np.array([valid[days==d].sum() for d in range(ndays)])
        starts=rng.integers(0,ndays if circular else ndays-block+1,(2000,int(np.ceil(ndays/block))))
        indices=(starts[:,:,None]+np.arange(block)).reshape(2000,-1)[:,:ndays]
        if circular:indices%=ndays
        numerator+=sums[indices].sum(1);denominator+=counts[indices].sum(1)
    assert (denominator>0).all()
    return np.quantile(numerator/denominator,[.025,.975]).tolist()


def load_case(source,month):
    dest=OUT/source/month
    record=json.loads((dest/'acceptance.json').read_text());assert record['passed']
    for name,digest in record['files'].items():assert sha(ROOT/name)==digest,name
    with np.load(dest/'predictions.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
    for key in ARMS:
        for seed in SEEDS:
            assert np.isfinite(p[f'{key}_{seed}']).all()
    for seed in SEEDS:
        for suffix in ('with','without'):
            np.testing.assert_array_equal(p[f'F_{suffix}_{seed}'],.5*(p['S']+p[f'N_{suffix}_{seed}']))
    return p


def main():
    lock=json.loads((OUT/'freeze.json').read_text())
    for name,digest in lock['files'].items():assert sha(ROOT/name)==digest,name
    monthly={};scopes={};checks=0
    for scope,(source,months) in SCOPES.items():
        parts=[]
        for month in months:
            p=load_case(source,month);parts.append(p)
            result=metrics(p);checks+=result['scalar_checks']
            for pair in result['pairs']:
                pair['intervals']={f'{"circular" if c else "noncircular"}_{b}':interval([p],[month],pair['reference'],pair['candidate'],c,b)
                                   for c in (False,True) for b in (7,14)}
            monthly[f'{source}/{month}']=result
        combined={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        result=metrics(combined);checks+=result['scalar_checks']
        for pair in result['pairs']:
            pair['intervals']={f'{"circular" if c else "noncircular"}_{b}':interval(parts,months,pair['reference'],pair['candidate'],c,b)
                               for c in (False,True) for b in (7,14)}
        result['unit']={'op':'kW','ams':'kWh','lcl':'source_raw'}[source]
        scopes[scope]=result
    record=dict(scopes=scopes,monthly=monthly,new_fits=30,new_numerical_data=False,
                scalar_checks=checks,old_files_unchanged=len(lock['files']),retrospective=True,
                primary='N_without minus N_with; positive means summary lowers neural MAE',
                intervals='within-month paired date blocks,2000 draws,seed20261113')
    (OUT/'results.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    rows=['# 当前直接拼接网络的历史摘要消融','',
          '2026-11-13。固定输入、预算与种子，新增30组无摘要训练；已有有摘要模型逐值重放。',
          '以下均为已使用时期上的匹配组件比较，不新增数据或选模。','',
          '| 范围 | 单位 | 有摘要神经 | 无摘要神经 | 有摘要融合 | 无摘要融合 | 摘要对神经MAE降幅 |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name,result in scopes.items():
        values=result['metrics']['all'];gain=result['pairs'][0]['gain_percent']
        rows.append('| '+' | '.join([name,result['unit']]+[f'{values[k]:.6f}' for k in ('N_with','N_without','F_with','F_without')]+[f'{gain:+.2f}%'])+' |')
    rows+=['','正降幅表示有摘要误差较低；融合始终为同一统计树和相应神经分支各50%。',
           '神经结果为三个种子误差的均值，不是种子预测集成。区间、每月及负荷分组见results.json。',
           f'核验：{checks}项独立标量MAE复算通过，{len(lock["files"])}个冻结文件未改变。']
    (ROOT/'DIRECT_SUMMARY_ABLATION_RESULTS_2026-11-13.md').write_text('\n'.join(rows)+'\n')
    print(json.dumps({k:v['metrics']['all'] for k,v in scopes.items()},indent=2))


if __name__=='__main__':main()
