"""Check finite prepared history/labels against the redistributed meter records.

This validates values on the frozen support; it does not reselect the support
or claim to reconstruct all historical feature caches from scratch.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from reproduce import ROOT, REPRO, PERIODS, verify, save
sys.path.insert(0,str(REPRO))
from src import direct_summary_ablation_v1 as ab

HOUR=3600*10**9


def raw_op():
    frame=pd.read_csv(REPRO/'data/public_optimise_time_v1/depot_load.csv',dtype=str)
    times=pd.to_datetime(frame.ev_load_timestamp,utc=True,format='mixed',errors='coerce')
    mask=times.notna()&(times>=pd.Timestamp('2021-06-24',tz='UTC'))&(times<=pd.Timestamp('2022-07-26',tz='UTC'))
    frame=frame.loc[mask].copy();times=times.loc[mask]
    frame['stamp']=times.astype('int64');frame['id']=frame.depot_id.astype(int)
    frame['boundary']=times.dt.ceil('h').astype('int64')
    frame['duplicate']=frame.duplicated(['id','stamp'],keep=False)
    frame=frame.sort_values('stamp').drop_duplicates(['id','boundary'],keep='last')
    v=pd.to_numeric(frame.ev_load,errors='coerce')
    valid=(frame.boundary-frame.stamp<300*10**9)&~frame.duplicate&np.isfinite(v)&(v>=0)
    return pd.Series(v.where(valid).to_numpy(),index=pd.MultiIndex.from_arrays([frame.id,frame.boundary]))


def raw_ams():
    from src.public_ams import load_ams,FILE_SHA256,FILE_BYTES,FILE_NAME
    frame,_=load_ams(REPRO/'data/public_ams_v1'/FILE_NAME,expected_sha256=FILE_SHA256,
        expected_bytes=FILE_BYTES,reveal_final_test=True)
    frame=frame.loc[frame.label_eligible]
    return pd.Series(frame.ams_kwh.to_numpy(),index=pd.MultiIndex.from_arrays([
        np.ones(len(frame),int),frame.local_start.astype('int64')]))


def raw_lcl():
    from diagnostics.intake_lcl_ev_metadata_v1 import parse_clock
    mapping=json.loads((REPRO/'artifacts/lcl_next_period_v1/2013-11/inputs/entity_mapping.json').read_text())['selected']
    parts=[]
    for item in mapping:
        frame=pd.read_csv(REPRO/'data/lcl_public_projection'/Path(item['member']).name,dtype=str,keep_default_na=False)
        times=pd.DatetimeIndex([parse_clock(s) for s in frame.ReadingDate])
        mask=(times.minute==0)&(times.second==0)
        values=pd.to_numeric(frame.loc[mask,'Active Power Total (Watts)'],errors='coerce')
        parts.append(pd.Series(values.to_numpy(),index=pd.MultiIndex.from_arrays([
            np.full(mask.sum(),item['id']),times[mask].asi8])))
    return pd.concat(parts)


def compare(grid,ids,origins,offsets,values):
    stamps=origins[:,None]+np.asarray(offsets)[None,:]*HOUR
    keys=pd.MultiIndex.from_arrays([np.repeat(np.asarray(ids,int),len(offsets)),stamps.ravel()])
    actual=grid.reindex(keys).to_numpy().reshape(values.shape)
    mask=np.isfinite(values)
    np.testing.assert_allclose(actual[mask],values[mask],atol=1e-12,rtol=0)
    return dict(finite_values_checked=int(mask.sum()),stored_missing=int((~mask).sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reproduced/raw-audit.json');args=p.parse_args()
    verify(('raw','data','models'))
    reports=[]
    for source,loader in [('op',raw_op),('ams',raw_ams),('lcl',raw_lcl)]:
        grid=loader()
        if not grid.index.is_unique:raise ValueError('Duplicate raw time keys')
        h_offsets=np.arange(-168,0) if source=='ams' else np.arange(-167,1)
        y_offsets=np.arange(24) if source=='ams' else np.arange(1,25)
        for month in PERIODS[source]:
            train,q,y,_,_,_,_=ab.case_inputs(source,month)
            report=dict(source=source,month=month)
            report['train_history']=compare(grid,train['ids'],train['origins'],h_offsets,train['history'])
            report['train_labels']=compare(grid,train['ids'],train['origins'],y_offsets,train['y'])
            report['query_history']=compare(grid,q['ids'],q['origins'],h_offsets,q['history'])
            report['observed_truth']=compare(grid,q['ids'],q['origins'],y_offsets,y)
            reports.append(report);print('Raw values checked',source,month,flush=True)
    save(args.output,dict(passed=True,scope='Finite values on frozen support; not full support or feature regeneration',cases=reports))


if __name__=='__main__':main()
