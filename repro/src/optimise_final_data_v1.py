"""Explicit one-time final release adapter; old development loaders unchanged."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimise_numeric_data_v1 import (SnapshotWindows,TRAIN_START,DEVELOPMENT_END,FINAL_END,
    RELEASE_START,NUMERIC_STATUSES,_validate_selections,_utc_clock,parse_power,AUDIT_COLUMNS)


def claim_release(marker,output,metadata,*,confirmed):
    if not confirmed: raise ValueError('Explicit final release confirmation required')
    marker,output=Path(marker),Path(output)
    if output.exists(): raise FileExistsError('Fixed final output already exists')
    with marker.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    output.mkdir(exist_ok=False)


def release_final_rows(rows,masked,*,authorized):
    if not authorized: raise ValueError('Final numerical release not authorized')
    prepared=_validate_selections(masked)
    output=[dict(r) for r in masked]; selected={}; counts={name:0 for name in NUMERIC_STATUSES}
    for index,s in enumerate(prepared):
        if not DEVELOPMENT_END<s.boundary<=FINAL_END: continue
        original=masked[index]
        if original['numeric_status']!='withheld_uninspected' or original['value_kw'] is not None:
            raise ValueError('Expected locally sealed final records')
        if not s.record['time_only_eligible']:
            output[index]['numeric_status']='time_unsupported'; continue
        selected[s.record['source_row']]=(index,s)
    accessed=set(); total=0
    for row_number,row in enumerate(rows):
        total+=1
        if len(row)!=5: raise ValueError('Raw row width changed')
        if row_number not in selected: continue
        index,s=selected[row_number]
        if row[0]!=s.record['depot_id'] or _utc_clock(row[2],field='source clock')!=s.actual:
            raise ValueError('Selected final raw row identity mismatch')
        value,status=parse_power(row[1])
        output[index].update(value_kw=value,numeric_status=status)
        counts[status]+=1; accessed.add(row_number)
    if accessed!=set(selected): raise ValueError('Selected final source rows absent')
    for old,new,s in zip(masked,output,prepared):
        if not DEVELOPMENT_END<s.boundary<=FINAL_END and old!=new:
            raise ValueError('Release altered prior or later sealed data')
        for key in s.record:
            if old[key]!=new[key]: raise ValueError('Release altered time/source provenance')
    return output,{'raw_rows_tokenized':total,'numeric_cells_accessed':len(accessed),'numeric_status_counts':counts,
                    'prior_and_later_records_unchanged':True,'capacity_headroom_accessed':False,
                    'nonselected_power_accessed':False}


def build_extended_windows(records,start,end,depot_ids,*,allow_final):
    start,end=_utc_clock(start,field='start'),_utc_clock(end,field='end')
    limit=FINAL_END if allow_final else DEVELOPMENT_END
    if not TRAIN_START<=start<=end<=limit: raise ValueError('Window scope not authorized')
    if any(v.hour or v.minute or v.second or v.microsecond for v in (start,end)):
        raise ValueError('Origins must be UTC midnight')
    ids=list(depot_ids)
    if len(set(ids))!=len(ids) or any(not isinstance(d,str) or not d.isdigit() or str(int(d))!=d for d in ids):
        raise ValueError('Invalid or repeated series ID')
    prepared=_validate_selections(records); lookup={}
    for original,s in zip(records,prepared):
        # Permission precedes reading numerical value or quality of later rows.
        if not RELEASE_START<=s.boundary<=limit: continue
        status=original['numeric_status']; value=original['value_kw']
        if not s.record['time_only_eligible']:
            if status!='time_unsupported' or value is not None: raise ValueError('Invalid duplicate status')
        elif status not in NUMERIC_STATUSES:
            raise ValueError('Numerical record not yet released')
        if status=='valid':
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Invalid admitted power')
        elif value is not None: raise ValueError('Invalid/missing labels must be null')
        lookup[(s.record['depot_id'],pd.Timestamp(s.boundary))]=(s,value)
    calendar=pd.date_range(start,end,freq='D',inclusive='left')
    origins=pd.DatetimeIndex([o for d in ids for o in calendar],name='origin_utc')
    origin_ids=np.repeat(ids,len(calendar)).astype(str); n=len(origins)
    X,Y=np.full((n,168),np.nan),np.full((n,24),np.nan)
    hx,hy=np.full((n,168),-1,dtype=np.int64),np.full((n,24),-1,dtype=np.int64)
    tx,ty=np.full((n,168),np.datetime64('NaT'),dtype='datetime64[ns]'),np.full((n,24),np.datetime64('NaT'),dtype='datetime64[ns]')
    audit=[]
    for index,(depot,origin) in enumerate(zip(origin_ids,origins)):
        counts={}
        for name,offsets,values,source,times in (('history',range(-167,1),X,hx,tx),('target',range(1,25),Y,hy,ty)):
            time_slots=0
            for column,h in enumerate(offsets):
                selected=lookup.get((depot,origin+pd.Timedelta(hours=h)))
                if selected is None: continue
                s,value=selected
                if (name=='history' and s.actual>origin) or (name=='target' and s.actual<=origin):
                    raise ValueError('Source actual timestamp crosses forecast boundary')
                times[index,column]=np.datetime64(s.actual.replace(tzinfo=None),'ns')
                if s.record['source_row'] is not None: source[index,column]=s.record['source_row']
                time_slots+=int(s.record['time_only_eligible'])
                if value is not None: values[index,column]=float(value)
            counts[f'{name}_time_only_slots']=time_slots
            counts[f'{name}_valid_slots']=int(np.isfinite(values[index]).sum())
            counts[f'{name}_eligible']=bool(np.isfinite(values[index]).all())
        iso=lambda t:t.isoformat().replace('+00:00','Z')
        audit.append({'depot_id':depot,'origin_utc':iso(origin),'history_start_utc':iso(origin-pd.Timedelta(hours=167)),
                     'history_end_utc':iso(origin),'target_start_utc':iso(origin+pd.Timedelta(hours=1)),
                     'target_end_utc':iso(origin+pd.Timedelta(hours=24)),**counts,
                     'joint_eligible':counts['history_eligible'] and counts['target_eligible']})
    return SnapshotWindows(X,Y,origins,origin_ids,np.isfinite(X).all(axis=1),np.isfinite(Y).all(axis=1),
                           hx,hy,tx,ty,pd.DataFrame(audit,columns=AUDIT_COLUMNS))
