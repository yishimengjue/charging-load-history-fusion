"""Rebuild neural inputs from raw records on the published frozen sample split."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from audit_raw_inputs import raw_op, raw_ams, raw_lcl, HOUR, ROOT, REPRO, ab
from reproduce import PERIODS, verify, save


def values(grid,identifier,origins,offsets):
    times=origins[:,None]+np.asarray(offsets)[None,:]*HOUR
    keys=pd.MultiIndex.from_arrays([np.full(times.size,int(identifier)),times.ravel()])
    return grid.reindex(keys).to_numpy().reshape(times.shape)


def rebuild(source,out):
    from src.historical_memory_v1 import eligible_history
    from src.global_shared_july_v1 import uniform_memory
    grid={'op':raw_op,'ams':raw_ams,'lcl':raw_lcl}[source]()
    start={'op':'2021-07-01','ams':'2019-01-01','lcl':'2013-02-08'}[source]
    end={'op':'2022-07-25','ams':'2020-01-31','lcl':'2014-01-31'}[source]
    clock=pd.date_range(start,end,freq='D').asi8
    ho=np.arange(-168,0) if source=='ams' else np.arange(-167,1)
    yo=np.arange(24) if source=='ams' else np.arange(1,25)
    rows={};cases={};need={}
    for month in PERIODS[source]:
        train,q,y,*_=ab.case_inputs(source,month)
        cases[month]=(train,q)
        for data in (train,q):
            for i,t in zip(data['ids'],data['origins']):need.setdefault(int(i),set()).add(int(t))
    for identifier,wanted in need.items():
        h=values(grid,identifier,clock,ho);y=values(grid,identifier,clock,yo)
        # The AMS library includes only complete-history and complete-target days.
        if source=='ams':
            admissible=np.isfinite(h).all(1)
            h,y,dates=h[admissible],y[admissible],clock[admissible]
        else:dates=clock
        ids=np.full(len(dates),identifier)
        for t in sorted(wanted):
            index=np.flatnonzero(dates==t)
            if len(index)!=1:raise ValueError('Missing frozen origin in raw reconstruction')
            ix=int(index[0])
            if source=='ams':
                refs=np.flatnonzero(np.isfinite(y).all(1)&(dates+24*HOUR<=t))
                xx=np.concatenate([h[refs],h[ix:ix+1]])
                yy=np.concatenate([y[refs],np.full((1,24),np.nan)])
                cc=np.r_[dates[refs],t]
                memory=ab.ams.pilot.m.pack_memory(len(refs),xx,yy,np.ones(len(cc),int),cc)[0]['S']
            elif source=='op':memory=ab.op.m.pack_memory(ix,h,y,ids,dates)[0]['S']
            else:memory=uniform_memory(ix,h,y,ids,dates)[0]
            rows[(identifier,t)]=(h[ix],y[ix],memory)
        print('Rebuilt raw history and memory',source,identifier,flush=True)
    report=[]
    for month,(train,q) in cases.items():
        for name,old in [('train',train),('query',q)]:
            selected=[rows[(int(i),int(t))] for i,t in zip(old['ids'],old['origins'])]
            history=np.stack([r[0] for r in selected]);target=np.stack([r[1] for r in selected]);memory=np.stack([r[2] for r in selected])
            np.testing.assert_allclose(history,old['history'],rtol=0,atol=1e-12,equal_nan=True)
            old_memory=ab.op.m.memory_for(old) if source=='op' else old['memory']
            np.testing.assert_allclose(memory,old_memory,rtol=0,atol=1e-12,equal_nan=True)
            if name=='train':np.testing.assert_allclose(target,old['y'],rtol=0,atol=1e-12,equal_nan=True)
            else:target=np.full_like(target,np.nan)
            dest=out/source/f'{month}_{name}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            # Other fields are frozen split/reference metadata, not newly selected inputs.
            data=dict(old,history=history,y=target,memory=memory)
            np.savez_compressed(dest,**data)
            report.append(dict(month=month,split=name,origins=len(history),history_values=history.size,memory_values=memory.size))
    save(out/source/'acceptance.json',dict(passed=True,source=source,from_raw=True,
         support='Published frozen origins, entity mapping and reference metadata',cases=report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',choices=PERIODS)
    parser.add_argument('--output',type=Path,default=ROOT/'reproduced-inputs');args=parser.parse_args()
    verify(('raw','data','models'))
    for source in ([args.source] if args.source else PERIODS):rebuild(source,args.output)
