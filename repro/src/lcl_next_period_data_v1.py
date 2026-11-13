"""Bounded new-period records and unchanged direct-model training inputs."""
import json
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import lcl_monthly_followup_v1 as old
from src import lcl_zero_distribution_v1 as util

neural = old.neural
ROOT = neural.ROOT
OUT = ROOT/'artifacts/lcl_next_period_v1'
MONTHS = ('2013-11','2013-12','2014-01')
SEEDS = neural.SEEDS
DAY = util.DAY


def bounds(month):
    if month not in MONTHS:
        raise ValueError('Month outside authorized new period')
    start = pd.Timestamp(month+'-01')
    return start, start+pd.offsets.MonthBegin()


def folder(month):
    bounds(month)
    return OUT/month


def read_month(stream, month):
    start,end = bounds(month)
    return old.read_month(stream,start.to_pydatetime(),end.to_pydatetime())


def training_rows(data, cutoff):
    valid = np.isfinite(data['y'])
    keep = (data['origins']+DAY<cutoff)&valid.any(1)
    if (data['y'][keep][valid[keep]]<0).any():
        raise ValueError('Negative training labels: stop nonnegative task')
    return keep


def hashes(paths):
    return {str(p.relative_to(ROOT)):util.sha(p) for p in paths}


def check(files):
    for name,digest in files.items():
        if util.sha(ROOT/name)!=digest:
            raise ValueError('Changed frozen file: '+name)


def completed(manifest, outputs):
    if manifest.exists():
        check(json.loads(manifest.read_text())['files'])
        return True
    if any(p.exists() for p in outputs):
        raise FileExistsError('Incomplete stage retained: '+str(manifest.parent))
    return False


def protect(month=None):
    lock=json.loads((OUT/'freeze.json').read_text())
    assert lock['months']==list(MONTHS)
    check(lock['files'])
    if month and (folder(month)/'inputs/acceptance.json').exists():
        check(json.loads((folder(month)/'inputs/acceptance.json').read_text())['files'])
    return lock


def freeze():
    from src import lcl_next_period_v1 as run
    old.protect()
    for month in old.MONTHS:
        acc=json.loads((old.folder(month)/'acceptance.json').read_text())
        assert acc['passed'];old.check_files(acc['files'])
    paths=[old.august.ARCHIVE,old.august.forward.UP/'training_grid.npz',
           old.august.OUT/'august_grid.npz',neural.UP/'entity_mapping.json',
           old.OUT/'freeze.json',old.folder('2013-10')/'inputs/full_training.npz',
           old.folder('2013-10')/'inputs/entity_mapping.json',old.folder('2013-10')/'predictions.npz',
           old.folder('2013-10')/'queries.npz']
    for month in old.MONTHS:
        paths.extend(old.folder(month)/name for name in ('evaluation_grid.npz','release.json','acceptance.json'))
    for seed in SEEDS:
        paths.extend(old.folder('2013-10')/f'models/old/direct_full_{seed}'/name
                     for name in ('fit.json','encoder.safetensors','heads.safetensors','probe.npz'))
    paths.extend((ROOT/'src').glob('*.py'))
    paths.extend(ROOT/name for name in ('experiments/lcl_next_period_v1.md',
        'tests/test_lcl_next_period_v1.py','diagnostics/verify_lcl_next_period_v1.py',
        'diagnostics/verify_lcl_zero_distribution_v1.py','diagnostics/intake_lcl_ev_metadata_v1.py'))
    paths.extend(neural.tx.VENDOR/p for p in neural.tx.TX_FILES)
    assert util.sha(old.august.ARCHIVE)==old.august.EXPECTED_HASH
    OUT.mkdir(exist_ok=False)
    util.dump(OUT/'freeze.json',dict(months=MONTHS,seeds=SEEDS,methods=run.METHODS,
        neural_config=neural.CONFIG,shared_config=run.shared.CONFIG,distribution_config=util.CONFIG,
        files=hashes(sorted(set(paths))),new_values_opened=False,
        authorized_start='2013-11-01',authorized_end_exclusive='2014-02-01'))


def known_grid(month):
    bounds(month)
    grid=util.read(old.august.forward.UP/'training_grid.npz')
    members=grid['members'];parts=[grid['values']]
    sources=[old.august.OUT/'august_grid.npz']+[old.folder(m)/'evaluation_grid.npz' for m in old.MONTHS]
    for previous in MONTHS[:MONTHS.index(month)]:
        acc=json.loads((folder(previous)/'acceptance.json').read_text())
        assert acc['passed'];check(acc['files'])
        sources.append(folder(previous)/'evaluation_grid.npz')
    for path in sources:
        d=util.read(path);np.testing.assert_array_equal(d['members'],members);parts.append(d['values'])
    times=np.arange(grid['times'][0],np.datetime64(bounds(month)[0]),np.timedelta64(1,'h'))
    values=np.concatenate(parts,axis=1)
    assert values.shape==(55,len(times))
    return dict(values=values,times=times,members=members)


def prepare(month):
    protect();dest=folder(month);up=dest/'inputs'
    if (up/'acceptance.json').exists():
        protect(month);return
    up.mkdir(parents=True,exist_ok=False)
    grid=known_grid(month);cutoff=bounds(month)[0].value
    original=json.loads((neural.UP/'entity_mapping.json').read_text())
    old_train=util.read(old.folder('2013-10')/'inputs/full_training.npz')
    pieces=[];mapping=[];thresholds=[]
    for entity in original['selected']:
        identifier=entity['id'];i=list(grid['members']).index(entity['member'])
        q,_=old.august.queries(grid['values'][i],grid['times'],identifier,pd.Timestamp('2013-02-01').value,cutoff)
        keep=training_rows(q,cutoff);q={k:v[keep] for k,v in q.items()}
        prefix=old_train['ids']==identifier
        ix=pd.Index(q['origins']).get_indexer(old_train['origins'][prefix]);assert (ix>=0).all()
        for key in q:
            np.testing.assert_array_equal(q[key][ix],old_train[key][prefix])
        valid=np.isfinite(q['y']);labels=q['y'][valid];positive=labels[labels>0]
        complete=int(valid.all(1).sum());assert complete>=30
        mapping.append(dict(entity,scale=max(float(np.quantile(labels,.9)),1.),full_origins=len(keep.nonzero()[0]),matched_origins=None))
        thresholds.append(dict(id=identifier,positive_p90=float(np.quantile(positive,.9)) if len(positive) else None,
                               complete_days=complete,labels=len(labels)))
        # The direct encoder ignores anchor; only legacy training probes need it.
        q['anchor']=np.zeros_like(q['y'])
        pieces.append(q)
    merged={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    np.savez_compressed(up/'full_training.npz',**merged)
    util.dump(up/'entity_mapping.json',dict(selected=mapping,unsupported=original['unsupported']))
    util.dump(up/'thresholds.json',thresholds)
    util.dump(up/'acceptance.json',dict(passed=True,origins=len(merged['ids']),labels=int(np.isfinite(merged['y']).sum()),
        latest_training_target=int(merged['origins'].max()+DAY),cutoff=cutoff,prefix_entities=48,
        anchor_role='artificial zero used only by legacy training probes; not a training input',
        files=hashes(list(up.iterdir())),new_month_values_read=False))
    print(f'{month}: prepared {len(merged["ids"])} full-label origins',flush=True)


def release(month):
    protect(month);dest=folder(month)
    fit=json.loads((dest/'fit_acceptance.json').read_text());assert fit['passed'];check(fit['files'])
    if (dest/'release_acceptance.json').exists():
        check(json.loads((dest/'release_acceptance.json').read_text())['files']);return
    start,end=bounds(month);grid=known_grid(month)
    mapping=json.loads((dest/'inputs/entity_mapping.json').read_text())
    util.dump(dest/'release.json',dict(month=month,start=start.isoformat(),end_exclusive=end.isoformat(),
        freeze_sha256=util.sha(OUT/'freeze.json'),fit_sha256=util.sha(dest/'fit_acceptance.json'),
        selected_members=[e['member'] for e in mapping['selected']],access_attempted=True))
    values=np.full((55,int((end-start).total_seconds()//3600)),np.nan)
    with zipfile.ZipFile(old.august.ARCHIVE) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read('EV Residential trials.zip'))) as inner:
            for e in mapping['selected']:
                i=list(grid['members']).index(e['member'])
                with inner.open(e['member']) as stream:
                    values[i]=read_month(stream,month)
    np.savez_compressed(dest/'evaluation_grid.npz',values=values,members=grid['members'])
    full=np.concatenate([grid['values'],values,np.full((55,1),np.nan)],axis=1)
    times=np.arange(grid['times'][0],np.datetime64(end)+np.timedelta64(1,'h'),np.timedelta64(1,'h'))
    parts=[];coverage=[]
    for e in mapping['selected']:
        i=list(grid['members']).index(e['member'])
        q,c=old.august.queries(full[i],times,e['id'],start.value,end.value)
        c.update(observed_targets=int(np.isfinite(q['y']).sum()),negative_targets=int((q['y']<0).sum()),
                 finite_exact_hour_records=int(np.isfinite(values[i]).sum()))
        parts.append(q);coverage.append(c)
    merged={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(dest/'queries.npz',**merged)
    util.dump(dest/'coverage.json',dict(selected=coverage,unsupported=mapping['unsupported'],
        unsupported_members_not_read=True,unit='source_raw',outside_month_values_converted=False))
    util.dump(dest/'release_acceptance.json',dict(passed=True,files=hashes([dest/n for n in
        ('release.json','evaluation_grid.npz','queries.npz','coverage.json')])))
    print(f'{month}: released {len(merged["ids"])} issued origins, {np.isfinite(merged["y"]).sum()} observed targets',flush=True)
