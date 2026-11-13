"""Portable entry points for the published, already-scored research periods."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT/'repro'
PERIODS = {'op':('2022-01','2022-02','2022-03','2022-07'),
           'ams':('2019-11','2019-12','2020-01'), 'lcl':('2013-11','2013-12','2014-01')}
SEEDS = (42,43,44)


def verify(groups=('data',)):
    manifest = json.loads((ROOT/'reproduction-manifest.json').read_text())
    for item in manifest['code']:
        with (ROOT/item['path']).open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=item['public_sha256']:raise ValueError('Changed experiment source: '+item['path'])
    checked = 0
    for item in manifest['files']:
        if item['package'] not in groups:continue
        path = ROOT/item['path']
        if not path.exists():raise FileNotFoundError('Run fetch_reproduction.py first: '+item['path'])
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=item['sha256']:raise ValueError('Changed input: '+item['path'])
        checked+=1
    print('Verified', checked, 'input files', flush=True)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def scores(out):
    import numpy as np
    from diagnostics import summarize_direct_summary_ablation_v1 as score
    expected = json.loads((ROOT/'research/summary-ablation-results.json').read_text())
    results={}
    for scope,(source,months) in score.SCOPES.items():
        parts=[]
        for month in months:
            with np.load(REPRO/f'artifacts/direct_summary_ablation_v1/{source}/{month}/predictions.npz',allow_pickle=False) as z:
                parts.append({k:z[k] for k in z.files})
        combined={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        result=score.metrics(combined)
        for group,arms in result['metrics'].items():
            for arm,value in arms.items():
                np.testing.assert_allclose(value,expected['scopes'][scope]['metrics'][group][arm],rtol=0,atol=1e-12)
        for pair,reference in zip(result['pairs'],expected['scopes'][scope]['pairs']):
            pair['intervals']={f'{"circular" if circular else "noncircular"}_{block}':score.interval(parts,months,pair['reference'],pair['candidate'],circular,block)
                               for circular in (False,True) for block in (7,14)}
            for name,value in pair['intervals'].items():
                np.testing.assert_allclose(value,reference['intervals'][name],rtol=0,atol=1e-12)
        results[scope]=result
        print(scope,result['metrics']['all'],flush=True)
    save(out/'scores.json',dict(passed=True,scopes=results))


def replay(out, source=None, month=None):
    import numpy as np
    from src import direct_summary_ablation_v1 as ab
    rows=[]
    cases=[(s,m) for s,months in PERIODS.items() for m in months if (source is None or source==s) and (month is None or month==m)]
    if not cases:raise ValueError('No approved cases')
    for s,m in cases:
        train,q,y,tree,high,scales,expected=ab.case_inputs(s,m)
        old=ab.reference_folder
        with np.load(REPRO/f'artifacts/direct_summary_ablation_v1/{s}/{m}/predictions.npz',allow_pickle=False) as z:
            for removed in (False,True):
                for seed in SEEDS:
                    folder=old(s,m,seed) if not removed else REPRO/f'artifacts/direct_summary_ablation_v1/{s}/{m}/models'/(f'direct_full_{seed}' if s=='lcl' else str(seed))
                    model,head,scale=ab.load_models(s,folder)
                    prediction=ab.neural_prediction(ab.adapter(s,removed),s,model,head,q,scale)
                    ref=z[f'N_{"without" if removed else "with"}_{seed}']
                    maximum=float(np.max(abs(prediction-ref)))
                    np.testing.assert_allclose(prediction,ref,rtol=1e-5,atol=5e-5)
                    rows.append(dict(source=s,month=m,removed=removed,seed=seed,max_abs=maximum,exact=bool(np.array_equal(prediction,ref))))
        print('Replayed',s,m,flush=True)
    save(out/'replay.json',dict(passed=True,tolerance=dict(rtol=1e-5,atol=5e-5),models=len(rows),cases=rows))


def train_case(out,source,month,seed,removed,input_dir=None):
    from src import direct_summary_ablation_v1 as ab
    import numpy as np
    train,q,y,s,high,scales,expected=ab.case_inputs(source,month)
    if input_dir:
        train=ab.read(input_dir/source/f'{month}_train.npz')
        q=ab.read(input_dir/source/f'{month}_query.npz')
        if source=='op':
            # Only replace the disk cache lookup; the model and encoded values are unchanged.
            ab.op.m.memory_for=lambda data:data['memory']
    adapter=ab.adapter(source,removed)
    folder=out/'model'
    if folder.exists():raise FileExistsError('Use a new output directory')
    folder.parent.mkdir(parents=True,exist_ok=True)
    if source=='op':
        model,info=adapter.fit(train,seed,folder/'encoder')
        actual=adapter.cache(model,train,info['scales']);empty=adapter.cache(model,train,info['scales'],zero=True)
        ab.op.a.fit_stage(actual,seed,.05,folder/'head',model.decoder,empty)
    elif source=='ams':adapter.fit(train,seed,'concat',folder)
    else:
        folder=out/f'direct_full_{seed}'
        up=ab.locations(source,month)[0]
        if input_dir:
            import shutil
            up=out/'rebuilt-training';up.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(up/'full_training.npz',**train)
            shutil.copyfile(ab.locations(source,month)[0]/'entity_mapping.json',up/'entity_mapping.json')
        fit=ab.clone_function(ab.lcl.fit,UP=up,OUT=out,
            protect=lambda:verify(('data','models')),encode=adapter.encode,predict=adapter.predict)
        fit('direct_full',seed)
    model,head,scale=ab.load_models(source,folder)
    p=ab.neural_prediction(adapter,source,model,head,q,scale)
    mask=np.isfinite(y);value=float(abs(p[mask]-y[mask]).mean())
    np.savez_compressed(out/'prediction.npz',prediction=p,y=y,origins=q['origins'],ids=q['ids'])
    save(out/'training-result.json',dict(source=source,month=month,seed=seed,without_summary=removed,
        mae=value,epochs='80 encoder + 20 local output',input_stage='raw-rebuilt inputs' if input_dir else 'published prepared inputs'))
    print('Retrained MAE',value,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('verify','scores','replay','train'))
    parser.add_argument('--source',choices=PERIODS)
    parser.add_argument('--month')
    parser.add_argument('--seed',type=int,choices=SEEDS,default=42)
    parser.add_argument('--without-summary',action='store_true')
    parser.add_argument('--input-dir',type=Path,help='Use inputs regenerated by rebuild_inputs.py')
    parser.add_argument('--output',type=Path,default=ROOT/'reproduced')
    args=parser.parse_args()
    sys.path.insert(0,str(REPRO))
    verify(('data','models') if args.stage in ('replay','train') else ('data',))
    if args.stage=='scores':scores(args.output)
    elif args.stage=='replay':replay(args.output,args.source,args.month)
    elif args.stage=='train':
        if not args.source or args.month not in PERIODS[args.source]:parser.error('train requires an approved source and month')
        train_case(args.output,args.source,args.month,args.seed,args.without_summary,args.input_dir)


if __name__ == '__main__':main()
