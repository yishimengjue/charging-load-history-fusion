"""Inspect trained branches on training inputs only; do not select checkpoints."""
import argparse
import json
import numpy as np
from safetensors.torch import load_file
from src import lcl_neural_training_v1 as n


def inspect(folder):
    info=json.loads((folder/'fit.json').read_text())
    arm=info['arm']; data=n.read(n.UP/('full_training.npz' if arm=='direct_full' else 'matched_training.npz'))
    scales={int(k):v for k,v in info['scales'].items()}
    model=n.make_model(arm=='residual_matched');model.load_state_dict(load_file(str(folder/'encoder.safetensors')))
    heads=n.make_heads(model.decoder);heads.load_state_dict(load_file(str(folder/'heads.safetensors')))
    e=n.encode(data,scales,arm=='residual_matched');t=n.tx.configure()
    base,h,_=n.cache(model,e);heads.eval()
    with t.no_grad():raw=(base+heads(t.from_numpy(h),t.from_numpy(e['ids'])).numpy())*e['scale'][:,None]
    assert np.isfinite(raw).all()
    positive=np.isfinite(data['y'])&(data['y']>0)
    record=dict(run=folder.name,training_only=True,test_values_read=False,
        training_output_positions=int(raw.size),
        raw_min=float(raw.min()),raw_max=float(raw.max()),
        raw_negative_fraction=float((raw<0).mean()),branch_zero_fraction=float((raw<=0).mean()),
        all_zero_branch=bool((raw<=0).all()),
        observed_training_labels=int(np.isfinite(data['y']).sum()),
        zero_training_labels=int((data['y']==0).sum()),
        positive_training_labels=int(positive.sum()),
        positive_labels_with_raw_negative=int((positive&(raw<0)).sum()),
        positive_label_negative_fraction=float((positive&(raw<0)).sum()/positive.sum()),
        parameters_finite=all(bool(t.isfinite(v).all()) for bank in (model,heads) for v in bank.state_dict().values()),
        clipped_training_objective=arm!='direct_full')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run');args=p.parse_args()
    folders=[n.OUT/args.run] if args.run else [n.OUT/f'{a}_{s}' for a in n.ARMS for s in n.SEEDS]
    records=[inspect(f) for f in folders]
    dest=n.OUT/('health_'+args.run+'.json' if args.run else 'training_health.json')
    n.dump(dest,records)
    print(json.dumps(records,indent=2),flush=True)
