"""Loss-only follow-up using the frozen trainer in an isolated namespace."""
import argparse
import hashlib
import json
from pathlib import Path
from types import FunctionType

import numpy as np
from safetensors.torch import load_file

from src import lcl_neural_training_v1 as old
from diagnostics.lcl_neural_health_v1 import inspect

OUT = old.ROOT / 'artifacts/lcl_training_projection_v1'
ARMS = old.ARMS[:2]


def loss(pr, pz, e, ix, arm):
    if arm not in ARMS:
        raise ValueError('Only matched-support arms are allowed')
    return old.direct.a.weighted_loss(
        pr, pz, e['y'][ix], e['mask'][ix], e['scale'][ix],
        e['valid_per_origin'], e['scale_reference'], .05)[0]


def protect():
    old.protect()
    lock = json.loads((OUT / 'freeze.json').read_text())
    assert lock['config'] == old.CONFIG
    for name, digest in lock['files'].items():
        assert old.sha(old.ROOT / name) == digest, name


def trainer():
    # Rebind only loss, destination and guard; never mutate the old module.
    namespace = dict(old.fit.__globals__, loss=loss, OUT=OUT, protect=protect)
    return FunctionType(old.fit.__code__, namespace, name='fit_projection')


def initial_check():
    t = old.tx.configure()
    records = []
    for arm in ARMS:
        for seed in old.SEEDS:
            t.manual_seed(seed)
            a = old.make_model(arm == 'residual_matched')
            t.manual_seed(seed)
            b = trainer().__globals__['make_model'](arm == 'residual_matched')
            digest = hashlib.sha256()
            for key, value in a.state_dict().items():
                assert t.equal(value, b.state_dict()[key]), key
                digest.update(key.encode())
                digest.update(value.detach().numpy().tobytes())
            records.append(dict(arm=arm, seed=seed, initial_parameters_sha256=digest.hexdigest()))
    return records


def freeze():
    old.protect()
    previous = json.loads((old.OUT / 'acceptance.json').read_text())
    assert previous['passed']
    for name, digest in previous['files'].items():
        assert old.sha(old.ROOT / name) == digest, name
    initial = initial_check()
    paths = [Path(__file__), old.ROOT / 'experiments/lcl_training_projection_v1.md',
             old.ROOT / 'tests/test_lcl_training_projection_v1.py',
             old.ROOT / 'diagnostics/lcl_neural_health_v1.py',
             old.OUT / 'freeze.json', old.OUT / 'acceptance.json']
    files = dict(previous['files'])
    files.update({str(p.relative_to(old.ROOT)): old.sha(p) for p in paths})
    OUT.mkdir(exist_ok=False)
    old.dump(OUT / 'freeze.json', dict(config=old.CONFIG, arms=ARMS, seeds=old.SEEDS,
        files=files, initial_checks=initial, training_projection=False,
        inference_projection=True, test_values_read=False))


def verify():
    protect()
    data = old.read(old.UP / 'matched_training.npz')
    records, health = [], []
    for arm in ARMS:
        for seed in old.SEEDS:
            folder = OUT / f'{arm}_{seed}'
            info = json.loads((folder / 'fit.json').read_text())
            previous = json.loads((old.OUT / folder.name / 'fit.json').read_text())
            for key in ('config', 'origins', 'labels', 'scales', 'latest_target_ns',
                        'encoder_parameters', 'head_parameters'):
                assert info[key] == previous[key], key
            assert not info['test_values_read']
            model = old.make_model(arm == 'residual_matched')
            model.load_state_dict(load_file(str(folder / 'encoder.safetensors')))
            heads = old.make_heads(model.decoder)
            heads.load_state_dict(load_file(str(folder / 'heads.safetensors')))
            probe = old.read(folder / 'probe.npz')
            q = {k: v[probe['indices']] for k, v in data.items()}
            q['y'] = np.full_like(q['y'], np.nan)
            predictions = old.predict(model, heads, q, {int(k): v for k, v in info['scales'].items()})
            for key, value in predictions.items():
                np.testing.assert_array_equal(value, probe[key])
                assert np.isfinite(value).all() and (value >= 0).all()
            row = inspect(folder)
            row['clipped_training_objective'] = False
            row['comparison_run'] = str((old.OUT / folder.name).relative_to(old.ROOT))
            health.append(row)
            records.append(info)
    old.dump(OUT / 'training_health.json', health)
    paths = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'acceptance.json']
    old.dump(OUT / 'acceptance.json', dict(passed=True, encoder_runs=6, head_runs=6,
        entity_probe_predictions=6*48*24, records=records, test_values_read=False,
        training_projection=False, inference_projection=True,
        files={str(p.relative_to(old.ROOT)): old.sha(p) for p in paths}))
    print(json.dumps(health, indent=2), flush=True)
    print('PASS: six matched fits; 6912 exact training-probe replays', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'train', 'verify'])
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'train':
        for arm in ARMS:
            for seed in old.SEEDS:
                trainer()(arm, seed)
    else:
        verify()
