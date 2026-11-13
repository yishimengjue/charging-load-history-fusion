"""Train-only LCL adapter with origin-frozen memory and forward tree anchors."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.global_local_forecast_v1 import CONFIG
from src.global_shared_july_v1 import uniform_memory
from src.historical_memory_v1 import DAY_NS
from src.optimise_missing_features_v1 import history_eligible
from src.optimise_structure_features_v1 import aligned_features

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "data/lcl_meter_followup_2026-11-09"
OUT = ROOT / "artifacts/lcl_forward_inputs_v1"
CUTOFF = pd.Timestamp("2013-08-01").value
STEP = 14*DAY_NS


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def windows(values, times):
    slots = np.arange(168, len(times)-24, 24)
    h = values[slots[:, None]+np.arange(-167, 1)]
    y = values[slots[:, None]+np.arange(1, 25)]
    origins = times[slots].astype("datetime64[ns]").astype(np.int64)
    assert (origins+DAY_NS < CUTOFF).all()
    return h, y, origins


def prepare_entity(values, times, identifier):
    h, y, clock = windows(values, times)
    keep = history_eligible(h) & np.isfinite(y).any(1)
    indices = np.flatnonzero(keep)
    ids = np.full(len(h), identifier)
    memories, max_times, counts = [], [], []
    for q in indices:
        summary, records = uniform_memory(q, h, y, ids, clock)
        max_time = max((r["candidate_ns"]+DAY_NS for r in records), default=-1)
        assert max_time <= clock[q]
        memories.append(summary)
        max_times.append(max_time)
        counts.append(len(records))
    history, target, origins = h[keep], y[keep], clock[keep]
    base, _ = aligned_features(history)
    days = pd.DatetimeIndex(origins).dayofweek.to_numpy()
    weekdays = np.repeat(np.eye(7)[days], 24, axis=0)
    c = np.column_stack([base.to_numpy(), weekdays]).reshape(-1, 24, 180)
    memory = np.asarray(memories).reshape(-1, 24, 27)
    raw = np.concatenate([c, memory], -1)
    return dict(history=history, y=target, origins=origins,
                ids=np.full(len(origins), identifier), memory=memory, C=c, S=raw,
                memory_max_label_time=np.array(max_times, np.int64),
                memory_memberships=np.array(counts, np.int64))


def block_starts(origins):
    first = CUTOFF-((CUTOFF-int(origins.min())+STEP-1)//STEP)*STEP
    return range(first, CUTOFF, STEP)


def block_masks(origins, cutoff):
    return origins+DAY_NS <= cutoff, (origins >= cutoff) & (origins < cutoff+STEP)


def freeze():
    acceptance = json.loads((UP / "acceptance.json").read_text())
    assert acceptance["passed"]
    for name, digest in acceptance["files"].items():
        assert sha(ROOT / name) == digest, name
    raw = ROOT / "data/independent_source_intake_2026-11-09/lcl_ev_load_profiles.zip"
    source = json.loads((UP / "training_grid_summary.json").read_text())
    assert sha(raw) == source["source_sha256"]
    OUT.mkdir(exist_ok=False)
    for name in ("prepared", "forward_models"):
        (OUT / name).mkdir()
    files = [raw, UP / "acceptance.json", UP / "training_grid.npz", UP / "training_grid_summary.json",
             Path(__file__), ROOT / "experiments/lcl_forward_inputs_v1.md",
             ROOT / "tests/test_lcl_forward_inputs_v1.py", ROOT / "diagnostics/verify_lcl_forward_inputs_v1.py"]
    files += [ROOT / "src" / (name+".py") for name in
              ("historical_memory_v1", "global_shared_july_v1", "global_local_forecast_v1",
               "optimise_structure_features_v1", "optimise_missing_features_v1")]
    dump(OUT / "freeze.json", dict(config=dict(CONFIG, n_estimators=300), cutoff_ns=CUTOFF,
         block_days=14, min_complete_days=30, source_clock="naive", unit="source_raw",
         test_values_read=False, files={str(p.relative_to(ROOT)): sha(p) for p in files}))


def protect():
    for name, digest in json.loads((OUT / "freeze.json").read_text())["files"].items():
        assert sha(ROOT / name) == digest, name


def prepare():
    protect()
    summary = json.loads((UP / "training_grid_summary.json").read_text())
    with np.load(UP / "training_grid.npz", allow_pickle=False) as f:
        values, times, members = f["values"], f["times"], f["members"]
    status = {r["member"]: r["record_status"] for r in summary["series"]}
    mapping, parts, logs = [], [], []
    t0 = time.perf_counter()
    selected = [i for i, member in enumerate(members) if status[member] in ("nonnegative_active", "zero_only")]
    assert len(selected) == 48
    for identifier, row in enumerate(selected):
        member = str(members[row])
        path = OUT / "prepared" / f"entity_{identifier:02d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as f:
                data = {k: f[k] for k in f.files}
            entity_logs = json.loads(path.with_suffix(".json").read_text())
        else:
            data = prepare_entity(values[row], times, identifier)
            anchor = np.full_like(data["y"], np.nan)
            cutoffs = np.full(len(anchor), -1, np.int64)
            entity_logs = []
            for start in block_starts(data["origins"]):
                tr, qe = block_masks(data["origins"], start)
                complete = int(np.isfinite(data["y"][tr]).all(1).sum())
                if not qe.any() or complete < 30:
                    continue
                observed = np.isfinite(data["y"][tr])
                model = lgb.LGBMRegressor(**CONFIG, n_estimators=300).fit(
                    data["S"][tr][observed], data["y"][tr][observed]).booster_
                model_path = OUT / "forward_models" / f"{identifier:02d}_{start}.txt"
                model.save_model(str(model_path))
                anchor[qe] = np.maximum(model.predict(data["S"][qe].reshape(-1, 207), num_threads=2), 0).reshape(-1, 24)
                cutoffs[qe] = start
                entity_logs.append(dict(id=identifier, cutoff_ns=int(start), complete_days=complete,
                    train_indices=np.flatnonzero(tr).tolist(), query_indices=np.flatnonzero(qe).tolist(),
                    labels=int(observed.sum()), latest_training_target=int((data["origins"][tr]+DAY_NS).max()),
                    model_path=str(model_path.relative_to(ROOT)), model_sha256=sha(model_path)))
            data.update(anchor=anchor, anchor_cutoff=cutoffs)
            np.savez_compressed(path, **data)
            dump(path.with_suffix(".json"), entity_logs)
        finite = data["y"][np.isfinite(data["y"])]
        scale = max(float(np.quantile(finite, .9)), 1.)
        mapping.append(dict(id=identifier, member=member, status=status[member], scale=scale,
            full_origins=len(data["origins"]), matched_origins=int(np.isfinite(data["anchor"]).all(1).sum())))
        logs += entity_logs
        parts.append(data)
        print(f"Prepared {identifier+1}/48: origins={len(data['origins'])}, forward_S={mapping[-1]['matched_origins']}, trees={len(entity_logs)}", flush=True)
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    matched = np.isfinite(merged["anchor"]).all(1)
    np.savez_compressed(OUT / "full_training.npz", **merged)
    np.savez_compressed(OUT / "matched_training.npz", **{k:v[matched] for k,v in merged.items()})
    dump(OUT / "entity_mapping.json", dict(selected=mapping, unsupported=[
        dict(member=str(m), status=status[m]) for m in members if status[m] not in ("nonnegative_active", "zero_only")]))
    dump(OUT / "forward_log.json", logs)
    dump(OUT / "preparation.json", dict(full_origins=len(matched), matched_origins=int(matched.sum()),
        full_labels=int(np.isfinite(merged["y"]).sum()), matched_labels=int(np.isfinite(merged["y"][matched]).sum()),
        forward_tree_fits=len(logs), elapsed_seconds=time.perf_counter()-t0, test_values_read=False,
        neural_models_trained=0, new_evaluation_mae=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "prepare"))
    args = parser.parse_args()
    freeze() if args.stage == "freeze" else prepare()
