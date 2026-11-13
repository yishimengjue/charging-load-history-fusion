"""One fixed TiRex-2 comparison on the already-seen July calendar."""

import json
import math
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import pandas as pd

from src.tirex2_adapter_v1 import (
    ROOT, MODEL, CODE_REVISION, MODEL_REVISION, BATCH_SIZE, OPTIONS, QUANTILES,
    load_pipeline, predict, validate_histories, sha, dump,
)


OUT = ROOT / "artifacts/tirex2_july_v1"
WINDOWS = ROOT / "artifacts/optimise_july_v1/windows.npz"
BASE = ROOT / "artifacts/rolling_history_screen_v1_july/predictions.parquet"
SHARED = ROOT / "artifacts/global_shared_july_v1/predictions.parquet"
JULY = ROOT / "artifacts/optimise_july_v1/predictions.parquet"
PREFLIGHT = ROOT / "artifacts/tirex2_preflight_v1/summary.json"
ARMS = ("C", "S", "Chronos_p50", "TiRex2_p50")
KEYS = ["depot_id", "origin_utc", "step"]
HOUR = 3600 * 10**9


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.any():
        return dict(points=0, mae_kw=None, rmse_kw=None, bias_kw=None,
                    underprediction_kw=None, underprediction_rate=None)
    error = p[valid] - y[valid]
    return dict(points=int(valid.sum()), mae_kw=float(np.abs(error).mean()),
                rmse_kw=float(np.sqrt(np.mean(error**2))), bias_kw=float(error.mean()),
                underprediction_kw=float(np.maximum(-error, 0).mean()),
                underprediction_rate=float(np.mean(error < 0)))


def gain(a, b):
    return 100 * (1 - b["mae_kw"] / a["mae_kw"]) if a["mae_kw"] else None


def paired_blocks(daily, length, repeats=2000):
    """Resample date blocks; sum losses and point counts before dividing."""
    n = len(daily)
    if not 1 <= length <= n:
        raise ValueError("Invalid block length")
    rng = np.random.default_rng(4200 + length)
    starts = rng.integers(0, n - length + 1, size=(repeats, math.ceil(n / length)))
    indices = (starts[:, :, None] + np.arange(length)).reshape(repeats, -1)[:, :n]
    counts = daily["points"].to_numpy()[indices].sum(axis=1)
    draws = daily["gain_sum"].to_numpy()[indices].sum(axis=1) / counts
    return dict(block_days=length, replicates=repeats, method="noncircular moving date blocks",
                gain_kw=float(daily.gain_sum.sum() / daily.points.sum()),
                low_kw=float(np.quantile(draws, .025)), high_kw=float(np.quantile(draws, .975))), draws


def prepare():
    frame = pd.read_parquet(BASE)
    frame["depot_id"] = frame.depot_id.astype(str)
    for path, names in ((SHARED, ["S", "history_complete"]), (JULY, ["Chronos_p50"])):
        other = pd.read_parquet(path)
        other["depot_id"] = other.depot_id.astype(str)
        frame = frame.merge(other[KEYS + names], on=KEYS, how="left", validate="one_to_one", sort=False)
    frame["TiRex2_p50"] = np.nan
    selected, rows = [], []
    with np.load(WINDOWS, allow_pickle=False) as w:
        x = w["histories"].copy()
        for i, (depot, origin_ns) in enumerate(zip(w["ids"], w["origin_ns"])):
            group = frame.loc[(frame.depot_id == depot) & (frame.origin_utc.astype("int64") == origin_ns)].sort_values("step")
            assert len(group) == 24
            np.testing.assert_array_equal(group.step, np.arange(1, 25))
            np.testing.assert_array_equal(group.actual_kw, w["truth"][i])
            np.testing.assert_array_equal(group.source_row, w["target_source_rows"][i])
            np.testing.assert_array_equal(group.target_boundary_utc.astype("int64"), origin_ns + HOUR * np.arange(1, 25))
            valid = np.isfinite(x[i])
            clock = w["history_actual_times"][i].astype("datetime64[ns]").astype("int64")
            boundary = origin_ns - HOUR * np.arange(167, -1, -1)
            assert ((clock[valid] <= boundary[valid]) & (clock[valid] > boundary[valid] - 300 * 10**9)).all()
            assert bool(valid.all()) == bool(group.history_complete.iloc[0])
            issued = group.C.notna()
            assert issued.all() or not issued.any()
            if issued.all():
                selected.append(i)
                rows.append(group.index.to_numpy())
        selected = np.asarray(selected)
        rows = np.asarray(rows)
        keys = dict(ids=w["ids"][selected].copy(), origin_ns=w["origin_ns"][selected].copy(),
                    window_index=selected, frame_rows=rows)
    assert len(frame) == 6600 and len(selected) == 217
    for arm in ARMS[:-1]:
        assert frame.loc[rows.ravel(), arm].notna().all()
    assert frame.loc[rows.ravel(), "actual_kw"].notna().sum() == 4775
    validate_histories(x[selected])
    return frame, x[selected], keys


def evaluate(frame):
    issued = frame.C.notna()
    assert np.array_equal(issued, frame.TiRex2_p50.notna())
    work = frame.loc[issued].copy()
    seen = work.actual_kw.notna()
    obs = work.loc[seen].copy()
    main = {a: metrics(obs.actual_kw, obs[a]) for a in ARMS}
    assert all(r["points"] == 4775 for r in main.values())
    assert abs(main["C"]["mae_kw"] - 3.5522) < .00005
    groups = []
    for axis in ("depot_id", "state", "step", "history_complete", "week"):
        for label, part in obs.groupby(axis, sort=True):
            m = {a: metrics(part.actual_kw, part[a]) for a in ARMS}
            groups.append(dict(axis=axis, label=str(label), models=m,
                               gain_vs_C_percent=gain(m["C"], m["TiRex2_p50"])))
    positive = obs.loc[obs.actual_kw > 0]
    m = {a: metrics(positive.actual_kw, positive[a]) for a in ARMS}
    groups.append(dict(axis="truth_positive", label="all", models=m,
                       gain_vs_C_percent=gain(m["C"], m["TiRex2_p50"])))
    leave_one_out = []
    for depot in sorted(obs.depot_id.unique(), key=int):
        part = obs.loc[obs.depot_id != depot]
        m = {a: metrics(part.actual_kw, part[a]) for a in ARMS}
        leave_one_out.append(dict(omitted_id=depot, models=m,
                                  gain_vs_C_percent=gain(m["C"], m["TiRex2_p50"])))
    dates = pd.date_range("2022-07-01", "2022-07-25", tz="UTC")
    daily = pd.DataFrame(index=dates)
    for a in ARMS:
        error = obs[a] - obs.actual_kw
        temp = pd.DataFrame(dict(date=obs.origin_utc, absolute=np.abs(error), square=error**2))
        daily[a + "_abs_sum"] = temp.groupby("date").absolute.sum().reindex(dates, fill_value=0)
    daily["points"] = obs.groupby("origin_utc").size().reindex(dates, fill_value=0)
    daily["gain_sum"] = daily.C_abs_sum - daily.TiRex2_p50_abs_sum
    daily["gain_kw"] = daily.gain_sum / daily.points
    daily.to_parquet(OUT / "daily_losses.parquet")
    intervals = []
    for length in (7, 14):
        result, draws = paired_blocks(daily, length)
        intervals.append(result)
        np.save(OUT / f"bootstrap_{length}.npy", draws)
    known_gain = float(daily.gain_sum.sum())
    absent = work.loc[~seen]
    radius = float(np.abs(absent.C - absent.TiRex2_p50).sum())
    bounds = dict(issued_points=len(work), missing_points=len(absent),
                  low_kw=(known_gain - radius) / len(work), high_kw=(known_gain + radius) / len(work),
                  interpretation="outer bounds for C absolute loss minus TiRex absolute loss, not confidence interval")
    zero = next(g for g in groups if g["axis"] == "state" and g["label"] == "zero")["models"]
    high = next(g for g in groups if g["axis"] == "state" and g["label"] == "high_train_p90")["models"]
    ids = [g for g in groups if g["axis"] == "depot_id"]
    wins = sum(g["gain_vs_C_percent"] > 0 for g in ids)
    checks = dict(mae_gain_at_least_3_percent=gain(main["C"], main["TiRex2_p50"]) >= 3,
                  majority_ids_improve=wins > len(ids) / 2,
                  excluding_id4_improves=next(r for r in leave_one_out if r["omitted_id"] == "4")["gain_vs_C_percent"] > 0,
                  rmse_not_worse=main["TiRex2_p50"]["rmse_kw"] <= main["C"]["rmse_kw"],
                  zero_mae_not_worse=zero["TiRex2_p50"]["mae_kw"] <= zero["C"]["mae_kw"],
                  high_underprediction_not_worse=high["TiRex2_p50"]["underprediction_kw"] <= high["C"]["underprediction_kw"])
    shares = []
    for depot, part in obs.groupby("depot_id", sort=True):
        delta = float((np.abs(part.C - part.actual_kw) - np.abs(part.TiRex2_p50 - part.actual_kw)).sum())
        shares.append(dict(depot_id=depot, points=len(part), gain_sum_kw=delta,
                           fraction_of_net_gain=delta / known_gain if known_gain else None))
    return dict(main=main, groups=groups, leave_one_id_out=leave_one_out,
                mae_gain_vs_C_percent=gain(main["C"], main["TiRex2_p50"]),
                mae_gain_vs_S_percent=gain(main["S"], main["TiRex2_p50"]),
                date_block_intervals=intervals, missing_truth_bounds=bounds,
                id_contributions=shares, improved_ids=wins, supported_ids=len(ids),
                dates_improved=int((daily.gain_sum > 0).sum()), date_count=len(daily),
                working_gate=checks, working_gate_passed=all(checks.values()),
                decision="retain_candidate_for_further_validation" if all(checks.values()) else "retain_C_stop_this_fixed_recipe",
                new_training_fits=0, independent_validation=False, pretraining_overlap_excluded=False)


def run():
    pre = json.loads(PREFLIGHT.read_text())
    if not pre["completed"]:
        raise ValueError("Synthetic preflight has not passed")
    started = json.loads(PREFLIGHT.with_name("started.json").read_text())
    assert started["adapter_sha256"] == sha(ROOT / "src/tirex2_adapter_v1.py")
    OUT.mkdir(exist_ok=False)
    protected = [WINDOWS, BASE, SHARED, JULY, PREFLIGHT, MODEL / "model.safetensors", MODEL / "config.json"]
    hashes = {str(p): sha(p) for p in protected}
    code = [Path(__file__), ROOT / "src/tirex2_adapter_v1.py", ROOT / "experiments/tirex2_july_v1.md"]
    dump(OUT / "started.json", dict(date="2026-10-26", model_revision=MODEL_REVISION,
         code_revision=CODE_REVISION, options=OPTIONS, batch_size=BATCH_SIZE, no_training=True,
         input_sha256=hashes, implementation_sha256={str(p): sha(p) for p in code},
         research_stage="already-seen exploration", minimum_gain_working_target_percent=3))
    frame, histories, keys = prepare()
    np.savez_compressed(OUT / "input_histories.npz", histories=histories, **keys)
    dump(OUT / "status.json", dict(status="predicting", total_curves=len(histories), completed_curves=0))
    model = load_pipeline()
    predictions, timings = [], []
    before = time.monotonic()
    for start in range(0, len(histories), BATCH_SIZE):
        begin = time.monotonic()
        predictions.append(predict(model, histories[start:start+BATCH_SIZE]))
        timings.append(time.monotonic() - begin)
        stop = min(start+BATCH_SIZE, len(histories))
        if stop % 20 == 0 or stop == len(histories):
            print(f"TiRex-2: {stop}/{len(histories)} curves, {time.monotonic()-before:.1f}s", flush=True)
            dump(OUT / "status.json", dict(status="predicting", completed_curves=stop, total_curves=len(histories)))
    raw = np.concatenate(predictions)
    seconds = time.monotonic() - before
    q = np.maximum(raw, 0.)
    np.savez_compressed(OUT / "raw_predictions.npz", raw_quantiles=raw, quantile_levels=QUANTILES, **keys)
    frame.loc[keys["frame_rows"].ravel(), "TiRex2_p50"] = q[:, QUANTILES.index(.5)].ravel()
    frame.to_parquet(OUT / "predictions.parquet", index=False)
    results = evaluate(frame)
    results["runtime"] = dict(inference_seconds=seconds, batch_seconds=timings,
                              peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              platform="macOS (ru_maxrss is bytes)", device="cpu", threads=2)
    results["output_checks"] = dict(raw_negative_quantiles=int((raw < 0).sum()),
         raw_negative_p50=int((raw[:, 4] < 0).sum()), adjacent_quantile_crossings=int((np.diff(raw, axis=1) < 0).sum()),
         candidate_origins=275, issued_origins=217, scored_points=4775, missing_truth=433,
         input_hashes_unchanged=all(sha(p) == expected for p, expected in hashes.items()))
    assert results["output_checks"]["input_hashes_unchanged"]
    dump(OUT / "results.json", results)
    dump(OUT / "status.json", dict(status="completed_pending_verification", new_training_fits=0,
         completed_curves=217, decision=results["decision"], mae_gain_percent=results["mae_gain_vs_C_percent"]))
    print(json.dumps(dict(main=results["main"], gate=results["working_gate"], decision=results["decision"]), indent=2), flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        if OUT.exists():
            dump(OUT / "failure.json", dict(traceback=traceback.format_exc()))
        raise
