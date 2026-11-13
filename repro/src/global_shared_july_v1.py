"""Exact uniform S memory and bounded continuation decision, not a new mechanism."""

import numpy as np

from src.historical_memory_v1 import DAY_NS, POOLS, candidate_selection, summarize_targets


def uniform_memory(query, histories, targets, ids, origins_ns):
    positions, ages, _, _, _, _ = candidate_selection(query, histories, ids, origins_ns)
    parts, records = [], []
    for days, weekday_only in POOLS:
        keep = ages <= days
        if weekday_only:
            keep &= (origins_ns[positions]//DAY_NS)%7 == (origins_ns[query]//DAY_NS)%7
        selected = positions[keep]
        if len(selected) and (origins_ns[selected]+DAY_NS > origins_ns[query]).any():
            raise ValueError("Immature memory target")
        parts.append(summarize_targets(targets[selected], np.zeros(len(selected)), ages[keep]))
        records.extend(dict(query=int(query), candidate=int(i), pool_days=days,
                            query_ns=int(origins_ns[query]), candidate_ns=int(origins_ns[i])) for i in selected)
    return np.concatenate(parts, axis=1), records


def continuation_decision(pairs):
    checks = []
    for baseline in ("S", "L350"):
        row = next(x for x in pairs if x["first"] == baseline and x["second"] == "G350")
        gain = row["gain_percent"]
        intervals = row["intervals"]
        positive = len(intervals) == 2 and all(x["available"] and x["ci_low"] > 0 for x in intervals)
        rmse_ok = row["candidate"]["rmse_kw"] <= 1.01*row["baseline"]["rmse_kw"]
        checks.append(dict(baseline=baseline, relative_mae_gain_percent=gain,
                           intervals_positive=positive, rmse_guard_passed=rmse_ok,
                           passed=gain >= 1 and positive and rmse_ok))
    passed = all(x["passed"] for x in checks)
    return dict(status="retain_shared_direct_candidate" if passed else "retain_local_direct_references",
                gate_passed=passed, checks=checks, algorithmic_novelty_established=False,
                independent_validation=False, all_ids_improve=False, automatic_tuning=False)
