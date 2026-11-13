"""Origin-frozen final-test windows without any file-reading or release API.

The caller owns one-time truth release and source/protocol hash checks. This
module also accepts the original masked frame for historical preflight only.
History eligibility determines forecast support; target eligibility never does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ams_forecast_data import (
    AMSWindows, AUDIT_COLUMNS, AVAILABILITY_ASSUMPTION, COUNT_KINDS,
    REQUIRED_COLUMNS, _civil_timestamp, _row_time_validity,
)
from src.public_ams import WINDOWS, civil_time_kind, partition_for


CONTEXT = 168
HORIZON = 24
FIT_CUTOFFS = frozenset(pd.Timestamp(value) for value in (
    "2019-11-01", "2019-12-01", "2020-01-01",
))
VALUE_STATUSES = frozenset({"valid", "missing", "nonfinite", "nonnumeric", "negative"})
FINAL_AUDIT_COLUMNS = (
    *AUDIT_COLUMNS, "history_eligible", "target_eligible",
    "history_withheld_hours", "target_withheld_hours",
)


def _require(condition, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _validate_input(frame: pd.DataFrame) -> str:
    """Validate parser semantics without asserting that released truth is complete."""
    _require(isinstance(frame, pd.DataFrame), "Expected the parsed AMS DataFrame")
    _require(frame.columns.is_unique, "Duplicate input column labels are forbidden")
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    _require(not missing, f"Parsed AMS columns missing: {sorted(missing)}")
    for name in ("local_start", "local_end"):
        _require(pd.api.types.is_datetime64_dtype(frame[name].dtype)
                 and not isinstance(frame[name].dtype, pd.DatetimeTZDtype),
                 f"{name} must be an offset-free local-civil datetime column")
    for name in ("time_eligible", "label_eligible"):
        _require(pd.api.types.is_bool_dtype(frame[name]) and frame[name].notna().all(),
                 f"{name} must contain nonmissing booleans")
    _require(pd.api.types.is_integer_dtype(frame["source_row"])
             and not pd.api.types.is_bool_dtype(frame["source_row"])
             and frame["source_row"].notna().all()
             and not frame["source_row"].duplicated().any()
             and frame["source_row"].ge(2).all(),
             "source_row must preserve unique integer CSV row identifiers starting at 2")
    _require(pd.api.types.is_numeric_dtype(frame["ams_kwh"])
             and not pd.api.types.is_bool_dtype(frame["ams_kwh"])
             and not pd.api.types.is_complex_dtype(frame["ams_kwh"]),
             "ams_kwh must be real numeric values, not strings or booleans")
    expected_partitions = [partition_for(
        None if pd.isna(start) else start.to_pydatetime(),
        None if pd.isna(end) else end.to_pydatetime(),
    ) for start, end in zip(frame["local_start"], frame["local_end"], strict=True)]
    _require(frame["partition"].notna().all()
             and np.array_equal(frame["partition"].to_numpy(), expected_partitions),
             "Source partitions must match fixed timestamp boundaries without relabeling")
    _require(frame["target_status"].notna().all(), "target_status cannot be missing")
    for partition, group in frame.groupby("partition", sort=False):
        allowed = VALUE_STATUSES if partition in WINDOWS else frozenset({f"withheld_{partition}"})
        if partition == "final_test":
            allowed = allowed | {"withheld_final_test"}
        _require(group["target_status"].isin(allowed).all(),
                 f"Unexpected target status for partition {partition}")

    values = frame["ams_kwh"].to_numpy(float)
    valid = frame["target_status"].eq("valid").to_numpy()
    _require(np.isfinite(values[valid]).all() and (values[valid] >= 0).all(),
             "A valid target status requires a finite nonnegative measurement")
    _require(np.isnan(values[~valid]).all(),
             "Invalid or withheld target statuses must have NaN values, not filled observations")
    expected_labels = frame["time_eligible"].to_numpy(bool) & valid
    _require(np.array_equal(frame["label_eligible"].to_numpy(bool), expected_labels),
             "label_eligible must equal time_eligible AND valid target status")
    final_statuses = frame.loc[frame["partition"].eq("final_test"), "target_status"]
    if not len(final_statuses):
        return "no_final_rows"
    withheld = final_statuses.eq("withheld_final_test")
    if withheld.all():
        return "all_masked"
    return "mixed" if withheld.any() else "all_released"


def build_final_windows(
    frame: pd.DataFrame,
    start="2019-01-03",
    end="2020-02-01",
) -> AMSWindows:
    """Return every complete-history origin; invalid target hours remain NaN.

    X is lag168..lag1. Y is the next 24 native-hourly measurements, individually
    masked by exactly the frozen time/label rules. Duplicate or absent source
    keys have source-row ID -1; an invalid but unique source row retains its ID.
    Scoring eligibility is isfinite(Y).all(axis=1), independently of forecasting.
    """
    start, end = _civil_timestamp(start, midnight=True), _civil_timestamp(end, midnight=True)
    _require(pd.Timestamp(WINDOWS["development"][0]) <= start <= end <= pd.Timestamp(WINDOWS["final_test"][1]),
             "Origin range must lie inside the fixed January 2019 to January 2020 partitions")
    input_state = _validate_input(frame)
    time_valid, dst_rows = _row_time_validity(frame)
    values = frame["ams_kwh"].to_numpy(float)
    label_valid = (frame["label_eligible"].to_numpy(bool)
                   & frame["target_status"].eq("valid").to_numpy()
                   & frame["partition"].isin(WINDOWS).to_numpy()
                   & np.isfinite(values) & (values >= 0))
    withheld_rows = frame["target_status"].str.startswith("withheld_").to_numpy(bool)
    source_rows = frame["source_row"].to_numpy(np.int64)
    lookup: dict[pd.Timestamp, list[int]] = {}
    for position, stamp in enumerate(frame["local_start"]):
        if not pd.isna(stamp):
            lookup.setdefault(stamp, []).append(position)

    def inspect(stamps: pd.DatetimeIndex):
        indices = np.full(len(stamps), -1, dtype=np.int64)
        valid = np.zeros(len(stamps), dtype=bool)
        counts = {kind: 0 for kind in COUNT_KINDS}
        withheld = 0
        for position, stamp in enumerate(stamps):
            found = lookup.get(stamp, [])
            intrinsic_dst = any(civil_time_kind(point.to_pydatetime()) != "unambiguous"
                                for point in (stamp, stamp + pd.Timedelta(hours=1)))
            counts["dst"] += int(intrinsic_dst or any(dst_rows[row] for row in found))
            withheld += int(any(withheld_rows[row] for row in found))
            if not found:
                counts["missing"] += 1
                continue
            if len(found) != 1:
                counts["duplicate"] += 1
                continue
            row = found[0]
            indices[position] = row
            counts["time_invalid"] += int(not time_valid[row])
            counts["label_invalid"] += int(not label_valid[row])
            valid[position] = time_valid[row] and label_valid[row] and not intrinsic_dst
        return indices, valid, counts, withheld

    candidates = (pd.date_range(start, end, inclusive="left", freq="D")
                  if start < end else pd.DatetimeIndex([]))
    histories, targets, origins, history_times, target_times = [], [], [], [], []
    history_ids, target_ids, audit = [], [], []
    for origin in candidates:
        history_keys = pd.date_range(origin - pd.Timedelta(hours=CONTEXT), periods=CONTEXT, freq="h")
        target_keys = pd.date_range(origin, periods=HORIZON, freq="h")
        hidx, hvalid, hcounts, hwithheld = inspect(history_keys)
        tidx, tvalid, tcounts, twithheld = inspect(target_keys)
        history_eligible = bool(hvalid.all())
        target_eligible = bool(tvalid.all())
        counts = {f"{part}_{kind}_hours": value
                  for part, group in (("history", hcounts), ("target", tcounts))
                  for kind, value in group.items()}
        reasons = [name.removesuffix("_hours") for name, value in counts.items() if value]
        partition = partition_for(origin.to_pydatetime(), (origin + pd.Timedelta(hours=HORIZON)).to_pydatetime())
        audit.append(dict(
            origin=origin, history_start=history_keys[0], target_end=origin + pd.Timedelta(hours=HORIZON),
            partition=partition, eligible=history_eligible and target_eligible,
            history_eligible=history_eligible, target_eligible=target_eligible,
            reasons=";".join(reasons), history_withheld_hours=hwithheld,
            target_withheld_hours=twithheld, **counts,
        ))
        if not history_eligible:
            continue
        actual = np.full(HORIZON, np.nan, dtype=float)
        actual[tvalid] = values[tidx[tvalid]]
        target_row_ids = np.full(HORIZON, -1, dtype=np.int64)
        present = tidx >= 0
        target_row_ids[present] = source_rows[tidx[present]]
        histories.append(values[hidx])
        targets.append(actual)
        origins.append(origin)
        history_times.append(history_keys.to_numpy(dtype="datetime64[ns]"))
        target_times.append(target_keys.to_numpy(dtype="datetime64[ns]"))
        history_ids.append(source_rows[hidx])
        target_ids.append(target_row_ids)

    origin_audit = pd.DataFrame(audit, columns=FINAL_AUDIT_COLUMNS)
    n = len(origins)
    Y = np.asarray(targets, dtype=float).reshape(n, HORIZON)
    scoreable = int(np.isfinite(Y).all(axis=1).sum())
    qa = dict(
        requested_start_inclusive=start.isoformat(), requested_end_exclusive=end.isoformat(),
        candidate_origins=len(candidates), retained_origins=n, retained_forecast_origins=n,
        excluded_history_origins=len(candidates)-n, forecast_hours=n*HORIZON,
        fully_scoreable_origins=scoreable, fully_scoreable_target_hours=scoreable*HORIZON,
        forecast_origins_with_incomplete_targets=n-scoreable,
        context_hours=CONTEXT, horizon_hours=HORIZON,
        time_representation="offset-free local civil; no UTC/fold selected",
        availability_assumption=AVAILABILITY_ASSUMPTION,
        target_unit="kWh per native source hourly interval",
        final_test_input_state=input_state,
        final_test_targets_withheld=input_state in {"all_masked", "mixed"},
        final_test_release_performed_by_builder=False, forecasting_performed=False,
        missing_strategy="No filling: complete history permits forecasts; invalid target hours stay NaN.",
        support_audit_scope="All requested origins, including incomplete history or targets",
        label_count_note=("label_invalid means unusable, including withheld labels; neither NaN nor "
                          "withheld counts establish actual final-test missingness."),
        selection_note="Only complete target days can be scored jointly; forecasts do not depend on that mask.",
        exclusion_reason_origins={name.removesuffix("_hours"): int(origin_audit[name].gt(0).sum())
                                  for name in AUDIT_COLUMNS if name.endswith("_hours")},
        exclusion_counts_overlap=True,
    )
    return AMSWindows(
        X=np.asarray(histories, dtype=float).reshape(n, CONTEXT), Y=Y,
        origins=pd.DatetimeIndex(origins),
        history_starts=np.asarray(history_times, dtype="datetime64[ns]").reshape(n, CONTEXT),
        target_starts=np.asarray(target_times, dtype="datetime64[ns]").reshape(n, HORIZON),
        history_source_rows=np.asarray(history_ids, dtype=np.int64).reshape(n, CONTEXT),
        target_source_rows=np.asarray(target_ids, dtype=np.int64).reshape(n, HORIZON),
        origin_audit=origin_audit, qa=qa,
    )


def train_before(windows: AMSWindows, cutoff) -> np.ndarray:
    """Monthly refits may use only full measured target days already ended."""
    cutoff = _civil_timestamp(cutoff)
    _require(cutoff in FIT_CUTOFFS, "Fit cutoff must be November 1, December 1, or January 1")
    n = len(windows.origins)
    _require(windows.X.shape == (n, CONTEXT) and windows.Y.shape == (n, HORIZON),
             "Expected aligned 168-history/24-target windows")
    _require(np.isfinite(windows.X).all() and (windows.X >= 0).all(), "Training histories must remain complete")
    complete_truth = np.isfinite(windows.Y).all(axis=1) & (windows.Y >= 0).all(axis=1)
    return np.asarray(windows.target_ends <= cutoff, dtype=bool) & complete_truth
