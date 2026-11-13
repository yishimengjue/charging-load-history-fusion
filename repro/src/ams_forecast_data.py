"""Complete, origin-frozen local-civil windows from masked native AMS hours.

Only development/validation are supported. The source does not establish online
latency; this benchmark assumes each interval is available exactly at its end.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.public_ams import WINDOWS, civil_time_kind


AVAILABILITY_ASSUMPTION = (
    "Ideal interval-end availability: each measured hour is available at its "
    "local interval end; the source provides no online reporting-latency evidence."
)
REQUIRED_COLUMNS = (
    "source_row", "local_start", "local_end", "partition", "ams_kwh",
    "target_status", "time_eligible", "label_eligible",
)
COUNT_KINDS = ("missing", "duplicate", "time_invalid", "label_invalid", "dst")
AUDIT_COLUMNS = (
    "origin", "history_start", "target_end", "partition", "eligible", "reasons",
    *(f"{part}_{kind}_hours" for part in ("history", "target") for kind in COUNT_KINDS),
)


@dataclass(frozen=True)
class AMSWindows:
    X: np.ndarray
    Y: np.ndarray
    origins: pd.DatetimeIndex
    history_starts: np.ndarray
    target_starts: np.ndarray
    history_source_rows: np.ndarray
    target_source_rows: np.ndarray
    origin_audit: pd.DataFrame
    qa: dict

    @property
    def target_ends(self) -> pd.DatetimeIndex:
        return self.origins + pd.Timedelta(hours=self.Y.shape[1])

    def subset(self, mask: np.ndarray) -> "AMSWindows":
        """Select retained windows, without discarding the parent support audit."""
        mask = np.asarray(mask)
        if mask.dtype != np.dtype(bool) or mask.shape != (len(self.origins),):
            raise ValueError("subset requires a boolean mask with one entry per origin")
        return replace(
            self, X=self.X[mask], Y=self.Y[mask], origins=self.origins[mask],
            history_starts=self.history_starts[mask], target_starts=self.target_starts[mask],
            history_source_rows=self.history_source_rows[mask],
            target_source_rows=self.target_source_rows[mask],
            qa={**self.qa, "subset_retained_origins": int(mask.sum()),
                "support_audit_scope": "parent requested origin range"},
        )


def _civil_timestamp(value, *, midnight: bool = False) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None:
        raise ValueError("Use valid offset-free local-civil timestamps, not UTC/aware time")
    if midnight and stamp != stamp.normalize():
        raise ValueError("Requested origin bounds must be local midnight")
    return stamp


def _validate_frame(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Masked AMS columns missing: {sorted(missing)}")
    for name in ("local_start", "local_end"):
        if (not pd.api.types.is_datetime64_dtype(frame[name].dtype)
                or isinstance(frame[name].dtype, pd.DatetimeTZDtype)):
            raise ValueError(f"{name} must be an offset-free datetime column")
    for name in ("time_eligible", "label_eligible"):
        if not pd.api.types.is_bool_dtype(frame[name]) or frame[name].isna().any():
            raise ValueError(f"{name} must contain nonmissing booleans")
    if (not pd.api.types.is_numeric_dtype(frame["ams_kwh"])
            or pd.api.types.is_bool_dtype(frame["ams_kwh"])):
        raise ValueError("ams_kwh must be the numeric masked-ingestion column")
    if (not pd.api.types.is_integer_dtype(frame["source_row"])
            or frame["source_row"].isna().any()
            or frame["source_row"].duplicated().any()):
        raise ValueError("source_row must retain unique integer source identifiers")

    final_start, final_end = WINDOWS["final_test"]
    final = (frame["partition"].eq("final_test")
             | (frame["local_start"].ge(final_start) & frame["local_start"].lt(final_end)))
    if (frame.loc[final, "ams_kwh"].notna().any()
            or frame.loc[final, "label_eligible"].any()
            or not frame.loc[final, "target_status"].eq("withheld_final_test").all()
            or not frame.loc[final, "partition"].eq("final_test").all()):
        raise ValueError("Final-test input must remain withheld; revealed targets are forbidden")


def _row_time_validity(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    starts, ends = frame["local_start"], frame["local_end"]
    parsed = starts.notna() & ends.notna()
    aligned = starts.eq(starts.dt.floor("h")) & ends.eq(ends.dt.floor("h"))
    one_hour = (ends - starts).eq(pd.Timedelta(hours=1))
    duplicate = starts.duplicated(keep=False) & starts.notna()
    # Preserve input order. A lookup later must not silently repair a reversed source.
    bad_order = (starts.diff().lt(pd.Timedelta(0)) | starts.lt(ends.shift())
                 | frame["source_row"].diff().lt(0))
    affected_order = bad_order | bad_order.shift(-1, fill_value=False)
    kinds = {
        stamp: civil_time_kind(stamp.to_pydatetime())
        for stamp in pd.concat([starts, ends]).dropna().unique()
    }
    dst = starts.map(kinds).isin(["ambiguous", "nonexistent"])
    dst |= ends.map(kinds).isin(["ambiguous", "nonexistent"])
    valid = (frame["time_eligible"] & parsed & aligned & one_hour & ~duplicate
             & ~affected_order & ~dst)
    return valid.to_numpy(dtype=bool), dst.to_numpy(dtype=bool)


def build_windows(
    frame: pd.DataFrame,
    start,
    end,
    *,
    context: int = 168,
    horizon: int = 24,
) -> AMSWindows:
    """Build common complete-day support without sorting, imputing, or revealing.

    Origin bounds are inclusive/exclusive local midnights. X is ordered from
    lag-context through lag-1; Y covers origin through the end of that civil day.
    Every historical interval must end at or before origin. Missing hours,
    invalid endpoints, DST labels, duplicates and invalid labels exclude the
    entire origin for every model. Exclusion counts overlap across reasons.
    """
    start, end = _civil_timestamp(start, midnight=True), _civil_timestamp(end, midnight=True)
    if start > end:
        raise ValueError("Origin start must not be later than end")
    if start < pd.Timestamp(WINDOWS["development"][0]):
        raise ValueError("Origin request is outside the frozen development/validation windows")
    if end > pd.Timestamp(WINDOWS["final_test"][0]) or start >= pd.Timestamp(WINDOWS["final_test"][0]):
        raise ValueError("Final-test forecasting is forbidden in this development/validation module")
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        raise ValueError("context must be a positive number of historical hours")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon != 24:
        raise ValueError("This benchmark forecasts exactly one complete 24-hour civil day")
    _validate_frame(frame)

    time_valid, dst_rows = _row_time_validity(frame)
    values = frame["ams_kwh"].to_numpy(dtype=float)
    label_valid = (frame["label_eligible"].to_numpy(dtype=bool)
                   & frame["target_status"].eq("valid").to_numpy()
                   & frame["partition"].isin(["development", "validation"]).to_numpy()
                   & np.isfinite(values) & (values >= 0))
    source_rows = frame["source_row"].to_numpy(dtype=np.int64)
    lookup: dict[pd.Timestamp, list[int]] = {}
    for pos, stamp in enumerate(frame["local_start"]):
        if not pd.isna(stamp):
            lookup.setdefault(stamp, []).append(pos)

    def inspect(stamps: pd.DatetimeIndex):
        indices = []
        counts = {kind: 0 for kind in COUNT_KINDS}
        for stamp in stamps:
            found = lookup.get(stamp, [])
            intrinsic_dst = any(
                civil_time_kind(point.to_pydatetime()) != "unambiguous"
                for point in (stamp, stamp + pd.Timedelta(hours=1))
            )
            counts["dst"] += int(intrinsic_dst or any(dst_rows[pos] for pos in found))
            if not found:
                counts["missing"] += 1
                continue
            if len(found) != 1:
                counts["duplicate"] += 1
                continue
            pos = found[0]
            indices.append(pos)
            counts["time_invalid"] += int(not time_valid[pos])
            counts["label_invalid"] += int(not label_valid[pos])
        return np.asarray(indices, dtype=np.int64), counts

    candidates = (pd.date_range(start, end, freq="D", inclusive="left")
                  if start < end else pd.DatetimeIndex([]))
    histories, targets, origins = [], [], []
    history_times, target_times, history_ids, target_ids, audit = [], [], [], [], []
    for origin in candidates:
        hist_stamps = pd.date_range(origin - pd.Timedelta(hours=context), periods=context, freq="h")
        target_stamps = pd.date_range(origin, periods=horizon, freq="h")
        hist_indices, hist_counts = inspect(hist_stamps)
        target_indices, target_counts = inspect(target_stamps)
        counts = {f"{part}_{kind}_hours": value
                  for part, count in (("history", hist_counts), ("target", target_counts))
                  for kind, value in count.items()}
        reasons = [name.removesuffix("_hours") for name, value in counts.items() if value]
        eligible = not reasons
        partition = "development" if origin < pd.Timestamp(WINDOWS["validation"][0]) else "validation"
        audit.append({
            "origin": origin, "history_start": hist_stamps[0],
            "target_end": origin + pd.Timedelta(hours=horizon), "partition": partition,
            "eligible": eligible, "reasons": ";".join(reasons), **counts,
        })
        if not eligible:
            continue
        histories.append(values[hist_indices])
        targets.append(values[target_indices])
        origins.append(origin)
        history_times.append(hist_stamps.to_numpy(dtype="datetime64[ns]"))
        target_times.append(target_stamps.to_numpy(dtype="datetime64[ns]"))
        history_ids.append(source_rows[hist_indices])
        target_ids.append(source_rows[target_indices])

    origin_audit = pd.DataFrame(audit, columns=AUDIT_COLUMNS)
    retained = len(origins)
    qa = {
        "requested_start_inclusive": start.isoformat(), "requested_end_exclusive": end.isoformat(),
        "candidate_origins": len(candidates), "retained_origins": retained,
        "excluded_origins": len(candidates) - retained, "retained_target_hours": retained * horizon,
        "context_hours": context, "horizon_hours": horizon,
        "time_representation": "offset-free local civil; no UTC/fold selected",
        "availability_assumption": AVAILABILITY_ASSUMPTION,
        "missing_strategy": "complete context and target day only; no filling or interpolation",
        "selection_note": "Complete-case results apply only to retained days; exclusions may be nonrandom.",
        "target_unit": "kWh per native source hourly interval",
        "final_test_revealed": False, "final_test_scored": False,
        "exclusion_reason_origins": {
            name.removesuffix("_hours"): int(origin_audit[name].gt(0).sum())
            for name in AUDIT_COLUMNS if name.endswith("_hours")
        },
        "exclusion_counts_overlap": True,
    }
    return AMSWindows(
        X=np.asarray(histories, dtype=float).reshape(retained, context),
        Y=np.asarray(targets, dtype=float).reshape(retained, horizon),
        origins=pd.DatetimeIndex(origins),
        history_starts=np.asarray(history_times, dtype="datetime64[ns]").reshape(retained, context),
        target_starts=np.asarray(target_times, dtype="datetime64[ns]").reshape(retained, horizon),
        history_source_rows=np.asarray(history_ids, dtype=np.int64).reshape(retained, context),
        target_source_rows=np.asarray(target_ids, dtype=np.int64).reshape(retained, horizon),
        origin_audit=origin_audit, qa=qa,
    )


def eligible_train(windows: AMSWindows, cutoff) -> np.ndarray:
    """Only full target days already ended at the fit cutoff may train a model."""
    cutoff = _civil_timestamp(cutoff)
    if cutoff > pd.Timestamp(WINDOWS["final_test"][0]):
        raise ValueError("Training cutoff must not enter the final-test period")
    return np.asarray(windows.target_ends <= cutoff, dtype=bool)


def direct_feature_names(context: int = 168) -> list[str]:
    return [f"lag_{lag}" for lag in range(context, 0, -1)] + ["forecast_step"]


def flatten_direct_features(windows: AMSWindows) -> tuple[np.ndarray, np.ndarray]:
    """Repeat the same origin history for each target; step is zero-based."""
    count, horizon = windows.Y.shape
    X = np.column_stack([
        np.repeat(windows.X, horizon, axis=0),
        np.tile(np.arange(horizon, dtype=float), count),
    ])
    return X, windows.Y.reshape(-1).copy()


def seasonal_naive(windows: AMSWindows, period: int = 168) -> np.ndarray:
    """Previous day/week profiles using only the common origin-frozen history."""
    if period not in (24, 168) or windows.X.shape[1] < period:
        raise ValueError("period must be 24 or 168 hours and fit inside the common history")
    columns = windows.X.shape[1] - period + np.arange(windows.Y.shape[1]) % period
    return windows.X[:, columns].copy()
