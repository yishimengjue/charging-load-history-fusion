"""Frozen numerical admission of v2-selected Optimise Prime kW snapshots.

Only time-eligible source rows inside the declared release can expose ev_load.
The module does not inspect capacity/headroom, infer missing targets, select a
replacement reading, or create final-period numerical windows. Hash/freeze and
artifact persistence are the caller's responsibility.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from numbers import Real
from os import PathLike
from pathlib import Path
import re
from typing import TextIO

import numpy as np
import pandas as pd


UTC = timezone.utc
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
TRAIN_START = datetime(2021, 7, 1, tzinfo=UTC)
TRAIN_END = datetime(2022, 1, 1, tzinfo=UTC)
DEVELOPMENT_END = datetime(2022, 4, 1, tzinfo=UTC)
FINAL_END = datetime(2022, 7, 1, tzinfo=UTC)
HISTORY_POINTS = 168
TARGET_POINTS = 24
RELEASE_START = TRAIN_START - (HISTORY_POINTS - 1) * HOUR
RELEASE_END = DEVELOPMENT_END
EXPECTED_COLUMNS = (
    "depot_id", "ev_load", "ev_load_timestamp", "capacity_kva", "final_headroom",
)
NUMERIC_STATUSES = ("valid", "missing_blank", "nonfinite", "nonnumeric", "negative")
SELECTION_FIELDS = (
    "depot_id", "boundary_utc", "latest_timestamp", "source_row",
    "latest_duplicate_count", "age_seconds", "time_only_eligible",
)
ID_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)")
CLOCK_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|\+00:00)"
)
DECIMAL_PATTERN = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)
NONFINITE_PATTERN = re.compile(r"[+-]?(?:nan|inf|infinity)", re.IGNORECASE)


@dataclass(frozen=True)
class SnapshotWindows:
    """All requested calendar origins, with NaN rather than imputed values.

Actual-time arrays have numpy datetime64[ns] dtype and refer to UTC, with NaT
for absent selections. Source rows use -1 for an absent or duplicate selection.
Provenance can remain present for a selected but numerically invalid reading.
"""

    X: np.ndarray
    Y: np.ndarray
    origins: pd.DatetimeIndex
    depot_ids: np.ndarray
    history_eligible: np.ndarray
    target_eligible: np.ndarray
    history_source_rows: np.ndarray
    target_source_rows: np.ndarray
    history_actual_times: np.ndarray
    target_actual_times: np.ndarray
    origin_audit: pd.DataFrame


@dataclass(frozen=True)
class _Selection:
    record: dict
    boundary: datetime
    actual: datetime
    authorized: bool


def _iso(clock: datetime) -> str:
    return clock.isoformat().replace("+00:00", "Z")


def _utc_clock(value, *, field: str) -> datetime:
    if isinstance(value, str):
        if CLOCK_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{field} must be an explicitly UTC timestamp")
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{field} has an invalid UTC calendar timestamp") from error
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be an aware UTC datetime or timestamp string")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must have explicit UTC offset zero")
    return value.astimezone(UTC)


def _canonical_id(value) -> bool:
    return isinstance(value, str) and ID_PATTERN.fullmatch(value) is not None


def _validate_selections(selected_metadata: Iterable[Mapping]) -> list[_Selection]:
    if isinstance(selected_metadata, Mapping):
        raise TypeError("Pass the selected_snapshots sequence, not its enclosing audit")
    prepared = []
    seen_slots, seen_sources = set(), set()
    for index, metadata in enumerate(selected_metadata):
        if not isinstance(metadata, Mapping):
            raise TypeError("Each selected snapshot must be a mapping")
        if not set(SELECTION_FIELDS).issubset(metadata):
            raise ValueError(f"Selected snapshot {index} lacks required provenance")
        record = {name: metadata[name] for name in SELECTION_FIELDS}
        depot_id = record["depot_id"]
        if not _canonical_id(depot_id):
            raise ValueError(f"Selected snapshot {index} has a noncanonical depot ID")
        if not isinstance(record["boundary_utc"], str) or not isinstance(record["latest_timestamp"], str):
            raise TypeError("Selection timestamps must be JSON-safe UTC strings")
        boundary = _utc_clock(record["boundary_utc"], field="boundary_utc")
        actual = _utc_clock(record["latest_timestamp"], field="latest_timestamp")
        if boundary.minute or boundary.second or boundary.microsecond:
            raise ValueError("A selected boundary must be an exact UTC hour")
        elapsed = (boundary - actual).total_seconds()
        if not 0 <= elapsed < 300:
            raise ValueError("A selected observation must be in (boundary-300s, boundary]")
        age = record["age_seconds"]
        if (isinstance(age, bool) or not isinstance(age, (int, float))
                or not math.isfinite(age) or age != elapsed):
            raise ValueError("age_seconds must exactly match the selected timestamp")
        count, eligible, source_row = (
            record["latest_duplicate_count"], record["time_only_eligible"], record["source_row"],
        )
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("latest_duplicate_count must be a positive integer")
        if not isinstance(eligible, bool) or eligible != (count == 1):
            raise ValueError("Time eligibility must equal uniqueness of the latest timestamp")
        if eligible:
            if isinstance(source_row, bool) or not isinstance(source_row, int) or source_row < 0:
                raise ValueError("A time-eligible selection needs its nonnegative integer source row")
            if source_row in seen_sources:
                raise ValueError("A raw source row cannot support multiple selected slots")
            seen_sources.add(source_row)
        elif source_row is not None:
            raise ValueError("A duplicated latest timestamp must have no selected source row")
        key = depot_id, boundary
        if key in seen_slots:
            raise ValueError("Selected depot/hour slots must be unique")
        seen_slots.add(key)
        prepared.append(_Selection(record, boundary, actual,
                                   eligible and RELEASE_START <= boundary <= RELEASE_END))
    return prepared


def parse_power(token: str) -> tuple[float | None, str]:
    """Admit only a finite, nonnegative ASCII decimal/scientific power token."""
    if not isinstance(token, str):
        raise TypeError("The selected power field must be a string")
    stripped = token.strip()
    if not stripped:
        return None, "missing_blank"
    if NONFINITE_PATTERN.fullmatch(stripped) is not None:
        return None, "nonfinite"
    if DECIMAL_PATTERN.fullmatch(stripped) is None:
        return None, "nonnumeric"
    value = float(stripped)
    if not math.isfinite(value):
        return None, "nonfinite"
    if value < 0:
        return None, "negative"
    return value, "valid"


def _empty_status_counts() -> dict[str, int]:
    return dict.fromkeys(NUMERIC_STATUSES, 0)


def intake_rows(
    rows: Iterable[Sequence[str]], selected_metadata: Iterable[Mapping],
    header: Sequence[str] = EXPECTED_COLUMNS,
) -> tuple[list[dict], dict]:
    """Read a tokenized CSV row stream, touching power only after authorization.

This seam permits row-index tripwire tests. ID/clock identity is checked for
every eligible selected row, even outside the numerical release. Other numeric
columns are never indexed, including for authorized rows.
"""
    if tuple(header) != EXPECTED_COLUMNS:
        raise ValueError("Expected the frozen ordered Table 27 source columns")
    prepared = _validate_selections(selected_metadata)
    records = []
    source_lookup = {}
    for selection in prepared:
        record = dict(selection.record)
        in_release = RELEASE_START <= selection.boundary <= RELEASE_END
        record.update(value_kw=None, numeric_status=("time_unsupported" if in_release
                                                    else "withheld_uninspected"))
        records.append(record)
        if record["time_only_eligible"]:
            source_lookup[record["source_row"]] = (selection, record)

    total_counts = _empty_status_counts()
    depot_counts, month_counts = {}, {}
    seen_sources = set()
    raw_rows = 0
    for source_row, row in enumerate(rows):
        raw_rows += 1
        if len(row) != len(header):
            raise ValueError(f"Column count mismatch at data row {source_row}")
        pair = source_lookup.get(source_row)
        if pair is None:
            continue
        selection, record = pair
        if row[0] != record["depot_id"]:
            raise ValueError(f"Selected depot identity mismatch at data row {source_row}")
        actual = _utc_clock(row[2], field=f"source timestamp at data row {source_row}")
        if actual != selection.actual:
            raise ValueError(f"Selected timestamp mismatch at data row {source_row}")
        seen_sources.add(source_row)
        if not selection.authorized:
            continue
        # Authorization precedes the only access to the source power column.
        value, status = parse_power(row[1])
        record.update(value_kw=value, numeric_status=status)
        total_counts[status] += 1
        depot = record["depot_id"]
        month = selection.boundary.strftime("%Y-%m")
        depot_counts.setdefault(depot, _empty_status_counts())[status] += 1
        month_counts.setdefault(month, _empty_status_counts())[status] += 1

    if seen_sources != set(source_lookup):
        raise ValueError("Selected source rows are absent from the raw stream")
    accessed = sum(total_counts.values())
    if accessed != sum(selection.authorized for selection in prepared):
        raise AssertionError("Each authorized selected power must be admitted exactly once")
    summary = dict(
        release_start_utc=_iso(RELEASE_START), release_end_utc=_iso(RELEASE_END),
        raw_rows_tokenized=raw_rows, selected_metadata_records=len(records),
        authorized_selected_rows=accessed, numeric_fields_accessed=accessed,
        numeric_status_counts=total_counts,
        per_depot=[dict(depot_id=depot, authorized_selected_rows=sum(counts.values()),
                        numeric_status_counts=counts)
                   for depot, counts in sorted(depot_counts.items(), key=lambda pair: int(pair[0]))],
        per_boundary_month=[dict(month_utc=month, authorized_selected_rows=sum(counts.values()),
                                 numeric_status_counts=counts)
                            for month, counts in sorted(month_counts.items())],
        outside_release_numeric_fields_accessed=False,
        nonselected_numeric_fields_accessed=False,
        capacity_or_headroom_fields_accessed=False,
    )
    return records, summary


def intake_csv(
    path_or_stream: str | PathLike | TextIO, selected_metadata: Iterable[Mapping],
) -> tuple[list[dict], dict]:
    """Admit selected power from a path or an already open text CSV stream."""
    if isinstance(path_or_stream, (str, PathLike)):
        with Path(path_or_stream).open(encoding="utf-8-sig", newline="") as stream:
            return intake_csv(stream, selected_metadata)
    reader = csv.reader(path_or_stream, strict=True)
    header = next(reader, None)
    if header is None:
        raise ValueError("The source CSV has no header")
    return intake_rows(reader, selected_metadata, header=header)


def _admitted_lookup(records: Iterable[Mapping]) -> dict[tuple[str, datetime], tuple]:
    supplied = list(records)
    prepared = _validate_selections(supplied)
    lookup = {}
    for original, selection in zip(supplied, prepared):
        if "numeric_status" not in original or "value_kw" not in original:
            raise ValueError("Admitted records require numeric_status and value_kw fields")
        status = original["numeric_status"]
        if not RELEASE_START <= selection.boundary <= RELEASE_END:
            if status != "withheld_uninspected":
                raise ValueError("Outside-release numeric status must remain withheld_uninspected")
            # Do not inspect even an artifact's power field outside the release.
            continue
        if not selection.authorized:
            if status != "time_unsupported":
                raise ValueError("Unsupported time selections cannot receive a numeric status")
            value = None
        else:
            if status not in NUMERIC_STATUSES:
                raise ValueError("Authorized selection has an unknown numerical admission status")
            value = original["value_kw"]
            if status == "valid":
                if (isinstance(value, bool) or not isinstance(value, Real)
                        or not math.isfinite(value) or value < 0):
                    raise ValueError("Valid admitted power must be finite and nonnegative")
                value = float(value)
            elif value is not None:
                raise ValueError("Invalid admitted power must be represented by null")
        key = selection.record["depot_id"], selection.boundary
        lookup[key] = selection, value
    return lookup


AUDIT_COLUMNS = (
    "depot_id", "origin_utc", "history_start_utc", "history_end_utc",
    "target_start_utc", "target_end_utc", "history_time_only_slots", "target_time_only_slots",
    "history_valid_slots", "target_valid_slots", "history_eligible", "target_eligible", "joint_eligible",
)


def build_windows(
    records: Iterable[Mapping], start: datetime | str, end: datetime | str,
    depot_ids: Iterable[str],
) -> SnapshotWindows:
    """Build every midnight origin in [start,end), never a final-period window.

Targets have endpoint labels (origin, origin+24h], so the final development
origin owns RELEASE_END itself. Histories end at the origin and may use the
history-only context before TRAIN_START. ID order is preserved; dates ascend
within each ID. Prediction eligibility depends only on the 168 history values.
"""
    start = _utc_clock(start, field="start")
    end = _utc_clock(end, field="end")
    if any(clock.hour or clock.minute or clock.second or clock.microsecond for clock in (start, end)):
        raise ValueError("Origin partition endpoints must be UTC midnights")
    if not TRAIN_START <= start <= end <= DEVELOPMENT_END:
        raise ValueError("Only frozen training/development origins may be built; final is withheld")
    if isinstance(depot_ids, str):
        raise TypeError("depot_ids must be a sequence, not a single string")
    identifiers = list(depot_ids)
    if any(not _canonical_id(value) for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("depot_ids must be unique canonical nonnegative integer strings")
    lookup = _admitted_lookup(records)
    calendar = [start + offset * DAY for offset in range((end - start).days)]
    origins = [origin for _ in identifiers for origin in calendar]
    origin_ids = [depot for depot in identifiers for _ in calendar]
    shape_x, shape_y = (len(origins), HISTORY_POINTS), (len(origins), TARGET_POINTS)
    X, Y = np.full(shape_x, np.nan), np.full(shape_y, np.nan)
    history_rows, target_rows = np.full(shape_x, -1, dtype=np.int64), np.full(shape_y, -1, dtype=np.int64)
    history_times = np.full(shape_x, np.datetime64("NaT"), dtype="datetime64[ns]")
    target_times = np.full(shape_y, np.datetime64("NaT"), dtype="datetime64[ns]")
    audit = []
    for row, (depot, origin) in enumerate(zip(origin_ids, origins)):
        parts = (
            (range(-HISTORY_POINTS + 1, 1), X, history_rows, history_times, "history"),
            (range(1, TARGET_POINTS + 1), Y, target_rows, target_times, "target"),
        )
        counts = {}
        for offsets, values, source_rows, actual_times, name in parts:
            time_slots = 0
            for column, offset in enumerate(offsets):
                boundary = origin + offset * HOUR
                admitted = lookup.get((depot, boundary))
                if admitted is None:
                    continue
                selection, value = admitted
                if name == "history" and selection.actual > origin:
                    raise AssertionError("A historical observation cannot follow its origin")
                if name == "target" and selection.actual <= origin:
                    raise AssertionError("A target observation must follow its origin")
                actual_times[row, column] = np.datetime64(selection.actual.replace(tzinfo=None), "ns")
                if selection.record["source_row"] is not None:
                    source_rows[row, column] = selection.record["source_row"]
                time_slots += int(selection.record["time_only_eligible"])
                if value is not None:
                    values[row, column] = value
            counts[f"{name}_time_only_slots"] = time_slots
            counts[f"{name}_valid_slots"] = int(np.isfinite(values[row]).sum())
            counts[f"{name}_eligible"] = bool(np.isfinite(values[row]).all())
        audit.append(dict(
            depot_id=depot, origin_utc=_iso(origin), history_start_utc=_iso(origin - 167 * HOUR),
            history_end_utc=_iso(origin), target_start_utc=_iso(origin + HOUR),
            target_end_utc=_iso(origin + 24 * HOUR), **counts,
            joint_eligible=counts["history_eligible"] and counts["target_eligible"],
        ))
    return SnapshotWindows(
        X=X, Y=Y, origins=pd.DatetimeIndex(origins, tz="UTC", name="origin_utc"),
        depot_ids=np.asarray(origin_ids, dtype=str), history_eligible=np.isfinite(X).all(axis=1),
        target_eligible=np.isfinite(Y).all(axis=1), history_source_rows=history_rows,
        target_source_rows=target_rows, history_actual_times=history_times,
        target_actual_times=target_times, origin_audit=pd.DataFrame(audit, columns=AUDIT_COLUMNS),
    )
