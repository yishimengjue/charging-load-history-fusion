"""UTC time-only selection of actual depot snapshots near fixed hour boundaries.

This module neither reads files nor accepts power values. For boundary g, use
the latest observed timestamp in (g-300s, g], only if that timestamp is globally
unique within its depot. Preserve the actual timestamp and original source row.
No interpolation, value-based selection, or fallback from a duplicate latest
timestamp is allowed. Frozen v1 modules are not imported or modified.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
import re


UTC = timezone.utc
HOUR = timedelta(hours=1)
WINDOW = timedelta(minutes=5)
CONTEXT_POINTS = 168
TARGET_POINTS = 24


@dataclass
class _Latest:
    timestamp: datetime
    first_source_row: int
    count: int = 1


@dataclass
class _Depot:
    total_rows: int = 0
    blank_rows: int = 0
    first_source_row: int | None = None
    last_source_row: int | None = None
    first_clock: datetime | None = None
    last_clock: datetime | None = None
    minimum_clock: datetime | None = None
    maximum_clock: datetime | None = None
    backwards: int = 0
    equal: int = 0
    exact_minute_rows: int = 0
    exact_hour_rows: int = 0
    in_window_rows: int = 0
    seen: set[datetime] = field(default_factory=set)
    duplicated: set[datetime] = field(default_factory=set)
    slots: dict[datetime, _Latest] = field(default_factory=dict)
    monthly_raw: dict[str, dict] = field(default_factory=dict)

    def observe(self, source_row: int, clock: datetime | None) -> None:
        if self.first_source_row is None:
            self.first_source_row = source_row
        self.last_source_row = source_row
        self.total_rows += 1
        if clock is None:
            self.blank_rows += 1
            return
        if self.last_clock is None:
            self.first_clock = clock
        else:
            self.backwards += int(clock < self.last_clock)
            self.equal += int(clock == self.last_clock)
        self.last_clock = clock
        self.minimum_clock = clock if self.minimum_clock is None else min(clock, self.minimum_clock)
        self.maximum_clock = clock if self.maximum_clock is None else max(clock, self.maximum_clock)
        is_minute = clock.second == clock.microsecond == 0
        is_hour = is_minute and clock.minute == 0
        self.exact_minute_rows += int(is_minute)
        self.exact_hour_rows += int(is_hour)
        month = _month(clock)
        if month not in self.monthly_raw:
            self.monthly_raw[month] = dict(valid_timestamp_rows=0, unique_timestamps=0,
                                          exact_minute_rows=0, exact_hour_rows=0)
        counts = self.monthly_raw[month]
        counts["valid_timestamp_rows"] += 1
        counts["exact_minute_rows"] += int(is_minute)
        counts["exact_hour_rows"] += int(is_hour)
        if clock in self.seen:
            self.duplicated.add(clock)
        else:
            self.seen.add(clock)
            counts["unique_timestamps"] += 1

        boundary = clock.replace(minute=0, second=0, microsecond=0)
        if boundary != clock:
            boundary = _shift(boundary, 1)
        if boundary is None or not timedelta(0) <= boundary - clock < WINDOW:
            return
        self.in_window_rows += 1
        previous = self.slots.get(boundary)
        if previous is None or clock > previous.timestamp:
            self.slots[boundary] = _Latest(clock, source_row)
        elif clock == previous.timestamp:
            previous.count += 1


def _month(clock: datetime) -> str:
    return f"{clock.year:04d}-{clock.month:02d}"


def _iso(clock: datetime | None) -> str | None:
    return None if clock is None else clock.isoformat().replace("+00:00", "Z")


def _shift(clock: datetime, hours: int) -> datetime | None:
    try:
        return clock + hours * HOUR
    except OverflowError:
        return None


def _support(keys: list[datetime | None], slots: dict[datetime, _Latest], *, exact=False) -> dict:
    found = [slots.get(key) for key in keys]
    if exact:
        found = [point if point is not None and point.timestamp == key else None
                 for key, point in zip(keys, found)]
    missing = sum(point is None for point in found)
    duplicated = sum(point is not None and point.count > 1 for point in found)
    return dict(
        required_points=len(keys), present_points=len(keys) - missing,
        unique_points=len(keys) - missing - duplicated,
        missing_points=missing, duplicated_points=duplicated,
        time_only_eligible=missing == duplicated == 0,
    )


def _age_summary(values: list[float]) -> dict:
    return dict(count=len(values), minimum=None if not values else min(values),
                maximum=None if not values else max(values),
                mean=None if not values else sum(values) / len(values))


def audit_depot_utc_snapshots(rows: Iterable[tuple[int, str, datetime | None]]) -> dict:
    """Return fixed-window and exact-hour necessary time-only support audits.

Source IDs must be nonnegative integers in globally strictly increasing order;
gaps are allowed. Depot IDs are canonical nonnegative integer strings, including
"0". A clock is either None (blank) or an aware datetime whose explicit UTC
offset is zero. Naive and nonzero-offset clocks are rejected, not guessed.

Observed UTC dates, not selected-snapshot dates, define inclusive midnight
origin candidates separately for every depot. History has 168 slots o-167h..o;
targets have 24 slots o+1h..o+24h. A slot is usable only when its latest timestamp
in (g-300s,g] occurs exactly once globally in that depot. File timestamp order
is diagnostic only. All source rows, including blanks, retain their row IDs.
"""
    depots: dict[str, _Depot] = {}
    total_rows = 0
    first_source_row = previous_source_row = None
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 3:
            raise TypeError("Each row must be a (source_row, depot_id, UTC_datetime_or_None) tuple")
        source_row, depot_id, clock = row
        if not isinstance(source_row, int) or isinstance(source_row, bool):
            raise TypeError("source_row must be an integer, not a boolean")
        if source_row < 0 or (previous_source_row is not None and source_row <= previous_source_row):
            raise ValueError("source_row must be nonnegative and globally strictly increasing")
        if not isinstance(depot_id, str) or re.fullmatch(r"0|[1-9][0-9]*", depot_id) is None:
            raise ValueError("depot_id must be a canonical nonnegative integer string")
        if clock is not None:
            if not isinstance(clock, datetime):
                raise TypeError("clock must be an aware UTC datetime or None")
            if clock.tzinfo is None or clock.utcoffset() != timedelta(0):
                raise ValueError("clock must be aware with explicit UTC offset zero")
            clock = clock.astimezone(UTC)
        if first_source_row is None:
            first_source_row = source_row
        previous_source_row = source_row
        total_rows += 1
        if depot_id not in depots:
            depots[depot_id] = _Depot()
        depots[depot_id].observe(source_row, clock)

    per_depot, origins, selected = [], [], []
    all_ages = []
    for depot_id, depot in depots.items():
        depot_origins, monthly_support, ages = [], {}, []
        grid_start = grid_end = None
        if depot.minimum_clock is not None:
            first_date, last_date = depot.minimum_clock.date(), depot.maximum_clock.date()
            grid_start = datetime.combine(first_date, time.min, tzinfo=UTC)
            last_midnight = datetime.combine(last_date, time.min, tzinfo=UTC)
            grid_end = _shift(last_midnight, 24)
            if grid_end is None:
                grid_end = last_midnight + 23 * HOUR
            boundary = grid_start
            while True:
                month = _month(boundary)
                if month not in monthly_support:
                    monthly_support[month] = dict(grid_slots=0, populated_snapshot_slots=0,
                                                  complete_snapshot_slots=0, missing_window_slots=0,
                                                  duplicate_latest_slots=0, exact_hour_complete_slots=0)
                counts = monthly_support[month]
                counts["grid_slots"] += 1
                latest = depot.slots.get(boundary)
                if latest is None:
                    counts["missing_window_slots"] += 1
                else:
                    eligible = latest.count == 1
                    age = (boundary - latest.timestamp).total_seconds()
                    counts["populated_snapshot_slots"] += 1
                    counts["complete_snapshot_slots"] += int(eligible)
                    counts["duplicate_latest_slots"] += int(not eligible)
                    counts["exact_hour_complete_slots"] += int(eligible and latest.timestamp == boundary)
                    ages.append(age)
                    selected.append(dict(
                        depot_id=depot_id, boundary_utc=_iso(boundary),
                        latest_timestamp=_iso(latest.timestamp),
                        source_row=latest.first_source_row if eligible else None,
                        latest_duplicate_count=latest.count,
                        age_seconds=age, time_only_eligible=eligible,
                    ))
                if boundary == grid_end:
                    break
                boundary += HOUR

            candidate_date = first_date
            while True:
                origin = datetime.combine(candidate_date, time.min, tzinfo=UTC)
                history = [_shift(origin, offset) for offset in range(-CONTEXT_POINTS + 1, 1)]
                targets = [_shift(origin, offset) for offset in range(1, TARGET_POINTS + 1)]
                parts = dict(history=history, target=targets, window=history + targets)
                depot_origins.append(dict(
                    depot_id=depot_id, origin_utc=_iso(origin),
                    history_start_utc=_iso(history[0]), history_end_utc=_iso(origin),
                    target_start_utc=_iso(targets[0]), target_end_utc=_iso(targets[-1]),
                    **{name: _support(keys, depot.slots) for name, keys in parts.items()},
                    exact_hour={name: _support(keys, depot.slots, exact=True) for name, keys in parts.items()},
                ))
                if candidate_date == last_date:
                    break
                candidate_date += timedelta(days=1)

        origins.extend(depot_origins)
        all_ages.extend(ages)
        valid_rows = depot.total_rows - depot.blank_rows
        excess = valid_rows - len(depot.seen)
        slot_totals = {name: sum(counts[name] for counts in monthly_support.values()) for name in (
            "grid_slots", "populated_snapshot_slots", "complete_snapshot_slots", "missing_window_slots",
            "duplicate_latest_slots", "exact_hour_complete_slots",
        )}
        per_depot.append(dict(
            depot_id=depot_id, total_rows=depot.total_rows, valid_timestamp_rows=valid_rows,
            blank_timestamp_rows=depot.blank_rows, unique_timestamps=len(depot.seen),
            first_source_row=depot.first_source_row, last_source_row=depot.last_source_row,
            first_timestamp_utc_in_source_order=_iso(depot.first_clock),
            last_timestamp_utc_in_source_order=_iso(depot.last_clock),
            minimum_timestamp_utc=_iso(depot.minimum_clock), maximum_timestamp_utc=_iso(depot.maximum_clock),
            backwards_transitions=depot.backwards, equal_transitions=depot.equal,
            distinct_duplicated_timestamps=len(depot.duplicated), duplicate_excess_timestamp_rows=excess,
            rows_at_duplicated_timestamps=excess + len(depot.duplicated),
            exact_minute_rows=depot.exact_minute_rows, exact_hour_rows=depot.exact_hour_rows,
            in_selection_window_rows=depot.in_window_rows,
            outside_selection_window_rows=valid_rows - depot.in_window_rows,
            reporting_grid_start_utc=_iso(grid_start), reporting_grid_end_utc=_iso(grid_end),
            **slot_totals,
            age_seconds=_age_summary(ages), candidate_origins=len(depot_origins),
            history_time_only_eligible_origins=sum(row["history"]["time_only_eligible"] for row in depot_origins),
            target_time_only_eligible_origins=sum(row["target"]["time_only_eligible"] for row in depot_origins),
            time_only_eligible_origins=sum(row["window"]["time_only_eligible"] for row in depot_origins),
            exact_hour_time_only_eligible_origins=sum(row["exact_hour"]["window"]["time_only_eligible"] for row in depot_origins),
            monthly_raw_timestamp_counts=[dict(month_utc=month, **counts)
                                          for month, counts in sorted(depot.monthly_raw.items())],
            monthly_snapshot_support=[dict(month_utc=month, **counts)
                                      for month, counts in sorted(monthly_support.items())],
        ))

    totals = {name: sum(item[name] for item in per_depot) for name in (
        "valid_timestamp_rows", "blank_timestamp_rows", "backwards_transitions", "equal_transitions",
        "distinct_duplicated_timestamps", "duplicate_excess_timestamp_rows", "rows_at_duplicated_timestamps",
        "exact_minute_rows", "exact_hour_rows", "in_selection_window_rows", "outside_selection_window_rows",
        "grid_slots", "populated_snapshot_slots", "complete_snapshot_slots", "missing_window_slots",
        "duplicate_latest_slots", "exact_hour_complete_slots", "candidate_origins",
        "history_time_only_eligible_origins", "target_time_only_eligible_origins", "time_only_eligible_origins",
        "exact_hour_time_only_eligible_origins",
    )}
    return dict(
        protocol=dict(
            version="optimise_utc_snapshot_selection_v2",
            target_kind="Actual published depot power snapshot selected near a boundary; not interval mean or energy",
            time_basis="UTC from explicit zero-offset aware input; no timezone inference",
            source_row_contract="Nonnegative zero-based integers, globally strictly increasing; gaps allowed",
            depot_id_contract="Canonical nonnegative integer strings including 0; no automatic ID merging",
            blank_rule="None clock counted, never assigned a date or value; blank-only depots retained",
            selection_window="(boundary_utc - 300 seconds, boundary_utc]",
            selection_age_rule="0 <= age_seconds < 300",
            selection_rule="Latest actual timestamp in the fixed window, globally unique within that depot",
            duplicate_latest_rule="Reject slot if latest timestamp repeats; do not fall back to an earlier unique time",
            latest_duplicate_count_definition="Number of records at the latest timestamp, including the first; 1 is unique",
            operation="Fixed-window observation selection/downsampling; no value-based selection or imputation",
            boundary_note="Boundary is a slot label, not a claim that a measurement occurred exactly at that time",
            origin_rule="UTC midnight on every raw-valid observed UTC date from minimum through maximum, inclusive",
            reporting_grid_rule="First raw-valid UTC date midnight through midnight after last raw-valid date, inclusive",
            reporting_grid_limit="At datetime maximum, stop at the last representable hour boundary",
            selected_rows_rule="One row for every populated boundary window; missing windows create no observation row",
            history_definition="168 slots from o-167 UTC hours through o inclusive",
            target_definition="24 slots from o+1 UTC hour through o+24 UTC hours inclusive",
            context_points=CONTEXT_POINTS, target_points=TARGET_POINTS, joint_points=CONTEXT_POINTS + TARGET_POINTS,
            exact_hour_comparison="Independent requirement of unique raw timestamps exactly at the same UTC slot boundaries",
            file_order_role="Timestamp order diagnostic only; observation choice uses clock keys, not file order",
            age_summary_scope="Latest timestamps in all populated slots, including duplicate-latest slots; time metadata only",
            raw_month_basis="Actual UTC observation timestamp; blanks have no assigned month",
            support_month_basis="UTC slot-boundary month; may differ from the actual sample month",
            limitation=("Eligibility is only a necessary time-support upper bound; it does not establish power validity, "
                        "normalization, unit correctness, aggregation quality, forecast-time availability, or label completeness"),
            value_fields_inspected=False, training_performed=False, timezone_inferred=False,
        ),
        summary=dict(total_rows=total_rows, depot_count=len(depots), first_source_row=first_source_row,
                     last_source_row=previous_source_row,
                     blank_only_depots=sum(item["valid_timestamp_rows"] == 0 for item in per_depot),
                     unique_depot_timestamps=sum(item["unique_timestamps"] for item in per_depot),
                     age_seconds=_age_summary(all_ages), **totals),
        per_depot=per_depot, origin_audit=origins, selected_snapshots=selected,
    )
