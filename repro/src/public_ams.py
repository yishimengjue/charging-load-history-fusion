"""Source-pinned, local-civil ingestion of the Norwegian Dataset 5 AMS meter."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


DATASET_ID = "jbks2rcwyj"
DATASET_VERSION = 1
DATASET_DOI = "10.17632/jbks2rcwyj.1"
FILE_ID = "9f9820d3-300a-4b96-ad77-57876adf0349"
FILE_NAME = "Dataset 5_AMS data from garage Bl2.csv"
FILE_SHA256 = "8e4d0d0deae085dd21f148d1930ce57e55e392dc89f603ba82f076b06799729e"
FILE_BYTES = 500329
PROTOCOL_SHA256 = "144bd13d8321d6e52fde3129abc2317d8e4e93917e0c7b2692ee143adafcce0a"
METADATA_URL = "https://data.mendeley.com/public-api/datasets/jbks2rcwyj?version=1"
CATALOG_URL = (
    "https://data.mendeley.com/public-api/datasets/jbks2rcwyj/files?folder_id=root&version=1"
)
ARTICLE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8134705/fullTextXML"
SOURCE_FORMAT = "%d.%m.%Y %H:%M"
TIME_RE = re.compile(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}\Z")
NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?\Z")
REQUIRED_FIELDS = ("date_from", "date_to", "AMS_kWh")
OSLO = ZoneInfo("Europe/Oslo")
WINDOWS = {
    "development": (datetime(2019, 1, 1), datetime(2019, 9, 1)),
    "validation": (datetime(2019, 9, 1), datetime(2019, 11, 1)),
    "final_test": (datetime(2019, 11, 1), datetime(2020, 2, 1)),
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: str | Path, sha256: str, size: int) -> dict:
    path = Path(path)
    actual_size = path.stat().st_size
    actual_sha = file_sha256(path)
    if actual_size != size or actual_sha != sha256:
        raise ValueError(f"Source bytes or SHA256 mismatch: {path.name}")
    return {"path": str(path.resolve()), "bytes": actual_size, "sha256": actual_sha}


def verify_source_metadata(metadata: dict, catalog: list) -> dict:
    if metadata.get("id") != DATASET_ID or metadata.get("version") != DATASET_VERSION:
        raise ValueError("Dataset identity/version mismatch")
    if metadata.get("doi", {}).get("id") != DATASET_DOI:
        raise ValueError("Dataset DOI mismatch")
    license_info = metadata.get("data_licence", {})
    accepted_urls = {
        "http://creativecommons.org/licenses/by/4.0",
        "https://creativecommons.org/licenses/by/4.0",
    }
    if (license_info.get("short_name") != "CC BY 4.0"
            or license_info.get("url", "").rstrip("/") not in accepted_urls):
        raise ValueError("Expected CC BY 4.0 source license")
    selected = [entry for entry in catalog if entry.get("id") == FILE_ID]
    if len(selected) != 1:
        raise ValueError("Expected exactly one pinned Dataset 5 file")
    entry = selected[0]
    details = entry.get("content_details", {})
    if (entry.get("filename") != FILE_NAME or entry.get("status") != "COMPLETED"
            or details.get("sha256_hash") != FILE_SHA256
            or details.get("size") != FILE_BYTES or entry.get("size") != FILE_BYTES):
        raise ValueError("Pinned file name/status/size/hash mismatch")
    expected_url = (
        f"https://data.mendeley.com/public-files/datasets/{DATASET_ID}/files/"
        f"{FILE_ID}/file_downloaded"
    )
    if details.get("download_url") != expected_url:
        raise ValueError("Unexpected source download URL")
    return {"file": entry, "license": license_info}


def parse_civil_time(raw: str) -> datetime | None:
    if not TIME_RE.fullmatch(raw):
        return None
    try:
        value = datetime.strptime(raw, SOURCE_FORMAT)
        pd.Timestamp(value).as_unit("ns")
    except (ValueError, OverflowError):
        return None
    return value if value.strftime(SOURCE_FORMAT) == raw else None


@lru_cache(maxsize=25000)
def civil_time_kind(value: datetime) -> str:
    """Classify civil labels without choosing a UTC instant or DST fold."""
    possible_offsets = set()
    for fold in (0, 1):
        candidate = value.replace(tzinfo=OSLO, fold=fold)
        roundtrip = candidate.astimezone(timezone.utc).astimezone(OSLO)
        if roundtrip.replace(tzinfo=None) == value:
            possible_offsets.add(candidate.utcoffset())
    if not possible_offsets:
        return "nonexistent"
    return "ambiguous" if len(possible_offsets) > 1 else "unambiguous"


def partition_for(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None or end <= start:
        return "invalid_time"
    for name, (lower, upper) in WINDOWS.items():
        if lower <= start < upper:
            return name if end <= upper else "crosses_partition_boundary"
    return "out_of_window"


def parse_ams_value(raw: str) -> tuple[float, str]:
    value = raw.strip()
    if value.lower() in {"", "na", "n/a", "nan", "null", "none"}:
        return math.nan, "missing"
    if value.lower() in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        return math.nan, "nonfinite"
    if not NUMBER_RE.fullmatch(value):
        return math.nan, "nonnumeric"
    number = float(value.replace(",", "."))
    if not math.isfinite(number):
        return math.nan, "nonfinite"
    if number < 0:
        return math.nan, "negative"
    return number, "valid"


def _iso(value) -> str | None:
    return None if pd.isna(value) else value.isoformat()


def _time_summary(frame: pd.DataFrame) -> dict:
    starts = frame["local_start"].dropna()
    ends = frame["local_end"].dropna()
    summary = {
        "rows": len(frame),
        "first_local_start": _iso(starts.min()) if len(starts) else None,
        "last_local_end": _iso(ends.max()) if len(ends) else None,
        "unparseable_endpoint_rows": int(frame["invalid_time_parse"].sum()),
        "non_hour_aligned_rows": int(frame["non_hour_aligned"].sum()),
        "not_one_civil_hour_rows": int(frame["not_one_civil_hour"].sum()),
        "ambiguous_endpoint_rows": int(frame["ambiguous_endpoint"].sum()),
        "nonexistent_endpoint_rows": int(frame["nonexistent_endpoint"].sum()),
        "duplicate_start_rows": int(frame["duplicate_start"].sum()),
        "duplicate_interval_rows": int(frame["duplicate_interval"].sum()),
        "backward_start_transitions": int(frame["backward_start"].sum()),
        "gap_before_rows": int(frame["gap_before"].sum()),
        "overlap_before_rows": int(frame["overlap_before"].sum()),
        "eligible_local_civil_hour_rows": int(frame["time_eligible"].sum()),
    }
    if len(starts):
        lower, upper = starts.min(), starts.max()
        if upper - lower <= pd.Timedelta(days=3660):
            nominal = pd.date_range(lower.ceil("h"), upper.floor("h"), freq="h")
            missing = nominal.difference(pd.DatetimeIndex(starts.unique()))
            summary["nominal_civil_grid_missing_starts"] = [x.isoformat() for x in missing]
            summary["nominal_grid_note"] = (
                "Civil-label gaps, not inferred missing meter measurements; DST can change labels."
            )
        else:
            summary["nominal_grid_note"] = "Grid not expanded: source range exceeds ten years."
    return summary


def load_ams(
    csv_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    reveal_final_test: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Return allowlisted rows and QA; final-test targets are withheld by default.

    Use both pinned integrity arguments for the real source. Omitting both is
    useful for synthetic parser tests but is explicitly reported as unverified.
    The reveal option is for a separately frozen final scorer, not ingestion QA.
    """
    if (expected_sha256 is None) != (expected_bytes is None):
        raise ValueError("Provide both source SHA256 and byte count, or neither")
    csv_path = Path(csv_path)
    integrity = (verify_file(csv_path, expected_sha256, expected_bytes)
                 if expected_sha256 is not None else None)
    records = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=";", strict=True)
        header = next(reader, None)
        if not header or len(header) != len(set(header)):
            raise ValueError("Missing or duplicate source column names")
        if any(field not in header for field in REQUIRED_FIELDS):
            raise ValueError("Required AMS/time columns are missing")
        positions = {field: header.index(field) for field in REQUIRED_FIELDS}
        for source_row, values in enumerate(reader, start=2):
            if len(values) != len(header):
                raise ValueError(f"Malformed source field count at record {source_row}")
            raw_start, raw_end = (values[positions[key]] for key in ("date_from", "date_to"))
            start, end = parse_civil_time(raw_start), parse_civil_time(raw_end)
            partition = partition_for(start, end)
            can_read_target = partition in {"development", "validation"}
            can_read_target |= reveal_final_test and partition == "final_test"
            if can_read_target:
                target, target_status = parse_ams_value(values[positions["AMS_kWh"]])
            else:
                target, target_status = math.nan, "withheld_" + partition
            start_kind = civil_time_kind(start) if start is not None else "invalid"
            end_kind = civil_time_kind(end) if end is not None else "invalid"
            records.append({
                "source_row": source_row,
                "date_from_raw": raw_start,
                "date_to_raw": raw_end,
                "local_start": start,
                "local_end": end,
                "partition": partition,
                "invalid_time_parse": start is None or end is None,
                "non_hour_aligned": any(x is not None and x.minute != 0 for x in (start, end)),
                "not_one_civil_hour": start is None or end is None
                    or (end - start).total_seconds() != 3600,
                "ambiguous_endpoint": "ambiguous" in (start_kind, end_kind),
                "nonexistent_endpoint": "nonexistent" in (start_kind, end_kind),
                "ams_kwh": target,
                "target_status": target_status,
            })
    if not records:
        raise ValueError("The source has no data rows")
    frame = pd.DataFrame.from_records(records)
    for name in ("local_start", "local_end"):
        frame[name] = pd.to_datetime(frame[name])
    parsed = ~frame["invalid_time_parse"]
    frame["duplicate_start"] = frame["local_start"].duplicated(keep=False) & parsed
    frame["duplicate_interval"] = frame.duplicated(["local_start", "local_end"], keep=False) & parsed
    frame["backward_start"] = frame["local_start"].diff().lt(pd.Timedelta(0))
    gap = frame["local_start"] - frame["local_end"].shift()
    frame["gap_before"] = gap.gt(pd.Timedelta(0))
    frame["overlap_before"] = gap.lt(pd.Timedelta(0))
    order_issue = frame["backward_start"] | frame["overlap_before"]
    frame["affected_by_order_defect"] = order_issue | order_issue.shift(-1, fill_value=False)
    disqualifiers = [
        "invalid_time_parse", "non_hour_aligned", "not_one_civil_hour",
        "ambiguous_endpoint", "nonexistent_endpoint", "duplicate_start",
        "duplicate_interval", "affected_by_order_defect",
    ]
    frame["time_eligible"] = ~frame[disqualifiers].any(axis=1)
    frame["label_eligible"] = frame["time_eligible"] & frame["target_status"].eq("valid")
    summary = {
        "source_integrity": integrity,
        "source_integrity_verified": integrity is not None,
        "source_columns": header,
        "selected_source_columns": list(REQUIRED_FIELDS),
        "excluded_source_columns": [key for key in header if key not in REQUIRED_FIELDS],
        "timestamp_format": SOURCE_FORMAT,
        "time_representation": "offset-free local civil; no UTC/fold selected",
        "timezone_classification_rule": "Europe/Oslo, supported by paper CET with DST in Norway",
        "target_unit": "kWh per native source hourly interval",
        "resolution": "native_hourly_only",
        "time_quality_all_rows": _time_summary(frame),
        "partitions": {},
        "target_quality_scope": "development and validation only",
        "final_test_revealed": bool(reveal_final_test),
        "final_test_target_statistics_computed": False,
        "forecasting_performed": False,
    }
    for name, (lower, upper) in WINDOWS.items():
        group = frame[frame["partition"] == name]
        report = {
            "requested_start_inclusive": lower.isoformat(),
            "requested_end_exclusive": upper.isoformat(),
            "time_quality": _time_summary(group),
        }
        if name in {"development", "validation"}:
            report["target_quality_counts"] = dict(Counter(group["target_status"]))
            report["eligible_labeled_rows"] = int(group["label_eligible"].sum())
        else:
            report["target_quality_counts"] = "not inspected"
        summary["partitions"][name] = report
    summary["other_partition_rows"] = {
        key: int(value) for key, value in frame["partition"].value_counts().items()
        if key not in WINDOWS
    }
    return frame, summary


def json_text(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
