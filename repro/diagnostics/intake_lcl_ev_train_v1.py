"""Summarize declared LCL training values; never convert held-out values."""

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from datetime import datetime
from pathlib import Path

from diagnostics.intake_lcl_ev_metadata_v1 import POWER, parse_clock

START = datetime(2013, 2, 1)
END = datetime(2013, 8, 1)
EXPECTED_HASH = "13bbb60211f45bd19c65d8ec5f1c81032ca068f169e43fb4ebec446df5481736"


def training_value(stamp, raw):
    if not START <= stamp < END:
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def inspect_train(stream):
    reader = csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
    fields = next(reader)
    if POWER not in fields:
        return None
    ti, pi = fields.index("ReadingDate"), fields.index(POWER)
    counts = dict(rows=0, finite=0, missing_or_nonfinite=0, negative=0, zero=0, positive=0)
    minimum, maximum = math.inf, -math.inf
    hours = {}
    for row in reader:
        if len(row) <= ti:
            continue
        try:
            stamp = parse_clock(row[ti])
        except ValueError:
            continue
        if not START <= stamp < END:
            continue
        counts["rows"] += 1
        value = training_value(stamp, row[pi] if len(row) > pi else None)
        if stamp.minute == stamp.second == 0:
            previous = hours.setdefault(stamp, {"rows": 0, "finite": 0})
            previous["rows"] += 1
            previous["finite"] += int(value is not None)
        if value is None:
            counts["missing_or_nonfinite"] += 1
            continue
        counts["finite"] += 1
        counts["negative" if value < 0 else "zero" if value == 0 else "positive"] += 1
        minimum, maximum = min(minimum, value), max(maximum, value)
    return {**counts, "minimum_raw": minimum if counts["finite"] else None,
            "maximum_raw": maximum if counts["finite"] else None,
            "unique_exact_hours": len(hours),
            "unambiguous_finite_hour_labels": sum(h["rows"] == h["finite"] == 1 for h in hours.values()),
            "duplicate_exact_hours": sum(h["rows"] > 1 for h in hours.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != EXPECTED_HASH:
        raise ValueError("Archive changed; rebind source before reading values")
    records = []
    with zipfile.ZipFile(args.archive) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("EV Residential trials.zip"))) as inner:
            for name in sorted(inner.namelist()):
                if not name.lower().endswith(".csv"):
                    continue
                with inner.open(name) as stream:
                    record = inspect_train(stream)
                if record is not None:
                    records.append({"member": name, **record})
    result = {"stage": "training_numeric_intake_only", "archive_sha256": digest,
              "source_unit_label": "Watts", "physical_scale_verified": False,
              "train_start": START.isoformat(), "train_end_exclusive": END.isoformat(),
              "test_values_converted": False, "models_trained": 0,
              "clock": "source-naive; timezone unresolved", "series": records}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Series:", len(records), "with finite values:", sum(r["finite"] > 0 for r in records), flush=True)
    print("Training hour labels:", sum(r["unambiguous_finite_hour_labels"] for r in records), flush=True)


if __name__ == "__main__":
    main()
