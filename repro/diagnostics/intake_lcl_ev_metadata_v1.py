"""Inventory LCL EV archive headers and clocks without inspecting load values."""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

POWER = "Active Power Total (Watts)"


def parse_clock(raw):
    parts = raw.strip().split()
    if not 1 <= len(parts) <= 2:
        raise ValueError("unexpected clock shape")
    day, month, year = map(int, parts[0].split("/"))
    # The source omits the time for midnight between 23:50 and 00:10.
    hour, minute, second = map(int, parts[1].split(":")) if len(parts) == 2 else (0, 0, 0)
    return datetime(year, month, day, hour, minute, second)


def inspect_csv(stream, clock=True):
    reader = csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
    fields = next(reader)
    result = {"fields": fields, "has_active_power_watts": POWER in fields}
    if not clock:
        return result
    index = fields.index("ReadingDate")
    stamps = Counter()
    invalid = 0
    dayfirst_evidence = 0
    date_only = 0
    for row in reader:
        if len(row) <= index:
            invalid += 1
            continue
        raw = row[index]
        try:
            stamp = parse_clock(raw)
        except ValueError:
            invalid += 1
            continue
        dayfirst_evidence += int(int(raw.split("/")[0]) > 12)
        date_only += int(len(raw.strip().split()) == 1)
        stamps[stamp] += 1
    ordered = sorted(stamps)
    steps = Counter(int((b - a).total_seconds()) for a, b in zip(ordered, ordered[1:]))
    result.update(
        rows=sum(stamps.values()) + invalid,
        invalid_clock_rows=invalid,
        date_only_midnight_rows=date_only,
        unique_timestamps=len(stamps),
        duplicate_clock_rows=sum(n - 1 for n in stamps.values()),
        first=ordered[0].isoformat() if ordered else None,
        last=ordered[-1].isoformat() if ordered else None,
        dayfirst_unambiguous_rows=dayfirst_evidence,
        common_steps_seconds=steps.most_common(5),
        exact_hour_timestamps=sum(t.minute == t.second == 0 for t in ordered),
        month_timestamp_counts=dict(sorted(Counter(t.strftime("%Y-%m") for t in ordered).items())),
    )
    return result


def build(archive):
    records = []
    docs = []
    with zipfile.ZipFile(archive) as outer:
        doc_name = "EV Vehicles Data  - Low Carbon London - Cover Note.docx"
        with zipfile.ZipFile(io.BytesIO(outer.read(doc_name))) as doc:
            root = ET.fromstring(doc.read("word/document.xml"))
            docs.append({"member": doc_name, "text": "\n".join(
                "".join(p.itertext()) for p in root.findall(
                    ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"))})
        for group in ("Residential", "Commercial"):
            member = f"EV {group} trials.zip"
            with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
                for name in sorted(inner.namelist()):
                    if name.lower().endswith("read me.txt"):
                        docs.append({"member": member + "/" + name,
                                     "text": inner.read(name).decode("utf-8-sig")})
                    if not name.lower().endswith(".csv"):
                        continue
                    with inner.open(name) as stream:
                        record = inspect_csv(stream, clock=group == "Residential")
                    records.append({"group": group, "member": name, **record})
    return {
        "source": "Low Carbon London Electric Vehicle Load Profiles",
        "landing_url": "https://data.london.gov.uk/dataset/low-carbon-london-electric-vehicle-load-profiles-2y5k8",
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.file_digest(archive.open("rb"), "sha256").hexdigest(),
        "stage": "metadata_only",
        "clock_parse": "dd/mm/yyyy [HH:MM:SS]; date-only means midnight; source-naive, timezone not assigned",
        "measurement_values_inspected": False,
        "survey_and_participant_workbooks_read": False,
        "external_validation_completed": False,
        "documents": docs,
        "series": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.archive)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for group in ("Residential", "Commercial"):
        rows = [r for r in result["series"] if r["group"] == group]
        print(group, "CSV files", len(rows), "active-power headers",
              sum(r["has_active_power_watts"] for r in rows))
    print("Saved metadata:", args.output)


if __name__ == "__main__":
    main()
