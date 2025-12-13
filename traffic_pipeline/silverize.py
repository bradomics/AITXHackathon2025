from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class SilverizeIncidentsStats:
    input_files: int
    output_rows: int


def _is_incidents_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            first = f.readline().lstrip("\ufeff")
    except FileNotFoundError:
        return False
    return "Traffic Report ID" in first


def silverize_incidents(*, bronze_dir: Path, out_path: Path, datetime_format: str) -> SilverizeIncidentsStats:
    files = sorted([p for p in bronze_dir.glob("*.csv") if _is_incidents_csv(p)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "traffic_report_id",
        "published_date",
        "issue_reported",
        "event_class",
        "latitude",
        "longitude",
        "address",
        "status",
        "status_date",
        "source_file",
    ]

    seen: set[str] = set()
    out_rows = 0

    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()

        for path in files:
            event_class = "collision" if "collision" in path.name.lower() else "traffic_incident"
            with path.open("r", encoding="utf-8", newline="") as f_in:
                r = csv.DictReader(f_in)
                for row in r:
                    rid = (row.get("Traffic Report ID") or "").strip()
                    if not rid or rid in seen:
                        continue
                    seen.add(rid)

                    published_raw = (row.get("Published Date") or "").strip()
                    if not published_raw:
                        continue
                    published_dt = datetime.strptime(published_raw, datetime_format)

                    status_raw = (row.get("Status Date") or "").strip()
                    status_dt_str = ""
                    if status_raw:
                        try:
                            status_dt_str = datetime.strptime(status_raw, datetime_format).strftime(datetime_format)
                        except ValueError:
                            status_dt_str = status_raw

                    w.writerow(
                        {
                            "traffic_report_id": rid,
                            "published_date": published_dt.strftime(datetime_format),
                            "issue_reported": (row.get("Issue Reported") or "").strip(),
                            "event_class": event_class,
                            "latitude": (row.get("Latitude") or "").strip(),
                            "longitude": (row.get("Longitude") or "").strip(),
                            "address": (row.get("Address") or "").strip(),
                            "status": (row.get("Status") or "").strip(),
                            "status_date": status_dt_str,
                            "source_file": path.name,
                        }
                    )
                    out_rows += 1

    return SilverizeIncidentsStats(input_files=len(files), output_rows=out_rows)

