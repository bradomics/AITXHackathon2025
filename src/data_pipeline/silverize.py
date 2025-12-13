from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

from util import parse_dt


@dataclass(frozen=True)
class SilverizeIncidentsStats:
    input_files: int
    input_rows: int
    output_rows: int
    deduped_rows: int


def _is_incidents_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            first = f.readline()
    except FileNotFoundError:
        return False
    return "Traffic Report ID" in first


_UNIFIED_SNAPSHOT_RE = re.compile(r".*_(\d{8})\.csv$", flags=re.IGNORECASE)


def _select_incident_files(bronze_dir: Path) -> list[Path]:
    """
    Prefer the unified snapshot CSV if present:
      Real-Time_Traffic_Incident_Reports_YYYYMMDD.csv
    Otherwise fall back to the legacy per-category CSVs.
    """

    snapshots = sorted(bronze_dir.glob("Real-Time_Traffic_Incident_Reports_*.csv"))
    snapshots = [p for p in snapshots if _is_incidents_csv(p)]
    if snapshots:
        dated: list[tuple[int, Path]] = []
        undated: list[Path] = []
        for p in snapshots:
            m = _UNIFIED_SNAPSHOT_RE.match(p.name)
            if not m:
                undated.append(p)
                continue
            try:
                dated.append((int(m.group(1)), p))
            except ValueError:
                undated.append(p)
        if dated:
            dated.sort(key=lambda x: x[0], reverse=True)
            return [dated[0][1]]
        undated.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [undated[0]]

    return sorted([p for p in bronze_dir.glob("*.csv") if _is_incidents_csv(p)])


def _is_collision_issue(issue_reported: str) -> bool:
    s = (issue_reported or "").strip().lower()
    if not s:
        return False
    return ("crash" in s) or ("collis" in s)


def silverize_incidents(*, bronze_dir: Path, out_path: Path, datetime_format: str) -> SilverizeIncidentsStats:
    files = _select_incident_files(bronze_dir)

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
    input_rows = 0
    out_rows = 0
    deduped_rows = 0

    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()

        for path in files:
            with path.open("r", encoding="utf-8-sig", newline="") as f_in:
                r = csv.DictReader(f_in)
                for row in r:
                    input_rows += 1
                    rid = (row.get("Traffic Report ID") or "").strip()
                    if not rid:
                        continue
                    if rid in seen:
                        deduped_rows += 1
                        continue
                    seen.add(rid)

                    published_raw = (row.get("Published Date") or "").strip()
                    if not published_raw:
                        continue
                    try:
                        published_dt = parse_dt(published_raw, datetime_format=datetime_format)
                    except ValueError:
                        continue

                    issue_reported = (row.get("Issue Reported") or "").strip()
                    event_class = (
                        "collision"
                        if ("collision" in path.name.lower() or _is_collision_issue(issue_reported))
                        else "traffic_incident"
                    )

                    status_raw = (row.get("Status Date") or "").strip()
                    status_dt_str = ""
                    if status_raw:
                        try:
                            status_dt_str = parse_dt(status_raw, datetime_format=datetime_format).strftime(datetime_format)
                        except ValueError:
                            status_dt_str = status_raw

                    w.writerow(
                        {
                            "traffic_report_id": rid,
                            "published_date": published_dt.strftime(datetime_format),
                            "issue_reported": issue_reported,
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

    return SilverizeIncidentsStats(
        input_files=len(files),
        input_rows=input_rows,
        output_rows=out_rows,
        deduped_rows=deduped_rows,
    )
