from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from traffic_pipeline.util import floor_dt


@dataclass(frozen=True)
class SilverizeWeatherStats:
    output_rows: int


def _read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []

    header = rows[0]
    if len(header) == 1 and "," in header[0]:
        parsed = [next(csv.reader([r[0]])) for r in rows if r]
        return parsed[0], parsed[1:]

    return rows[0], rows[1:]


def silverize_weather_hourly(
    *,
    bronze_path: Path,
    out_path: Path,
    datetime_format: str,
    bucket_minutes: int,
) -> SilverizeWeatherStats:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not bronze_path.exists():
        with out_path.open("w", encoding="utf-8", newline="") as f_out:
            csv.writer(f_out).writerow(["bucket_start"])
        return SilverizeWeatherStats(output_rows=0)

    header, rows = _read_rows(bronze_path)
    if not header:
        with out_path.open("w", encoding="utf-8", newline="") as f_out:
            csv.writer(f_out).writerow(["bucket_start"])
        return SilverizeWeatherStats(output_rows=0)

    header_l = [h.strip() for h in header]
    dt_col = None
    for cand in ("formatted_date", "date", "datetime", header_l[0]):
        if cand in header_l:
            dt_col = cand
            break
    if dt_col is None:
        dt_col = header_l[0]

    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    numeric_cols: set[str] = set()

    idx = {name: i for i, name in enumerate(header_l)}

    for raw in rows:
        if not raw:
            continue
        row = {name: raw[i] if i < len(raw) else "" for name, i in idx.items()}

        dt_raw = (row.get(dt_col) or "").strip()
        if not dt_raw:
            continue
        dt = datetime.strptime(dt_raw, datetime_format)
        bucket = floor_dt(dt, bucket_minutes=bucket_minutes)
        bucket_str = bucket.strftime(datetime_format)

        for col, val_raw in row.items():
            if col == dt_col:
                continue
            val_s = (val_raw or "").strip()
            if not val_s:
                continue
            try:
                v = float(val_s)
            except ValueError:
                continue
            sums[bucket_str][col] += v
            counts[bucket_str][col] += 1
            numeric_cols.add(col)

    cols_sorted = sorted(numeric_cols)
    buckets_sorted = sorted(sums.keys(), key=lambda s: datetime.strptime(s, datetime_format))

    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=["bucket_start", *cols_sorted])
        w.writeheader()
        for b in buckets_sorted:
            out = {"bucket_start": b}
            for col in cols_sorted:
                n = counts[b].get(col, 0)
                out[col] = "" if n == 0 else f"{sums[b][col] / n:.6f}"
            w.writerow(out)

    return SilverizeWeatherStats(output_rows=len(buckets_sorted))

