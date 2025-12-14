from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SilverizeAadtStats:
    input_rows: int
    unique_stations: int
    output_rows: int


def _pick_col(fieldnames: list[str], candidates: list[str]) -> str | None:
    m = {n.lower(): n for n in fieldnames}
    for c in candidates:
        hit = m.get(c.lower())
        if hit:
            return hit
    return None


def silverize_aadt_stations(*, bronze_path: Path, out_path: Path) -> SilverizeAadtStats:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not bronze_path.exists():
        with out_path.open("w", encoding="utf-8", newline="") as f_out:
            csv.writer(f_out).writerow(["station_id", "year", "latitude", "longitude", "aadt", "aadt_log1p"])
        return SilverizeAadtStats(input_rows=0, unique_stations=0, output_rows=0)

    with bronze_path.open("r", encoding="utf-8-sig", newline="") as f_in:
        r = csv.DictReader(f_in)
        if r.fieldnames is None:
            raise ValueError("AADT CSV missing header row")

        station_col = _pick_col(
            r.fieldnames,
            ["station_id", "Station_ID", "STATION_ID", "Station ID", "TRFC_STATN_ID", "traffic_station_id"],
        )
        year_col = _pick_col(r.fieldnames, ["year", "Year", "YEAR", "AADT_RPT_YEAR"])
        aadt_col = _pick_col(r.fieldnames, ["aadt", "AADT", "Annual Average Daily Traffic", "AADT_RPT_QTY"])
        lat_col = _pick_col(r.fieldnames, ["latitude", "Latitude", "LATITUDE", "lat", "Lat"])
        lon_col = _pick_col(r.fieldnames, ["longitude", "Longitude", "LONGITUDE", "lon", "Lon", "lng", "LNG"])

        if not station_col or not year_col or not aadt_col or not lat_col or not lon_col:
            missing = [n for n, v in [("station_id", station_col), ("year", year_col), ("aadt", aadt_col), ("lat", lat_col), ("lon", lon_col)] if not v]
            raise ValueError(f"AADT CSV missing required columns: {', '.join(missing)}")

        best: dict[str, tuple[int, dict[str, str]]] = {}
        input_rows = 0
        for row in r:
            input_rows += 1
            sid = (row.get(station_col) or "").strip()
            if not sid:
                continue
            try:
                year = int(float((row.get(year_col) or "").strip() or "0"))
            except ValueError:
                year = 0

            cur = best.get(sid)
            if cur is None or year > cur[0]:
                best[sid] = (year, row)

    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        w = csv.DictWriter(
            f_out,
            fieldnames=["station_id", "year", "latitude", "longitude", "aadt", "aadt_log1p"],
        )
        w.writeheader()

        output_rows = 0
        for sid, (year, row) in sorted(best.items(), key=lambda kv: kv[0]):
            lat = float((row.get(lat_col) or "").strip() or "nan")
            lon = float((row.get(lon_col) or "").strip() or "nan")
            aadt = float((row.get(aadt_col) or "").strip() or "nan")
            w.writerow(
                {
                    "station_id": sid,
                    "year": year,
                    "latitude": f"{lat:.6f}",
                    "longitude": f"{lon:.6f}",
                    "aadt": f"{aadt:.3f}",
                    "aadt_log1p": f"{math.log1p(aadt):.6f}" if aadt == aadt else "",
                }
            )
            output_rows += 1

    return SilverizeAadtStats(input_rows=input_rows, unique_stations=len(best), output_rows=output_rows)
