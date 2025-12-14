from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from traffic_pipeline.util import floor_dt, haversine_km, slugify


@dataclass(frozen=True)
class HotspotFeatureStats:
    output_rows: int
    unique_cells: int
    unique_cell_buckets: int


def _season(month: int) -> int:
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def _time_features(dt: datetime) -> dict[str, float]:
    how = dt.weekday() * 24 + dt.hour
    angle = 2 * math.pi * (how / 168.0)
    return {
        "hour_of_week": float(how),
        "how_sin": math.sin(angle),
        "how_cos": math.cos(angle),
        "month": float(dt.month),
        "season": float(_season(dt.month)),
    }


def _read_weather(weather_hourly_csv: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    if not weather_hourly_csv.exists():
        return [], {}

    with weather_hourly_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return [], {}

        cols = [c for c in r.fieldnames if c != "bucket_start"]
        by_bucket: dict[str, dict[str, float]] = {}
        for row in r:
            b = (row.get("bucket_start") or "").strip()
            if not b:
                continue
            out: dict[str, float] = {}
            for c in cols:
                v = (row.get(c) or "").strip()
                if not v:
                    continue
                try:
                    out[c] = float(v)
                except ValueError:
                    continue
            by_bucket[b] = out
        return cols, by_bucket


def _read_aadt(aadt_stations_csv: Path) -> list[dict[str, float | str]]:
    if not aadt_stations_csv.exists():
        return []

    with aadt_stations_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return []

        out: list[dict[str, float | str]] = []
        for row in r:
            try:
                lat = float((row.get("latitude") or "").strip())
                lon = float((row.get("longitude") or "").strip())
            except ValueError:
                continue
            if not (lat == lat and lon == lon):
                continue

            aadt_s = (row.get("aadt") or "").strip()
            aadt = float(aadt_s) if aadt_s else float("nan")
            aadt_log_s = (row.get("aadt_log1p") or "").strip()
            aadt_log = float(aadt_log_s) if aadt_log_s else float("nan")
            year_s = (row.get("year") or "").strip()
            year = float(year_s) if year_s else float("nan")

            out.append(
                {
                    "station_id": (row.get("station_id") or "").strip(),
                    "latitude": lat,
                    "longitude": lon,
                    "aadt": aadt,
                    "aadt_log1p": aadt_log,
                    "year": year,
                }
            )
        return out


def _build_aadt_index(stations: list[dict[str, float | str]], *, max_distance_km: float) -> dict[tuple[int, int], list[int]]:
    if max_distance_km <= 0:
        return {}
    cell_deg = max_distance_km / 111.0
    if cell_deg <= 0:
        return {}

    index: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, s in enumerate(stations):
        lat = float(s["latitude"])
        lon = float(s["longitude"])
        key = (int(lat / cell_deg), int(lon / cell_deg))
        index[key].append(i)
    return index


def _nearest_station(
    *,
    lat: float,
    lon: float,
    stations: list[dict[str, float | str]],
    index: dict[tuple[int, int], list[int]],
    max_distance_km: float,
) -> tuple[dict[str, float | str] | None, float | None]:
    if not stations or max_distance_km <= 0:
        return None, None

    cell_deg = max_distance_km / 111.0
    base_key = (int(lat / cell_deg), int(lon / cell_deg))
    candidates: list[int] = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            candidates.extend(index.get((base_key[0] + di, base_key[1] + dj), []))

    best_i = None
    best_d = None
    for i in candidates:
        s = stations[i]
        d = haversine_km(lat, lon, float(s["latitude"]), float(s["longitude"]))
        if d > max_distance_km:
            continue
        if best_d is None or d < best_d:
            best_d = d
            best_i = i

    return (stations[best_i], best_d) if best_i is not None else (None, None)


def build_hotspot_features(
    *,
    silver_incidents_csv: Path,
    weather_hourly_csv: Path,
    aadt_stations_csv: Path,
    out_csv: Path,
    datetime_format: str,
    bucket_minutes: int,
    cell_round_decimals: int,
    lookback_hours: list[int],
    label_horizon_hours: list[int],
    aadt_max_distance_km: float,
    ema_half_life_hours: float,
) -> HotspotFeatureStats:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with silver_incidents_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("incidents.csv missing header row")
        if "published_date" not in r.fieldnames:
            raise ValueError("incidents.csv missing published_date")

        rows = list(r)

    if not rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f_out:
            csv.writer(f_out).writerow(["bucket_start", "cell_lat", "cell_lon"])
        return HotspotFeatureStats(output_rows=0, unique_cells=0, unique_cell_buckets=0)

    # (cell_key, bucket_start) -> counts
    counts: dict[tuple[str, str], dict[str, object]] = {}
    cell_coords: dict[str, tuple[float, float]] = {}
    issue_slugs: set[str] = set()
    cells_to_buckets: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        try:
            dt = datetime.strptime((row.get("published_date") or "").strip(), datetime_format)
        except ValueError:
            continue
        bucket = floor_dt(dt, bucket_minutes=bucket_minutes)
        bucket_str = bucket.strftime(datetime_format)

        lat_s = (row.get("latitude") or "").strip()
        lon_s = (row.get("longitude") or "").strip()
        if not lat_s or not lon_s:
            continue
        try:
            lat = float(lat_s)
            lon = float(lon_s)
        except ValueError:
            continue

        cell_lat = round(lat, cell_round_decimals)
        cell_lon = round(lon, cell_round_decimals)
        cell_key = f"{cell_lat:.{cell_round_decimals}f},{cell_lon:.{cell_round_decimals}f}"
        cell_coords[cell_key] = (cell_lat, cell_lon)
        cells_to_buckets[cell_key].add(bucket_str)

        issue = slugify((row.get("issue_reported") or "").strip())
        issue_slugs.add(issue)

        key = (cell_key, bucket_str)
        rec = counts.get(key)
        if rec is None:
            rec = {"n_collisions": 0, "n_traffic_incidents": 0, "issue": defaultdict(int)}
            counts[key] = rec

        if (row.get("event_class") or "").strip() == "collision":
            rec["n_collisions"] = int(rec["n_collisions"]) + 1
        else:
            rec["n_traffic_incidents"] = int(rec["n_traffic_incidents"]) + 1
        rec["issue"][issue] += 1  # type: ignore[index]

    weather_cols, weather_by_bucket = _read_weather(weather_hourly_csv)
    stations = _read_aadt(aadt_stations_csv)
    station_index = _build_aadt_index(stations, max_distance_km=aadt_max_distance_km)

    issue_cols = [f"n_issue_{s}" for s in sorted(issue_slugs)]
    lookbacks_sorted = sorted({int(h) for h in lookback_hours if int(h) > 0})
    horizons_sorted = sorted({int(h) for h in label_horizon_hours if int(h) > 0})

    lb_cols = []
    for h in lookbacks_sorted:
        lb_cols.extend([f"n_collisions_lb_{h}h", f"n_traffic_incidents_lb_{h}h"])

    label_cols = []
    for h in horizons_sorted:
        label_cols.extend([f"y_collisions_next_{h}h", f"y_traffic_incidents_next_{h}h"])

    base_cols = [
        "bucket_start",
        "cell_lat",
        "cell_lon",
        "n_collisions",
        "n_traffic_incidents",
        *lb_cols,
        "ema_collisions",
        "ema_traffic_incidents",
        "hrs_since_last_collision",
        "hrs_since_last_traffic_incident",
        *label_cols,
        "hour_of_week",
        "how_sin",
        "how_cos",
        "month",
        "season",
        *issue_cols,
        *weather_cols,
        "aadt",
        "aadt_log1p",
        "aadt_year",
        "aadt_distance_km",
        "aadt_station_id",
    ]

    out_rows = 0

    with out_csv.open("w", encoding="utf-8", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=base_cols)
        w.writeheader()

        for cell_key in sorted(cell_coords.keys()):
            lat, lon = cell_coords[cell_key]
            station, station_dist = _nearest_station(
                lat=lat,
                lon=lon,
                stations=stations,
                index=station_index,
                max_distance_km=aadt_max_distance_km,
            )

            buckets = sorted(cells_to_buckets[cell_key], key=lambda s: datetime.strptime(s, datetime_format))
            bucket_dts = [datetime.strptime(b, datetime_format) for b in buckets]

            coll_by_t = {b: int(counts[(cell_key, b)]["n_collisions"]) for b in buckets if (cell_key, b) in counts}
            inc_by_t = {
                b: int(counts[(cell_key, b)]["n_traffic_incidents"]) for b in buckets if (cell_key, b) in counts
            }

            last_collision_dt: datetime | None = None
            last_inc_dt: datetime | None = None
            ema_coll = 0.0
            ema_inc = 0.0
            last_dt: datetime | None = None

            for i, (bucket_str, bucket_dt) in enumerate(zip(buckets, bucket_dts, strict=True)):
                # --- lookbacks
                lb_vals: dict[str, int] = {}
                for h in lookbacks_sorted:
                    start_dt = bucket_dt - timedelta(hours=h)
                    c = 0
                    n = 0
                    for j in range(i - 1, -1, -1):
                        if bucket_dts[j] < start_dt:
                            break
                        c += coll_by_t.get(buckets[j], 0)
                        n += inc_by_t.get(buckets[j], 0)
                    lb_vals[f"n_collisions_lb_{h}h"] = c
                    lb_vals[f"n_traffic_incidents_lb_{h}h"] = n

                # --- EMA (history only)
                ema_coll_hist = 0.0
                ema_inc_hist = 0.0
                if last_dt is not None and ema_half_life_hours > 0:
                    dt_hours = (bucket_dt - last_dt).total_seconds() / 3600.0
                    decay = math.exp(-math.log(2.0) * (dt_hours / ema_half_life_hours))
                    ema_coll = ema_coll * decay
                    ema_inc = ema_inc * decay
                    ema_coll_hist = ema_coll
                    ema_inc_hist = ema_inc
                else:
                    ema_coll = 0.0
                    ema_inc = 0.0

                # --- time since last
                hrs_since_collision = (
                    (bucket_dt - last_collision_dt).total_seconds() / 3600.0 if last_collision_dt is not None else ""
                )
                hrs_since_inc = (bucket_dt - last_inc_dt).total_seconds() / 3600.0 if last_inc_dt is not None else ""

                # --- future labels
                max_h = horizons_sorted[-1] if horizons_sorted else 0
                end_dt = bucket_dt + timedelta(hours=max_h)
                y_coll: dict[int, int] = {h: 0 for h in horizons_sorted}
                y_inc: dict[int, int] = {h: 0 for h in horizons_sorted}
                for j in range(i + 1, len(bucket_dts)):
                    if bucket_dts[j] > end_dt:
                        break
                    dt_h = int((bucket_dts[j] - bucket_dt).total_seconds() // 3600)
                    for h in horizons_sorted:
                        if 1 <= dt_h <= h:
                            y_coll[h] += coll_by_t.get(buckets[j], 0)
                            y_inc[h] += inc_by_t.get(buckets[j], 0)

                rec = counts.get((cell_key, bucket_str))
                n_collisions = int(rec["n_collisions"]) if rec else 0
                n_incidents = int(rec["n_traffic_incidents"]) if rec else 0

                # issue mix for this bucket
                issue_counts = rec["issue"] if rec else defaultdict(int)
                issue_out = {f"n_issue_{s}": int(issue_counts.get(s, 0)) for s in issue_slugs}

                weather = weather_by_bucket.get(bucket_str, {})
                tfeat = _time_features(bucket_dt)

                out: dict[str, object] = {
                    "bucket_start": bucket_str,
                    "cell_lat": f"{lat:.6f}",
                    "cell_lon": f"{lon:.6f}",
                    "n_collisions": n_collisions,
                    "n_traffic_incidents": n_incidents,
                    "ema_collisions": f"{ema_coll_hist:.6f}",
                    "ema_traffic_incidents": f"{ema_inc_hist:.6f}",
                    "hrs_since_last_collision": "" if hrs_since_collision == "" else f"{hrs_since_collision:.3f}",
                    "hrs_since_last_traffic_incident": "" if hrs_since_inc == "" else f"{hrs_since_inc:.3f}",
                    **lb_vals,
                    **{f"y_collisions_next_{h}h": y_coll[h] for h in horizons_sorted},
                    **{f"y_traffic_incidents_next_{h}h": y_inc[h] for h in horizons_sorted},
                    **tfeat,
                    **issue_out,
                    **{c: weather.get(c, "") for c in weather_cols},
                }

                if station is None:
                    out.update(
                        {
                            "aadt": "",
                            "aadt_log1p": "",
                            "aadt_year": "",
                            "aadt_distance_km": "",
                            "aadt_station_id": "",
                        }
                    )
                else:
                    out.update(
                        {
                            "aadt": station.get("aadt", ""),
                            "aadt_log1p": station.get("aadt_log1p", ""),
                            "aadt_year": station.get("year", ""),
                            "aadt_distance_km": "" if station_dist is None else f"{station_dist:.3f}",
                            "aadt_station_id": station.get("station_id", ""),
                        }
                    )

                w.writerow(out)
                out_rows += 1

                # update state
                if n_collisions > 0:
                    last_collision_dt = bucket_dt
                if n_incidents > 0:
                    last_inc_dt = bucket_dt
                if ema_half_life_hours > 0:
                    ema_coll = ema_coll_hist + n_collisions
                    ema_inc = ema_inc_hist + n_incidents
                last_dt = bucket_dt

    return HotspotFeatureStats(
        output_rows=out_rows,
        unique_cells=len(cell_coords),
        unique_cell_buckets=len(counts),
    )
