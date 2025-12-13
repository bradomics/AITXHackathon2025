from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from traffic_pipeline.util import floor_dt, haversine_km


@dataclass(frozen=True)
class TokenizeH3Stats:
    n_time_steps: int
    n_cells: int
    collisions_events: int
    traffic_incidents_events: int
    collisions_pairs: int
    traffic_incidents_pairs: int
    skipped_missing_time_bucket: int
    skipped_missing_latlon: int
    skipped_outside_radius: int


def _season(month: int) -> int:
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def _time_features(dt: datetime) -> list[float]:
    how = dt.weekday() * 24 + dt.hour
    angle = 2 * math.pi * (how / 168.0)
    return [float(how), math.sin(angle), math.cos(angle), float(dt.month), float(_season(dt.month))]


def _read_weather(weather_hourly_csv: Path) -> tuple[list[str], list[str], list[list[float]]]:
    if not weather_hourly_csv.exists():
        return [], [], []

    with weather_hourly_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return [], [], []

        weather_cols = [c for c in r.fieldnames if c != "bucket_start"]
        buckets: list[str] = []
        values: list[list[float]] = []

        for row in r:
            b = (row.get("bucket_start") or "").strip()
            if not b:
                continue
            buckets.append(b)
            vrow: list[float] = []
            for c in weather_cols:
                s = (row.get(c) or "").strip()
                if not s:
                    vrow.append(0.0)
                    continue
                try:
                    vrow.append(float(s))
                except ValueError:
                    vrow.append(0.0)
            values.append(vrow)

        return weather_cols, buckets, values


def tokenize_h3_time_series(
    *,
    incidents_csv: Path,
    weather_hourly_csv: Path,
    out_dir: Path,
    h3_resolution: int,
    austin_center_lat: float,
    austin_center_lon: float,
    austin_radius_km: float,
    context_steps: int,
    datetime_format: str,
    bucket_minutes: int,
) -> TokenizeH3Stats:
    import numpy as np
    import h3

    out_dir.mkdir(parents=True, exist_ok=True)

    # Timeline from weather if available, otherwise from incidents.
    weather_cols, buckets, weather_vals = _read_weather(weather_hourly_csv)
    if not buckets:
        with incidents_csv.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if r.fieldnames is None:
                raise ValueError("incidents.csv missing header row")
            buckets_set: set[str] = set()
            for row in r:
                dt_raw = (row.get("published_date") or "").strip()
                if not dt_raw:
                    continue
                dt = datetime.strptime(dt_raw, datetime_format)
                b = floor_dt(dt, bucket_minutes=bucket_minutes).strftime(datetime_format)
                buckets_set.add(b)
        buckets = sorted(buckets_set)
        weather_vals = [[0.0 for _ in weather_cols] for _ in buckets]

    bucket_to_t = {b: i for i, b in enumerate(buckets)}

    # Build X (time features + weather).
    time_feats = []
    for b in buckets:
        dt = datetime.strptime(b, datetime_format)
        time_feats.append(_time_features(dt))
    X = np.array([tf + wv for tf, wv in zip(time_feats, weather_vals, strict=True)], dtype=np.float32)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std = np.where(X_std == 0, 1.0, X_std)

    # Sparse labels per time bucket.
    coll_by_t: list[set[int]] = [set() for _ in range(len(buckets))]
    inc_by_t: list[set[int]] = [set() for _ in range(len(buckets))]

    cell_to_idx: dict[str, int] = {}
    collisions_events = 0
    traffic_incidents_events = 0
    skipped_missing_time_bucket = 0
    skipped_missing_latlon = 0
    skipped_outside_radius = 0

    with incidents_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("incidents.csv missing header row")

        for row in r:
            dt_raw = (row.get("published_date") or "").strip()
            if not dt_raw:
                continue
            dt = datetime.strptime(dt_raw, datetime_format)
            bucket = floor_dt(dt, bucket_minutes=bucket_minutes).strftime(datetime_format)
            t = bucket_to_t.get(bucket)
            if t is None:
                skipped_missing_time_bucket += 1
                continue

            lat_s = (row.get("latitude") or "").strip()
            lon_s = (row.get("longitude") or "").strip()
            if not lat_s or not lon_s:
                skipped_missing_latlon += 1
                continue
            try:
                lat = float(lat_s)
                lon = float(lon_s)
            except ValueError:
                skipped_missing_latlon += 1
                continue

            if haversine_km(lat, lon, austin_center_lat, austin_center_lon) > austin_radius_km:
                skipped_outside_radius += 1
                continue

            cell = h3.latlng_to_cell(lat, lon, h3_resolution)
            idx = cell_to_idx.get(cell)
            if idx is None:
                idx = len(cell_to_idx)
                cell_to_idx[cell] = idx

            if (row.get("event_class") or "").strip() == "collision":
                coll_by_t[t].add(idx)
                collisions_events += 1
            else:
                inc_by_t[t].add(idx)
                traffic_incidents_events += 1

    n_cells = len(cell_to_idx)
    n_time = len(buckets)

    def to_csr(sets_by_t: list[set[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ptr = np.zeros(n_time + 1, dtype=np.int64)
        idxs: list[int] = []
        vals: list[float] = []
        pos = 0
        for t in range(n_time):
            ptr[t] = pos
            s = sorted(sets_by_t[t])
            idxs.extend(s)
            vals.extend([1.0] * len(s))
            pos += len(s)
        ptr[n_time] = pos
        return ptr, np.array(idxs, dtype=np.int64), np.array(vals, dtype=np.float32)

    coll_ptr, coll_idx, coll_val = to_csr(coll_by_t)
    inc_ptr, inc_idx, inc_val = to_csr(inc_by_t)

    collisions_pairs = sum(len(s) for s in coll_by_t)
    traffic_incidents_pairs = sum(len(s) for s in inc_by_t)

    # Write artifacts.
    with (out_dir / "buckets.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "bucket_start"])
        w.writeheader()
        for i, b in enumerate(buckets):
            w.writerow({"t": i, "bucket_start": b})

    # cells.csv: idx -> h3 cell + center lat/lon
    cells = sorted(cell_to_idx.items(), key=lambda kv: kv[1])
    with (out_dir / "cells.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell_idx", "h3_cell", "center_lat", "center_lon"])
        w.writeheader()
        for cell, idx in cells:
            clat, clon = h3.cell_to_latlng(cell)
            w.writerow({"cell_idx": idx, "h3_cell": cell, "center_lat": f"{clat:.6f}", "center_lon": f"{clon:.6f}"})

    feature_names = ["hour_of_week", "how_sin", "how_cos", "month", "season", *weather_cols]
    meta = {
        "h3_resolution": h3_resolution,
        "austin_center_lat": austin_center_lat,
        "austin_center_lon": austin_center_lon,
        "austin_radius_km": austin_radius_km,
        "context_steps": context_steps,
        "n_time_steps": n_time,
        "n_cells": n_cells,
        "feature_names": feature_names,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    np.savez_compressed(
        out_dir / "dataset.npz",
        X=X,
        X_mean=X_mean,
        X_std=X_std,
        coll_ptr=coll_ptr,
        coll_idx=coll_idx,
        coll_val=coll_val,
        inc_ptr=inc_ptr,
        inc_idx=inc_idx,
        inc_val=inc_val,
    )

    return TokenizeH3Stats(
        n_time_steps=n_time,
        n_cells=n_cells,
        collisions_events=collisions_events,
        traffic_incidents_events=traffic_incidents_events,
        collisions_pairs=collisions_pairs,
        traffic_incidents_pairs=traffic_incidents_pairs,
        skipped_missing_time_bucket=skipped_missing_time_bucket,
        skipped_missing_latlon=skipped_missing_latlon,
        skipped_outside_radius=skipped_outside_radius,
    )
