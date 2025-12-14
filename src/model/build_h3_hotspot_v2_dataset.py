from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import load_config
from data_pipeline.feature_factory import _build_spatial_index, _nearest_point, _read_aadt, _read_radar_detectors, _read_radar_hourly
from util import floor_dt, haversine_km


@dataclass(frozen=True)
class BuildH3HotspotV2Stats:
    n_cells: int
    n_buckets: int
    n_samples: int
    n_pos_rows: int
    n_neg_rows: int
    collisions_events: int
    traffic_incidents_events: int
    out_dir: Path


def _latlng_to_cell(h3: Any, *, lat: float, lon: float, res: int) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return str(h3.latlng_to_cell(lat, lon, res))
    return str(h3.geo_to_h3(lat, lon, res))


def _cell_to_latlng(h3: Any, cell: str) -> tuple[float, float]:
    if hasattr(h3, "cell_to_latlng"):
        lat, lon = h3.cell_to_latlng(cell)
        return float(lat), float(lon)
    lat, lon = h3.h3_to_geo(cell)
    return float(lat), float(lon)


def _circle_polygon(
    *, center_lat: float, center_lon: float, radius_km: float, n_points: int = 60
) -> list[tuple[float, float]]:
    if n_points < 12:
        n_points = 12
    lat_scale = radius_km / 111.0
    lon_scale = radius_km / (111.0 * max(math.cos(math.radians(center_lat)), 1e-6))

    pts: list[tuple[float, float]] = []
    for i in range(n_points):
        ang = 2.0 * math.pi * (i / n_points)
        lat = center_lat + lat_scale * math.sin(ang)
        lon = center_lon + lon_scale * math.cos(ang)
        pts.append((lat, lon))
    pts.append(pts[0])
    return pts


def _polyfill_circle(h3: Any, *, center_lat: float, center_lon: float, radius_km: float, res: int) -> set[str]:
    boundary = _circle_polygon(center_lat=center_lat, center_lon=center_lon, radius_km=radius_km)

    if hasattr(h3, "LatLngPoly") and hasattr(h3, "polygon_to_cells"):
        try:
            poly = h3.LatLngPoly(boundary)
            return {str(c) for c in h3.polygon_to_cells(poly, res)}
        except Exception:
            pass

    geo = {
        "type": "Polygon",
        "coordinates": [[[lon, lat] for lat, lon in boundary]],
    }
    if hasattr(h3, "polyfill"):
        return {str(c) for c in h3.polyfill(geo, res, geo_json_conformant=True)}

    raise RuntimeError("h3 module missing polygon fill helpers (need h3.polyfill or h3.polygon_to_cells)")


def _read_weather_hourly(
    weather_hourly_csv: Path, *, datetime_format: str
) -> tuple[list[str], list[datetime], list[list[float]]]:
    by_dt: dict[datetime, list[float]] = {}
    with weather_hourly_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return [], [], []

        cols = [c for c in r.fieldnames if c != "bucket_start"]
        for row in r:
            b = (row.get("bucket_start") or "").strip()
            if not b:
                continue
            try:
                dt = datetime.strptime(b, datetime_format)
            except ValueError:
                continue
            vrow: list[float] = []
            for c in cols:
                s = (row.get(c) or "").strip()
                if not s:
                    vrow.append(0.0)
                    continue
                try:
                    vrow.append(float(s))
                except ValueError:
                    vrow.append(0.0)
            by_dt[dt] = vrow

    buckets = sorted(by_dt.keys())
    values = [by_dt[dt] for dt in buckets]
    return cols, buckets, values


def _season(month: int) -> int:
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def _time_features(dt: datetime) -> tuple[list[float], int]:
    how = dt.weekday() * 24 + dt.hour
    ang = 2 * math.pi * (how / 168.0)
    return [float(how), math.sin(ang), math.cos(ang), float(dt.month), float(_season(dt.month))], int(how)


def _is_collision_issue(issue_reported: str) -> bool:
    s = (issue_reported or "").strip().lower()
    return ("collision" in s) or ("collis" in s) or ("crash" in s)


def _compute_radar_baselines(
    *,
    radar_hourly: dict[tuple[str, datetime], list[float]],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Baselines by detector + hour_of_week (0..167).

    Returns:
      detid -> {"vol": (168,), "speed": (168,), "occ": (168,)}
    """
    vol_sum: dict[str, np.ndarray] = {}
    vol_n: dict[str, np.ndarray] = {}
    speed_sum: dict[str, np.ndarray] = {}
    speed_n: dict[str, np.ndarray] = {}
    occ_sum: dict[str, np.ndarray] = {}
    occ_n: dict[str, np.ndarray] = {}

    for (detid, dt), rec in radar_hourly.items():
        how = dt.weekday() * 24 + dt.hour
        if how < 0 or how >= 168:
            continue

        if detid not in vol_sum:
            vol_sum[detid] = np.zeros(168, dtype=np.float64)
            vol_n[detid] = np.zeros(168, dtype=np.float64)
            speed_sum[detid] = np.zeros(168, dtype=np.float64)
            speed_n[detid] = np.zeros(168, dtype=np.float64)
            occ_sum[detid] = np.zeros(168, dtype=np.float64)
            occ_n[detid] = np.zeros(168, dtype=np.float64)

        vol_sum[detid][how] += float(rec[0])
        vol_n[detid][how] += 1.0
        speed_sum[detid][how] += float(rec[1])
        speed_n[detid][how] += float(rec[2])
        occ_sum[detid][how] += float(rec[3])
        occ_n[detid][how] += float(rec[4])

    out: dict[str, dict[str, np.ndarray]] = {}
    for detid in vol_sum.keys():
        vol = np.zeros(168, dtype=np.float32)
        np.divide(vol_sum[detid], vol_n[detid], out=vol, where=vol_n[detid] > 0)
        vol = np.where(np.isfinite(vol), vol, 0.0).astype(np.float32)

        speed = np.zeros(168, dtype=np.float32)
        np.divide(speed_sum[detid], speed_n[detid], out=speed, where=speed_n[detid] > 0)
        speed = np.where(np.isfinite(speed), speed, 0.0).astype(np.float32)

        occ = np.zeros(168, dtype=np.float32)
        np.divide(occ_sum[detid], occ_n[detid], out=occ, where=occ_n[detid] > 0)
        occ = np.where(np.isfinite(occ), occ, 0.0).astype(np.float32)
        out[detid] = {"vol": vol, "speed": speed, "occ": occ}
    return out


def build_h3_hotspot_v2_dataset(
    *,
    incidents_csv: Path,
    weather_hourly_csv: Path,
    aadt_stations_csv: Path,
    traffic_counts_dir: Path | None,
    out_dir: Path,
    datetime_format: str,
    bucket_minutes: int,
    h3_resolution: int,
    austin_center_lat: float,
    austin_center_lon: float,
    austin_radius_km: float,
    aadt_max_distance_km: float,
    lookback_hours: list[int],
    ema_half_life_hours: float,
    neg_per_hour: int,
    seed: int = 13,
) -> BuildH3HotspotV2Stats:
    import h3  # venv dependency

    bucket_min = int(bucket_minutes)
    if bucket_min <= 0:
        raise ValueError(f"bucket_minutes must be > 0 (got {bucket_minutes})")
    bucket_hours = float(bucket_min) / 60.0

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- timeline (weather)
    weather_cols, bucket_dts, weather_vals = _read_weather_hourly(weather_hourly_csv, datetime_format=datetime_format)
    if not bucket_dts:
        raise ValueError(f"missing/empty weather_hourly.csv: {weather_hourly_csv}")
    if len(bucket_dts) != len(weather_vals):
        raise ValueError("weather_hourly.csv parse mismatch")
    bucket_to_t = {dt.strftime(datetime_format): i for i, dt in enumerate(bucket_dts)}

    # --- cells (full disk)
    cells = sorted(
        _polyfill_circle(
            h3,
            center_lat=austin_center_lat,
            center_lon=austin_center_lon,
            radius_km=austin_radius_km,
            res=h3_resolution,
        )
    )
    cell_to_idx = {c: i for i, c in enumerate(cells)}
    n_cells = len(cells)

    # cells.csv
    with (out_dir / "cells.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell_idx", "h3_cell", "center_lat", "center_lon"])
        w.writeheader()
        for i, cell in enumerate(cells):
            lat, lon = _cell_to_latlng(h3, cell)
            w.writerow({"cell_idx": i, "h3_cell": cell, "center_lat": f"{lat:.6f}", "center_lon": f"{lon:.6f}"})

    # buckets.csv
    with (out_dir / "buckets.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "bucket_start"])
        w.writeheader()
        for i, dt in enumerate(bucket_dts):
            w.writerow({"t": i, "bucket_start": dt.strftime(datetime_format)})

    # --- static joins (AADT + radar baseline by hour_of_week)
    stations = _read_aadt(aadt_stations_csv)
    station_index = _build_spatial_index(stations, max_distance_km=aadt_max_distance_km)

    radar_hourly: dict[tuple[str, datetime], list[float]] = {}
    radar_detectors: list[dict[str, float | str]] = []
    radar_index: dict[tuple[int, int], list[int]] = {}
    radar_baselines: dict[str, dict[str, np.ndarray]] = {}
    if traffic_counts_dir is not None:
        radar_dir = traffic_counts_dir / "counts" / "i626-g7ub"
        detectors_csv = traffic_counts_dir / "lookups" / "qpuw-8eeb.csv"
        if radar_dir.exists() and detectors_csv.exists():
            radar_hourly, radar_detids = _read_radar_hourly(radar_dir, bucket_minutes=bucket_minutes)
            radar_detectors = _read_radar_detectors(detectors_csv, allowed_detector_ids=radar_detids)
            radar_index = _build_spatial_index(radar_detectors, max_distance_km=aadt_max_distance_km)
            radar_baselines = _compute_radar_baselines(radar_hourly=radar_hourly)

    aadt_log1p = np.zeros(n_cells, dtype=np.float32)
    aadt_dist_km = np.zeros(n_cells, dtype=np.float32)
    has_aadt = np.zeros(n_cells, dtype=np.float32)

    radar_dist_km = np.zeros(n_cells, dtype=np.float32)
    has_radar = np.zeros(n_cells, dtype=np.float32)
    radar_vol = np.zeros((n_cells, 168), dtype=np.float32)
    radar_speed = np.zeros((n_cells, 168), dtype=np.float32)
    radar_occ = np.zeros((n_cells, 168), dtype=np.float32)

    for i, cell in enumerate(cells):
        lat, lon = _cell_to_latlng(h3, cell)
        if haversine_km(lat, lon, austin_center_lat, austin_center_lon) > austin_radius_km + 1.0:
            continue

        station, sdist = _nearest_point(
            lat=lat, lon=lon, points=stations, index=station_index, max_distance_km=aadt_max_distance_km
        )
        if station is not None:
            has_aadt[i] = 1.0
            aadt_log1p[i] = float(station.get("aadt_log1p") or 0.0)
            aadt_dist_km[i] = float(sdist or 0.0)

        det, ddist = _nearest_point(
            lat=lat, lon=lon, points=radar_detectors, index=radar_index, max_distance_km=aadt_max_distance_km
        )
        detid = (det.get("detector_id") or "").strip() if det else ""
        if detid and detid in radar_baselines:
            has_radar[i] = 1.0
            radar_dist_km[i] = float(ddist or 0.0)
            b = radar_baselines[detid]
            radar_vol[i, :] = b["vol"]
            radar_speed[i, :] = b["speed"]
            radar_occ[i, :] = b["occ"]

    np.savez_compressed(
        out_dir / "cell_static.npz",
        aadt_log1p=aadt_log1p,
        aadt_distance_km=aadt_dist_km,
        has_aadt=has_aadt,
        radar_distance_km=radar_dist_km,
        has_radar=has_radar,
        radar_vol_baseline=radar_vol,
        radar_speed_baseline=radar_speed,
        radar_occ_baseline=radar_occ,
    )

    # --- incidents -> sparse events_by_t (counts per cell per bucket)
    events_by_t: dict[int, dict[int, list[int]]] = {}
    collisions_events = 0
    traffic_incidents_events = 0

    with incidents_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("incidents.csv missing header row")

        for row in r:
            dt_raw = (row.get("published_date") or "").strip()
            if not dt_raw:
                continue
            try:
                dt = datetime.strptime(dt_raw, datetime_format)
            except ValueError:
                continue
            bucket_dt = floor_dt(dt, bucket_minutes=bucket_minutes)
            t = bucket_to_t.get(bucket_dt.strftime(datetime_format))
            if t is None:
                continue

            lat_s = (row.get("latitude") or "").strip()
            lon_s = (row.get("longitude") or "").strip()
            if not lat_s or not lon_s:
                continue
            try:
                lat = float(lat_s)
                lon = float(lon_s)
            except ValueError:
                continue

            if haversine_km(lat, lon, austin_center_lat, austin_center_lon) > austin_radius_km:
                continue

            cell = _latlng_to_cell(h3, lat=lat, lon=lon, res=h3_resolution)
            idx = cell_to_idx.get(cell)
            if idx is None:
                continue

            d = events_by_t.get(t)
            if d is None:
                d = {}
                events_by_t[t] = d
            rec = d.get(idx)
            if rec is None:
                rec = [0, 0]  # [coll, inc]
                d[idx] = rec

            event_class = (row.get("event_class") or "").strip()
            if event_class == "collision" or _is_collision_issue((row.get("issue_reported") or "").strip()):
                rec[0] += 1
                collisions_events += 1
            else:
                rec[1] += 1
                traffic_incidents_events += 1

    # --- dataset assembly (sampled negatives + all positives)
    rng = random.Random(int(seed))
    n_time = len(bucket_dts)
    lookbacks = sorted({int(h) for h in lookback_hours if int(h) > 0})
    lookback_steps = {h: max(1, int(math.ceil(float(h) / max(bucket_hours, 1e-9)))) for h in lookbacks}
    max_lb_steps = max(lookback_steps.values()) if lookback_steps else 0

    # AR state (history < current bucket)
    ema_coll = np.zeros(n_cells, dtype=np.float32)
    ema_inc = np.zeros(n_cells, dtype=np.float32)
    last_coll_t = np.full(n_cells, -1, dtype=np.int32)
    last_inc_t = np.full(n_cells, -1, dtype=np.int32)
    window_pos = 0
    coll_window = np.zeros((n_cells, max_lb_steps), dtype=np.float32) if max_lb_steps > 0 else None
    inc_window = np.zeros((n_cells, max_lb_steps), dtype=np.float32) if max_lb_steps > 0 else None

    def lb_sum(window: np.ndarray, *, idx: int, steps: int) -> float:
        total = 0.0
        for k in range(1, steps + 1):
            j = (window_pos - k) % max_lb_steps
            total += float(window[idx, j])
        return total

    weather_vals_arr = np.asarray(weather_vals, dtype=np.float32)

    try:
        import holidays  # type: ignore
    except Exception:  # pragma: no cover
        holidays = None  # type: ignore[assignment]
    us_holidays = holidays.US() if holidays is not None else None

    def holiday_flag(dt: datetime) -> float:
        if us_holidays is None:
            return 0.0
        return 1.0 if dt.date() in us_holidays else 0.0

    feature_names: list[str] = [
        "hour_of_week",
        "how_sin",
        "how_cos",
        "month",
        "season",
        "is_holiday",
        *weather_cols,
        "aadt_log1p",
        "aadt_distance_km",
        "has_aadt",
        "radar_vol_baseline",
        "radar_speed_baseline",
        "radar_occ_baseline",
        "radar_distance_km",
        "has_radar",
    ]
    for h in lookbacks:
        feature_names.extend([f"n_collisions_lb_{h}h", f"n_traffic_incidents_lb_{h}h"])
    feature_names.extend(
        [
            "ema_collisions",
            "ema_traffic_incidents",
            "hrs_since_last_collision",
            "hrs_since_last_traffic_incident",
        ]
    )
    n_features = len(feature_names)

    X_blocks: list[np.ndarray] = []
    y_coll_blocks: list[np.ndarray] = []
    y_inc_blocks: list[np.ndarray] = []
    cell_blocks: list[np.ndarray] = []
    t_blocks: list[np.ndarray] = []

    n_pos_rows = 0
    n_neg_rows = 0
    last_dt: datetime | None = None

    for t, bucket_dt in enumerate(bucket_dts):
        # EMA decay to the current time (history only).
        if last_dt is not None and ema_half_life_hours > 0:
            dt_hours = (bucket_dt - last_dt).total_seconds() / 3600.0
            decay = float(math.exp(-math.log(2.0) * (dt_hours / float(ema_half_life_hours))))
            ema_coll *= decay
            ema_inc *= decay

        time_vec, how = _time_features(bucket_dt)
        time_vec = list(time_vec) + [holiday_flag(bucket_dt)]
        weather_vec = weather_vals_arr[t].tolist()

        events = events_by_t.get(t, {})
        pos_cells = sorted([i for i, (c, n) in events.items() if (c > 0 or n > 0)])
        pos_set = set(pos_cells)

        neg_n = max(int(neg_per_hour), 0)
        neg_cells: set[int] = set()
        if neg_n > 0:
            while len(neg_cells) < neg_n and len(neg_cells) < n_cells - len(pos_set):
                i = rng.randrange(n_cells)
                if i in pos_set:
                    continue
                neg_cells.add(i)

        chosen = pos_cells + sorted(neg_cells)
        if not chosen:
            # No positives and no negatives requested.
            last_dt = bucket_dt
            continue

        Xb = np.zeros((len(chosen), n_features), dtype=np.float32)
        ycb = np.zeros(len(chosen), dtype=np.float32)
        yib = np.zeros(len(chosen), dtype=np.float32)
        cb = np.zeros(len(chosen), dtype=np.int32)
        tb = np.full(len(chosen), t, dtype=np.int32)

        for j, cell_idx in enumerate(chosen):
            cb[j] = int(cell_idx)

            rec = events.get(cell_idx) or [0, 0]
            y_coll = 1.0 if int(rec[0]) > 0 else 0.0
            y_inc = 1.0 if int(rec[1]) > 0 else 0.0
            ycb[j] = y_coll
            yib[j] = y_inc
            if y_coll > 0 or y_inc > 0:
                n_pos_rows += 1
            else:
                n_neg_rows += 1

            feats: list[float] = []
            feats.extend(time_vec)
            feats.extend(weather_vec)

            feats.extend([float(aadt_log1p[cell_idx]), float(aadt_dist_km[cell_idx]), float(has_aadt[cell_idx])])
            feats.extend(
                [
                    float(radar_vol[cell_idx, how]),
                    float(radar_speed[cell_idx, how]),
                    float(radar_occ[cell_idx, how]),
                    float(radar_dist_km[cell_idx]),
                    float(has_radar[cell_idx]),
                ]
            )

            if max_lb_steps > 0 and coll_window is not None and inc_window is not None:
                for h in lookbacks:
                    steps = int(lookback_steps[h])
                    feats.append(lb_sum(coll_window, idx=cell_idx, steps=steps))
                    feats.append(lb_sum(inc_window, idx=cell_idx, steps=steps))
            else:
                for _h in lookbacks:
                    feats.extend([0.0, 0.0])

            last_coll_hrs = (
                999.0
                if last_coll_t[cell_idx] < 0
                else float((bucket_dt - bucket_dts[int(last_coll_t[cell_idx])]).total_seconds() / 3600.0)
            )
            last_inc_hrs = (
                999.0
                if last_inc_t[cell_idx] < 0
                else float((bucket_dt - bucket_dts[int(last_inc_t[cell_idx])]).total_seconds() / 3600.0)
            )
            feats.extend(
                [
                    float(ema_coll[cell_idx]),
                    float(ema_inc[cell_idx]),
                    last_coll_hrs,
                    last_inc_hrs,
                ]
            )

            Xb[j, :] = np.asarray(feats, dtype=np.float32)

        X_blocks.append(Xb)
        y_coll_blocks.append(ycb)
        y_inc_blocks.append(yib)
        cell_blocks.append(cb)
        t_blocks.append(tb)

        # Update history with this bucket's realized events.
        if max_lb_steps > 0 and coll_window is not None and inc_window is not None:
            coll_window[:, window_pos] = 0.0
            inc_window[:, window_pos] = 0.0
            for cell_idx, (c, n) in events.items():
                if c:
                    coll_window[cell_idx, window_pos] = float(c)
                if n:
                    inc_window[cell_idx, window_pos] = float(n)
            window_pos = (window_pos + 1) % max_lb_steps

        if ema_half_life_hours > 0:
            for cell_idx, (c, n) in events.items():
                if c:
                    ema_coll[cell_idx] += float(c)
                if n:
                    ema_inc[cell_idx] += float(n)

        for cell_idx, (c, n) in events.items():
            if c:
                last_coll_t[cell_idx] = int(t)
            if n:
                last_inc_t[cell_idx] = int(t)

        last_dt = bucket_dt

    X = np.concatenate(X_blocks, axis=0) if X_blocks else np.zeros((0, n_features), dtype=np.float32)
    y_coll = np.concatenate(y_coll_blocks, axis=0) if y_coll_blocks else np.zeros((0,), dtype=np.float32)
    y_inc = np.concatenate(y_inc_blocks, axis=0) if y_inc_blocks else np.zeros((0,), dtype=np.float32)
    cell_idx = np.concatenate(cell_blocks, axis=0) if cell_blocks else np.zeros((0,), dtype=np.int32)
    bucket_idx = np.concatenate(t_blocks, axis=0) if t_blocks else np.zeros((0,), dtype=np.int32)

    if X.shape[0] == 0:
        raise ValueError("no samples produced (check inputs)")

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std = np.where(X_std == 0, 1.0, X_std)

    np.savez_compressed(
        out_dir / "dataset.npz",
        X=X.astype(np.float32),
        X_mean=X_mean.astype(np.float32),
        X_std=X_std.astype(np.float32),
        y_coll=y_coll.astype(np.float32),
        y_inc=y_inc.astype(np.float32),
        cell_idx=cell_idx.astype(np.int32),
        bucket_idx=bucket_idx.astype(np.int32),
    )

    meta = {
        "h3_resolution": int(h3_resolution),
        "austin_center_lat": float(austin_center_lat),
        "austin_center_lon": float(austin_center_lon),
        "austin_radius_km": float(austin_radius_km),
        "bucket_minutes": int(bucket_minutes),
        "neg_per_hour": int(neg_per_hour),
        "lookback_hours": [int(h) for h in lookbacks],
        "ema_half_life_hours": float(ema_half_life_hours),
        "n_cells": int(n_cells),
        "n_buckets": int(n_time),
        "n_samples": int(X.shape[0]),
        "feature_names": feature_names,
        "labels": ["y_coll", "y_inc"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return BuildH3HotspotV2Stats(
        n_cells=n_cells,
        n_buckets=n_time,
        n_samples=int(X.shape[0]),
        n_pos_rows=int(n_pos_rows),
        n_neg_rows=int(n_neg_rows),
        collisions_events=int(collisions_events),
        traffic_incidents_events=int(traffic_incidents_events),
        out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build v2 enriched H3 hotspot dataset (gold tokens) from silver + joins")
    ap.add_argument("--config", default="configs/pipeline.toml")
    ap.add_argument("--out-dir", default=None, help="Override config hotspot_v2.output_dir")
    ap.add_argument("--neg-per-hour", type=int, default=None, help="Override config hotspot_v2.neg_per_hour")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    cfg = load_config(args.config)

    incidents_csv = cfg.paths.silver_dir / cfg.silverize.incidents_output_name
    weather_csv = cfg.paths.silver_dir / cfg.silverize.weather_output_name
    aadt_csv = cfg.paths.silver_dir / cfg.silverize.aadt_output_name
    traffic_counts_dir = cfg.paths.bronze_dir / "austin_traffic_counts"

    out_dir = Path(args.out_dir) if args.out_dir else cfg.hotspot_v2.output_dir
    neg_per_hour = int(args.neg_per_hour) if args.neg_per_hour is not None else int(cfg.hotspot_v2.neg_per_hour)

    stats = build_h3_hotspot_v2_dataset(
        incidents_csv=incidents_csv,
        weather_hourly_csv=weather_csv,
        aadt_stations_csv=aadt_csv,
        traffic_counts_dir=traffic_counts_dir if traffic_counts_dir.exists() else None,
        out_dir=out_dir,
        datetime_format=cfg.silverize.datetime_format,
        bucket_minutes=int(cfg.features.bucket_minutes),
        h3_resolution=int(cfg.tokenizer.h3_resolution),
        austin_center_lat=float(cfg.tokenizer.austin_center_lat),
        austin_center_lon=float(cfg.tokenizer.austin_center_lon),
        austin_radius_km=float(cfg.tokenizer.austin_radius_km),
        aadt_max_distance_km=float(cfg.features.aadt_max_distance_km),
        lookback_hours=cfg.features.lookback_hours,
        ema_half_life_hours=float(cfg.features.ema_half_life_hours),
        neg_per_hour=neg_per_hour,
        seed=int(args.seed),
    )

    print(
        json.dumps(
            {
                "out_dir": str(stats.out_dir),
                "cells": stats.n_cells,
                "buckets": stats.n_buckets,
                "samples": stats.n_samples,
                "pos_rows": stats.n_pos_rows,
                "neg_rows": stats.n_neg_rows,
                "collisions_events": stats.collisions_events,
                "traffic_incidents_events": stats.traffic_incidents_events,
            }
        )
    )


if __name__ == "__main__":
    main()
