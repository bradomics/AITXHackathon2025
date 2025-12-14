from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import load_config
from util import floor_dt


def _latlng_to_cell(h3: Any, *, lat: float, lon: float, res: int) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return str(h3.latlng_to_cell(lat, lon, res))
    return str(h3.geo_to_h3(lat, lon, res))


def _season_id(month: int) -> int:
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def _is_collision_issue(issue_reported: str) -> bool:
    s = (issue_reported or "").strip().lower()
    return ("collision" in s) or ("collis" in s) or ("crash" in s)


def _time_features(*, dt: datetime, us_holidays: Any) -> tuple[list[float], int]:
    how = dt.weekday() * 24 + dt.hour
    ang = 2.0 * math.pi * (how / 168.0)
    is_holiday = 1.0 if us_holidays is not None and dt.date() in us_holidays else 0.0
    return [
        float(how),
        float(math.sin(ang)),
        float(math.cos(ang)),
        float(dt.month),
        float(_season_id(dt.month)),
        float(is_holiday),
    ], int(how)


def _open_text_out(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def _read_cells(cells_csv: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    h3_cells: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    with cells_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            h3_cells.append(str(row.get("h3_cell") or "").strip())
            lats.append(float(row["center_lat"]))
            lons.append(float(row["center_lon"]))
    return h3_cells, np.asarray(lats, dtype=np.float32), np.asarray(lons, dtype=np.float32)


def _read_buckets(buckets_csv: Path, *, datetime_format: str) -> tuple[list[str], list[datetime]]:
    starts: list[str] = []
    dts: list[datetime] = []
    with buckets_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            s = str(row.get("bucket_start") or "").strip()
            if not s:
                continue
            starts.append(s)
            dts.append(datetime.strptime(s, datetime_format))
    return starts, dts


def _read_weather(weather_csv: Path) -> tuple[list[str], dict[str, list[float]]]:
    if not weather_csv.exists():
        raise FileNotFoundError(weather_csv)
    by_bucket: dict[str, list[float]] = {}
    with weather_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"missing header: {weather_csv}")
        cols = [c for c in r.fieldnames if c != "bucket_start"]
        for row in r:
            b = str(row.get("bucket_start") or "").strip()
            if not b:
                continue
            vals: list[float] = []
            for c in cols:
                s = str(row.get(c) or "").strip()
                if not s:
                    vals.append(0.0)
                    continue
                try:
                    vals.append(float(s))
                except ValueError:
                    vals.append(0.0)
            by_bucket[b] = vals
    return cols, by_bucket


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a dense (cell×bucket) v2 feature grid CSV")
    ap.add_argument("--config", default="configs/pipeline.toml")
    ap.add_argument("--data-dir", default=None, help="Defaults to config hotspot_v2.output_dir")
    ap.add_argument("--out-csv", default="data/gold/features/hotspot_features_v2_grid.csv")
    ap.add_argument("--max-buckets", type=int, default=0, help="Debug: only write first N buckets (0=all)")
    ap.add_argument("--progress-every", type=int, default=1000, help="Log progress every N buckets (0=quiet)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else cfg.hotspot_v2.output_dir

    ds_path = data_dir / "dataset.npz"
    meta_path = data_dir / "meta.json"
    cells_csv = data_dir / "cells.csv"
    buckets_csv = data_dir / "buckets.csv"
    static_npz = data_dir / "cell_static.npz"

    for p in (ds_path, meta_path, cells_csv, buckets_csv, static_npz):
        if not p.exists():
            raise FileNotFoundError(p)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_names = list(meta.get("feature_names") or [])
    if not feature_names:
        raise ValueError(f"missing feature_names in {meta_path}")

    bucket_minutes = int(meta.get("bucket_minutes") or 0)
    if bucket_minutes <= 0:
        raise ValueError("invalid meta: bucket_minutes")

    h3_resolution = int(meta.get("h3_resolution") or 0)
    if h3_resolution <= 0:
        raise ValueError("invalid meta: h3_resolution")

    ema_half_life_hours = float(meta.get("ema_half_life_hours") or 0.0)

    lookbacks = sorted({int(h) for h in (meta.get("lookback_hours") or []) if int(h) > 0})

    time_cols = ["hour_of_week", "how_sin", "how_cos", "month", "season", "is_holiday"]
    if feature_names[: len(time_cols)] != time_cols:
        raise ValueError("unexpected v2 feature schema (time columns)")

    weather_cols, weather_by_bucket = _read_weather(cfg.paths.silver_dir / cfg.silverize.weather_output_name)
    w0 = len(time_cols)
    w1 = w0 + len(weather_cols)
    if feature_names[w0:w1] != weather_cols:
        raise ValueError("unexpected v2 feature schema (weather columns)")

    static_cols = [
        "aadt_log1p",
        "aadt_distance_km",
        "has_aadt",
        "radar_vol_baseline",
        "radar_speed_baseline",
        "radar_occ_baseline",
        "radar_distance_km",
        "has_radar",
    ]
    s0 = w1
    s1 = s0 + len(static_cols)
    if feature_names[s0:s1] != static_cols:
        raise ValueError("unexpected v2 feature schema (static columns)")

    lb_cols: list[str] = []
    for h in lookbacks:
        lb_cols.extend([f"n_collisions_lb_{h}h", f"n_traffic_incidents_lb_{h}h"])
    lb0 = s1
    lb1 = lb0 + len(lb_cols)
    if feature_names[lb0:lb1] != lb_cols:
        raise ValueError("unexpected v2 feature schema (lookback columns)")

    tail_cols = [
        "ema_collisions",
        "ema_traffic_incidents",
        "hrs_since_last_collision",
        "hrs_since_last_traffic_incident",
    ]
    t0 = lb1
    t1 = t0 + len(tail_cols)
    if feature_names[t0:t1] != tail_cols or t1 != len(feature_names):
        raise ValueError("unexpected v2 feature schema (tail columns)")

    h3_cells, cell_lats, cell_lons = _read_cells(cells_csv)
    n_cells = int(cell_lats.shape[0])
    if int(meta.get("n_cells") or 0) != n_cells:
        raise ValueError("meta n_cells mismatch")

    bucket_starts, bucket_dts = _read_buckets(buckets_csv, datetime_format=cfg.silverize.datetime_format)
    n_buckets = len(bucket_starts)
    if int(meta.get("n_buckets") or 0) != n_buckets:
        raise ValueError("meta n_buckets mismatch")

    bucket_to_t = {s: i for i, s in enumerate(bucket_starts)}

    weather_mat = np.zeros((n_buckets, len(weather_cols)), dtype=np.float32)
    for i, s in enumerate(bucket_starts):
        v = weather_by_bucket.get(s)
        if v is None:
            continue
        if len(v) != len(weather_cols):
            raise ValueError("weather column width mismatch")
        weather_mat[i, :] = np.asarray(v, dtype=np.float32)

    static = np.load(static_npz)
    aadt_log1p = static["aadt_log1p"].astype(np.float32)
    aadt_distance_km = static["aadt_distance_km"].astype(np.float32)
    has_aadt = static["has_aadt"].astype(np.float32)
    radar_distance_km = static["radar_distance_km"].astype(np.float32)
    has_radar = static["has_radar"].astype(np.float32)
    radar_vol_baseline = static["radar_vol_baseline"].astype(np.float32)
    radar_speed_baseline = static["radar_speed_baseline"].astype(np.float32)
    radar_occ_baseline = static["radar_occ_baseline"].astype(np.float32)

    if (
        aadt_log1p.shape[0] != n_cells
        or radar_distance_km.shape[0] != n_cells
        or radar_vol_baseline.shape[0] != n_cells
        or radar_vol_baseline.shape[1] != 168
    ):
        raise ValueError("cell_static.npz shape mismatch")

    cell_to_idx = {c: i for i, c in enumerate(h3_cells) if c}
    events_by_t: dict[int, dict[int, list[int]]] = {}

    import h3  # venv dependency

    incidents_csv = cfg.paths.silver_dir / cfg.silverize.incidents_output_name
    with incidents_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("incidents.csv missing header row")

        for row in r:
            dt_raw = str(row.get("published_date") or "").strip()
            if not dt_raw:
                continue
            try:
                dt = datetime.strptime(dt_raw, cfg.silverize.datetime_format)
            except ValueError:
                continue

            bucket_dt = floor_dt(dt, bucket_minutes=bucket_minutes)
            t = bucket_to_t.get(bucket_dt.strftime(cfg.silverize.datetime_format))
            if t is None:
                continue

            lat_s = str(row.get("latitude") or "").strip()
            lon_s = str(row.get("longitude") or "").strip()
            if not lat_s or not lon_s:
                continue
            try:
                lat = float(lat_s)
                lon = float(lon_s)
            except ValueError:
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
                rec = [0, 0]  # [collisions, traffic_incidents]
                d[idx] = rec

            event_class = str(row.get("event_class") or "").strip()
            if event_class == "collision" or _is_collision_issue(str(row.get("issue_reported") or "").strip()):
                rec[0] += 1
            else:
                rec[1] += 1

    try:
        import holidays  # type: ignore
    except Exception:  # pragma: no cover
        holidays = None  # type: ignore[assignment]
    us_holidays = holidays.US() if holidays is not None else None

    bucket_hours = float(bucket_minutes) / 60.0
    lookback_steps = {h: max(1, int(math.ceil(float(h) / max(bucket_hours, 1e-9)))) for h in lookbacks}
    max_lb_steps = max(lookback_steps.values()) if lookback_steps else 0

    ema_coll = np.zeros(n_cells, dtype=np.float32)
    ema_inc = np.zeros(n_cells, dtype=np.float32)
    last_coll_t = np.full(n_cells, -1, dtype=np.int32)
    last_inc_t = np.full(n_cells, -1, dtype=np.int32)
    window_pos = 0
    coll_window = np.zeros((n_cells, max_lb_steps), dtype=np.float32) if max_lb_steps > 0 else None
    inc_window = np.zeros((n_cells, max_lb_steps), dtype=np.float32) if max_lb_steps > 0 else None

    out_path = Path(args.out_csv)
    with _open_text_out(out_path) as f_out:
        w = csv.writer(f_out)
        w.writerow(
            [
                "bucket_start",
                "bucket_idx",
                "cell_idx",
                "h3_cell",
                "center_lat",
                "center_lon",
                *feature_names,
                "y_coll",
                "y_inc",
            ]
        )

        last_dt: datetime | None = None
        max_buckets = int(args.max_buckets)
        progress_every = int(args.progress_every)

        wrote_rows = 0
        wrote_buckets = 0

        for t in range(n_buckets):
            if max_buckets > 0 and t >= max_buckets:
                break

            bucket_dt = bucket_dts[t]
            if last_dt is not None and ema_half_life_hours > 0:
                dt_hours = (bucket_dt - last_dt).total_seconds() / 3600.0
                if dt_hours > 0:
                    decay = float(math.exp(-math.log(2.0) * (dt_hours / float(ema_half_life_hours))))
                    ema_coll *= decay
                    ema_inc *= decay

            time_vec, how = _time_features(dt=bucket_dt, us_holidays=us_holidays)
            weather_vec = weather_mat[t, :].tolist()

            lb_coll_by_h: dict[int, np.ndarray] = {}
            lb_inc_by_h: dict[int, np.ndarray] = {}
            if max_lb_steps > 0 and coll_window is not None and inc_window is not None:
                for h, steps in lookback_steps.items():
                    idxs = [(window_pos - k) % max_lb_steps for k in range(1, steps + 1)]
                    lb_coll_by_h[h] = coll_window[:, idxs].sum(axis=1)
                    lb_inc_by_h[h] = inc_window[:, idxs].sum(axis=1)

            hrs_since_coll = np.full(n_cells, 999.0, dtype=np.float32)
            hrs_since_inc = np.full(n_cells, 999.0, dtype=np.float32)
            for i in range(n_cells):
                lc = int(last_coll_t[i])
                if lc >= 0:
                    hrs_since_coll[i] = float((bucket_dt - bucket_dts[lc]).total_seconds() / 3600.0)
                li = int(last_inc_t[i])
                if li >= 0:
                    hrs_since_inc[i] = float((bucket_dt - bucket_dts[li]).total_seconds() / 3600.0)

            events = events_by_t.get(t, {})

            for i in range(n_cells):
                c, n = events.get(i, [0, 0])
                y_coll = 1 if int(c) > 0 else 0
                y_inc = 1 if int(n) > 0 else 0

                row: list[object] = [
                    bucket_starts[t],
                    int(t),
                    int(i),
                    h3_cells[i],
                    float(cell_lats[i]),
                    float(cell_lons[i]),
                ]
                row.extend(time_vec)
                row.extend(weather_vec)
                row.extend(
                    [
                        float(aadt_log1p[i]),
                        float(aadt_distance_km[i]),
                        float(has_aadt[i]),
                        float(radar_vol_baseline[i, how]),
                        float(radar_speed_baseline[i, how]),
                        float(radar_occ_baseline[i, how]),
                        float(radar_distance_km[i]),
                        float(has_radar[i]),
                    ]
                )
                for h in lookbacks:
                    row.append(float(lb_coll_by_h.get(h, 0.0)[i] if h in lb_coll_by_h else 0.0))
                    row.append(float(lb_inc_by_h.get(h, 0.0)[i] if h in lb_inc_by_h else 0.0))
                row.extend([float(ema_coll[i]), float(ema_inc[i]), float(hrs_since_coll[i]), float(hrs_since_inc[i])])
                row.extend([int(y_coll), int(y_inc)])
                w.writerow(row)
                wrote_rows += 1

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
            wrote_buckets += 1

            if progress_every > 0 and (wrote_buckets % progress_every == 0):
                print(f"[export_v2_grid] buckets={wrote_buckets}/{n_buckets} rows={wrote_rows}", flush=True)

    print(
        json.dumps(
            {
                "out_csv": str(out_path),
                "n_cells": int(n_cells),
                "n_buckets_written": int(wrote_buckets),
                "rows_written": int(wrote_rows),
            }
        )
    )


if __name__ == "__main__":
    main()
