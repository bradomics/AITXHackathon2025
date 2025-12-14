from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from util import floor_dt

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


_DEFAULT_TZ = "America/Chicago"


@dataclass
class ArStateV2:
    processed_bytes: int
    bucket_minutes: int
    last_ema_bucket_id: int
    ema_coll: np.ndarray
    ema_inc: np.ndarray
    last_coll_bucket_id: np.ndarray
    last_inc_bucket_id: np.ndarray
    events_by_bucket: dict[int, dict[int, tuple[int, int]]]


def _latlng_to_cell(h3: Any, *, lat: float, lon: float, res: int) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return str(h3.latlng_to_cell(lat, lon, res))
    return str(h3.geo_to_h3(lat, lon, res))


def _bucket_id(dt_local: datetime, *, tz_name: str, bucket_minutes: int) -> int:
    if ZoneInfo is None:
        # Fallback: treat naive dt as already epoch-compatible (best-effort).
        sec = float(dt_local.timestamp())
        return int(sec // (max(int(bucket_minutes), 1) * 60))
    tz = ZoneInfo(tz_name)
    aware = dt_local.replace(tzinfo=tz)
    sec = float(aware.timestamp())
    return int(sec // (max(int(bucket_minutes), 1) * 60))


def _parse_dt_local(s: str, *, datetime_format: str) -> datetime:
    return datetime.strptime((s or "").strip(), datetime_format)


def _is_collision_issue(issue_reported: str) -> bool:
    s = (issue_reported or "").strip().lower()
    return ("collision" in s) or ("collis" in s) or ("crash" in s)


def load_ar_state_v2(*, state_path: Path, n_cells: int, bucket_minutes: int) -> ArStateV2:
    bucket_min = int(bucket_minutes)
    if bucket_min <= 0:
        raise ValueError(f"bucket_minutes must be > 0 (got {bucket_minutes})")

    if not state_path.exists():
        return ArStateV2(
            processed_bytes=0,
            bucket_minutes=bucket_min,
            last_ema_bucket_id=-1,
            ema_coll=np.zeros(n_cells, dtype=np.float32),
            ema_inc=np.zeros(n_cells, dtype=np.float32),
            last_coll_bucket_id=np.full(n_cells, -1, dtype=np.int64),
            last_inc_bucket_id=np.full(n_cells, -1, dtype=np.int64),
            events_by_bucket={},
        )

    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw_bucket_min = int(raw.get("bucket_minutes") or 60)
    if int(raw.get("n_cells") or 0) != int(n_cells):
        return ArStateV2(
            processed_bytes=0,
            bucket_minutes=bucket_min,
            last_ema_bucket_id=-1,
            ema_coll=np.zeros(n_cells, dtype=np.float32),
            ema_inc=np.zeros(n_cells, dtype=np.float32),
            last_coll_bucket_id=np.full(n_cells, -1, dtype=np.int64),
            last_inc_bucket_id=np.full(n_cells, -1, dtype=np.int64),
            events_by_bucket={},
        )
    if raw_bucket_min != bucket_min:
        return ArStateV2(
            processed_bytes=0,
            bucket_minutes=bucket_min,
            last_ema_bucket_id=-1,
            ema_coll=np.zeros(n_cells, dtype=np.float32),
            ema_inc=np.zeros(n_cells, dtype=np.float32),
            last_coll_bucket_id=np.full(n_cells, -1, dtype=np.int64),
            last_inc_bucket_id=np.full(n_cells, -1, dtype=np.int64),
            events_by_bucket={},
        )

    def arr(name: str, *, default: float | int, dtype: Any) -> np.ndarray:
        v = raw.get(name)
        if not isinstance(v, list) or len(v) != n_cells:
            return np.full(n_cells, default, dtype=dtype)
        return np.asarray(v, dtype=dtype)

    version = int(raw.get("version") or 1)

    # Back-compat: v1 stored hour IDs and hourly buckets. That matches bucket_minutes=60.
    if version == 1:
        if bucket_min != 60:
            return ArStateV2(
                processed_bytes=0,
                bucket_minutes=bucket_min,
                last_ema_bucket_id=-1,
                ema_coll=np.zeros(n_cells, dtype=np.float32),
                ema_inc=np.zeros(n_cells, dtype=np.float32),
                last_coll_bucket_id=np.full(n_cells, -1, dtype=np.int64),
                last_inc_bucket_id=np.full(n_cells, -1, dtype=np.int64),
                events_by_bucket={},
            )
        events_by_bucket: dict[int, dict[int, tuple[int, int]]] = {}
        raw_events = raw.get("events_by_hour") or {}
        if isinstance(raw_events, dict):
            for h_s, d in raw_events.items():
                try:
                    bid = int(h_s)
                except ValueError:
                    continue
                if not isinstance(d, dict):
                    continue
                bucket: dict[int, tuple[int, int]] = {}
                for k_s, pair in d.items():
                    try:
                        cell_idx = int(k_s)
                    except ValueError:
                        continue
                    if not isinstance(pair, list) or len(pair) != 2:
                        continue
                    try:
                        c = int(pair[0])
                        n = int(pair[1])
                    except Exception:
                        continue
                    if c == 0 and n == 0:
                        continue
                    bucket[cell_idx] = (c, n)
                if bucket:
                    events_by_bucket[bid] = bucket
        return ArStateV2(
            processed_bytes=int(raw.get("processed_bytes") or 0),
            bucket_minutes=bucket_min,
            last_ema_bucket_id=int(raw.get("last_ema_hour_id") or -1),
            ema_coll=arr("ema_coll", default=0.0, dtype=np.float32),
            ema_inc=arr("ema_inc", default=0.0, dtype=np.float32),
            last_coll_bucket_id=arr("last_coll_hour_id", default=-1, dtype=np.int64),
            last_inc_bucket_id=arr("last_inc_hour_id", default=-1, dtype=np.int64),
            events_by_bucket=events_by_bucket,
        )

    events_by_bucket: dict[int, dict[int, tuple[int, int]]] = {}
    raw_events = raw.get("events_by_bucket") or {}
    if isinstance(raw_events, dict):
        for b_s, d in raw_events.items():
            try:
                bid = int(b_s)
            except ValueError:
                continue
            if not isinstance(d, dict):
                continue
            bucket: dict[int, tuple[int, int]] = {}
            for k_s, pair in d.items():
                try:
                    cell_idx = int(k_s)
                except ValueError:
                    continue
                if not isinstance(pair, list) or len(pair) != 2:
                    continue
                try:
                    c = int(pair[0])
                    n = int(pair[1])
                except Exception:
                    continue
                if c == 0 and n == 0:
                    continue
                bucket[cell_idx] = (c, n)
            if bucket:
                events_by_bucket[bid] = bucket

    return ArStateV2(
        processed_bytes=int(raw.get("processed_bytes") or 0),
        bucket_minutes=bucket_min,
        last_ema_bucket_id=int(raw.get("last_ema_bucket_id") or -1),
        ema_coll=arr("ema_coll", default=0.0, dtype=np.float32),
        ema_inc=arr("ema_inc", default=0.0, dtype=np.float32),
        last_coll_bucket_id=arr("last_coll_bucket_id", default=-1, dtype=np.int64),
        last_inc_bucket_id=arr("last_inc_bucket_id", default=-1, dtype=np.int64),
        events_by_bucket=events_by_bucket,
    )


def save_ar_state_v2(*, state_path: Path, state: ArStateV2, n_cells: int) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    events_out: dict[str, dict[str, list[int]]] = {}
    for bid, bucket in sorted(state.events_by_bucket.items(), key=lambda kv: kv[0]):
        events_out[str(bid)] = {str(i): [int(c), int(n)] for i, (c, n) in bucket.items() if c or n}

    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "n_cells": int(n_cells),
                "processed_bytes": int(state.processed_bytes),
                "bucket_minutes": int(state.bucket_minutes),
                "last_ema_bucket_id": int(state.last_ema_bucket_id),
                "ema_coll": state.ema_coll.astype(float).tolist(),
                "ema_inc": state.ema_inc.astype(float).tolist(),
                "last_coll_bucket_id": state.last_coll_bucket_id.astype(int).tolist(),
                "last_inc_bucket_id": state.last_inc_bucket_id.astype(int).tolist(),
                "events_by_bucket": events_out,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def update_ar_state_v2(
    *,
    state: ArStateV2,
    incidents_ndjson: Path,
    cell_to_idx: dict[str, int],
    h3_resolution: int,
    datetime_format: str,
    ema_half_life_hours: float,
    lookback_hours: list[int],
    bucket_minutes: int,
    tz_name: str = _DEFAULT_TZ,
) -> None:
    if not incidents_ndjson.exists():
        return
    import h3  # venv dependency

    bucket_min = int(bucket_minutes)
    bucket_hours = float(bucket_min) / 60.0
    lookbacks = [int(h) for h in lookback_hours if int(h) > 0]
    max_steps = max([max(1, int(math.ceil(float(h) / max(bucket_hours, 1e-9)))) for h in lookbacks], default=0)

    def apply_bucket(bid: int, bucket: dict[int, tuple[int, int]]) -> None:
        if state.last_ema_bucket_id >= 0 and ema_half_life_hours > 0:
            dt_hours = float(bid - int(state.last_ema_bucket_id)) * bucket_hours
            if dt_hours > 0:
                decay = float(math.exp(-math.log(2.0) * (dt_hours / float(ema_half_life_hours))))
                state.ema_coll *= decay
                state.ema_inc *= decay

        if state.last_ema_bucket_id < 0:
            state.last_ema_bucket_id = int(bid)
        else:
            state.last_ema_bucket_id = max(int(state.last_ema_bucket_id), int(bid))

        for cell_idx, (c, n) in bucket.items():
            if c:
                state.ema_coll[cell_idx] += float(c)
                state.last_coll_bucket_id[cell_idx] = int(bid)
            if n:
                state.ema_inc[cell_idx] += float(n)
                state.last_inc_bucket_id[cell_idx] = int(bid)

        cur = state.events_by_bucket.get(bid) or {}
        for cell_idx, (c, n) in bucket.items():
            pc, pn = cur.get(cell_idx, (0, 0))
            cur[cell_idx] = (int(pc + c), int(pn + n))
        if cur:
            state.events_by_bucket[bid] = cur

        if max_steps > 0:
            keep_after = bid - max_steps - 2
            for old in [k for k in state.events_by_bucket.keys() if k < keep_after]:
                del state.events_by_bucket[old]

    by_bucket: dict[int, dict[int, list[int]]] = {}

    with incidents_ndjson.open("rb") as f:
        f.seek(max(int(state.processed_bytes), 0))
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            state.processed_bytes = int(f.tell())
            try:
                rec = json.loads(line.decode("utf-8"))
            except Exception:
                continue

            pub_local = str(rec.get("published_date_local") or "").strip()
            if not pub_local:
                continue
            try:
                dt = _parse_dt_local(pub_local, datetime_format=datetime_format)
            except ValueError:
                continue
            bucket_dt = floor_dt(dt, bucket_minutes=bucket_min)
            bid = _bucket_id(bucket_dt, tz_name=tz_name, bucket_minutes=bucket_min)

            lat = rec.get("latitude")
            lon = rec.get("longitude")
            try:
                lat_f = float(lat)
                lon_f = float(lon)
            except Exception:
                continue
            if not (lat_f == lat_f and lon_f == lon_f):
                continue

            cell = _latlng_to_cell(h3, lat=lat_f, lon=lon_f, res=h3_resolution)
            cell_idx = cell_to_idx.get(cell)
            if cell_idx is None:
                continue

            is_coll = _is_collision_issue(str(rec.get("issue_reported") or ""))
            d = by_bucket.get(bid)
            if d is None:
                d = {}
                by_bucket[bid] = d
            pair = d.get(cell_idx)
            if pair is None:
                pair = [0, 0]
                d[cell_idx] = pair
            if is_coll:
                pair[0] += 1
            else:
                pair[1] += 1

            _ = pos  # keep local var used (future: debugging)

    for bid in sorted(by_bucket.keys()):
        bucket = {cell_idx: (int(c), int(n)) for cell_idx, (c, n) in by_bucket[bid].items() if c or n}
        if bucket:
            apply_bucket(bid, bucket)


def compute_ar_features_v2(
    *,
    state: ArStateV2,
    target_bucket: datetime,
    n_cells: int,
    lookback_hours: list[int],
    ema_half_life_hours: float,
    bucket_minutes: int,
    tz_name: str = _DEFAULT_TZ,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    bucket_min = int(bucket_minutes)
    bucket_hours = float(bucket_min) / 60.0
    target_dt = floor_dt(target_bucket, bucket_minutes=bucket_min)
    target_bid = _bucket_id(target_dt, tz_name=tz_name, bucket_minutes=bucket_min)

    # EMA decayed to the target hour (no assumed new events).
    ema_coll = state.ema_coll.astype(np.float32, copy=True)
    ema_inc = state.ema_inc.astype(np.float32, copy=True)
    if state.last_ema_bucket_id >= 0 and ema_half_life_hours > 0:
        dt_hours = float(target_bid - int(state.last_ema_bucket_id)) * bucket_hours
        if dt_hours > 0:
            decay = float(math.exp(-math.log(2.0) * (dt_hours / float(ema_half_life_hours))))
            ema_coll *= decay
            ema_inc *= decay
    out["ema_collisions"] = ema_coll
    out["ema_traffic_incidents"] = ema_inc

    # Time since last event (hours).
    last_c = state.last_coll_bucket_id
    last_i = state.last_inc_bucket_id
    hrs_c = np.where(
        last_c >= 0,
        ((target_bid - last_c).astype(np.float32) * float(bucket_hours)),
        np.full(n_cells, 999.0, dtype=np.float32),
    )
    hrs_i = np.where(
        last_i >= 0,
        ((target_bid - last_i).astype(np.float32) * float(bucket_hours)),
        np.full(n_cells, 999.0, dtype=np.float32),
    )
    out["hrs_since_last_collision"] = hrs_c
    out["hrs_since_last_traffic_incident"] = hrs_i

    # Lookbacks (history only): sum over prior hours.
    lookbacks = sorted({int(h) for h in lookback_hours if int(h) > 0})
    for h in lookbacks:
        coll = np.zeros(n_cells, dtype=np.float32)
        inc = np.zeros(n_cells, dtype=np.float32)
        steps = max(1, int(math.ceil(float(h) / max(bucket_hours, 1e-9))))
        for off in range(1, steps + 1):
            bucket = state.events_by_bucket.get(target_bid - off)
            if not bucket:
                continue
            for cell_idx, (c, n) in bucket.items():
                if c:
                    coll[cell_idx] += float(c)
                if n:
                    inc[cell_idx] += float(n)
        out[f"n_collisions_lb_{h}h"] = coll
        out[f"n_traffic_incidents_lb_{h}h"] = inc

    return out


def seed_ar_state_v2_from_silver(
    *,
    state: ArStateV2,
    incidents_csv: Path,
    cell_to_idx: dict[str, int],
    h3_resolution: int,
    datetime_format: str,
    target_bucket: datetime,
    ema_half_life_hours: float,
    lookback_hours: list[int],
    bucket_minutes: int,
    seed_hours: int = 168,
    tz_name: str = _DEFAULT_TZ,
) -> None:
    """
    One-time initialization: backfill a short window from silver incidents so the AR buffer isn't empty on first run.
    """
    import csv
    import h3  # venv dependency

    if not incidents_csv.exists():
        return

    bucket_min = int(bucket_minutes)
    bucket_hours = float(bucket_min) / 60.0
    max_steps = max(
        [max(1, int(math.ceil(float(h) / max(bucket_hours, 1e-9)))) for h in lookback_hours if int(h) > 0],
        default=0,
    )
    target_dt = floor_dt(target_bucket, bucket_minutes=bucket_min)
    target_bid = _bucket_id(target_dt, tz_name=tz_name, bucket_minutes=bucket_min)
    seed_steps = max(1, int(math.ceil(float(seed_hours) / max(bucket_hours, 1e-9))))
    start_bid = target_bid - max(seed_steps, max_steps + 1, 1)

    by_bucket: dict[int, dict[int, list[int]]] = {}
    with incidents_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            dt_raw = (row.get("published_date") or "").strip()
            if not dt_raw:
                continue
            try:
                dt = datetime.strptime(dt_raw, datetime_format)
            except ValueError:
                continue
            bucket_dt = floor_dt(dt, bucket_minutes=bucket_min)
            bid = _bucket_id(bucket_dt, tz_name=tz_name, bucket_minutes=bucket_min)
            if bid < start_bid or bid >= target_bid:
                continue

            lat_s = (row.get("latitude") or "").strip()
            lon_s = (row.get("longitude") or "").strip()
            if not lat_s or not lon_s:
                continue
            try:
                lat = float(lat_s)
                lon = float(lon_s)
            except Exception:
                continue
            if not (lat == lat and lon == lon):
                continue

            cell = _latlng_to_cell(h3, lat=lat, lon=lon, res=h3_resolution)
            cell_idx = cell_to_idx.get(cell)
            if cell_idx is None:
                continue

            event_class = (row.get("event_class") or "").strip()
            issue = (row.get("issue_reported") or "").strip()
            is_coll = event_class == "collision" or _is_collision_issue(issue)

            d = by_bucket.get(bid)
            if d is None:
                d = {}
                by_bucket[bid] = d
            pair = d.get(cell_idx)
            if pair is None:
                pair = [0, 0]
                d[cell_idx] = pair
            if is_coll:
                pair[0] += 1
            else:
                pair[1] += 1

    if not by_bucket:
        return

    for bid in sorted(by_bucket.keys()):
        bucket = {cell_idx: (int(c), int(n)) for cell_idx, (c, n) in by_bucket[bid].items() if c or n}
        if not bucket:
            continue

        if state.last_ema_bucket_id >= 0 and ema_half_life_hours > 0:
            dt_hours = float(bid - int(state.last_ema_bucket_id)) * bucket_hours
            if dt_hours > 0:
                decay = float(math.exp(-math.log(2.0) * (dt_hours / float(ema_half_life_hours))))
                state.ema_coll *= decay
                state.ema_inc *= decay

        state.last_ema_bucket_id = int(bid if state.last_ema_bucket_id < 0 else max(state.last_ema_bucket_id, bid))

        for cell_idx, (c, n) in bucket.items():
            if c:
                state.ema_coll[cell_idx] += float(c)
                state.last_coll_bucket_id[cell_idx] = int(bid)
            if n:
                state.ema_inc[cell_idx] += float(n)
                state.last_inc_bucket_id[cell_idx] = int(bid)

        state.events_by_bucket[bid] = {cell_idx: (int(c), int(n)) for cell_idx, (c, n) in bucket.items() if c or n}

    # prune to the most recent lookback window
    if max_steps > 0:
        keep_after = target_bid - max_steps - 2
        for old in [k for k in state.events_by_bucket.keys() if k < keep_after]:
            del state.events_by_bucket[old]
