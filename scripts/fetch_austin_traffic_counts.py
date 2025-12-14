from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import gzip
import json
import os
from pathlib import Path
import time
from typing import Any

import requests


_BASE_URL = "https://data.austintexas.gov"
_DEFAULT_LIMIT = 50_000


@dataclass(frozen=True)
class SocrataDataset:
    name: str
    dataset_id: str
    date_field: str | None
    stable_order_fields: list[str]
    where_all: str | None = None
    date_has_z_suffix: bool = False


_CAMERA = SocrataDataset(
    name="camera_traffic_counts",
    dataset_id="sh59-i6y9",
    date_field="read_date",
    stable_order_fields=["read_date", "record_id"],
    date_has_z_suffix=False,
)

_RADAR = SocrataDataset(
    name="radar_traffic_counts",
    dataset_id="i626-g7ub",
    date_field="curdatetime",
    stable_order_fields=["curdatetime", "row_id"],
    where_all=(
        "curdatetime >= '2017-01-01T00:00:00.000Z' "
        "AND curdatetime < '2022-01-01T00:00:00.000Z'"
    ),
    date_has_z_suffix=True,
)

_TRAFFIC_DETECTORS = SocrataDataset(
    name="traffic_detectors",
    dataset_id="qpuw-8eeb",
    date_field=None,
    stable_order_fields=["detector_id"],
)

_TRAVEL_SENSORS = SocrataDataset(
    name="travel_sensors",
    dataset_id="6yd9-yz29",
    date_field=None,
    stable_order_fields=["reader_id"],
)


def _parse_iso_datetime(s: str) -> datetime:
    s = (s or "").strip()
    if not s:
        raise ValueError("empty datetime")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _add_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _month_boundaries_inclusive(min_dt: datetime, max_dt: datetime) -> list[date]:
    months: list[date] = []
    cur = _month_start(min_dt.date())
    last = _month_start(max_dt.date())
    while cur <= last:
        months.append(cur)
        cur = _add_month(cur)
    return months


def _format_boundary(d: date, *, with_z: bool) -> str:
    s = f"{d.isoformat()}T00:00:00.000"
    return s + ("Z" if with_z else "")


def _build_where_month(
    ds: SocrataDataset,
    *,
    month_start: date,
    month_end: date,
) -> str:
    if not ds.date_field:
        raise ValueError("dataset has no date_field")

    where = (
        f"{ds.date_field} >= '{_format_boundary(month_start, with_z=ds.date_has_z_suffix)}' "
        f"AND {ds.date_field} < '{_format_boundary(month_end, with_z=ds.date_has_z_suffix)}'"
    )
    if ds.where_all:
        return f"({where}) AND ({ds.where_all})"
    return where


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "hackathon-traffic-fetch/1.0"})
    app_token = (os.getenv("SODA_APP_TOKEN") or "").strip()
    if app_token:
        s.headers.update({"X-App-Token": app_token})
    return s


def _request_with_retries(
    session: requests.Session,
    *,
    url: str,
    params: dict[str, Any] | None = None,
    stream: bool = False,
    timeout_s: int = 120,
    max_attempts: int = 8,
) -> requests.Response:
    backoff_s = 2
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, params=params, stream=stream, timeout=timeout_s)
        except Exception as e:  # noqa: BLE001 (simple retry loop)
            last_exc = e
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, 60)
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            retry_after = (resp.headers.get("Retry-After") or "").strip()
            resp.close()
            if retry_after.isdigit():
                time.sleep(min(int(retry_after), 120))
            else:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 60)
            continue

        resp.raise_for_status()
        return resp

    if last_exc:
        raise last_exc
    raise RuntimeError(f"failed request after {max_attempts} attempts: {url}")


def _soda_json(
    session: requests.Session, *, dataset_id: str, params: dict[str, Any]
) -> list[dict[str, Any]]:
    url = f"{_BASE_URL}/resource/{dataset_id}.json"
    resp = _request_with_retries(session, url=url, params=params, stream=False)
    try:
        return resp.json()
    finally:
        resp.close()


def _soda_count_rows(session: requests.Session, *, ds: SocrataDataset, where: str | None) -> int:
    params: dict[str, Any] = {"$select": "count(1) as n"}
    if where:
        params["$where"] = where
    rows = _soda_json(session, dataset_id=ds.dataset_id, params=params)
    return int(rows[0]["n"])


def _soda_min_max_datetime(
    session: requests.Session,
    *,
    ds: SocrataDataset,
) -> tuple[datetime, datetime, int]:
    if not ds.date_field:
        raise ValueError("dataset has no date_field")

    params: dict[str, Any] = {
        "$select": (
            f"min({ds.date_field}) as min_date, "
            f"max({ds.date_field}) as max_date, "
            "count(1) as n"
        )
    }
    if ds.where_all:
        params["$where"] = ds.where_all

    rows = _soda_json(session, dataset_id=ds.dataset_id, params=params)
    min_dt = _parse_iso_datetime(rows[0]["min_date"])
    max_dt = _parse_iso_datetime(rows[0]["max_date"])
    n = int(rows[0]["n"])
    return min_dt, max_dt, n


def _stream_csv_response(
    resp: requests.Response,
    *,
    out_fp,
    keep_header: bool,
) -> None:
    header_processed = False
    buf = b""

    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        if not chunk:
            continue
        if header_processed:
            out_fp.write(chunk)
            continue

        buf += chunk
        nl = buf.find(b"\n")
        if nl == -1:
            continue

        header_line = buf[: nl + 1]
        rest = buf[nl + 1 :]
        if keep_header:
            out_fp.write(header_line)
        out_fp.write(rest)
        header_processed = True
        buf = b""


def _download_full_csv(
    session: requests.Session,
    *,
    ds: SocrataDataset,
    out_path: Path,
    limit: int,
    overwrite: bool,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = _soda_count_rows(session, ds=ds, where=ds.where_all)
    meta: dict[str, Any] = {
        "dataset_id": ds.dataset_id,
        "rows": row_count,
        "path": str(out_path),
        "where": ds.where_all,
    }

    if out_path.exists() and not overwrite:
        meta["skipped"] = True
        return meta

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    wrote_header = False
    with tmp_path.open("wb") as f_out:
        for offset in range(0, row_count, limit):
            params: dict[str, Any] = {
                "$limit": limit,
                "$offset": offset,
                "$order": ",".join(ds.stable_order_fields),
            }
            if ds.where_all:
                params["$where"] = ds.where_all

            url = f"{_BASE_URL}/resource/{ds.dataset_id}.csv"
            resp = _request_with_retries(session, url=url, params=params, stream=True, timeout_s=300)
            try:
                _stream_csv_response(resp, out_fp=f_out, keep_header=not wrote_header)
                wrote_header = True
            finally:
                resp.close()

    tmp_path.replace(out_path)
    return meta


def _download_month_partitioned_gz(
    session: requests.Session,
    *,
    ds: SocrataDataset,
    out_dir: Path,
    limit: int,
    overwrite: bool,
    start_month: date | None = None,
    end_month: date | None = None,
) -> dict[str, Any]:
    if not ds.date_field:
        raise ValueError("dataset has no date_field")

    min_dt, max_dt, total_rows = _soda_min_max_datetime(session, ds=ds)
    months = _month_boundaries_inclusive(min_dt, max_dt)
    if start_month:
        months = [m for m in months if m >= start_month]
    if end_month:
        months = [m for m in months if m <= end_month]

    meta: dict[str, Any] = {
        "dataset_id": ds.dataset_id,
        "date_field": ds.date_field,
        "where_all": ds.where_all,
        "min_date": min_dt.isoformat(),
        "max_date": max_dt.isoformat(),
        "rows": total_rows,
        "partitions": [],
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    for m in months:
        m_end = _add_month(m)
        where = _build_where_month(ds, month_start=m, month_end=m_end)
        month_rows = _soda_count_rows(session, ds=ds, where=where)

        part_name = f"{ds.dataset_id}_{m.year:04d}-{m.month:02d}.csv.gz"
        out_path = out_dir / part_name

        part_meta = {
            "month": f"{m.year:04d}-{m.month:02d}",
            "rows": month_rows,
            "path": str(out_path),
            "where": where,
        }

        if month_rows == 0:
            meta["partitions"].append(part_meta)
            continue

        if out_path.exists() and not overwrite:
            part_meta["skipped"] = True
            meta["partitions"].append(part_meta)
            continue

        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        if tmp_path.exists():
            tmp_path.unlink()

        print(f"[{ds.name}] {m.year:04d}-{m.month:02d} rows={month_rows} -> {out_path}")

        wrote_header = False
        with gzip.open(tmp_path, "wb", compresslevel=6) as gz_out:
            for offset in range(0, month_rows, limit):
                params: dict[str, Any] = {
                    "$where": where,
                    "$limit": limit,
                    "$offset": offset,
                    "$order": ",".join(ds.stable_order_fields),
                }

                url = f"{_BASE_URL}/resource/{ds.dataset_id}.csv"
                resp = _request_with_retries(session, url=url, params=params, stream=True, timeout_s=300)
                try:
                    _stream_csv_response(resp, out_fp=gz_out, keep_header=not wrote_header)
                    wrote_header = True
                finally:
                    resp.close()

        tmp_path.replace(out_path)
        meta["partitions"].append(part_meta)

    return meta


def _write_view_metadata(session: requests.Session, *, dataset_ids: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset_id in dataset_ids:
        url = f"{_BASE_URL}/api/views/{dataset_id}.json"
        resp = _request_with_retries(session, url=url, stream=False)
        try:
            payload = resp.json()
        finally:
            resp.close()
        out_path = out_dir / f"{dataset_id}.view.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_month_arg(s: str | None) -> date | None:
    if not s:
        return None
    raw = s.strip()
    if not raw:
        return None
    parts = raw.split("-")
    if len(parts) < 2:
        raise ValueError(f"invalid month '{s}' (expected YYYY-MM)")
    return date(int(parts[0]), int(parts[1]), 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download Austin historic traffic counts (camera + radar) and lookup tables from data.austintexas.gov"
    )
    ap.add_argument("--out-dir", default="data/bronze/austin_traffic_counts")
    ap.add_argument("--limit", type=int, default=_DEFAULT_LIMIT)
    ap.add_argument("--overwrite", action="store_true", help="re-download and overwrite existing files")
    ap.add_argument("--start-month", help="inclusive start month for counts, e.g. 2020-01")
    ap.add_argument("--end-month", help="inclusive end month for counts, e.g. 2020-12")
    ap.add_argument("--skip-camera", action="store_true")
    ap.add_argument("--skip-radar", action="store_true")
    ap.add_argument("--skip-lookups", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    limit = int(args.limit)
    overwrite = bool(args.overwrite)
    start_month = _parse_month_arg(args.start_month)
    end_month = _parse_month_arg(args.end_month)

    meta_dir = out_dir / "meta"
    view_dir = meta_dir / "views"
    counts_dir = out_dir / "counts"
    lookups_dir = out_dir / "lookups"

    with _session() as sess:
        _write_view_metadata(
            sess,
            dataset_ids=[
                _CAMERA.dataset_id,
                _RADAR.dataset_id,
                _TRAFFIC_DETECTORS.dataset_id,
                _TRAVEL_SENSORS.dataset_id,
            ],
            out_dir=view_dir,
        )

        detectors_meta: dict[str, Any] | None = None
        sensors_meta: dict[str, Any] | None = None
        if not args.skip_lookups:
            detectors_meta = _download_full_csv(
                sess,
                ds=_TRAFFIC_DETECTORS,
                out_path=lookups_dir / f"{_TRAFFIC_DETECTORS.dataset_id}.csv",
                limit=limit,
                overwrite=overwrite,
            )
            sensors_meta = _download_full_csv(
                sess,
                ds=_TRAVEL_SENSORS,
                out_path=lookups_dir / f"{_TRAVEL_SENSORS.dataset_id}.csv",
                limit=limit,
                overwrite=overwrite,
            )

        camera_meta: dict[str, Any] | None = None
        radar_meta: dict[str, Any] | None = None
        if not args.skip_radar:
            radar_meta = _download_month_partitioned_gz(
                sess,
                ds=_RADAR,
                out_dir=counts_dir / _RADAR.dataset_id,
                limit=limit,
                overwrite=overwrite,
                start_month=start_month,
                end_month=end_month,
            )
        if not args.skip_camera:
            camera_meta = _download_month_partitioned_gz(
                sess,
                ds=_CAMERA,
                out_dir=counts_dir / _CAMERA.dataset_id,
                limit=limit,
                overwrite=overwrite,
                start_month=start_month,
                end_month=end_month,
            )

    manifest = {
        "downloaded_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "base_url": _BASE_URL,
        "camera_counts": camera_meta,
        "radar_counts": radar_meta,
        "traffic_detectors": detectors_meta,
        "travel_sensors": sensors_meta,
        "views_dir": str(view_dir),
    }
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote {meta_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
