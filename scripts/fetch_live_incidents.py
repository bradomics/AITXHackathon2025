from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any

import requests

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


_BASE_URL = "https://data.austintexas.gov"
_DATASET_ID = "dx9v-zd7x"


def _parse_iso_z(s: str) -> datetime:
    s = (s or "").strip()
    if not s:
        raise ValueError("empty datetime")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _format_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    s = dt.isoformat(timespec="milliseconds")
    return s.replace("+00:00", "Z")


def _to_local_string(dt_utc: datetime, *, tz_name: str, out_format: str) -> str:
    if ZoneInfo is None:
        # Fallback: keep UTC as naive (better than failing).
        dt_local = dt_utc.replace(tzinfo=None)
    else:
        tz = ZoneInfo(tz_name)
        dt_local = dt_utc.astimezone(tz).replace(tzinfo=None)
    return dt_local.strftime(out_format)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "aitx-live-incidents/1.0"})
    app_token = (os.getenv("SODA_APP_TOKEN") or "").strip()
    if app_token:
        s.headers.update({"X-App-Token": app_token})
    return s


def _request_json(session: requests.Session, *, url: str, params: dict[str, Any], timeout_s: int = 60) -> list[dict[str, Any]]:
    resp = session.get(url, params=params, timeout=timeout_s)
    resp.raise_for_status()
    try:
        return resp.json()
    finally:
        resp.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch live Austin traffic incidents (Socrata) into a local ndjson buffer")
    ap.add_argument("--out", default=".runtime/hotspot_v2/incidents.ndjson")
    ap.add_argument("--state", default=".runtime/hotspot_v2/incidents_fetch_state.json")
    ap.add_argument("--seed-hours", type=int, default=168, help="Initial backfill window if no state exists")
    ap.add_argument("--tz", default="America/Chicago")
    ap.add_argument("--datetime-format", default="%Y-%m-%d %H:%M:%S")
    ap.add_argument("--limit", type=int, default=50_000)
    args = ap.parse_args()

    out_path = Path(args.out)
    state_path = Path(args.state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    last_published = ""
    last_id = ""
    if state_path.exists():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        last_published = str(st.get("last_published_date") or "")
        last_id = str(st.get("last_traffic_report_id") or "")

    if last_published:
        where = (
            f"published_date > '{last_published}' "
            f"OR (published_date = '{last_published}' AND traffic_report_id > '{last_id}')"
        )
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=int(args.seed_hours))
        where = f"published_date >= '{_format_utc(cutoff)}'"

    url = f"{_BASE_URL}/resource/{_DATASET_ID}.json"
    session = _session()

    offset = 0
    total = 0
    max_pages = 50
    limit = max(int(args.limit), 1)

    newest_published = last_published
    newest_id = last_id

    with out_path.open("a", encoding="utf-8") as f_out:
        for _page in range(max_pages):
            params: dict[str, Any] = {
                "$select": "traffic_report_id,published_date,issue_reported,latitude,longitude",
                "$order": "published_date,traffic_report_id",
                "$where": where,
                "$limit": limit,
                "$offset": offset,
            }
            rows = _request_json(session, url=url, params=params)
            if not rows:
                break

            for row in rows:
                rid = str(row.get("traffic_report_id") or "").strip()
                pub = str(row.get("published_date") or "").strip()
                if not rid or not pub:
                    continue

                try:
                    dt_utc = _parse_iso_z(pub)
                except ValueError:
                    continue

                lat = row.get("latitude")
                lon = row.get("longitude")
                try:
                    lat_f = float(lat) if lat is not None else float("nan")
                    lon_f = float(lon) if lon is not None else float("nan")
                except ValueError:
                    lat_f = float("nan")
                    lon_f = float("nan")

                rec = {
                    "traffic_report_id": rid,
                    "published_date_utc": pub,
                    "published_date_local": _to_local_string(dt_utc, tz_name=str(args.tz), out_format=str(args.datetime_format)),
                    "issue_reported": str(row.get("issue_reported") or "").strip(),
                    "latitude": lat_f,
                    "longitude": lon_f,
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
                newest_published = pub
                newest_id = rid

            if len(rows) < limit:
                break
            offset += limit

    state_path.write_text(
        json.dumps(
            {
                "last_published_date": newest_published,
                "last_traffic_report_id": newest_id,
                "rows_written_total": int(total),
                "updated_at_utc": _format_utc(datetime.now(timezone.utc)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"out": str(out_path), "rows_appended": int(total), "where": where}))


if __name__ == "__main__":
    main()

