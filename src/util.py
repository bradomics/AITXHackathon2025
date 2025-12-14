from __future__ import annotations

import math
import re
from datetime import datetime


def floor_dt(dt: datetime, *, bucket_minutes: int) -> datetime:
    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be > 0")
    minutes_since_midnight = dt.hour * 60 + dt.minute
    bucket_start = (minutes_since_midnight // bucket_minutes) * bucket_minutes
    hour = bucket_start // 60
    minute = bucket_start % 60
    return dt.replace(hour=hour, minute=minute, second=0, microsecond=0)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * r_km * math.asin(math.sqrt(a))


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    s = (text or "").strip().lower()
    s = _SLUG_RE.sub("_", s).strip("_")
    return s or "unknown"


_ALT_DT_FORMATS = (
    "%Y %b %d %I:%M:%S %p",  # e.g. "2019 May 31 11:27:00 PM"
)


def parse_dt(value: str, *, datetime_format: str) -> datetime:
    s = (value or "").strip()
    if not s:
        raise ValueError("empty datetime")

    try:
        return datetime.strptime(s, datetime_format)
    except ValueError:
        pass

    for fmt in _ALT_DT_FORMATS:
        if fmt == datetime_format:
            continue
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt
