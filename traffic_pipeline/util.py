from __future__ import annotations

import math
import re
from datetime import datetime


def floor_dt(dt: datetime, *, bucket_minutes: int) -> datetime:
    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be > 0")
    minutes = (dt.minute // bucket_minutes) * bucket_minutes
    return dt.replace(minute=minutes, second=0, microsecond=0)


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

