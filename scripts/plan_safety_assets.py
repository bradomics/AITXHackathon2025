from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class HeatPoint:
    lat: float
    lon: float
    weight: float


_TARGET_BUCKET_RE = re.compile(r"^//\s*target_bucket=(.+)\s*$")
_POINT_RE = re.compile(
    r"""
    \{\s*
      position:\s*\[\s*(?P<lon>-?\d+(?:\.\d+)?)\s*,\s*(?P<lat>-?\d+(?:\.\d+)?)\s*\]\s*,\s*
      weight:\s*(?P<weight>-?\d+(?:\.\d+)?)\s*
    \}\s*,?\s*
    """,
    flags=re.VERBOSE,
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * r_km * math.asin(math.sqrt(a))


def _parse_phase1_heatpoints(path: Path) -> tuple[str, list[HeatPoint], list[HeatPoint]]:
    """
    Parse the TypeScript-like HeatPoint arrays emitted by src/model/infer_hotspot.py.
    Returns: (target_bucket, collision_points, incident_points)
    """
    target_bucket = ""
    collision: list[HeatPoint] = []
    incident: list[HeatPoint] = []

    section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()

        if not target_bucket:
            m = _TARGET_BUCKET_RE.match(line)
            if m:
                target_bucket = m.group(1).strip()

        if line.startswith("const COLLISION_POINTS"):
            section = "collision"
            continue
        if line.startswith("const INCIDENT_POINTS"):
            section = "incident"
            continue
        if line.startswith("];"):
            section = None
            continue
        if section is None:
            continue

        m = _POINT_RE.search(line)
        if not m:
            continue

        pt = HeatPoint(
            lat=float(m.group("lat")),
            lon=float(m.group("lon")),
            weight=float(m.group("weight")),
        )
        if section == "collision":
            collision.append(pt)
        else:
            incident.append(pt)

    return target_bucket, collision, incident


def _assign_assets(
    points: list[HeatPoint],
    *,
    points_per_asset: int,
    min_coverage: int,
    balance_weight: float,
    n_assets: int,
) -> list[set[int]]:
    if not points:
        return []

    n_points = len(points)
    required = n_points * min_coverage
    n_assets = max(int(n_assets), 0)
    if n_assets <= 0:
        n_assets = int(math.ceil(required / points_per_asset))
    n_assets = max(n_assets, min_coverage, 1)

    capacity = n_assets * points_per_asset
    if capacity < required:
        raise ValueError(f"Not enough assets: capacity={capacity} required={required}")

    order = sorted(range(n_points), key=lambda i: points[i].weight, reverse=True)

    # Pairwise distances.
    dist: list[list[float]] = [[0.0 for _ in range(n_points)] for _ in range(n_points)]
    max_dist = 0.0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d = _haversine_km(points[i].lat, points[i].lon, points[j].lat, points[j].lon)
            dist[i][j] = d
            dist[j][i] = d
            if d > max_dist:
                max_dist = d
    if max_dist <= 0:
        max_dist = 1.0

    total_weight = sum(p.weight for p in points)
    target_cluster_weight = (total_weight * min_coverage) / n_assets if n_assets > 0 else 0.0

    asset_points: list[set[int]] = [set() for _ in range(n_assets)]
    asset_weight: list[float] = [0.0 for _ in range(n_assets)]

    def centroid_distance_km(j: int, *, i: int) -> float:
        s = asset_points[j]
        if not s:
            return 0.0
        lat_c = sum(points[k].lat for k in s) / len(s)
        lon_c = sum(points[k].lon for k in s) / len(s)
        return _haversine_km(points[i].lat, points[i].lon, lat_c, lon_c)

    # Ensure each point is covered min_coverage times.
    for i in order:
        for _ in range(min_coverage):
            candidates = [
                j for j in range(n_assets) if len(asset_points[j]) < points_per_asset and i not in asset_points[j]
            ]
            if not candidates:
                raise ValueError(f"Unable to assign point {i} to reach coverage={min_coverage}")

            min_load = min(len(asset_points[j]) for j in candidates)
            candidates = [j for j in candidates if len(asset_points[j]) == min_load]

            best_j: int | None = None
            best_score = float("inf")
            for j in candidates:
                d = centroid_distance_km(j, i=i) / max_dist
                if target_cluster_weight > 0 and balance_weight > 0:
                    new_w = asset_weight[j] + points[i].weight
                    bal = abs(new_w - target_cluster_weight) / target_cluster_weight
                else:
                    bal = 0.0
                score = d + (balance_weight * bal)
                if score < best_score - 1e-12:
                    best_score = score
                    best_j = j

            if best_j is None:
                raise ValueError(f"Unable to assign point {i} to reach coverage={min_coverage}")
            asset_points[best_j].add(i)
            asset_weight[best_j] += points[i].weight

    # Fill remaining capacity (optional redundancy) so each asset covers ~points_per_asset points.
    for j in range(n_assets):
        while len(asset_points[j]) < points_per_asset and len(asset_points[j]) < n_points:
            best_i: int | None = None
            best_score = float("inf")

            target_now = (sum(asset_weight) / n_assets) if n_assets > 0 else 0.0
            denom = target_now if target_now > 0 else 1.0

            for i in order:
                if i in asset_points[j]:
                    continue
                if asset_points[j]:
                    d = min(dist[i][m] for m in asset_points[j]) / max_dist
                else:
                    d = 0.0
                if balance_weight > 0:
                    new_w = asset_weight[j] + points[i].weight
                    bal = abs(new_w - target_now) / denom
                else:
                    bal = 0.0
                score = d + (balance_weight * bal)
                if score < best_score - 1e-12:
                    best_score = score
                    best_i = i

            if best_i is None:
                break
            asset_points[j].add(best_i)
            asset_weight[j] += points[best_i].weight

    return asset_points


def _assets_to_records(
    points: list[HeatPoint],
    *,
    asset_sets: list[set[int]],
    asset_type: str,
    cover_count: list[int],
) -> list[dict[str, object]]:
    if not points or not asset_sets:
        return []

    n = len(points)
    dist: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _haversine_km(points[i].lat, points[i].lon, points[j].lat, points[j].lon)
            dist[i][j] = d
            dist[j][i] = d

    out: list[dict[str, object]] = []
    for j, s in enumerate(asset_sets):
        cluster = list(s)
        if not cluster:
            continue

        medoid = min(cluster, key=lambda m: sum(dist[m][o] for o in cluster))
        center = points[medoid]

        expected_hit = 0.0
        covers = []
        dists = []
        for i in sorted(cluster, key=lambda idx: points[idx].weight, reverse=True):
            p = points[i]
            c = cover_count[i]
            if c > 0:
                expected_hit += p.weight / c
            d = dist[medoid][i]
            dists.append(d)
            covers.append(
                {
                    "lat": round(p.lat, 6),
                    "lon": round(p.lon, 6),
                    "weight": round(p.weight, 6),
                    "distance_km": round(d, 3),
                    "cover_count": int(c),
                }
            )

        mean_dist = (sum(dists) / len(dists)) if dists else 0.0
        out.append(
            {
                "asset_id": f"{asset_type}_{len(out) + 1}",
                "asset_type": asset_type,
                "lat": round(center.lat, 6),
                "lon": round(center.lon, 6),
                "expected_hit": round(expected_hit, 6),
                "mean_distance_km": round(mean_dist, 3),
                "covers": covers,
            }
        )

    return out


def _plan_assets(
    points: list[HeatPoint],
    *,
    asset_type: str,
    points_per_asset: int,
    min_coverage: int,
    balance_weight: float,
    n_assets: int,
) -> list[dict[str, object]]:
    asset_sets = _assign_assets(
        points,
        points_per_asset=points_per_asset,
        min_coverage=min_coverage,
        balance_weight=balance_weight,
        n_assets=n_assets,
    )

    cover_count = [0 for _ in range(len(points))]
    for s in asset_sets:
        for i in s:
            cover_count[i] += 1

    return _assets_to_records(points, asset_sets=asset_sets, asset_type=asset_type, cover_count=cover_count)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan safety asset locations from phase1 hotspot HeatPoints")
    ap.add_argument("--in", dest="in_path", default="output/phase1_output.json")
    ap.add_argument("--out", dest="out_path", default="output/phase1_safety_output.json")
    ap.add_argument("--points-per-asset", type=int, default=4)
    ap.add_argument("--min-coverage", type=int, default=2)
    ap.add_argument("--balance-weight", type=float, default=0.5)
    ap.add_argument("--n-collision-assets", type=int, default=0, help="0=auto")
    ap.add_argument("--n-incident-assets", type=int, default=0, help="0=auto")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"missing input: {in_path}")

    target_bucket, coll, inc = _parse_phase1_heatpoints(in_path)
    ts = datetime.now().isoformat(timespec="seconds")

    type1 = _plan_assets(
        coll,
        asset_type="type1",
        points_per_asset=int(args.points_per_asset),
        min_coverage=int(args.min_coverage),
        balance_weight=float(args.balance_weight),
        n_assets=int(args.n_collision_assets),
    )
    type2 = _plan_assets(
        inc,
        asset_type="type2",
        points_per_asset=int(args.points_per_asset),
        min_coverage=int(args.min_coverage),
        balance_weight=float(args.balance_weight),
        n_assets=int(args.n_incident_assets),
    )

    out = {
        "generated_at": ts,
        "source": str(in_path),
        "target_bucket": target_bucket,
        "params": {
            "points_per_asset": int(args.points_per_asset),
            "min_coverage": int(args.min_coverage),
            "balance_weight": float(args.balance_weight),
            "n_collision_assets": int(args.n_collision_assets) if int(args.n_collision_assets) > 0 else "auto",
            "n_incident_assets": int(args.n_incident_assets) if int(args.n_incident_assets) > 0 else "auto",
        },
        "assets": {
            "type1_collisions": type1,
            "type2_incidents": type2,
        },
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "type1_assets": len(type1), "type2_assets": len(type2)}))


if __name__ == "__main__":
    main()

