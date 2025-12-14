from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from xml.etree.ElementTree import iterparse


@dataclass(frozen=True)
class NetGeoMap:
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def geo_to_xy(self, *, lon: float, lat: float) -> tuple[float, float]:
        x = self.x_min + (lon - self.lon_min) * (self.x_max - self.x_min) / (self.lon_max - self.lon_min)
        y = self.y_min + (lat - self.lat_min) * (self.y_max - self.y_min) / (self.lat_max - self.lat_min)
        return x, y


def _parse_point_wkt(s: str) -> tuple[float, float] | None:
    s = (s or "").strip()
    if not s:
        return None
    left = s.find("(")
    right = s.rfind(")")
    if left == -1 or right == -1 or right <= left:
        return None
    inner = s[left + 1 : right].strip()
    parts = inner.split()
    if len(parts) < 2:
        return None
    try:
        lon = float(parts[0])
        lat = float(parts[1])
    except ValueError:
        return None
    if not (lat == lat and lon == lon):
        return None
    return lat, lon


def _read_net_geomap(net_xml: Path) -> NetGeoMap:
    for _ev, elem in iterparse(net_xml, events=("start",)):
        if elem.tag != "location":
            continue
        conv = (elem.attrib.get("convBoundary") or "").strip()
        orig = (elem.attrib.get("origBoundary") or "").strip()
        if not conv or not orig:
            break
        try:
            x0, y0, x1, y1 = [float(x) for x in conv.split(",")]
            lon0, lat0, lon1, lat1 = [float(x) for x in orig.split(",")]
        except Exception as e:
            raise ValueError(f"Failed to parse net <location> boundaries from {net_xml}") from e
        return NetGeoMap(
            lon_min=min(lon0, lon1),
            lat_min=min(lat0, lat1),
            lon_max=max(lon0, lon1),
            lat_max=max(lat0, lat1),
            x_min=min(x0, x1),
            y_min=min(y0, y1),
            x_max=max(x0, x1),
            y_max=max(y0, y1),
        )
    raise ValueError(f"Missing <location> element with convBoundary/origBoundary in {net_xml}")


def _iter_vehicle_routes(routes_xml: Path):
    for _ev, elem in iterparse(routes_xml, events=("end",)):
        if elem.tag != "vehicle":
            continue
        route_elem = None
        for child in elem:
            if child.tag == "route":
                route_elem = child
                break
        if route_elem is None:
            elem.clear()
            continue
        edges = (route_elem.attrib.get("edges") or "").strip().split()
        if edges:
            yield edges
        elem.clear()


def _source_routes(routes_xml: Path) -> tuple[list[str], dict[str, list[str]]]:
    counts: Counter[str] = Counter()
    first_route: dict[str, list[str]] = {}
    for edges in _iter_vehicle_routes(routes_xml):
        src = edges[0]
        counts[src] += 1
        if src not in first_route:
            first_route[src] = edges
    return [s for s, _n in counts.most_common()], first_route


def _edge_centroids(net_xml: Path, *, edge_ids: set[str]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if not edge_ids:
        return out
    for _ev, elem in iterparse(net_xml, events=("end",)):
        if elem.tag != "edge":
            continue
        edge_id = (elem.attrib.get("id") or "").strip()
        if edge_id not in edge_ids:
            elem.clear()
            continue
        if (elem.attrib.get("function") or "").strip() == "internal":
            elem.clear()
            continue
        lane = None
        for child in elem:
            if child.tag == "lane":
                lane = child
                break
        if lane is None:
            elem.clear()
            continue
        shape = (lane.attrib.get("shape") or "").strip()
        if not shape:
            elem.clear()
            continue
        pts = []
        for part in shape.split():
            try:
                xs, ys = part.split(",")
                pts.append((float(xs), float(ys)))
            except Exception:
                continue
        if not pts:
            elem.clear()
            continue
        x = sum(p[0] for p in pts) / float(len(pts))
        y = sum(p[1] for p in pts) / float(len(pts))
        out[edge_id] = (x, y)
        elem.clear()
    return out


def _read_detector_points(detectors_csv: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    with detectors_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            detid = (row.get("detector_id") or "").strip()
            if not detid or detid in out:
                continue
            pt = _parse_point_wkt((row.get("location") or "").strip())
            if pt is None:
                continue
            lat, lon = pt
            out[detid] = (lat, lon)
    return out


def _estimate_detid_baselines_vph(
    *,
    radar_counts_dir: Path,
    allowed_detids: set[str],
    bin_duration_s: int,
    max_files: int = 0,
) -> tuple[dict[str, float], dict[str, int]]:
    sums: dict[str, float] = defaultdict(float)
    ns: dict[str, int] = defaultdict(int)
    files = sorted([p for p in radar_counts_dir.glob("*.gz") if not p.name.endswith(".part")])
    if int(max_files) > 0:
        files = files[: int(max_files)]
    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                detid = (row.get("detid") or "").strip().strip('"')
                if not detid or detid not in allowed_detids:
                    continue
                try:
                    vol = float(row.get("volume") or 0.0)
                except Exception:
                    continue
                vph = vol * (3600.0 / float(bin_duration_s))
                sums[detid] += float(vph)
                ns[detid] += 1
    out: dict[str, float] = {}
    for detid, total in sums.items():
        n = ns.get(detid, 0)
        if n > 0:
            out[detid] = float(total) / float(n)
    return out, ns


def _safe_id(s: str) -> str:
    out = []
    for ch in (s or ""):
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "x"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Auto-generate sim controls by mapping radar detectors (with coords) to nearest SUMO source edges."
    )
    ap.add_argument("--net-xml", default="sumo/austin/austin.net.xml")
    ap.add_argument("--routes-xml", default="sumo/austin/routes.rou.xml")
    ap.add_argument("--radar-counts-dir", default="data/bronze/austin_traffic_counts/counts/i626-g7ub")
    ap.add_argument("--detectors-csv", default="data/bronze/austin_traffic_counts/lookups/qpuw-8eeb.csv")
    ap.add_argument("--out", default="sim/controls_austin_radar_auto.json")
    ap.add_argument("--control-interval-s", type=float, default=900.0)
    ap.add_argument("--context-steps", type=int, default=12)
    ap.add_argument("--max-controls", type=int, default=32, help="Keep top-N detectors by mean volume (0=all).")
    ap.add_argument("--max-distance-km", type=float, default=3.0, help="Skip detectors whose nearest source edge is farther.")
    ap.add_argument("--bin-duration-s", type=int, default=900)
    ap.add_argument("--max-count-files", type=int, default=0, help="0=scan all radar count files; >0=scan only first N.")
    args = ap.parse_args()

    net_xml = Path(args.net_xml)
    routes_xml = Path(args.routes_xml)
    radar_dir = Path(args.radar_counts_dir)
    detectors_csv = Path(args.detectors_csv)

    geomap = _read_net_geomap(net_xml)
    sources, first_route_by_src = _source_routes(routes_xml)
    source_set = set(sources)
    edge_xy = _edge_centroids(net_xml, edge_ids=source_set)
    sources_xy = [(s, edge_xy[s]) for s in sources if s in edge_xy]
    if not sources_xy:
        raise SystemExit("No source edges found in net XML (check routes/net match).")

    det_points = _read_detector_points(detectors_csv)
    if not det_points:
        raise SystemExit(f"No detector points found in {detectors_csv}")

    baselines_vph, det_n = _estimate_detid_baselines_vph(
        radar_counts_dir=radar_dir,
        allowed_detids=set(det_points.keys()),
        bin_duration_s=int(args.bin_duration_s),
        max_files=int(args.max_count_files),
    )

    detids_sorted = sorted(
        [d for d in baselines_vph.keys() if det_n.get(d, 0) > 0],
        key=lambda d: baselines_vph.get(d, 0.0),
        reverse=True,
    )
    if int(args.max_controls) > 0:
        detids_sorted = detids_sorted[: int(args.max_controls)]

    assigned_edges: set[str] = set()
    controls = []
    assignments = []

    for detid in detids_sorted:
        lat, lon = det_points[detid]
        dx, dy = geomap.geo_to_xy(lon=lon, lat=lat)

        best_edge = None
        best_d2 = None
        for edge_id, (ex, ey) in sources_xy:
            if edge_id in assigned_edges:
                continue
            d2 = (dx - ex) * (dx - ex) + (dy - ey) * (dy - ey)
            if best_d2 is None or d2 < best_d2:
                best_edge = edge_id
                best_d2 = d2

        if best_edge is None or best_d2 is None:
            continue
        dist_km = math.sqrt(best_d2) / 1000.0
        if float(dist_km) > float(args.max_distance_km):
            continue

        route_edges = first_route_by_src.get(best_edge) or [best_edge]
        controls.append(
            {
                "control_id": f"radar_{_safe_id(detid)}",
                "edge_id": best_edge,
                "baseline_veh_per_hour": 0.0,
                "route_edges": route_edges,
                "radar_detids": [detid],
            }
        )
        assignments.append(
            {
                "detid": detid,
                "lat": lat,
                "lon": lon,
                "mean_vph_est": baselines_vph.get(detid, 0.0),
                "rows_used": det_n.get(detid, 0),
                "edge_id": best_edge,
                "distance_km": dist_km,
            }
        )
        assigned_edges.add(best_edge)

    out = {
        "control_interval_s": float(args.control_interval_s),
        "context_steps": int(args.context_steps),
        "controls": controls,
        "meta": {
            "net_xml": str(net_xml),
            "routes_xml": str(routes_xml),
            "radar_counts_dir": str(radar_dir),
            "detectors_csv": str(detectors_csv),
            "max_distance_km": float(args.max_distance_km),
            "max_controls": int(args.max_controls),
            "detectors_with_coords": len(det_points),
            "detectors_with_counts": len(baselines_vph),
            "controls_written": len(controls),
            "assignments": assignments,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[gen_controls_auto] wrote {out_path} controls={len(controls)} sources={len(sources_xy)}", flush=True)


if __name__ == "__main__":
    main()

