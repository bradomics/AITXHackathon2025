from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from xml.etree.ElementTree import iterparse


def _safe_id(s: str) -> str:
    out = []
    for ch in (s or ""):
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "x"


def _iter_vehicle_routes(routes_xml: Path):
    """
    Yields (depart_seconds, route_edges_list) for each <vehicle> in a SUMO route file.
    Supports the common format:
      <vehicle ... depart="..."><route edges="..."/></vehicle>
    """
    for _ev, elem in iterparse(routes_xml, events=("end",)):
        if elem.tag != "vehicle":
            continue
        depart_s = float(elem.attrib.get("depart", "0") or "0")
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
            yield depart_s, edges
        elem.clear()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a sim controls JSON from a SUMO routes.rou.xml file.")
    ap.add_argument("--routes-xml", default="sumo/austin/routes.rou.xml")
    ap.add_argument("--out", default="sim/controls_austin_from_routes.json")
    ap.add_argument("--top-k", type=int, default=50, help="Number of source edges to keep.")
    ap.add_argument("--duration-s", type=float, default=3600.0, help="Simulation duration used for baseline vph.")
    ap.add_argument("--control-interval-s", type=float, default=60.0)
    ap.add_argument("--context-steps", type=int, default=12)
    args = ap.parse_args()

    routes_xml = Path(args.routes_xml)
    if not routes_xml.exists():
        raise FileNotFoundError(routes_xml)

    counts_by_src: Counter[str] = Counter()
    first_route_by_src: dict[str, list[str]] = {}
    max_depart = 0.0

    for depart_s, edges in _iter_vehicle_routes(routes_xml):
        max_depart = max(max_depart, depart_s)
        src = edges[0]
        counts_by_src[src] += 1
        if src not in first_route_by_src:
            first_route_by_src[src] = edges

    if not counts_by_src:
        raise SystemExit(f"No <vehicle><route> entries found in {routes_xml}")

    duration_s = float(args.duration_s)
    if duration_s <= 0:
        duration_s = max_depart if max_depart > 0 else 3600.0
    hours = max(duration_s / 3600.0, 1e-6)

    controls = []
    for i, (src, n) in enumerate(counts_by_src.most_common(max(0, int(args.top_k)))):
        baseline_vph = float(n) / hours
        route_edges = first_route_by_src.get(src) or [src]
        controls.append(
            {
                "control_id": f"cp_{i}_{_safe_id(src)}",
                "edge_id": src,
                "baseline_veh_per_hour": baseline_vph,
                "route_edges": route_edges,
            }
        )

    out = {
        "control_interval_s": float(args.control_interval_s),
        "context_steps": int(args.context_steps),
        "controls": controls,
        "meta": {
            "routes_xml": str(routes_xml),
            "duration_s": duration_s,
            "unique_sources": len(counts_by_src),
            "max_depart_s": max_depart,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[gen_controls] wrote {out_path} controls={len(controls)} sources_total={len(counts_by_src)}")


if __name__ == "__main__":
    main()

