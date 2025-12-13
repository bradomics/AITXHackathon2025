from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ControlPoint:
    control_id: str
    edge_id: str
    baseline_veh_per_hour: float
    route_edges: list[str] | None = None


@dataclass(frozen=True)
class ControlsConfig:
    control_interval_s: float
    context_steps: int
    controls: list[ControlPoint]


def load_controls(path: Path) -> ControlsConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    interval_s = float(raw.get("control_interval_s", 60))
    context_steps = int(raw.get("context_steps", 12))
    controls_raw = raw.get("controls", [])
    controls: list[ControlPoint] = []
    for c in controls_raw:
        controls.append(
            ControlPoint(
                control_id=str(c["control_id"]),
                edge_id=str(c["edge_id"]),
                baseline_veh_per_hour=float(c.get("baseline_veh_per_hour", 0.0)),
                route_edges=list(c.get("route_edges")) if c.get("route_edges") else None,
            )
        )
    return ControlsConfig(control_interval_s=interval_s, context_steps=context_steps, controls=controls)


def how_sin_cos(dt: datetime) -> tuple[float, float]:
    how = dt.weekday() * 24 + dt.hour
    ang = 2.0 * math.pi * (how / 168.0)
    return math.sin(ang), math.cos(ang)

